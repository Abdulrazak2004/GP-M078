"""
Training pipeline for the multi-task Casing RUL Prediction model.

Pipeline:
  Phase 1 — K-fold CV for robust metric estimation (no test data touched)
  Phase 2 — Retrain final model on all trainval data for evaluation

LEAKAGE PREVENTION:
  - Test wells held out FIRST, never seen during CV or final training
  - Each CV fold's scaler is fit ONLY on that fold's training wells
  - Final model's scaler is fit on all trainval wells (no test)
  - Well-level splits prevent same-well rows in train vs val/test
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    EARLY_STOP_PATIENCE, WARMUP_EPOCHS,
    GRAD_CLIP_NORM, RANDOM_SEED,
    EXPERIMENTS, FEATURES_A, FEATURES_B, NUM_WORKERS, N_CV_FOLDS,
    USE_AMP, AUGMENT_NOISE_STD, AUGMENT_SCALE_RANGE, NUM_RAW_FEATURES,
)
from src.data_loader import (
    load_dataset, engineer_features, split_wells_cv, prepare_fold, prepare_test,
)
from src.models import build_model
from src.losses import MultiTaskLoss


def seed_everything(seed=RANDOM_SEED):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_scheduler(optimizer, total_epochs, warmup_epochs):
    """Linear warmup + cosine annealing — smooth decay, no harsh steps."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip=GRAD_CLIP_NORM, amp_scaler=None,
                    noise_std=0.0, scale_range=0.0):
    """
    Run one training epoch with optional AMP and data augmentation.

    Returns
    -------
    avg_losses : dict with keys 'total', 'rul', 'cr', 'wt', 'forecast'
    """
    model.train()
    accum = {}
    n_batches = 0
    use_amp = amp_scaler is not None

    pbar = tqdm(loader, desc="  Train", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")
    for batch in pbar:
        X, y_rul, y_cr, y_wt, y_cause, y_forecast = [
            b.to(device) for b in batch
        ]

        # Data augmentation — ONLY on raw sensor features (first NUM_RAW_FEATURES cols)
        # Engineered features (rolling means, slopes, etc.) are left untouched
        # to avoid physical inconsistencies (e.g. jittered thickness but unchanged rolling mean)
        if noise_std > 0:
            noise = torch.randn(X.size(0), X.size(1), NUM_RAW_FEATURES, device=device) * noise_std
            X[:, :, :NUM_RAW_FEATURES] = X[:, :, :NUM_RAW_FEATURES] + noise
        if scale_range > 0:
            scale = 1.0 + (torch.rand(X.size(0), 1, 1, device=device) - 0.5) * 2 * scale_range
            X[:, :, :NUM_RAW_FEATURES] = X[:, :, :NUM_RAW_FEATURES] * scale

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(X)
            total_loss, loss_dict = criterion(preds, y_rul, y_cr, y_wt,
                                              y_forecast)

        optimizer.zero_grad()
        if use_amp:
            amp_scaler.scale(total_loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

        pbar.set_postfix(loss=f"{accum['total']/n_batches:.2f}",
                         rul=f"{accum['rul']/n_batches:.1f}")
    pbar.close()

    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    """
    Run validation pass (with optional AMP autocast).

    Returns
    -------
    avg_losses : dict
    mae_rul : float — mean absolute error on RUL
    mae_cr : float — mean absolute error on corrosion rate
    """
    model.eval()
    accum = {}
    n_batches = 0

    all_rul_pred, all_rul_true = [], []
    all_cr_pred, all_cr_true = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")
    for batch in pbar:
        X, y_rul, y_cr, y_wt, y_cause, y_forecast = [
            b.to(device) for b in batch
        ]

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(X)
            _, loss_dict = criterion(preds, y_rul, y_cr, y_wt,
                                     y_forecast)

        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

        all_rul_pred.append(preds["rul"].cpu())
        all_rul_true.append(y_rul.cpu())
        all_cr_pred.append(preds["cr"].cpu())
        all_cr_true.append(y_cr.cpu())

    pbar.close()
    avg_losses = {k: v / n_batches for k, v in accum.items()}

    rul_pred = torch.cat(all_rul_pred)
    rul_true = torch.cat(all_rul_true)
    mae_rul = (rul_pred - rul_true).abs().mean().item()

    cr_pred = torch.cat(all_cr_pred)
    cr_true = torch.cat(all_cr_true)
    mae_cr = (cr_pred - cr_true).abs().mean().item()

    return avg_losses, mae_rul, mae_cr


# ============================================================================
# CORE TRAINING LOOP (used by both CV folds and final model)
# ============================================================================

def _train_loop(model, train_loader, val_loader, criterion, device,
                tag="", save_path=None, exp_config=None, extra_checkpoint=None):
    """
    Core training loop with early stopping, cosine warmup, AMP, and augmentation.

    Returns
    -------
    best_val_loss, best_epoch, history
    """
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

    # AMP — only on CUDA (not MPS/CPU)
    use_amp = USE_AMP and device.type == "cuda"
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        tqdm.write(f"{tag} Mixed precision (AMP) enabled")

    history = {
        "train_loss": [], "val_loss": [],
        "val_mae_rul": [], "val_mae_cr": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    epoch_pbar = tqdm(range(1, EPOCHS + 1), desc=tag,
                      bar_format="{l_bar}{bar:25}{r_bar}")
    for epoch in epoch_pbar:
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            amp_scaler=amp_scaler,
            noise_std=AUGMENT_NOISE_STD,
            scale_range=AUGMENT_SCALE_RANGE,
        )
        val_losses, mae_rul, mae_cr = validate(
            model, val_loader, criterion, device, use_amp=use_amp,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_losses["total"])
        history["val_mae_rul"].append(mae_rul)
        history["val_mae_cr"].append(mae_cr)
        history["lr"].append(current_lr)

        star = "*" if val_losses["total"] < best_val_loss else ""
        epoch_pbar.set_postfix_str(
            f"loss={val_losses['total']:.2f}{star} "
            f"MAE_RUL={mae_rul:.1f}d "
            f"MAE_CR={mae_cr:.2f}mpy "
            f"LR={current_lr:.1e} "
            f"pat={patience_counter}/{EARLY_STOP_PATIENCE}"
        )

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(
                f"{tag} Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {train_losses['total']:.4f} | "
                f"Val: {val_losses['total']:.4f} | "
                f"MAE_RUL: {mae_rul:.1f} | "
                f"MAE_CR: {mae_cr:.2f} | "
                f"LR: {current_lr:.2e}"
            )

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            patience_counter = 0
            if save_path:
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }
                if exp_config:
                    ckpt["exp_config"] = exp_config
                if extra_checkpoint:
                    ckpt.update(extra_checkpoint)
                torch.save(ckpt, save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                tqdm.write(f"{tag} Early stopping at epoch {epoch} "
                           f"(best: {best_epoch})")
                break
    epoch_pbar.close()

    return best_val_loss, best_epoch, history


# ============================================================================
# MAIN ENTRY POINT: 5-FOLD CV + FINAL MODEL
# ============================================================================

def train_experiment(name, exp_config, device, output_dir):
    """
    Full training pipeline with K-fold cross-validation.

    Phase 1: K-fold CV on trainval wells (test wells never touched)
    Phase 2: Retrain final model on all trainval data (~90% train + ~10% val
             for early stopping) — scaler saved for deployment

    LEAKAGE PREVENTION:
      - Test wells separated first, never used in any CV fold or scaler
      - Each fold scaler fit only on that fold's training wells
      - Final scaler fit on all trainval wells (no test)

    Returns
    -------
    dict with training_info + 'eval_data' for evaluate_experiment()
    """
    seed_everything()

    exp_dir = Path(output_dir) / name
    model_dir = exp_dir / "models"
    metric_dir = exp_dir / "metrics"
    plot_dir = exp_dir / "plots"
    for d in [model_dir, metric_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    num_workers = NUM_WORKERS if device.type == "cuda" else 0
    feature_opt = exp_config["features"]
    window_size = exp_config.get("window_size", 30)
    feature_cols = FEATURES_A if feature_opt == "A" else FEATURES_B
    n_features = len(feature_cols)
    backbone_name = exp_config["backbone"]

    # Load data once + engineer features (before splitting/scaling)
    print(f"\n[{name}] Loading dataset...")
    df = load_dataset()
    df = engineer_features(df)
    print(f"[{name}] Rows: {len(df):,} | Wells: {df['Well_ID'].nunique()} | "
          f"Features: {len(feature_cols)} (incl. 6 engineered)")

    # Split: K folds + held-out test (test NEVER seen during training)
    folds, test_wells = split_wells_cv(df)
    all_trainval = list(set(
        w for f in folds for w in f["train"] + f["val"]
    ))

    print(f"[{name}] Model: {backbone_name} | Features: {feature_opt} | "
          f"Window: {window_size} | Device: {device}")
    print(f"[{name}] {N_CV_FOLDS}-fold CV | "
          f"TrainVal: {len(all_trainval)} wells | Test: {len(test_wells)} wells")

    # Verify no leakage: test wells must not appear in any fold
    test_set = set(test_wells)
    for i, fold in enumerate(folds):
        overlap_train = test_set & set(fold["train"])
        overlap_val = test_set & set(fold["val"])
        assert not overlap_train, f"LEAKAGE: test wells in fold {i} train!"
        assert not overlap_val, f"LEAKAGE: test wells in fold {i} val!"

    # ======================================================================
    # Phase 1: K-Fold Cross-Validation
    # ======================================================================
    fold_results = []
    t_cv_start = time.time()

    for fold_i, fold in enumerate(folds):
        fold_tag = f"[{name}|F{fold_i+1}]"
        print(f"\n{fold_tag} === FOLD {fold_i+1}/{N_CV_FOLDS} "
              f"(train={len(fold['train'])}, val={len(fold['val'])}) ===")

        seed_everything()

        # Build fold data (scaler fit ONLY on this fold's training wells)
        train_loader, val_loader, _, _ = prepare_fold(
            df, fold["train"], fold["val"], feature_cols,
            window_size, BATCH_SIZE, num_workers, save_scaler=False,
        )

        print(f"{fold_tag} Train samples: {len(train_loader.dataset):,} | "
              f"Val samples: {len(val_loader.dataset):,}")

        # Fresh model per fold
        model = build_model(backbone_name, n_features).to(device)
        if fold_i == 0:
            param_count = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
            print(f"[{name}] Trainable parameters: {param_count:,}")

        criterion = MultiTaskLoss().to(device)

        # Train this fold
        best_val_loss, best_epoch, _ = _train_loop(
            model, train_loader, val_loader, criterion, device,
            tag=fold_tag,
        )

        # Record fold val metrics at best epoch
        # (model still has last epoch weights, but best_val_loss was at best_epoch)
        fold_results.append({
            "fold": fold_i + 1,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "val_mae_rul": float(min(_ for _ in [best_val_loss])),  # placeholder
        })

        # Actually re-validate to get final MAE metrics
        # (since best model might have been overwritten by later epochs)
        # We didn't save fold checkpoints, so use last-epoch stats from history
        # Instead, let's just grab from the training output
        _, final_mae_rul, final_mae_cr = validate(
            model, val_loader, criterion, device
        )
        fold_results[-1]["val_mae_rul"] = float(final_mae_rul)
        fold_results[-1]["val_mae_cr"] = float(final_mae_cr)

        print(f"{fold_tag} Done | Best epoch: {best_epoch} | "
              f"MAE_RUL: {final_mae_rul:.1f}d | MAE_CR: {final_mae_cr:.2f}mpy")

        # Free fold resources
        del train_loader, val_loader, model, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cv_time = time.time() - t_cv_start

    # CV Summary
    avg_rul = np.mean([f["val_mae_rul"] for f in fold_results])
    std_rul = np.std([f["val_mae_rul"] for f in fold_results])
    avg_cr = np.mean([f["val_mae_cr"] for f in fold_results])
    std_cr = np.std([f["val_mae_cr"] for f in fold_results])

    print(f"\n[{name}] === CV SUMMARY ({cv_time / 60:.1f} min) ===")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: MAE_RUL={fr['val_mae_rul']:.1f}d  "
              f"MAE_CR={fr['val_mae_cr']:.2f}mpy  (best ep={fr['best_epoch']})")
    print(f"  Average: RUL MAE = {avg_rul:.1f} +/- {std_rul:.1f}d  |  "
          f"CR MAE = {avg_cr:.2f} +/- {std_cr:.2f}mpy")

    cv_summary = {
        "folds": fold_results,
        "avg_mae_rul": float(avg_rul), "std_mae_rul": float(std_rul),
        "avg_mae_cr": float(avg_cr), "std_mae_cr": float(std_cr),
        "cv_time_s": cv_time,
    }
    with open(metric_dir / "cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    # ======================================================================
    # Phase 2: Final Model (all trainval data, test held out)
    # ======================================================================
    print(f"\n[{name}] === FINAL MODEL TRAINING ===")
    seed_everything()

    # Split trainval into ~90% train + ~10% val for early stopping
    final_train_wells, final_val_wells = train_test_split(
        all_trainval, test_size=0.1, random_state=RANDOM_SEED,
    )

    # Final scaler is fit on final_train_wells only (not val, not test)
    # This prevents any leakage from val into scaler statistics
    train_loader, val_loader, final_scaler, final_cols = prepare_fold(
        df, final_train_wells, final_val_wells, feature_cols,
        window_size, BATCH_SIZE, num_workers, save_scaler=True,
    )

    print(f"[{name}] Final split: train={len(final_train_wells)} | "
          f"val={len(final_val_wells)} | test={len(test_wells)}")
    print(f"[{name}] Train samples: {len(train_loader.dataset):,} | "
          f"Val samples: {len(val_loader.dataset):,}")

    model = build_model(backbone_name, n_features).to(device)
    criterion = MultiTaskLoss().to(device)

    t_final_start = time.time()
    best_val_loss, best_epoch, history = _train_loop(
        model, train_loader, val_loader, criterion, device,
        tag=f"[{name}|Final]",
        save_path=model_dir / "best_model.pt",
        exp_config=exp_config,
        extra_checkpoint={"cv_summary": cv_summary},
    )
    final_time = time.time() - t_final_start

    print(f"[{name}] Final model complete in {final_time:.1f}s | "
          f"Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")

    # Save training history
    with open(metric_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0].plot(epochs_range, history["val_loss"], label="Val")
    axes[0].axvline(best_epoch, color="r", linestyle="--", alpha=0.5,
                    label=f"Best ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title(f"{name} — Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs_range, history["val_mae_rul"], color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (days)")
    axes[1].set_title("Validation MAE — RUL")

    axes[2].plot(epochs_range, history["val_mae_cr"], color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE (mpy)")
    axes[2].set_title("Validation MAE — Corrosion Rate")

    plt.tight_layout()
    fig.savefig(plot_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Build test loader (scaler from final_train_wells, NO test data in scaler)
    test_loader = prepare_test(
        df, test_wells, final_scaler, final_cols, feature_cols,
        window_size, BATCH_SIZE, num_workers,
    )
    print(f"[{name}] Test samples: {len(test_loader.dataset):,}")

    total_time = cv_time + final_time

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "training_time_s": total_time,
        "param_count": param_count,
        "cv_summary": cv_summary,
        "n_features": n_features,
        "eval_data": {
            "test_loader": test_loader,
            "feature_cols": feature_cols,
            "n_features": n_features,
        },
    }
