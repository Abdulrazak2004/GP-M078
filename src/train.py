"""
Training pipeline for the multi-task Casing RUL Prediction model.

Functions:
  - train_one_epoch()    — single training pass
  - validate()           — validation pass with metrics
  - train_experiment()   — full pipeline: data → model → train → save
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR,
    GRAD_CLIP_NORM, RANDOM_SEED,
    EXPERIMENTS, FEATURES_A, FEATURES_B, NUM_WORKERS,
)
from src.data_loader import prepare_data
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


def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip=GRAD_CLIP_NORM):
    """
    Run one training epoch.

    Returns
    -------
    avg_losses : dict with keys 'total', 'rul', 'cr', 'wt', 'cause', 'forecast'
    """
    model.train()
    accum = {}
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")
    for batch in pbar:
        X, y_rul, y_cr, y_wt, y_cause, y_forecast = [
            b.to(device) for b in batch
        ]

        preds = model(X)
        total_loss, loss_dict = criterion(preds, y_rul, y_cr, y_wt,
                                          y_cause, y_forecast)

        optimizer.zero_grad()
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
def validate(model, loader, criterion, device):
    """
    Run validation pass.

    Returns
    -------
    avg_losses : dict
    mae_rul : float — mean absolute error on RUL
    accuracy_cause : float — classification accuracy for corrosion cause
    """
    model.eval()
    accum = {}
    n_batches = 0

    all_rul_pred, all_rul_true = [], []
    all_cause_pred, all_cause_true = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")
    for batch in pbar:
        X, y_rul, y_cr, y_wt, y_cause, y_forecast = [
            b.to(device) for b in batch
        ]

        preds = model(X)
        _, loss_dict = criterion(preds, y_rul, y_cr, y_wt,
                                 y_cause, y_forecast)

        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1

        all_rul_pred.append(preds["rul"].cpu())
        all_rul_true.append(y_rul.cpu())
        all_cause_pred.append(preds["cause"].argmax(dim=1).cpu())
        all_cause_true.append(y_cause.cpu())

    pbar.close()
    avg_losses = {k: v / n_batches for k, v in accum.items()}

    rul_pred = torch.cat(all_rul_pred)
    rul_true = torch.cat(all_rul_true)
    mae_rul = (rul_pred - rul_true).abs().mean().item()

    cause_pred = torch.cat(all_cause_pred)
    cause_true = torch.cat(all_cause_true)
    accuracy_cause = (cause_pred == cause_true).float().mean().item()

    return avg_losses, mae_rul, accuracy_cause


def train_experiment(name, exp_config, device, output_dir):
    """
    Full training pipeline for a single experiment.

    Parameters
    ----------
    name : str
        Experiment name (e.g. 'exp3_bilstm_optA').
    exp_config : dict
        Experiment config from EXPERIMENTS dict.
    device : torch.device
        Target device.
    output_dir : Path
        Root output directory for this experiment.

    Returns
    -------
    dict with 'best_val_loss', 'best_epoch', 'training_time_s'
    """
    seed_everything()

    exp_dir = Path(output_dir) / name
    model_dir = exp_dir / "models"
    metric_dir = exp_dir / "metrics"
    plot_dir = exp_dir / "plots"
    for d in [model_dir, metric_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Determine num_workers: use NUM_WORKERS for CUDA, 0 otherwise
    num_workers = NUM_WORKERS if device.type == "cuda" else 0

    # Prepare data
    feature_opt = exp_config["features"]
    window_size = exp_config.get("window_size", 30)
    data = prepare_data(
        feature_option=feature_opt,
        window_size=window_size,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    n_features = data["n_features"]

    # Build model
    backbone_name = exp_config["backbone"]
    model = build_model(backbone_name, n_features).to(device)
    print(f"\n[{name}] Model: {backbone_name} | Features: {feature_opt} | "
          f"Window: {window_size} | Device: {device}")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Trainable parameters: {param_count:,}")

    # Optimizer + scheduler + loss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_REDUCE_PATIENCE,
        factor=LR_REDUCE_FACTOR, verbose=False,
    )
    criterion = MultiTaskLoss().to(device)

    # Training loop
    history = {
        "train_loss": [], "val_loss": [],
        "val_mae_rul": [], "val_acc_cause": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    t_start = time.time()

    epoch_pbar = tqdm(range(1, EPOCHS + 1), desc=f"[{name}]",
                      bar_format="{l_bar}{bar:25}{r_bar}")
    for epoch in epoch_pbar:
        # Train
        train_losses = train_one_epoch(model, train_loader, criterion,
                                       optimizer, device)

        # Validate
        val_losses, mae_rul, acc_cause = validate(model, val_loader,
                                                  criterion, device)

        # Scheduler step
        scheduler.step(val_losses["total"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Record
        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_losses["total"])
        history["val_mae_rul"].append(mae_rul)
        history["val_acc_cause"].append(acc_cause)
        history["lr"].append(current_lr)

        # Update epoch progress bar
        star = "*" if val_losses["total"] < best_val_loss else ""
        epoch_pbar.set_postfix_str(
            f"loss={val_losses['total']:.2f}{star} "
            f"MAE={mae_rul:.1f}d "
            f"Acc={acc_cause:.3f} "
            f"LR={current_lr:.1e} "
            f"pat={patience_counter}/{EARLY_STOP_PATIENCE}"
        )

        # Log detailed every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(
                f"[{name}] Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {train_losses['total']:.4f} | "
                f"Val: {val_losses['total']:.4f} | "
                f"MAE_RUL: {mae_rul:.1f} | "
                f"Acc_Cause: {acc_cause:.3f} | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "exp_config": exp_config,
            }, model_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[{name}] Early stopping at epoch {epoch} "
                           f"(best: {best_epoch})")
                break
    epoch_pbar.close()

    training_time = time.time() - t_start
    print(f"[{name}] Training complete in {training_time:.1f}s | "
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

    axes[2].plot(epochs_range, history["val_acc_cause"], color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Validation Accuracy — Cause")

    plt.tight_layout()
    fig.savefig(plot_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "training_time_s": training_time,
        "param_count": param_count,
    }
