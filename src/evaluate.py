"""
Evaluation module for the multi-task Casing RUL Prediction pipeline.

Functions:
  - predict_deterministic()   — standard single-pass inference
  - predict_mc_dropout()      — MC Dropout with uncertainty estimation
  - compute_metrics()         — MAE/RMSE/R² + accuracy/F1 + CI calibration
  - run_naive_baseline()      — linear extrapolation comparison
  - run_critical_tests()      — TEST-11 through TEST-22 pass/fail
  - generate_plots()          — 12 plots (PLOT-01 through PLOT-12)
  - time_inference()          — benchmark inference speed
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MC_DROPOUT_SAMPLES, CI_LOWER_QUANTILE, CI_UPPER_QUANTILE,
    NUM_CAUSE_CLASSES, FORECAST_HORIZONS, NUM_FORECAST_HORIZONS,
    RUL_CAP, FEATURES_A,
)
from src.models import NaiveBaseline
from src.cfi import compute_cfi, cfi_label


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def predict_deterministic(model, loader, device):
    """
    Standard inference — single forward pass.

    Returns dict of numpy arrays:
        'rul', 'cr', 'wt', 'cause_logits', 'forecast',
        'y_rul', 'y_cr', 'y_wt', 'y_cause', 'y_forecast'
    """
    model.eval()
    preds = {"rul": [], "cr": [], "wt": [], "cause_logits": [], "forecast": []}
    targets = {"y_rul": [], "y_cr": [], "y_wt": [], "y_cause": [], "y_forecast": []}

    for batch in tqdm(loader, desc="  Inference", leave=False,
                      bar_format="{l_bar}{bar:20}{r_bar}"):
        X, y_rul, y_cr, y_wt, y_cause, y_forecast = [
            b.to(device) for b in batch
        ]
        out = model(X)

        preds["rul"].append(out["rul"].cpu().numpy())
        preds["cr"].append(out["cr"].cpu().numpy())
        preds["wt"].append(out["wt"].cpu().numpy())
        preds["cause_logits"].append(out["cause"].cpu().numpy())
        preds["forecast"].append(out["forecast"].cpu().numpy())

        targets["y_rul"].append(y_rul.cpu().numpy())
        targets["y_cr"].append(y_cr.cpu().numpy())
        targets["y_wt"].append(y_wt.cpu().numpy())
        targets["y_cause"].append(y_cause.cpu().numpy())
        targets["y_forecast"].append(y_forecast.cpu().numpy())

    result = {}
    for k in preds:
        result[k] = np.concatenate(preds[k])
    for k in targets:
        result[k] = np.concatenate(targets[k])

    return result


def predict_mc_dropout(model, loader, device, n_samples=MC_DROPOUT_SAMPLES):
    """
    MC Dropout inference — n_samples forward passes with dropout enabled.

    Returns dict:
        'rul_mean', 'rul_lower', 'rul_upper',
        'cr_mean', 'wt_mean', 'forecast_mean',
        'cause_probs',
        + all targets
    """
    model.train()  # Keep dropout active

    all_rul_samples = []
    all_cr_samples = []
    all_wt_samples = []
    all_forecast_samples = []
    all_cause_samples = []
    targets_collected = False
    targets = {}

    with torch.no_grad():
        for s in tqdm(range(n_samples), desc="  MC Dropout", leave=False,
                      bar_format="{l_bar}{bar:20}{r_bar}"):
            rul_s, cr_s, wt_s, forecast_s, cause_s = [], [], [], [], []

            for batch in loader:
                X, y_rul, y_cr, y_wt, y_cause, y_forecast = [
                    b.to(device) for b in batch
                ]
                out = model(X)

                rul_s.append(out["rul"].cpu().numpy())
                cr_s.append(out["cr"].cpu().numpy())
                wt_s.append(out["wt"].cpu().numpy())
                forecast_s.append(out["forecast"].cpu().numpy())
                cause_s.append(F.softmax(out["cause"], dim=1).cpu().numpy())

                if not targets_collected:
                    targets.setdefault("y_rul", []).append(y_rul.cpu().numpy())
                    targets.setdefault("y_cr", []).append(y_cr.cpu().numpy())
                    targets.setdefault("y_wt", []).append(y_wt.cpu().numpy())
                    targets.setdefault("y_cause", []).append(y_cause.cpu().numpy())
                    targets.setdefault("y_forecast", []).append(y_forecast.cpu().numpy())

            targets_collected = True
            all_rul_samples.append(np.concatenate(rul_s))
            all_cr_samples.append(np.concatenate(cr_s))
            all_wt_samples.append(np.concatenate(wt_s))
            all_forecast_samples.append(np.concatenate(forecast_s))
            all_cause_samples.append(np.concatenate(cause_s))

    model.eval()

    # Stack: (n_samples, N, ...)
    rul_all = np.stack(all_rul_samples)
    cr_all = np.stack(all_cr_samples)
    wt_all = np.stack(all_wt_samples)
    forecast_all = np.stack(all_forecast_samples)
    cause_all = np.stack(all_cause_samples)

    # Aggregate targets
    for k in targets:
        targets[k] = np.concatenate(targets[k])

    q_lo = CI_LOWER_QUANTILE
    q_hi = CI_UPPER_QUANTILE

    result = {
        "rul_mean": rul_all.mean(axis=0),
        "rul_lower": np.quantile(rul_all, q_lo, axis=0),
        "rul_upper": np.quantile(rul_all, q_hi, axis=0),
        "rul_std": rul_all.std(axis=0),
        "cr_mean": cr_all.mean(axis=0),
        "cr_lower": np.quantile(cr_all, q_lo, axis=0),
        "cr_upper": np.quantile(cr_all, q_hi, axis=0),
        "wt_mean": wt_all.mean(axis=0),
        "wt_lower": np.quantile(wt_all, q_lo, axis=0),
        "wt_upper": np.quantile(wt_all, q_hi, axis=0),
        "forecast_mean": forecast_all.mean(axis=0),
        "cause_probs": cause_all.mean(axis=0),
    }
    result.update(targets)

    return result


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(det_results, mc_results=None):
    """
    Compute all evaluation metrics.

    Returns dict with regression metrics (MAE, RMSE, R²), classification
    metrics (accuracy, F1), and optionally CI calibration.
    """
    metrics = {}

    # --- Regression: RUL ---
    y_true = det_results["y_rul"]
    y_pred = det_results["rul"]
    metrics["rul_mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["rul_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["rul_r2"] = float(r2_score(y_true, y_pred))

    # --- Regression: Corrosion Rate ---
    y_true = det_results["y_cr"]
    y_pred = det_results["cr"]
    metrics["cr_mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["cr_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["cr_r2"] = float(r2_score(y_true, y_pred))

    # --- Regression: Wall Thickness ---
    y_true = det_results["y_wt"]
    y_pred = det_results["wt"]
    metrics["wt_mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["wt_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["wt_r2"] = float(r2_score(y_true, y_pred))

    # --- Regression: Forecast (per-horizon MAE, ignoring NaN) ---
    y_true_f = det_results["y_forecast"]
    y_pred_f = det_results["forecast"]
    horizon_maes = []
    for h in range(NUM_FORECAST_HORIZONS):
        valid = ~np.isnan(y_true_f[:, h])
        if valid.sum() > 0:
            mae_h = float(mean_absolute_error(y_true_f[valid, h], y_pred_f[valid, h]))
        else:
            mae_h = float("nan")
        horizon_maes.append(mae_h)
    metrics["forecast_horizon_mae"] = horizon_maes
    valid_maes = [m for m in horizon_maes if not np.isnan(m)]
    metrics["forecast_avg_mae"] = float(np.mean(valid_maes)) if valid_maes else float("nan")

    # --- Classification: Corrosion Cause ---
    y_true_c = det_results["y_cause"]
    y_pred_c = det_results["cause_logits"].argmax(axis=1)
    metrics["cause_accuracy"] = float(accuracy_score(y_true_c, y_pred_c))
    metrics["cause_f1_macro"] = float(f1_score(y_true_c, y_pred_c, average="macro",
                                               zero_division=0))
    metrics["cause_f1_weighted"] = float(f1_score(y_true_c, y_pred_c, average="weighted",
                                                  zero_division=0))
    metrics["cause_confusion_matrix"] = confusion_matrix(y_true_c, y_pred_c).tolist()

    # --- CI Calibration (if MC dropout results available) ---
    if mc_results is not None:
        rul_true = mc_results["y_rul"]
        rul_lo = mc_results["rul_lower"]
        rul_hi = mc_results["rul_upper"]
        in_ci = ((rul_true >= rul_lo) & (rul_true <= rul_hi))
        metrics["rul_ci_coverage"] = float(in_ci.mean())
        metrics["rul_ci_mean_width"] = float((rul_hi - rul_lo).mean())

    return metrics


# ============================================================================
# NAIVE BASELINE
# ============================================================================

def run_naive_baseline(test_loader, feature_cols):
    """
    Run NaiveBaseline (linear extrapolation) on test data.

    Returns dict with 'rul_pred', 'rul_true', metrics.
    """
    thickness_idx = feature_cols.index("Current_Thickness_mm")
    baseline = NaiveBaseline(thickness_feature_idx=thickness_idx)

    all_X, all_rul = [], []
    for batch in test_loader:
        X = batch[0].numpy()
        y_rul = batch[1].numpy()
        all_X.append(X)
        all_rul.append(y_rul)

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_rul)
    rul_pred = baseline.predict(X_all)

    metrics = {
        "baseline_rul_mae": float(mean_absolute_error(y_all, rul_pred)),
        "baseline_rul_rmse": float(np.sqrt(mean_squared_error(y_all, rul_pred))),
        "baseline_rul_r2": float(r2_score(y_all, rul_pred)),
    }

    return {"rul_pred": rul_pred, "rul_true": y_all, "metrics": metrics}


# ============================================================================
# CRITICAL TESTS (TEST-11 through TEST-22)
# ============================================================================

def run_critical_tests(metrics, mc_results, baseline_metrics, training_info):
    """
    Evaluate pass/fail for critical tests.

    Returns list of dicts: [{id, description, passed, value, threshold}]
    """
    tests = []

    # TEST-11: RUL MAE < 60 days
    tests.append({
        "id": "TEST-11",
        "description": "RUL MAE < 60 days",
        "passed": metrics["rul_mae"] < 60,
        "value": metrics["rul_mae"],
        "threshold": 60,
    })

    # TEST-12: RUL R² > 0.70
    tests.append({
        "id": "TEST-12",
        "description": "RUL R² > 0.70",
        "passed": metrics["rul_r2"] > 0.70,
        "value": metrics["rul_r2"],
        "threshold": 0.70,
    })

    # TEST-13: Corrosion Rate MAE < 2 mpy
    tests.append({
        "id": "TEST-13",
        "description": "Corrosion Rate MAE < 2 mpy",
        "passed": metrics["cr_mae"] < 2.0,
        "value": metrics["cr_mae"],
        "threshold": 2.0,
    })

    # TEST-14: Wall Thickness MAE < 0.5 mm
    tests.append({
        "id": "TEST-14",
        "description": "Wall Thickness MAE < 0.5 mm",
        "passed": metrics["wt_mae"] < 0.5,
        "value": metrics["wt_mae"],
        "threshold": 0.5,
    })

    # TEST-15: Cause Classification Accuracy > 0.60
    tests.append({
        "id": "TEST-15",
        "description": "Cause Classification Accuracy > 60%",
        "passed": metrics["cause_accuracy"] > 0.60,
        "value": metrics["cause_accuracy"],
        "threshold": 0.60,
    })

    # TEST-16: Forecast Avg MAE < 1.0 mm
    forecast_mae = metrics.get("forecast_avg_mae", float("inf"))
    tests.append({
        "id": "TEST-16",
        "description": "Forecast Average MAE < 1.0 mm",
        "passed": forecast_mae < 1.0,
        "value": forecast_mae,
        "threshold": 1.0,
    })

    # TEST-17: 95% CI coverage > 0.85
    ci_coverage = metrics.get("rul_ci_coverage", 0.0)
    tests.append({
        "id": "TEST-17",
        "description": "95% CI Coverage > 85%",
        "passed": ci_coverage > 0.85,
        "value": ci_coverage,
        "threshold": 0.85,
    })

    # TEST-18: Beat NaiveBaseline on RUL MAE
    baseline_mae = baseline_metrics.get("baseline_rul_mae", float("inf"))
    tests.append({
        "id": "TEST-18",
        "description": "RUL MAE < NaiveBaseline MAE",
        "passed": metrics["rul_mae"] < baseline_mae,
        "value": metrics["rul_mae"],
        "threshold": baseline_mae,
    })

    # TEST-19: Training converged (best epoch > 5)
    best_epoch = training_info.get("best_epoch", 0)
    tests.append({
        "id": "TEST-19",
        "description": "Training converged (best epoch > 5)",
        "passed": best_epoch > 5,
        "value": best_epoch,
        "threshold": 5,
    })

    # TEST-20: Cause F1 macro > 0.50
    tests.append({
        "id": "TEST-20",
        "description": "Cause F1 (macro) > 0.50",
        "passed": metrics["cause_f1_macro"] > 0.50,
        "value": metrics["cause_f1_macro"],
        "threshold": 0.50,
    })

    # TEST-21: RUL RMSE < 80 days
    tests.append({
        "id": "TEST-21",
        "description": "RUL RMSE < 80 days",
        "passed": metrics["rul_rmse"] < 80,
        "value": metrics["rul_rmse"],
        "threshold": 80,
    })

    # TEST-22: CR R² > 0.50
    tests.append({
        "id": "TEST-22",
        "description": "Corrosion Rate R² > 0.50",
        "passed": metrics["cr_r2"] > 0.50,
        "value": metrics["cr_r2"],
        "threshold": 0.50,
    })

    return tests


# ============================================================================
# INFERENCE TIMING
# ============================================================================

def time_inference(model, loader, device, n_warmup=3, n_runs=10):
    """
    Benchmark inference speed.

    Returns dict:
        'single_sample_ms': time for a single sample
        'batch_ms': time for a full batch
        'total_samples': number of test samples
        'ms_per_sample': average ms per sample over full test set
    """
    model.eval()

    # Get one batch for single-sample timing
    batch = next(iter(loader))
    X_batch = batch[0].to(device)

    # Single sample
    X_single = X_batch[:1]
    with torch.no_grad():
        for _ in range(n_warmup):
            model(X_single)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_runs):
            model(X_single)
            if device.type == "cuda":
                torch.cuda.synchronize()
        single_ms = (time.time() - t0) / n_runs * 1000

    # Full batch
    with torch.no_grad():
        for _ in range(n_warmup):
            model(X_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_runs):
            model(X_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
        batch_ms = (time.time() - t0) / n_runs * 1000

    # Full test set
    total_samples = 0
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for b in loader:
            X = b[0].to(device)
            model(X)
            total_samples += X.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_ms = (time.time() - t0) * 1000

    return {
        "single_sample_ms": round(single_ms, 3),
        "batch_ms": round(batch_ms, 3),
        "batch_size": int(X_batch.shape[0]),
        "total_samples": total_samples,
        "total_test_ms": round(total_ms, 3),
        "ms_per_sample": round(total_ms / max(total_samples, 1), 3),
    }


# ============================================================================
# PLOTS  (PLOT-01 through PLOT-12)
# ============================================================================

CAUSE_NAMES = ["CO2", "H2S", "MIC", "Erosion", "O2", "Combined"]


def generate_plots(det_results, mc_results, metrics, baseline_results,
                   plot_dir, exp_name=""):
    """Generate 12 diagnostic plots."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_rul = det_results["y_rul"]
    p_rul = det_results["rul"]
    y_cr = det_results["y_cr"]
    p_cr = det_results["cr"]
    y_wt = det_results["y_wt"]
    p_wt = det_results["wt"]
    y_cause = det_results["y_cause"]
    p_cause = det_results["cause_logits"].argmax(axis=1)

    # PLOT-01: RUL — Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_rul, p_rul, alpha=0.15, s=8, c="steelblue")
    lims = [0, max(y_rul.max(), p_rul.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual RUL (days)")
    ax.set_ylabel("Predicted RUL (days)")
    ax.set_title(f"PLOT-01: RUL Predicted vs Actual — {exp_name}")
    ax.text(0.05, 0.92, f"MAE={metrics['rul_mae']:.1f}d  R²={metrics['rul_r2']:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(plot_dir / "plot01_rul_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-02: RUL residual distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    residuals = p_rul - y_rul
    ax.hist(residuals, bins=80, edgecolor="k", alpha=0.7, color="steelblue")
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (Predicted - Actual) days")
    ax.set_ylabel("Count")
    ax.set_title(f"PLOT-02: RUL Residual Distribution — {exp_name}")
    ax.text(0.05, 0.92, f"Mean={residuals.mean():.1f}  Std={residuals.std():.1f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(plot_dir / "plot02_rul_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-03: Corrosion Rate — Predicted vs Actual
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_cr, p_cr, alpha=0.15, s=8, c="darkorange")
    lims = [0, max(y_cr.max(), p_cr.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual CR (mpy)")
    ax.set_ylabel("Predicted CR (mpy)")
    ax.set_title(f"PLOT-03: Corrosion Rate Pred vs Actual — {exp_name}")
    ax.text(0.05, 0.92, f"MAE={metrics['cr_mae']:.2f}  R²={metrics['cr_r2']:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(plot_dir / "plot03_cr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-04: Wall Thickness — Predicted vs Actual
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_wt, p_wt, alpha=0.15, s=8, c="forestgreen")
    lims = [0, max(y_wt.max(), p_wt.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Thickness (mm)")
    ax.set_ylabel("Predicted Thickness (mm)")
    ax.set_title(f"PLOT-04: Wall Thickness Pred vs Actual — {exp_name}")
    ax.text(0.05, 0.92, f"MAE={metrics['wt_mae']:.3f}  R²={metrics['wt_r2']:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(plot_dir / "plot04_wt_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-05: Confusion Matrix — Corrosion Cause
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_cause, p_cause, labels=range(NUM_CAUSE_CLASSES))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CAUSE_NAMES, yticklabels=CAUSE_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"PLOT-05: Corrosion Cause Confusion Matrix — {exp_name}")
    fig.savefig(plot_dir / "plot05_cause_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-06: Forecast MAE per Horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    horizon_months = [h // 30 for h in FORECAST_HORIZONS]
    horizon_maes = metrics["forecast_horizon_mae"]
    valid_idx = [i for i, m in enumerate(horizon_maes) if not np.isnan(m)]
    if valid_idx:
        ax.bar([horizon_months[i] for i in valid_idx],
               [horizon_maes[i] for i in valid_idx],
               color="steelblue", alpha=0.8)
    ax.set_xlabel("Forecast Horizon (months)")
    ax.set_ylabel("MAE (mm)")
    ax.set_title(f"PLOT-06: Forecast MAE per Monthly Horizon — {exp_name}")
    fig.savefig(plot_dir / "plot06_forecast_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-07: MC Dropout — RUL Uncertainty (sorted by actual RUL)
    if mc_results is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        sort_idx = np.argsort(mc_results["y_rul"])
        n_show = min(500, len(sort_idx))
        idx = sort_idx[np.linspace(0, len(sort_idx) - 1, n_show, dtype=int)]
        x = np.arange(n_show)
        ax.fill_between(x, mc_results["rul_lower"][idx],
                        mc_results["rul_upper"][idx],
                        alpha=0.3, color="steelblue", label="95% CI")
        ax.plot(x, mc_results["rul_mean"][idx], "b-", linewidth=0.8,
                label="MC Mean")
        ax.plot(x, mc_results["y_rul"][idx], "r.", markersize=2,
                label="Actual")
        ax.set_xlabel("Test Samples (sorted by actual RUL)")
        ax.set_ylabel("RUL (days)")
        ax.set_title(f"PLOT-07: MC Dropout Uncertainty — {exp_name}")
        ax.legend()
        fig.savefig(plot_dir / "plot07_mc_uncertainty.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    else:
        _save_placeholder(plot_dir / "plot07_mc_uncertainty.png",
                          "PLOT-07: MC Dropout (skipped)")

    # PLOT-08: Uncertainty vs Error scatter
    if mc_results is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        error = np.abs(mc_results["rul_mean"] - mc_results["y_rul"])
        uncertainty = mc_results["rul_std"]
        ax.scatter(uncertainty, error, alpha=0.1, s=5, c="purple")
        ax.set_xlabel("Predictive Std Dev (days)")
        ax.set_ylabel("Absolute Error (days)")
        ax.set_title(f"PLOT-08: Uncertainty vs Error — {exp_name}")
        # Add correlation
        corr = np.corrcoef(uncertainty, error)[0, 1]
        ax.text(0.05, 0.92, f"Corr={corr:.3f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        fig.savefig(plot_dir / "plot08_uncertainty_vs_error.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    else:
        _save_placeholder(plot_dir / "plot08_uncertainty_vs_error.png",
                          "PLOT-08: Uncertainty vs Error (skipped)")

    # PLOT-09: NaiveBaseline vs Model RUL scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if baseline_results is not None:
        bl_pred = baseline_results["rul_pred"]
        bl_true = baseline_results["rul_true"]
        axes[0].scatter(bl_true, bl_pred, alpha=0.15, s=8, c="gray")
        lims = [0, max(bl_true.max(), bl_pred.max()) * 1.05]
        axes[0].plot(lims, lims, "r--")
        bl_mae = baseline_results["metrics"]["baseline_rul_mae"]
        axes[0].set_title(f"NaiveBaseline (MAE={bl_mae:.1f}d)")
    axes[0].set_xlabel("Actual RUL")
    axes[0].set_ylabel("Predicted RUL")

    axes[1].scatter(y_rul, p_rul, alpha=0.15, s=8, c="steelblue")
    lims = [0, max(y_rul.max(), p_rul.max()) * 1.05]
    axes[1].plot(lims, lims, "r--")
    axes[1].set_title(f"Model (MAE={metrics['rul_mae']:.1f}d)")
    axes[1].set_xlabel("Actual RUL")
    axes[1].set_ylabel("Predicted RUL")

    fig.suptitle(f"PLOT-09: NaiveBaseline vs Model — {exp_name}")
    fig.savefig(plot_dir / "plot09_baseline_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # PLOT-10: Per-cause RUL MAE
    fig, ax = plt.subplots(figsize=(8, 5))
    cause_maes = []
    for c in range(NUM_CAUSE_CLASSES):
        mask = y_cause == c
        if mask.sum() > 0:
            mae_c = float(mean_absolute_error(y_rul[mask], p_rul[mask]))
        else:
            mae_c = 0.0
        cause_maes.append(mae_c)
    bars = ax.bar(CAUSE_NAMES, cause_maes, color="steelblue", alpha=0.8)
    ax.set_ylabel("MAE (days)")
    ax.set_title(f"PLOT-10: RUL MAE by Corrosion Cause — {exp_name}")
    for bar, v in zip(bars, cause_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(plot_dir / "plot10_per_cause_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-11: RUL Error by RUL Bucket
    fig, ax = plt.subplots(figsize=(8, 5))
    buckets = [(0, 50), (50, 100), (100, 200), (200, 350), (350, 500)]
    bucket_labels = []
    bucket_maes = []
    for lo, hi in buckets:
        mask = (y_rul >= lo) & (y_rul < hi)
        label = f"{lo}-{hi}"
        bucket_labels.append(label)
        if mask.sum() > 0:
            bucket_maes.append(float(mean_absolute_error(y_rul[mask], p_rul[mask])))
        else:
            bucket_maes.append(0.0)
    ax.bar(bucket_labels, bucket_maes, color="darkorange", alpha=0.8)
    ax.set_xlabel("Actual RUL Bucket (days)")
    ax.set_ylabel("MAE (days)")
    ax.set_title(f"PLOT-11: RUL MAE by RUL Range — {exp_name}")
    fig.savefig(plot_dir / "plot11_rul_bucket_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PLOT-12: Learning Rate Schedule (from training history if available)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, "See loss_curves.png for LR schedule",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title(f"PLOT-12: Placeholder — {exp_name}")
    fig.savefig(plot_dir / "plot12_lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved 12 plots to {plot_dir}")


def _save_placeholder(path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.text(0.5, 0.5, title, ha="center", va="center",
            transform=ax.transAxes, fontsize=14)
    ax.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FULL EVALUATION PIPELINE
# ============================================================================

def evaluate_experiment(model, data, device, output_dir, exp_name,
                        training_info):
    """
    Run complete evaluation pipeline for one experiment.

    Parameters
    ----------
    model : nn.Module (with best weights loaded)
    data : dict from prepare_data()
    device : torch.device
    output_dir : Path — experiment output directory
    exp_name : str
    training_info : dict from train_experiment()

    Returns
    -------
    metrics : dict
    """
    metric_dir = Path(output_dir) / "metrics"
    plot_dir = Path(output_dir) / "plots"
    metric_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    test_loader = data["test_loader"]
    feature_cols = data["feature_cols"]

    print(f"\n[{exp_name}] === EVALUATION ===")

    # Deterministic predictions
    print(f"[{exp_name}] Deterministic inference...")
    det_results = predict_deterministic(model, test_loader, device)

    # MC Dropout
    print(f"[{exp_name}] MC Dropout inference ({MC_DROPOUT_SAMPLES} samples)...")
    mc_results = predict_mc_dropout(model, test_loader, device)

    # Metrics
    print(f"[{exp_name}] Computing metrics...")
    metrics = compute_metrics(det_results, mc_results)

    # Naive Baseline
    print(f"[{exp_name}] Running NaiveBaseline...")
    baseline_results = run_naive_baseline(test_loader, feature_cols)
    metrics.update(baseline_results["metrics"])

    # Critical Tests
    print(f"[{exp_name}] Running critical tests (TEST-11..TEST-22)...")
    critical_tests = run_critical_tests(metrics, mc_results,
                                        baseline_results["metrics"],
                                        training_info)
    n_passed = sum(1 for t in critical_tests if t["passed"])
    print(f"[{exp_name}] Critical tests: {n_passed}/{len(critical_tests)} passed")
    for t in critical_tests:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  {t['id']}: {status} — {t['description']} "
              f"(value={t['value']:.4f}, threshold={t['threshold']})")

    # Inference Timing
    print(f"[{exp_name}] Timing inference...")
    timing = time_inference(model, test_loader, device)
    print(f"[{exp_name}] Inference: {timing['single_sample_ms']:.1f} ms/sample, "
          f"{timing['ms_per_sample']:.3f} ms/sample (batched)")

    # Save metrics
    with open(metric_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(metric_dir / "critical_tests.json", "w") as f:
        json.dump(critical_tests, f, indent=2)
    with open(metric_dir / "inference_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # Generate plots
    print(f"[{exp_name}] Generating plots...")
    generate_plots(det_results, mc_results, metrics, baseline_results,
                   plot_dir, exp_name)

    return metrics
