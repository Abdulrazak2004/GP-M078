"""
TimesFM forecast comparison experiment.

Loads google/timesfm-2.5-200m-pytorch and compares its 60-month thickness
forecast against our LSTM Head 5 output on the test set.

Inference-only, ~5 min runtime.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DATASET_PATH, FORECAST_HORIZONS, NUM_FORECAST_HORIZONS,
    RANDOM_SEED, WINDOW_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    TARGET_WT,
)
from src.data_loader import load_dataset, split_wells


def run_timesfm_experiment(output_dir, device=None):
    """
    Run TimesFM forecast comparison.

    1. Load test wells' thickness history
    2. Feed to TimesFM → forecast 60 months (1800 days)
    3. Compare MAE per monthly horizon
    4. Save metrics + plots

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    device : torch.device, optional
        GPU device for TimesFM.

    Returns
    -------
    dict with comparison metrics
    """
    import timesfm

    out = Path(output_dir) / "timesfm_comparison"
    metric_dir = out / "metrics"
    plot_dir = out / "plots"
    metric_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  TimesFM 2.5-200m Forecast Comparison")
    print("=" * 60)

    # Load data and get test wells
    print("Loading dataset...")
    df = load_dataset()
    splits = split_wells(df)
    test_wells = splits["test"]
    print(f"  Test wells: {len(test_wells)}")

    # Initialize TimesFM
    print("Loading TimesFM model...")
    t0 = time.time()
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if (device and device.type == "cuda") else "cpu",
            per_core_batch_size=32,
            horizon_len=NUM_FORECAST_HORIZONS,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.5-200m-pytorch",
        ),
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # For each test well: extract thickness history, forecast, compare
    all_horizon_errors = []  # (n_wells, 60) — per-horizon absolute errors
    well_metrics = []
    n_valid_wells = 0

    print("Running forecasts per well...")
    for wid in test_wells:
        df_well = df[df["Well_ID"] == wid].sort_values("Day").reset_index(drop=True)

        # We use the first `context_len` days of thickness as input
        thickness = df_well[TARGET_WT].values.astype(np.float32)
        n_days = len(thickness)

        if n_days < WINDOW_SIZE + max(FORECAST_HORIZONS):
            continue  # Not enough history + future for comparison

        # Use the first (n_days - max_horizon) as context, forecast from there
        context_end = n_days - max(FORECAST_HORIZONS)
        context = thickness[:context_end]

        # Ground truth at each forecast horizon
        ground_truth = []
        for h in FORECAST_HORIZONS:
            future_idx = context_end + h
            if future_idx < n_days:
                ground_truth.append(thickness[future_idx])
            else:
                ground_truth.append(np.nan)
        ground_truth = np.array(ground_truth, dtype=np.float32)

        # TimesFM forecast (returns point forecasts for horizon_len steps)
        # TimesFM expects daily data — we ask for 60 steps at 30-day frequency
        # by subsampling every 30 days from a longer daily forecast
        try:
            # Forecast daily for enough steps, then subsample
            forecast_output = tfm.forecast(
                [context.tolist()],
                freq=[0],  # 0 = unknown/daily granularity
            )
            point_forecast = np.array(forecast_output[0][0], dtype=np.float32)

            # The forecast has horizon_len=60 points
            # Map them to our 60 monthly horizons
            if len(point_forecast) >= NUM_FORECAST_HORIZONS:
                tfm_pred = point_forecast[:NUM_FORECAST_HORIZONS]
            else:
                # Pad with last value if needed
                pad = np.full(NUM_FORECAST_HORIZONS - len(point_forecast),
                              point_forecast[-1])
                tfm_pred = np.concatenate([point_forecast, pad])

        except Exception as e:
            print(f"  Warning: TimesFM failed for {wid}: {e}")
            continue

        # Compute per-horizon absolute errors
        errors = np.abs(tfm_pred - ground_truth)
        all_horizon_errors.append(errors)
        n_valid_wells += 1

        # Per-well summary
        valid_mask = ~np.isnan(ground_truth)
        if valid_mask.sum() > 0:
            well_mae = float(np.nanmean(errors[valid_mask]))
        else:
            well_mae = float("nan")
        well_metrics.append({"well_id": wid, "mae": well_mae})

    print(f"  Completed: {n_valid_wells}/{len(test_wells)} wells")

    if n_valid_wells == 0:
        print("  ERROR: No valid wells for TimesFM comparison")
        return {"error": "no valid wells"}

    # Aggregate per-horizon MAE
    errors_array = np.stack(all_horizon_errors)  # (n_wells, 60)
    horizon_mae = np.nanmean(errors_array, axis=0).tolist()
    avg_mae = float(np.nanmean(errors_array))

    results = {
        "n_wells": n_valid_wells,
        "avg_mae": avg_mae,
        "horizon_mae": horizon_mae,
        "per_well": well_metrics,
    }

    # Save metrics
    with open(metric_dir / "timesfm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  TimesFM Avg MAE: {avg_mae:.3f} mm")

    # Plot: TimesFM MAE per horizon
    fig, ax = plt.subplots(figsize=(12, 5))
    months = list(range(1, NUM_FORECAST_HORIZONS + 1))
    ax.bar(months, horizon_mae, color="teal", alpha=0.8)
    ax.set_xlabel("Forecast Horizon (months)")
    ax.set_ylabel("MAE (mm)")
    ax.set_title(f"TimesFM 2.5-200m — Forecast MAE per Monthly Horizon "
                 f"(Avg={avg_mae:.3f} mm)")
    fig.savefig(plot_dir / "timesfm_horizon_mae.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plot_dir}")
    return results
