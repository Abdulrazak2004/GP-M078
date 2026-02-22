#!/usr/bin/env python3
"""
Parallel launcher for all experiments across 4 GPUs.

Uses a work-stealing scheduler: each GPU picks the next experiment from a
priority queue (longest-first). Experiments run as isolated subprocesses
with separate CUDA contexts.

Usage:
    python run_all.py                    # auto-detect GPUs
    python run_all.py --gpus 4           # override GPU count
    python run_all.py --gpus 1           # sequential on GPU 0
"""

import argparse
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch


# Estimated runtimes (minutes) for longest-first scheduling
# 500 wells + 5-fold CV = ~5x longer per experiment
EXPERIMENT_PRIORITY = OrderedDict([
    ("exp3_bilstm_optA", 120),
    ("exp4_cnnlstm_optA", 90),
    ("exp5_bilstm_w15", 90),
    ("exp1_lstm_optB", 60),
    ("timesfm", 10),
])


def run_single_experiment(experiment, gpu_id, output_dir):
    """
    Launch a single experiment as a subprocess.

    Returns (experiment, gpu_id, returncode, elapsed_s)
    """
    cmd = [
        sys.executable, "-u", "run_experiment.py",  # -u for unbuffered output
        "--experiment", experiment,
        "--gpu", str(gpu_id),
        "--output-dir", str(output_dir),
    ]

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment}.log"

    tag = f"[GPU{gpu_id}|{experiment}]"
    print(f"  {tag} Starting → {log_path}")

    t0 = time.time()
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent),
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            log_file.write(line + "\n")
            # Print key lines live (epochs, results, errors)
            if any(k in line for k in ["Epoch", "Early stop", "PASS", "FAIL",
                                        "MAE", "MAPE", "complete", "ERROR",
                                        "Traceback", "Using GPU", "Experiment:",
                                        "parameters", "COMPLETE", "Detection"]):
                print(f"  {tag} {line}")
        proc.wait()
    elapsed = time.time() - t0

    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"  {tag} DONE — {status} in {elapsed / 60:.1f} min")

    return experiment, gpu_id, proc.returncode, elapsed


def build_comparison_table(output_dir):
    """
    After all experiments, build a comparison CSV + PLOT-13.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    experiments = [e for e in EXPERIMENT_PRIORITY if e != "timesfm"]

    for exp_name in experiments:
        metric_path = output_dir / exp_name / "metrics" / "test_metrics.json"
        timing_path = output_dir / exp_name / "metrics" / "inference_timing.json"
        history_path = output_dir / exp_name / "metrics" / "training_history.json"

        if not metric_path.exists():
            print(f"  Warning: No metrics for {exp_name}")
            continue

        with open(metric_path) as f:
            m = json.load(f)

        row = {
            "experiment": exp_name,
            "rul_mae": m.get("rul_mae"),
            "rul_rmse": m.get("rul_rmse"),
            "rul_r2": m.get("rul_r2"),
            "cr_mae": m.get("cr_mae"),
            "cr_mape": m.get("cr_mape"),
            "cr_r2": m.get("cr_r2"),
            "wt_mae": m.get("wt_mae"),
            "wt_r2": m.get("wt_r2"),
            "wt_loss_detect": m.get("wt_loss_detection_accuracy"),
            "forecast_avg_mae": m.get("forecast_avg_mae"),
            "baseline_rul_mae": m.get("baseline_rul_mae"),
        }

        if timing_path.exists():
            with open(timing_path) as f:
                t = json.load(f)
            row["ms_per_sample"] = t.get("ms_per_sample")

        rows.append(row)

    if not rows:
        print("  No experiment results found for comparison.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(comp_dir / "comparison_table.csv", index=False)
    print(f"\n  Comparison table saved: {comp_dir / 'comparison_table.csv'}")
    print(df.to_string(index=False))

    # PLOT-13: Model Comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RUL MAE
    ax = axes[0, 0]
    ax.barh(df["experiment"], df["rul_mae"], color="steelblue")
    # Add baseline line if available
    if "baseline_rul_mae" in df.columns:
        bl = df["baseline_rul_mae"].iloc[0]
        if bl is not None:
            ax.axvline(bl, color="red", linestyle="--", label=f"Baseline ({bl:.0f}d)")
            ax.legend()
    ax.set_xlabel("MAE (days)")
    ax.set_title("RUL MAE")

    # RUL R²
    ax = axes[0, 1]
    ax.barh(df["experiment"], df["rul_r2"], color="forestgreen")
    ax.set_xlabel("R²")
    ax.set_title("RUL R²")

    # CR MAPE (spec S2)
    ax = axes[1, 0]
    if "cr_mape" in df.columns:
        ax.barh(df["experiment"], df["cr_mape"], color="darkorange")
        ax.axvline(5.0, color="red", linestyle="--", label="Spec S2 (5%)")
        ax.legend()
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Corrosion Rate MAPE (spec S2)")

    # Forecast MAE
    ax = axes[1, 1]
    valid = df["forecast_avg_mae"].dropna()
    if len(valid) > 0:
        ax.barh(df.loc[valid.index, "experiment"], valid, color="teal")
    ax.set_xlabel("MAE (mm)")
    ax.set_title("Forecast Avg MAE")

    fig.suptitle("PLOT-13: Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(comp_dir / "plot13_model_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  PLOT-13 saved: {comp_dir / 'plot13_model_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments in parallel")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs (auto-detected if not set)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Auto-detect GPUs
    if args.gpus is not None:
        n_gpus = args.gpus
    elif torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1  # CPU fallback

    from src.config import OUTPUT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = list(EXPERIMENT_PRIORITY.keys())

    print(f"{'=' * 60}")
    print(f"  Parallel Experiment Launcher")
    print(f"  GPUs: {n_gpus} | Experiments: {len(experiments)}")
    for i in range(min(n_gpus, torch.cuda.device_count())):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    t_start = time.time()

    # Work-stealing: use ProcessPoolExecutor with n_gpus workers
    # Each worker gets a GPU ID assigned round-robin as tasks complete
    gpu_queue = list(range(n_gpus))
    futures = {}
    results = []

    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        # Submit initial batch (one per GPU)
        exp_iter = iter(experiments)
        for gpu_id in range(min(n_gpus, len(experiments))):
            exp = next(exp_iter)
            fut = executor.submit(run_single_experiment, exp, gpu_id, output_dir)
            futures[fut] = (exp, gpu_id)

        # As each completes, submit next experiment on freed GPU
        for fut in as_completed(futures):
            exp, gpu_id = futures[fut]
            result = fut.result()
            results.append(result)

            # Try to launch next experiment on this GPU
            try:
                next_exp = next(exp_iter)
                new_fut = executor.submit(run_single_experiment, next_exp,
                                          gpu_id, output_dir)
                futures[new_fut] = (next_exp, gpu_id)
            except StopIteration:
                pass

    total_time = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  All experiments complete in {total_time / 60:.1f} min")
    print(f"{'=' * 60}")

    # Summary
    for exp, gpu, rc, elapsed in sorted(results, key=lambda x: x[0]):
        status = "OK" if rc == 0 else "FAILED"
        print(f"  {exp:25s} | GPU {gpu} | {status:6s} | {elapsed / 60:.1f} min")

    # Build comparison table
    print("\nBuilding comparison table...")
    build_comparison_table(output_dir)

    # Check for failures
    failures = [r for r in results if r[2] != 0]
    if failures:
        print(f"\n  WARNING: {len(failures)} experiment(s) failed!")
        for exp, gpu, rc, _ in failures:
            log_path = output_dir / "logs" / f"{exp}.log"
            print(f"    {exp}: see {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
