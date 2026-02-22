#!/usr/bin/env python3
"""
Single experiment CLI runner with 5-fold CV.

Usage:
    python run_experiment.py --experiment exp3_bilstm_optA --gpu 0
    python run_experiment.py --experiment timesfm --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--experiment", required=True,
                        help="Experiment name (e.g. exp3_bilstm_optA, timesfm)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Set CUDA device BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from src.config import EXPERIMENTS, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Always 0 since CUDA_VISIBLE_DEVICES is set
        print(f"Using GPU: {torch.cuda.get_device_name(0)} "
              f"(physical GPU {args.gpu})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Handle TimesFM experiment separately
    if args.experiment == "timesfm":
        print("\n=== TimesFM Experiment ===")
        from src.timesfm_experiment import run_timesfm_experiment
        results = run_timesfm_experiment(output_dir, device)
        print(f"\nTimesFM completed. Avg MAE: {results.get('avg_mae', 'N/A')}")
        return

    # Standard experiment
    if args.experiment not in EXPERIMENTS:
        print(f"ERROR: Unknown experiment '{args.experiment}'")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    exp_config = EXPERIMENTS[args.experiment]
    exp_name = args.experiment

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {exp_name}")
    print(f"  {exp_config['description']}")
    print(f"  Backbone: {exp_config['backbone']} | "
          f"Features: {exp_config['features']} | "
          f"Window: {exp_config['window_size']}")
    print(f"  Batch size: {BATCH_SIZE} | 5-fold CV enabled")
    print(f"{'=' * 60}")

    # Import after CUDA_VISIBLE_DEVICES is set
    from src.train import train_experiment, seed_everything
    from src.evaluate import evaluate_experiment
    from src.models import build_model

    seed_everything()

    # Train (5-fold CV + final model)
    training_info = train_experiment(exp_name, exp_config, device, output_dir)

    # Load best model for evaluation
    exp_dir = output_dir / exp_name
    checkpoint = torch.load(exp_dir / "models" / "best_model.pt",
                            map_location=device, weights_only=False)

    model = build_model(
        exp_config["backbone"], training_info["n_features"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate using data from training (test loader already prepared)
    eval_data = training_info["eval_data"]
    metrics = evaluate_experiment(model, eval_data, device, exp_dir, exp_name,
                                  training_info)

    # Print CV summary alongside test results
    cv = training_info.get("cv_summary", {})
    print(f"\n{'=' * 60}")
    print(f"  {exp_name} COMPLETE")
    if cv:
        print(f"  CV RUL MAE: {cv['avg_mae_rul']:.1f} +/- {cv['std_mae_rul']:.1f} days")
        print(f"  CV CR MAE:  {cv['avg_mae_cr']:.2f} +/- {cv['std_mae_cr']:.2f} mpy")
    print(f"  Test RUL MAE: {metrics['rul_mae']:.1f} days  |  R2: {metrics['rul_r2']:.3f}")
    print(f"  Test CR MAE: {metrics['cr_mae']:.2f} mpy  |  NMAE: {metrics['cr_nmae']:.1f}%")
    print(f"  Test WT MAE: {metrics['wt_mae']:.3f} mm  |  Detection: {metrics.get('wt_loss_detection_accuracy', 0):.1%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
