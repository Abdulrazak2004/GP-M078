#!/usr/bin/env python3
"""
Single experiment CLI runner with 5-fold CV.

Usage:
    # Full run (5-fold CV + final model + evaluation)
    python run_experiment.py --experiment exp3_bilstm_optA --gpu 0

    # Stress test (2 epochs)
    python run_experiment.py --experiment exp1_lstm_optB --gpu 0 --epochs 2

    # Parallel folds across GPUs (run in 4 terminals):
    python run_experiment.py --experiment exp1_lstm_optB --gpu 0 --fold 1 &
    python run_experiment.py --experiment exp1_lstm_optB --gpu 1 --fold 2 &
    python run_experiment.py --experiment exp1_lstm_optB --gpu 2 --fold 3 &
    python run_experiment.py --experiment exp1_lstm_optB --gpu 3 --fold 4 &
    wait
    python run_experiment.py --experiment exp1_lstm_optB --gpu 0 --fold 5
    # Then run without --fold for Phase 2 (final model + eval)

    # Re-run ONLY evaluation (after a crash during eval):
    python run_experiment.py --experiment exp3_bilstm_optA --gpu 0 --eval-only
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
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (e.g. 2 for stress test)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: 1024)")
    parser.add_argument("--fold", type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help="Run only this fold (1-5) for parallel CV across GPUs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load saved checkpoint and run evaluation only")
    args = parser.parse_args()

    # Set CUDA device BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Override config values (must happen before importing train)
    import src.config
    if args.epochs is not None:
        src.config.EPOCHS = args.epochs
    if args.batch_size is not None:
        src.config.BATCH_SIZE = args.batch_size

    import torch

    output_dir = Path(args.output_dir) if args.output_dir else src.config.OUTPUT_DIR
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
    if args.experiment not in src.config.EXPERIMENTS:
        print(f"ERROR: Unknown experiment '{args.experiment}'")
        print(f"Available: {list(src.config.EXPERIMENTS.keys())}")
        sys.exit(1)

    exp_config = src.config.EXPERIMENTS[args.experiment]
    exp_name = args.experiment

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {exp_name}")
    print(f"  {exp_config['description']}")
    print(f"  Backbone: {exp_config['backbone']} | "
          f"Features: {exp_config['features']} | "
          f"Window: {exp_config['window_size']}")
    print(f"  Batch size: {src.config.BATCH_SIZE} | "
          f"Epochs: {src.config.EPOCHS} | "
          f"VRAM preload: {'YES' if torch.cuda.is_available() else 'no'}")
    if args.eval_only:
        print(f"  Mode: EVALUATION ONLY (loading saved checkpoint)")
    elif args.fold:
        print(f"  Mode: SINGLE FOLD {args.fold} (parallel CV)")
    else:
        print(f"  Mode: Full pipeline (5-fold CV + final model + eval)")
    print(f"{'=' * 60}")

    # Import after CUDA_VISIBLE_DEVICES is set
    from src.train import train_experiment, seed_everything
    from src.evaluate import evaluate_experiment
    from src.models import build_model
    from src.data_loader import (
        load_dataset, engineer_features, split_wells_cv,
        prepare_test, fit_scaler,
    )

    seed_everything()

    # --eval-only: skip training, just load checkpoint and evaluate
    if args.eval_only:
        exp_dir = output_dir / exp_name
        ckpt_path = exp_dir / "models" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"ERROR: No checkpoint found at {ckpt_path}")
            sys.exit(1)

        feature_opt = exp_config["features"]
        feature_cols = (src.config.FEATURES_A if feature_opt == "A"
                        else src.config.FEATURES_B)
        n_features = len(feature_cols)
        window_size = exp_config.get("window_size", 30)

        # Load and prepare test data
        print(f"[{exp_name}] Loading dataset for evaluation...")
        df = load_dataset()
        df = engineer_features(df)
        _, test_wells = split_wells_cv(df)

        # Use all non-test wells to fit scaler (same as training)
        all_wells = df["Well_ID"].unique().tolist()
        trainval_wells = [w for w in all_wells if w not in set(test_wells)]
        df_trainval = df[df["Well_ID"].isin(trainval_wells)]
        scaler, cols_to_scale = fit_scaler(df_trainval, feature_cols, save=False)

        preload = device if device.type == "cuda" else None
        test_loader = prepare_test(
            df, test_wells, scaler, cols_to_scale, feature_cols,
            window_size, src.config.BATCH_SIZE, 0,
            preload_device=preload,
        )
        print(f"[{exp_name}] Test samples: {len(test_loader.dataset):,}")

        # Load model
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(exp_config["backbone"], n_features).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[{exp_name}] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

        eval_data = {
            "test_loader": test_loader,
            "feature_cols": feature_cols,
            "n_features": n_features,
        }
        training_info = {
            "best_epoch": checkpoint.get("epoch", 0),
            "best_val_loss": checkpoint.get("val_loss", 0),
        }
        metrics = evaluate_experiment(model, eval_data, device, exp_dir,
                                       exp_name, training_info)

        print(f"\n{'=' * 60}")
        print(f"  {exp_name} EVALUATION COMPLETE")
        print(f"  Test RUL MAE: {metrics['rul_mae']:.1f} days  |  R2: {metrics['rul_r2']:.3f}")
        print(f"  Test CR MAE: {metrics['cr_mae']:.2f} mpy  |  NMAE: {metrics['cr_nmae']:.1f}%")
        print(f"  Test WT MAE: {metrics['wt_mae']:.3f} mm  |  "
              f"Detection: {metrics.get('wt_loss_detection_accuracy', 0):.1%}")
        print(f"{'=' * 60}")
        return

    # Train (5-fold CV + final model)
    training_info = train_experiment(exp_name, exp_config, device, output_dir,
                                     fold_only=args.fold)

    # If single fold, we're done (no Phase 2 / evaluation)
    if args.fold is not None:
        return

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
