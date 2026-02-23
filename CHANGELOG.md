# Changelog — Casing RUL Prediction Pipeline

All notable changes to the pipeline, in reverse chronological order.

---

## [2026-02-23] Physics-Informed Architecture + GPU Optimization

### Architecture Changes

#### Metadata One-Hot Features
- **`src/config.py`**: Added `METADATA_ONEHOT_FEATURES` (6 columns: 3 Reservoir_Type + 3 Casing_Grade)
- **`src/config.py`**: Expanded `FEATURES_A` from 20 → 26 features (14 raw + 6 engineered + 6 metadata)
- **`src/config.py`**: Expanded `FEATURES_B` from 21 → 27 features
- **`src/config.py`**: Added metadata features to `BINARY_FEATURES` (skip MinMaxScaler)
- **`src/data_loader.py`**: Added one-hot encoding in `engineer_features()` for Reservoir_Type (Carbonate/Clastic/Mixed) and Casing_Grade (N80/L80/P110)

#### Physics-Informed CR Head
- **`src/models.py`**: `MultiTaskHeads` now takes dual inputs: `temporal_dim` (from backbone) and `raw_dim` (n_features)
- **`src/models.py`**: CR head changed to standalone MLP: `raw → 64 → 32 → 1` using only raw last-timestep features
- **`src/models.py`**: RUL/WT/Forecast heads unchanged, still use temporal backbone features
- **`src/models.py`**: All 4 backbones (SimpleLSTM, BiLSTMAttention, CNNLSTM, TransformerBackbone) extract `raw_last = x[:, -1, :]` and pass both temporal + raw to heads
- **`src/models.py`**: Removed old CNNLSTM `head_cr_skip` (replaced by physics-informed CR head)

#### WT-CR Physics Consistency Loss
- **`src/config.py`**: Added `MPY_TO_MMDAY = 0.0254 / 365.0` conversion constant
- **`src/config.py`**: Added `HUBER_DELTA_PHYSICS = 1.0` for physics constraint
- **`src/losses.py`**: Added 5th task to Kendall uncertainty weighting: `loss_physics`
- **`src/losses.py`**: Physics constraint: `WT(t) ≈ WT(t-1) - CR(t) * MPY_TO_MMDAY`
- **`src/losses.py`**: Added `log_var_physics` learnable parameter
- **`src/data_loader.py`**: Added `y_wt_prev` (wall thickness at t-1) as 7th element in `CasingDataset`
- **`src/data_loader.py`**: Updated `create_windows_for_well()` to extract WT at t-1
- **`src/train.py`**: 7-element batch unpacking, `y_wt_prev` passed to criterion
- **`src/evaluate.py`**: 7-element batch unpacking in `predict_deterministic()` and `predict_mc_dropout()`

### GPU Performance Optimization

#### VRAM Preloading
- **`src/data_loader.py`**: Added `CasingDataset.to(device)` method to move entire dataset to GPU memory
- **`src/data_loader.py`**: Added `preload_device` parameter to `prepare_fold()` and `prepare_test()`
- **`src/data_loader.py`**: When preloading to VRAM, sets `num_workers=0` and `pin_memory=False` (data already on GPU)
- **`src/train.py`**: Passes `preload_device=device` when CUDA is available

#### Batch Size + Learning Rate Scaling
- **`src/config.py`**: `BATCH_SIZE` increased from 128 → 1024
- **`src/config.py`**: `LEARNING_RATE` scaled from 5e-4 → 2e-3 (linear scaling with batch size)

#### Parallel Fold Training
- **`run_experiment.py`**: Added `--fold N` CLI flag (1-5) to run a single CV fold
- **`src/train.py`**: Added `fold_only` parameter to `train_experiment()` for single-fold mode
- **`src/train.py`**: Single fold saves result to `fold{N}_result.json` and exits without Phase 2
- Enables parallel CV: `python run_experiment.py --experiment exp3_bilstm_optA --gpu 0 --fold 1 &` (one per GPU)

### CLI Improvements

- **`run_experiment.py`**: Added `--epochs N` flag for stress testing (overrides config)
- **`run_experiment.py`**: Added `--batch-size N` flag to override batch size
- **`run_experiment.py`**: Added `--eval-only` flag to re-run evaluation from saved checkpoint without retraining
- **`src/train.py`**: Changed to `import src.config as cfg` so runtime overrides of `cfg.EPOCHS` propagate correctly

### Test Inference During Training
- **`src/train.py`**: `_train_loop()` accepts optional `test_loader` and `test_every` parameters
- **`src/train.py`**: Test MAE (RUL + CR) logged every 5 epochs during Phase 2

### Bug Fixes
- **`src/evaluate.py`**: Fixed `.numpy()` on GPU tensors in `run_naive_baseline()` — added `.cpu()` before `.numpy()` (VRAM preload compatibility)
- **`src/train.py`**: Fixed `param_count` undefined when `fold_only != 1` — changed condition from `fold_i == 0` to `fold_i == fold_indices[0]`
- **`src/train.py`**: Fixed epochs override not taking effect — was importing EPOCHS as value copy, now uses `cfg.EPOCHS` module reference

---

## [2026-02-22] Right-Sized Models + Kendall Loss

### commit `eb1aed2`

- **`src/config.py`**: Reduced LSTM hidden sizes (256/128 → 128/64) for ~100K training windows
- **`src/config.py`**: Added dropout config: `DROPOUT_LSTM=0.2`, `DROPOUT_BILSTM=0.3`, `DROPOUT_HEAD=0.2`
- **`src/losses.py`**: Implemented Kendall et al. (2018) uncertainty weighting with learnable log-variance per task
- **`src/models.py`**: Added CNNLSTM `head_cr_skip` — direct skip connection for CR prediction
- **`src/train.py`**: Added data augmentation (Gaussian jitter + magnitude scaling) on raw sensor features only
- **`src/config.py`**: Added `NUM_RAW_FEATURES = 14` — augmentation only touches first 14 cols
- **`src/config.py`**: Added `AUGMENT_NOISE_STD = 0.01`, `AUGMENT_SCALE_RANGE = 0.05`

---

## [2026-02-22] Batch Size + Worker Tuning

### commits `46d0563`, `4b8c253`

- **`src/config.py`**: Batch size 32 → 256 → 128 (settled at 128 for GPU utilization vs generalization)
- **`src/config.py`**: NUM_WORKERS 0 → 4

---

## [2026-02-22] 500-Well Scale + 5-Fold CV

### commit `0fdab36`

- Scaled dataset from 100 → 500 wells
- Added 5-fold cross-validation in `src/train.py`
- Reduced from 6 → 4 experiments (removed redundant window ablations)
- Set stride = 5 for manageable dataset size (~100K windows)

---

## [2026-02-22] Core Pipeline

### commit `511c172`

- **`src/train.py`**: Full training pipeline with Phase 1 (K-fold CV) + Phase 2 (final model)
- **`src/evaluate.py`**: 12-plot evaluation suite, MC Dropout uncertainty, critical tests
- **`src/models.py`**: 4 architectures (SimpleLSTM, BiLSTMAttention, CNNLSTM, Transformer)
- **`src/losses.py`**: Multi-task Huber loss
- **`src/data_loader.py`**: Sliding window dataset with well-level splits, leakage prevention
- **`src/config.py`**: Central configuration hub
- **`run_experiment.py`**: Single-experiment CLI runner
- **`run_all.py`**: Batch runner for all experiments + comparison
- **`setup_vastai.sh`**: Automated vast.ai GPU setup script

---

## Project Structure

```
GP/
├── src/
│   ├── config.py          # All hyperparameters, paths, feature lists
│   ├── data_loader.py     # CSV loading, preprocessing, windowing, DataLoaders
│   ├── models.py          # Neural architectures (LSTM, BiLSTM, CNN-LSTM, Transformer)
│   ├── losses.py          # Multi-task Kendall loss with physics constraint
│   ├── attention.py       # Temporal attention module
│   ├── train.py           # Training pipeline (5-fold CV + final model)
│   ├── evaluate.py        # Evaluation, plots, critical tests, MC Dropout
│   ├── cause_model.py     # Standalone cause classification (separate from multi-task)
│   └── timesfm_experiment.py  # Google TimesFM baseline experiment
├── data/
│   └── synthetic_corrosion_dataset.csv  # 500 wells, ~365K rows
├── data_generation/
│   └── ...                # Dataset generation scripts
├── outputs/               # All experiment outputs (models, plots, metrics, logs)
│   ├── exp1_lstm_optB/    # Pipeline verification (cheating baseline)
│   ├── exp3_bilstm_optA/  # Primary model (best results)
│   ├── exp4_cnnlstm_optA/ # CNN-LSTM comparison
│   ├── exp5_bilstm_w15/   # Window size ablation
│   ├── comparison/        # Cross-experiment comparison table + plot
│   ├── logs/              # Training logs per experiment
│   └── scalers/           # Saved MinMaxScaler objects
├── run_experiment.py      # Single experiment runner (CLI)
├── run_all.py             # Batch runner for all experiments
├── setup_vastai.sh        # GPU machine setup script
└── requirements.txt       # Python dependencies
```
