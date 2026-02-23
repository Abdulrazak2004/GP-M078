# LSTM Casing RUL Prediction — Project Plan & Test Specification

**Project:** Remaining Useful Life prediction for oil well casings using LSTM  
**Dataset:** `synthetic_corrosion_dataset.csv` — 80 wells, ~85K rows, 16 columns  
**Target:** `RUL_days` (regression)

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Total rows | 84,981 |
| Total wells | 80 |
| Wells that hit failure (RUL=0) | 56 |
| Wells that survived full 2000 days | 24 |
| Days per well | 133–1999 (mean: 1061) |
| Features | 13 operational + 1 binary (Inhibitor) |
| Target | `RUL_days` (continuous, 0–2000) |

### Columns

**Input Features:** `Status`, `Pressure_bar`, `Temp_C`, `pH`, `Water_Cut`, `Flow_Rate_m3d`, `Flow_Velocity_ms`, `Shear_Stress_Pa`, `CO2_Partial_Pressure_bar`, `Fluid_Density_kgm3`, `Corrosion_Rate_mm_yr`, `Current_Thickness_mm`, `Inhibitor_Active`

**Target:** `RUL_days`

**Metadata (exclude from features):** `Well_ID`, `Day`

---

## Phase 1: EDA & Data Validation

### 1.1 Sanity Checks (Automated Tests)

```
TEST-01: No NaN/Inf values in any column
TEST-02: All wells have monotonically increasing Day column
TEST-03: RUL_days is monotonically decreasing per well (except during shut-ins where it may plateau)
TEST-04: Current_Thickness_mm is monotonically non-increasing per well
TEST-05: When Status=0 (shut-in), Flow_Velocity_ms ≈ 0 and Corrosion_Rate ≈ 0
TEST-06: Corrosion_Rate_mm_yr is always ≥ 0
TEST-07: Water_Cut is in [0, 100], pH in [3.5, 7.0], Temp_C in [20, 160]
TEST-08: For failed wells, final row has RUL_days = 0
TEST-09: Current_Thickness_mm never goes negative
TEST-10: Feature correlation matrix — Pressure should correlate with CO2_Partial_Pressure
```

### 1.2 Visualizations

- Per-well trajectory plots (sample 5 wells): Thickness decay, RUL curve, feature evolution
- Feature distributions (histograms) across all wells
- Correlation heatmap (verify physics makes sense)
- RUL distribution — check for class imbalance at low RUL values
- Time-series stationarity check per feature

---

## Phase 2: Feature Engineering & Preprocessing

### 2.1 Feature Decisions

**Drop from model input:**
- `Well_ID` — metadata only (used for splitting)
- `Day` — replaced by sequence position
- `Corrosion_Rate_mm_yr` — **CRITICAL DECISION: This is a near-direct leakage path to RUL.** Two options:
  - **Option A (Realistic):** DROP it. In real deployments, you don't have the true corrosion rate — you measure operational parameters and *predict* degradation. This forces the model to learn the NORSOK physics implicitly.
  - **Option B (Easier):** KEEP it as a feature. The model will lean heavily on it + Current_Thickness. Easier to train, but doesn't generalize to real data.
  - **Recommendation: Option A for the primary model, Option B as a "cheating baseline" to verify the pipeline works.**

**Engineered Features (optional, add if baseline underperforms):**
- `delta_thickness` — daily change in wall thickness (first difference)
- `rolling_mean_CR_7d` — 7-day rolling average of corrosion rate (if kept)
- `rolling_mean_temp_7d` — smoothed temperature
- `pressure_rate_of_change` — first derivative of pressure
- `time_since_last_shutin` — operational context
- `cumulative_shutin_days` — total downtime so far

### 2.2 Normalization

- **Strategy:** Per-feature Min-Max scaling to [0, 1] fitted on TRAINING SET ONLY
- **Alternative:** StandardScaler (zero mean, unit variance) — try both
- **Important:** Fit scaler on train set, transform val/test with same scaler (no leakage)
- Save scaler object with `joblib` for inference

### 2.3 Sequence Windowing

For LSTM, convert each well's time series into overlapping sliding windows:

```
WINDOW_SIZE = 30 days (hyperparameter — try 15, 30, 60)
STRIDE = 1 (every day is a prediction point)

For each well:
  For t in range(WINDOW_SIZE, len(well)):
    X = features[t-WINDOW_SIZE : t]   → shape (WINDOW_SIZE, n_features)
    y = RUL_days[t]                    → scalar
```

- Wells shorter than WINDOW_SIZE are dropped (or zero-padded from the left)
- This produces ~83K samples for window=30

### 2.4 Train / Val / Test Split

**CRITICAL: Split by Well_ID, NOT by row.**

Random row splitting causes temporal leakage (model sees future days of same well in training).

```
Split Ratio: 60 / 20 / 20 by well count
- Train: 48 wells (~51K rows)
- Val:   16 wells (~17K rows)  
- Test:  16 wells (~17K rows)

Stratification: Ensure each split has proportional failed vs. survived wells
- Train: ~34 failed, ~14 survived
- Val:   ~11 failed, ~5 survived
- Test:  ~11 failed, ~5 survived
```

---

## Phase 3: Model Architecture

### 3.1 Baseline — Simple LSTM

```
Input: (batch, WINDOW_SIZE, n_features)
  → LSTM(64 units, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32 units, return_sequences=False)
  → Dropout(0.2)
  → Dense(16, activation='relu')
  → Dense(1, activation='relu')  # RUL ≥ 0
Output: scalar (predicted RUL in days)
```

**Loss:** Huber Loss (delta=100) — robust to outliers in high-RUL regime  
**Optimizer:** Adam, lr=1e-3 with ReduceLROnPlateau  
**Batch Size:** 256  
**Epochs:** 100 with EarlyStopping (patience=15, monitor=val_loss)

### 3.2 Improved — Bidirectional LSTM + Attention

```
Input: (batch, WINDOW_SIZE, n_features)
  → Bidirectional LSTM(64, return_sequences=True)
  → Dropout(0.3)
  → Bidirectional LSTM(32, return_sequences=True)
  → Temporal Attention Layer (learnable weighted sum over timesteps)
  → Dense(32, activation='relu')
  → Dropout(0.2)
  → Dense(1, activation='relu')
```

### 3.3 Comparison — 1D CNN + LSTM Hybrid

```
Input: (batch, WINDOW_SIZE, n_features)
  → Conv1D(32, kernel=3, activation='relu')
  → Conv1D(64, kernel=3, activation='relu')
  → MaxPool1D(2)
  → LSTM(64, return_sequences=False)
  → Dense(32, activation='relu')
  → Dense(1, activation='relu')
```

### 3.4 (Stretch) Transformer Encoder

```
Input: (batch, WINDOW_SIZE, n_features)
  → Positional Encoding
  → TransformerEncoderLayer(d_model=64, nhead=4, dim_ff=128) × 2
  → Global Average Pooling over time
  → Dense(32) → Dense(1)
```

---

## Phase 4: Training Pipeline

### 4.1 Training Script Structure

```
project/
├── data/
│   └── synthetic_corrosion_dataset.csv
├── src/
│   ├── config.py          # All hyperparameters in one place
│   ├── data_loader.py     # Dataset class, windowing, splitting
│   ├── models.py          # All model architectures
│   ├── train.py           # Training loop with logging
│   ├── evaluate.py        # Metrics, plots, error analysis
│   └── utils.py           # Scaler, helpers
├── notebooks/
│   └── 01_EDA.ipynb       # Optional, for exploration
├── outputs/
│   ├── models/            # Saved checkpoints
│   ├── plots/             # Evaluation figures
│   └── metrics/           # JSON/CSV results
└── requirements.txt
```

### 4.2 Framework

- **PyTorch** (preferred for flexibility with custom attention, variable-length sequences)
- Alternatively: TensorFlow/Keras if that's your team's stack
- `torch.utils.data.Dataset` + `DataLoader` for batching
- Mixed precision training (`torch.cuda.amp`) if GPU available

### 4.3 Training Callbacks / Logging

- EarlyStopping on val_loss (patience=15)
- ReduceLROnPlateau (factor=0.5, patience=5)
- Model checkpoint saving (best val_loss)
- TensorBoard or Weights & Biases logging
- Log per-epoch: train_loss, val_loss, val_MAE, val_RMSE

---

## Phase 5: Evaluation & Testing

### 5.1 Metrics

| Metric | Formula | Why |
|--------|---------|-----|
| **MAE** | mean(\|y - ŷ\|) | Primary metric — interpretable in "days off" |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Penalizes large errors more |
| **R²** | 1 - SS_res/SS_tot | Overall fit quality |
| **MAPE** | mean(\|y - ŷ\|/y) × 100 | Percentage error (exclude RUL=0 to avoid div/0) |
| **Late Prediction Rate** | % where ŷ > y by >50 days | Safety-critical: overestimating RUL is dangerous |
| **Early Prediction Rate** | % where ŷ < y by >50 days | Conservative but wasteful |

### 5.2 Expected Performance Benchmarks

These are rough targets based on similar synthetic RUL problems (C-MAPSS, turbofan, etc.):

| Model | Expected Test MAE | Expected R² | Notes |
|-------|-------------------|-------------|-------|
| Naive baseline (linear extrapolation from thickness) | 80–150 days | 0.85–0.90 | Must beat this |
| Simple LSTM (Option B, with CR feature) | 20–50 days | 0.95–0.98 | "Cheating" baseline |
| Simple LSTM (Option A, no CR feature) | 40–80 days | 0.92–0.96 | Realistic target |
| Bi-LSTM + Attention | 30–60 days | 0.94–0.97 | Should improve on baseline |
| CNN-LSTM Hybrid | 30–65 days | 0.93–0.97 | Often competitive |
| Transformer | 25–55 days | 0.94–0.98 | Best if enough data |

**"Good enough" threshold:** MAE < 60 days on test set with Late Prediction Rate < 10%

### 5.3 Evaluation Plots (Generate for each model)

```
PLOT-01: Predicted vs. Actual RUL scatter (45° line = perfect)
PLOT-02: Error distribution histogram (should be centered near 0)
PLOT-03: MAE by RUL bucket (0-200, 200-500, 500-1000, 1000-2000)
         → Models typically struggle at extremes
PLOT-04: Per-well RUL trajectory — overlay predicted vs. actual for 4 test wells
PLOT-05: Training curves (loss vs. epoch for train and val)
PLOT-06: Feature importance via permutation importance or attention weights
PLOT-07: Residual vs. predicted RUL (check for heteroscedasticity)
```

### 5.4 Critical Tests

```
TEST-11: Model predicts RUL ≥ 0 for all samples (relu output)
TEST-12: Test MAE beats naive linear baseline
TEST-13: No single test well has MAE > 200 days (no catastrophic failures)
TEST-14: Late prediction rate < 15% (safety requirement)
TEST-15: Val and test metrics are within 20% of each other (no overfitting to val)
TEST-16: Model performance doesn't degrade for wells with many shut-ins
TEST-17: Ablation — removing each feature one at a time, verify no feature 
         causes >50% MAE increase (model isn't over-dependent on one signal)
TEST-18: Window size sensitivity — MAE should improve from w=15 → w=30, 
         plateau by w=60
```

---

## Phase 6: Execution Order for Claude Code

Here's the step-by-step build order. Each step should be a standalone, testable script:

### Step 1: `src/config.py`
All hyperparameters: window size, batch size, learning rate, split ratios, feature lists, random seed.

### Step 2: `src/data_loader.py`
- Load CSV
- Run sanity checks (TEST-01 through TEST-10)
- Split by Well_ID (stratified by failure status)
- Fit scaler on train, transform all
- Create sliding windows
- Return PyTorch DataLoaders

### Step 3: `src/models.py`
- `NaiveBaseline` — linear extrapolation from thickness slope
- `SimpleLSTM`
- `BiLSTMAttention`
- `CNNLSTM`
- All with consistent `.forward()` interface

### Step 4: `src/train.py`
- Training loop with early stopping
- Logging (tensorboard or print-based)
- Save best model checkpoint
- Support training any model from `models.py`

### Step 5: `src/evaluate.py`
- Load best checkpoint
- Compute all metrics (5.1)
- Generate all plots (5.3)
- Run critical tests (5.4)
- Export results to JSON

### Step 6: `run_experiment.py`
- Master script that runs the full pipeline
- Trains all models, compares them
- Generates final comparison table and plots

---

## Key Gotchas to Watch For

1. **RUL Capping:** Consider capping RUL at a maximum (e.g., 500 days). The model doesn't need to distinguish between 1500 and 1800 days remaining — both mean "far from failure." This is standard practice (see NASA C-MAPSS benchmarks). This dramatically improves learning at the critical low-RUL end.

2. **Asymmetric Loss:** Late predictions (overestimating RUL) are more dangerous than early ones. Consider a custom loss that penalizes late predictions 2× more than early ones.

3. **Shut-in Handling:** During shut-ins (Status=0), features drop to near-zero. The model needs to learn that shut-ins *pause* degradation, not accelerate it. Make sure the window captures the transition in/out of shut-ins.

4. **Thickness as Feature:** `Current_Thickness_mm` is the strongest predictor and is partially a "leaky" feature since it directly encodes cumulative damage. Decide upfront whether to keep it (easier) or drop it (forces learning from operational params only).

5. **Piece-wise Linear RUL Target:** Instead of raw RUL, consider a piece-wise target: `min(RUL, RUL_CAP)`. This flattens the target at high RUL values and concentrates model capacity on the critical degradation phase.
