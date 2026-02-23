# HANDOFF DOCUMENT — Casing RUL Prediction Project
## For Claude Code: Full Context & Build Instructions

---

## 1. WHAT THIS PROJECT IS

We are building an **LSTM-based deep learning system** to predict the **Remaining Useful Life (RUL)** of oil well casings. The target user is a field engineer in an oil company who needs to know "how many days until this well's casing fails?" so they can schedule maintenance.

This is a **university capstone / senior project**. The deliverables are:
1. A trained LSTM model that predicts RUL from operational sensor data
2. Evaluation metrics and plots proving it works
3. (Later) A Streamlit dashboard for live visualization

---

## 2. WHAT HAS ALREADY BEEN DONE (DO NOT REDO)

### 2.1 Synthetic Data Generation (COMPLETE)

A full physics-based synthetic data pipeline has been built and executed. The code is in:

**File: `synthetic_corrosion_pipeline.py`** (656 lines)

This script implements:
- **NORSOK M-506 corrosion rate model** — the oil & gas industry standard for CO2 corrosion prediction
- **Formula:** `CR = K_T × f_CO2^0.62 × (S/19)^(0.146 + 0.0324×log(f_CO2)) × f(pH)_T`
- Lookup table interpolation for K_T (temperature-dependent) and f(pH) (2D: temperature × pH)
- Realistic scenario engine: reservoir pressure decline, water breakthrough sigmoid, seasonal temperature cycles, random shut-ins, choke changes
- Corrosion inhibitor effects, pitting factors, sensor noise
- Wall thickness degradation tracking and RUL label computation

**You do NOT need to modify or rerun this script.** The data is already generated.

### 2.2 Generated Dataset (COMPLETE)

**File: `synthetic_corrosion_dataset.csv`** (~8.3 MB)

Dataset stats:
```
Rows:           84,981
Wells:          80 unique wells (Well_ID: WELL-001 through WELL-080)
Failed wells:   56 (hit RUL=0, meaning thickness < 3mm)
Survived wells: 24 (lasted full 2000 days without failure)
Days per well:  133 to 1999 (mean: 1061)
```

**Columns (16 total):**
```
Well_ID                    - string, e.g. "WELL-001" (metadata, don't use as feature)
Day                        - int, 0 to 1999 (metadata, don't use as feature)
Status                     - int, 1=operating, 0=shut-in
Pressure_bar               - float, reservoir pressure (declines over time)
Temp_C                     - float, temperature with seasonal cycling
pH                         - float, formation water pH (~5.0-5.7)
Water_Cut                  - float, water percentage (sigmoid increase over time)
Flow_Rate_m3d              - float, flow rate in m³/day
Flow_Velocity_ms           - float, flow velocity in m/s
Shear_Stress_Pa            - float, wall shear stress
CO2_Partial_Pressure_bar   - float, CO2 partial pressure
Fluid_Density_kgm3         - float, fluid density
Corrosion_Rate_mm_yr       - float, instantaneous corrosion rate from NORSOK model
Current_Thickness_mm       - float, current wall thickness (starts 10-13mm, degrades)
Inhibitor_Active           - int, 1=inhibitor working, 0=inhibitor failed
RUL_days                   - int, REMAINING USEFUL LIFE — THIS IS THE TARGET VARIABLE
```

### 2.3 Visualization Plots (COMPLETE)

Two plots have been generated:
- **`well_history_plot.png`** — Full operational history of WELL-003 (9 subplots showing temp, pressure, water cut, pH, flow velocity, CO2, corrosion rate, thickness, and RUL)
- **`fleet_summary_plot.png`** — Fleet-level summary (thickness trajectories for all 80 wells, corrosion rate distribution, RUL distribution, feature correlation matrix)

These are for reference/verification only. You will generate your own evaluation plots.

### 2.4 Project Plan (COMPLETE)

**File: `LSTM_RUL_Project_Plan.md`**

This contains the full specification for what needs to be built, including:
- 10 data sanity checks (TEST-01 through TEST-10)
- Feature engineering decisions
- Normalization and windowing specs
- 4 model architectures (Naive Baseline, Simple LSTM, Bi-LSTM+Attention, CNN-LSTM)
- Training pipeline structure
- 6 evaluation metrics
- 8 critical tests (TEST-11 through TEST-18)
- 7 evaluation plots
- Expected performance benchmarks
- Key gotchas and decisions

**READ THIS FILE CAREFULLY BEFORE BUILDING ANYTHING.**

---

## 3. WHAT NEEDS TO BE BUILT NOW

Build the following project structure:

```
casing_rul_prediction/
├── data/
│   └── synthetic_corrosion_dataset.csv      ← (provided, just copy here)
├── src/
│   ├── config.py          # All hyperparameters
│   ├── data_loader.py     # Load, validate, split, window, scale
│   ├── models.py          # All model architectures
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Metrics + plots
│   └── utils.py           # Helpers
├── outputs/
│   ├── models/            # Saved .pt checkpoints
│   ├── plots/             # Evaluation figures
│   └── metrics/           # JSON results
├── run_experiment.py      # Master script
└── requirements.txt
```

### 3.1 config.py

All hyperparameters in one place:
```python
RANDOM_SEED = 42
WINDOW_SIZE = 30          # Try 15, 30, 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5
DROPOUT = 0.2
LSTM_HIDDEN_1 = 64
LSTM_HIDDEN_2 = 32
RUL_CAP = 500             # Cap RUL target at 500 days (piece-wise linear)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
HUBER_DELTA = 100

# IMPORTANT FEATURE DECISIONS:
# Option A (realistic): Drop Corrosion_Rate_mm_yr — in production you don't have this
# Option B (cheating baseline): Keep it — model will be very accurate but unrealistic
# Build BOTH variants and compare.

FEATURES_OPTION_A = [
    'Status', 'Pressure_bar', 'Temp_C', 'pH', 'Water_Cut',
    'Flow_Rate_m3d', 'Flow_Velocity_ms', 'Shear_Stress_Pa',
    'CO2_Partial_Pressure_bar', 'Fluid_Density_kgm3',
    'Current_Thickness_mm', 'Inhibitor_Active'
]

FEATURES_OPTION_B = FEATURES_OPTION_A + ['Corrosion_Rate_mm_yr']

TARGET = 'RUL_days'
```

### 3.2 data_loader.py

Must implement:
1. **Load CSV** with pandas
2. **Sanity checks** — run TEST-01 through TEST-10 (see project plan) and print pass/fail
3. **RUL capping** — `RUL = min(RUL, RUL_CAP)` — this is critical for performance
4. **Split by Well_ID** — NOT by row. Stratify so each split has proportional failed vs survived wells:
   - Train: 48 wells
   - Val: 16 wells
   - Test: 16 wells
5. **Min-Max scaling** — fit on train set ONLY, transform val/test with same scaler. Save scaler with joblib.
6. **Sliding window creation:**
   ```
   For each well:
     For t in range(WINDOW_SIZE, len(well)):
       X = features[t-WINDOW_SIZE : t]   → shape (WINDOW_SIZE, n_features)
       y = RUL_days[t]                    → scalar
   ```
7. Return PyTorch `DataLoader` objects for train/val/test

### 3.3 models.py

Implement these architectures (all with consistent interface):

**Model 1: NaiveBaseline**
- Not a neural net. Simple linear extrapolation from the slope of `Current_Thickness_mm` over the window to estimate when it hits 3mm.
- This is the "must beat" baseline.

**Model 2: SimpleLSTM**
```
Input (batch, 30, n_features)
→ LSTM(64, return_sequences=True)
→ Dropout(0.2)
→ LSTM(32, return_sequences=False)
→ Dropout(0.2)
→ Dense(16, ReLU)
→ Dense(1, ReLU)      ← ReLU on output to enforce RUL ≥ 0
```

**Model 3: BiLSTMAttention**
```
Input (batch, 30, n_features)
→ Bidirectional LSTM(64, return_sequences=True)
→ Dropout(0.3)
→ Bidirectional LSTM(32, return_sequences=True)
→ Temporal Attention (learnable weighted sum over timesteps)
→ Dense(32, ReLU)
→ Dropout(0.2)
→ Dense(1, ReLU)
```

**Model 4: CNNLSTM**
```
Input (batch, 30, n_features)
→ Conv1D(32, kernel=3, ReLU)
→ Conv1D(64, kernel=3, ReLU)
→ MaxPool1D(2)
→ LSTM(64, return_sequences=False)
→ Dense(32, ReLU)
→ Dense(1, ReLU)
```

### 3.4 train.py

- **Loss:** Huber Loss (delta=100) — NOT MSE
- **Optimizer:** Adam, lr=1e-3
- **Early stopping:** patience=15 on val_loss
- **LR scheduler:** ReduceLROnPlateau(factor=0.5, patience=5)
- **Save best model** checkpoint (.pt file)
- **Log per epoch:** train_loss, val_loss, val_MAE, val_RMSE
- Must work with any model from models.py

### 3.5 evaluate.py

**Metrics to compute:**
| Metric | Notes |
|--------|-------|
| MAE | Primary metric (days off) |
| RMSE | Penalizes large errors |
| R² | Overall fit |
| MAPE | Exclude RUL=0 rows to avoid div/0 |
| Late Prediction Rate | % where predicted > actual by >50 days (DANGEROUS) |
| Early Prediction Rate | % where predicted < actual by >50 days |

**Plots to generate:**
- PLOT-01: Predicted vs Actual scatter (with 45° line)
- PLOT-02: Error distribution histogram
- PLOT-03: MAE by RUL bucket (0-100, 100-200, 200-500, 500+)
- PLOT-04: 4 test wells — overlay predicted vs actual RUL trajectory
- PLOT-05: Training loss curves (train + val)
- PLOT-06: Feature importance (permutation importance)
- PLOT-07: Residual vs predicted (heteroscedasticity check)

**Critical tests (print pass/fail):**
- TEST-11: All predictions ≥ 0
- TEST-12: Test MAE beats naive baseline
- TEST-13: No single well has MAE > 200 days
- TEST-14: Late prediction rate < 15%
- TEST-15: Val and test MAE within 20% of each other
- TEST-16: Wells with many shut-ins don't have significantly worse MAE
- TEST-17: No single feature removal causes >50% MAE increase
- TEST-18: Window size 30 outperforms window size 15

### 3.6 run_experiment.py

Master script that:
1. Loads and validates data
2. Trains all 4 models (SimpleLSTM with Option A, SimpleLSTM with Option B, BiLSTMAttention Option A, CNNLSTM Option A)
3. Evaluates each on test set
4. Prints comparison table
5. Generates all plots for the best model
6. Saves everything to outputs/

---

## 4. EXPECTED RESULTS

| Model | Expected MAE | Expected R² |
|-------|-------------|-------------|
| Naive baseline | 80–150 days | 0.85–0.90 |
| SimpleLSTM Option B (with CR) | 20–50 days | 0.95–0.98 |
| SimpleLSTM Option A (no CR) | 40–80 days | 0.92–0.96 |
| BiLSTM+Attention Option A | 30–60 days | 0.94–0.97 |
| CNN-LSTM Option A | 30–65 days | 0.93–0.97 |

**"Good enough" = MAE < 60 days, Late Prediction Rate < 10%**

---

## 5. KEY GOTCHAS — READ BEFORE CODING

1. **SPLIT BY WELL_ID, NOT BY ROW.** This is the #1 mistake. Row-level splitting leaks temporal info and gives fake 99% R².

2. **CAP RUL AT 500 DAYS.** Without this, the model wastes capacity on the 1000-2000 day range where prediction doesn't matter. Use `y = min(RUL, 500)`.

3. **HUBER LOSS, NOT MSE.** MSE explodes on outliers in the high-RUL zone. Huber with delta=100 is robust.

4. **RELU ON OUTPUT LAYER.** RUL cannot be negative. Enforce this architecturally.

5. **FIT SCALER ON TRAIN ONLY.** Then transform val/test with the same fitted scaler. Otherwise you leak test statistics into training.

6. **SHUT-INS (Status=0):** During these periods, many features drop to ~0. The model must learn that shut-ins PAUSE degradation. Don't filter them out.

7. **WELLS SHORTER THAN WINDOW_SIZE:** Some wells are only 133 days. If window=30, they still work. If you try window=150, you'll lose some wells. Handle this gracefully.

8. **`Corrosion_Rate_mm_yr` IS LEAKAGE.** It's the direct output of the NORSOK physics model that drives thickness loss. Keeping it makes the problem trivially easy. The realistic model (Option A) drops it.

9. **`Current_Thickness_mm` IS SEMI-LEAKAGE.** It encodes cumulative damage directly. We keep it in both options because in reality you CAN measure thickness with UT gauges. But be aware the model will lean heavily on it.

---

## 6. FRAMEWORK & DEPENDENCIES

```
Python 3.10+
PyTorch (torch, torch.nn, torch.optim, torch.utils.data)
pandas
numpy
scikit-learn (MinMaxScaler, train_test_split utilities)
matplotlib
seaborn (optional, for prettier plots)
joblib (to save scaler)
```

`requirements.txt`:
```
torch>=2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

---

## 7. FILES PROVIDED IN THIS PACKAGE

| File | Description | Action |
|------|-------------|--------|
| `synthetic_corrosion_dataset.csv` | The dataset (85K rows, 80 wells) | Place in `data/` |
| `synthetic_corrosion_pipeline.py` | The script that generated the data | Reference only, do not modify |
| `LSTM_RUL_Project_Plan.md` | Detailed project plan with tests & benchmarks | Read for full specs |
| `well_history_plot.png` | Single well visualization | Reference only |
| `fleet_summary_plot.png` | Fleet-level summary | Reference only |
| `HANDOFF_TO_CLAUDE_CODE.md` | THIS FILE — the master briefing | Read first |

---

## 8. BUILD ORDER

Execute in this exact sequence:
1. `config.py` (5 min)
2. `data_loader.py` + run sanity checks (30 min)
3. `models.py` — all 4 architectures (20 min)
4. `train.py` — training loop (20 min)
5. `evaluate.py` — metrics + plots (20 min)
6. `run_experiment.py` — wire it all together (10 min)
7. Run full experiment, verify results match expected benchmarks
8. Save outputs (models, plots, metrics JSON)

Total estimated: ~2 hours of Claude Code work.

---

**END OF HANDOFF. START BUILDING.**
