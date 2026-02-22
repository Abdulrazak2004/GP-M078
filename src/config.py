"""
Central configuration for the Casing RUL Prediction pipeline.

All hyperparameters, feature lists, paths, and multi-task weights live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
METRIC_DIR = OUTPUT_DIR / "metrics"
SCALER_DIR = OUTPUT_DIR / "scalers"

DATASET_PATH = DATA_DIR / "synthetic_corrosion_dataset.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data splitting  (by Well_ID, stratified)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.60   # 48 wells
VAL_RATIO = 0.20     # 16 wells
TEST_RATIO = 0.20    # 16 wells

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
RUL_CAP = 500        # Cap RUL target at 500 days (piece-wise linear)
WINDOW_SIZE = 30     # Sliding window length (also try 15, 60)
STRIDE = 1           # Window stride

# Forecast horizons in days (every 30 days for 5 years = 60 monthly values)
FORECAST_HORIZONS = list(range(30, 1830, 30))
NUM_FORECAST_HORIZONS = 60

# DataLoader workers (0 for local/MPS, 2 for multi-GPU)
NUM_WORKERS = 2

# ---------------------------------------------------------------------------
# Feature definitions  (column names from the generated CSV)
# ---------------------------------------------------------------------------
# Metadata columns (not model inputs)
META_COLS = [
    "Well_ID", "Field_Name", "Sub_Area", "Latitude", "Longitude",
    "Reservoir_Type", "Casing_Grade", "Casing_OD_in", "Initial_Thickness_mm",
]

# Option A — realistic (no corrosion rate, no thickness loss pct)
FEATURES_A = [
    "Status",
    "Pressure_psi",
    "Temp_F",
    "pH",
    "Water_Cut_pct",
    "Production_Rate_bpd",
    "Flow_Velocity_fps",
    "Shear_Stress_Pa",
    "CO2_Partial_Pressure_psi",
    "H2S_Partial_Pressure_psi",
    "Fluid_Density_ppg",
    "Viscosity_cP",
    "Current_Thickness_mm",
    "Inhibitor_Active",
]  # 14 features

# Option B — cheating baseline (adds corrosion rate)
FEATURES_B = FEATURES_A + ["Corrosion_Rate_mpy"]  # 15 features

# Binary features (not scaled)
BINARY_FEATURES = ["Status", "Inhibitor_Active"]

# Target columns
TARGET_RUL = "RUL_days"
TARGET_CR = "Corrosion_Rate_mpy"
TARGET_WT = "Current_Thickness_mm"
TARGET_CAUSE = "Corrosion_Cause"

# Number of corrosion cause classes
NUM_CAUSE_CLASSES = 6

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LSTM_HIDDEN_1 = 64
LSTM_HIDDEN_2 = 32
DROPOUT_LSTM = 0.3
DROPOUT_BILSTM = 0.4
DROPOUT_HEAD = 0.3
CNN_FILTERS_1 = 32
CNN_FILTERS_2 = 64
CNN_KERNEL = 3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
EARLY_STOP_PATIENCE = 20
LR_REDUCE_PATIENCE = 7
LR_REDUCE_FACTOR = 0.5
GRAD_CLIP_NORM = 1.0
HUBER_DELTA_RUL = 20.0
HUBER_DELTA_CR = 3.0
HUBER_DELTA_WT = 1.5
HUBER_DELTA_FORECAST = 1.5

# Multi-task loss weights
LOSS_WEIGHT_RUL = 1.0       # Primary objective
LOSS_WEIGHT_CR = 2.0        # Corrosion rate (boosted)
LOSS_WEIGHT_WT = 1.0        # Wall thickness (boosted)
LOSS_WEIGHT_CAUSE = 8.0     # Cause classification (heavily boosted — was drowning)
LOSS_WEIGHT_FORECAST = 0.5  # 60-month forecast

# MC Dropout inference
MC_DROPOUT_SAMPLES = 50
CI_LOWER_QUANTILE = 0.025
CI_UPPER_QUANTILE = 0.975

# ---------------------------------------------------------------------------
# CFI weights and thresholds
# ---------------------------------------------------------------------------
CFI_WEIGHT_WT = 0.35
CFI_WEIGHT_CR = 0.25
CFI_WEIGHT_RUL = 0.25
CFI_WEIGHT_CAUSE = 0.15

# Cause severity weights (CO2, H2S, MIC, Erosion, O2, Combined)
CFI_CAUSE_SEVERITY = [40, 70, 60, 55, 50, 80]

# CFI thresholds
CFI_GREEN = 25
CFI_YELLOW = 50
CFI_ORANGE = 75
# Above 75 = Red (Critical)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "exp1_lstm_optB": {
        "backbone": "SimpleLSTM",
        "features": "B",
        "window_size": 30,
        "description": "Pipeline verification (cheating baseline)",
    },
    "exp2_lstm_optA": {
        "backbone": "SimpleLSTM",
        "features": "A",
        "window_size": 30,
        "description": "Realistic baseline",
    },
    "exp3_bilstm_optA": {
        "backbone": "BiLSTMAttention",
        "features": "A",
        "window_size": 30,
        "description": "Primary model (expected best)",
    },
    "exp4_cnnlstm_optA": {
        "backbone": "CNNLSTM",
        "features": "A",
        "window_size": 30,
        "description": "CNN-LSTM comparison",
    },
    "exp5_bilstm_w15": {
        "backbone": "BiLSTMAttention",
        "features": "A",
        "window_size": 15,
        "description": "Window size ablation (15)",
    },
    "exp6_bilstm_w60": {
        "backbone": "BiLSTMAttention",
        "features": "A",
        "window_size": 60,
        "description": "Window size ablation (60)",
    },
}
