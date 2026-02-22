"""
Data loading, preprocessing, and DataLoader creation for the multi-task
Casing RUL Prediction pipeline.

Pipeline:
  1. Load CSV
  2. Cap RUL at RUL_CAP
  3. Split by Well_ID (stratified by failure status + corrosion cause)
  4. Min-Max scale (fit on train only)
  5. Create sliding windows with multi-task targets
  6. Return PyTorch DataLoaders
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

from src.config import (
    DATASET_PATH, SCALER_DIR, RANDOM_SEED,
    RUL_CAP, WINDOW_SIZE, STRIDE,
    FEATURES_A, FEATURES_B, BINARY_FEATURES,
    TARGET_RUL, TARGET_CR, TARGET_WT, TARGET_CAUSE,
    FORECAST_HORIZONS, NUM_FORECAST_HORIZONS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    BATCH_SIZE, NUM_WORKERS,
)


# ============================================================================
# 1. LOAD + BASIC PREPROCESSING
# ============================================================================

def load_dataset(path=None):
    """Load CSV and apply RUL capping."""
    path = path or DATASET_PATH
    df = pd.read_csv(path)

    # Cap RUL at RUL_CAP
    df[TARGET_RUL] = df[TARGET_RUL].clip(upper=RUL_CAP)

    return df


# ============================================================================
# 2. WELL-LEVEL TRAIN / VAL / TEST SPLIT
# ============================================================================

def split_wells(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    """
    Split wells into train / val / test sets.

    Stratified by:
      - Failed vs survived (failure = any row where RUL_days == 0)
      - Dominant corrosion cause

    Returns dict of {'train': [...], 'val': [...], 'test': [...]} well IDs.
    """
    well_info = df.groupby("Well_ID").agg(
        failed=("RUL_days", lambda x: int((x == 0).any())),
        cause=("Corrosion_Cause", "first"),
        field=("Field_Name", "first"),
    ).reset_index()

    # Create stratification key combining failure status + cause
    well_info["strat_key"] = (
        well_info["failed"].astype(str) + "_" + well_info["cause"].astype(str)
    )

    # Some strata may have only 1 member — merge rare strata into "other"
    strat_counts = well_info["strat_key"].value_counts()
    rare = strat_counts[strat_counts < 3].index
    well_info.loc[well_info["strat_key"].isin(rare), "strat_key"] = "other"

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_wells, valtest_wells = train_test_split(
        well_info["Well_ID"].values,
        test_size=val_test_ratio,
        stratify=well_info["strat_key"],
        random_state=seed,
    )

    # Second split: val vs test (re-merge rare strata in the smaller pool)
    valtest_info = well_info[well_info["Well_ID"].isin(valtest_wells)].copy()
    vt_counts = valtest_info["strat_key"].value_counts()
    vt_rare = vt_counts[vt_counts < 2].index
    valtest_info.loc[valtest_info["strat_key"].isin(vt_rare), "strat_key"] = "other"

    relative_test = test_ratio / val_test_ratio
    val_wells, test_wells = train_test_split(
        valtest_info["Well_ID"].values,
        test_size=relative_test,
        stratify=valtest_info["strat_key"],
        random_state=seed,
    )

    return {
        "train": list(train_wells),
        "val": list(val_wells),
        "test": list(test_wells),
    }


# ============================================================================
# 3. SCALING
# ============================================================================

def fit_scaler(df_train, feature_cols):
    """Fit MinMaxScaler on training data only. Binary features are excluded."""
    cols_to_scale = [c for c in feature_cols if c not in BINARY_FEATURES]
    scaler = MinMaxScaler()
    scaler.fit(df_train[cols_to_scale].values)

    # Save for inference
    SCALER_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_DIR / "feature_scaler.joblib")
    joblib.dump(cols_to_scale, SCALER_DIR / "scaled_columns.joblib")

    return scaler, cols_to_scale


def apply_scaler(df, scaler, cols_to_scale, feature_cols):
    """Apply fitted scaler to a dataframe. Returns scaled feature array."""
    arr = df[feature_cols].values.copy()
    # Find indices of columns to scale
    scale_idx = [feature_cols.index(c) for c in cols_to_scale]
    arr[:, scale_idx] = scaler.transform(df[cols_to_scale].values)
    return arr.astype(np.float32)


# ============================================================================
# 4. SLIDING WINDOW DATASET
# ============================================================================

class CasingDataset(Dataset):
    """
    PyTorch Dataset for sliding-window multi-task casing prediction.

    Each sample:
      X:          (window_size, n_features)  — scaled input window
      y_rul:      scalar                     — capped RUL at end of window
      y_cr:       scalar                     — corrosion rate at end of window
      y_wt:       scalar                     — wall thickness at end of window
      y_cause:    int                        — corrosion cause label
      y_forecast: (60,)                      — thickness at +30d..+1800d every 30 days (NaN if unavailable)
    """

    def __init__(self, windows, targets_rul, targets_cr, targets_wt,
                 targets_cause, targets_forecast):
        self.windows = torch.from_numpy(windows)           # (N, W, F)
        self.targets_rul = torch.from_numpy(targets_rul)   # (N,)
        self.targets_cr = torch.from_numpy(targets_cr)     # (N,)
        self.targets_wt = torch.from_numpy(targets_wt)     # (N,)
        self.targets_cause = torch.from_numpy(targets_cause).long()  # (N,)
        self.targets_forecast = torch.from_numpy(targets_forecast)   # (N, 60)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            self.windows[idx],
            self.targets_rul[idx],
            self.targets_cr[idx],
            self.targets_wt[idx],
            self.targets_cause[idx],
            self.targets_forecast[idx],
        )


def create_windows_for_well(scaled_features, df_well, window_size=WINDOW_SIZE,
                            stride=STRIDE):
    """
    Create sliding windows and multi-task targets for a single well.

    Returns arrays or None if the well is too short.
    """
    n = len(df_well)
    if n < window_size + 1:
        return None

    rul = df_well[TARGET_RUL].values.astype(np.float32)
    cr = df_well[TARGET_CR].values.astype(np.float32)
    wt = df_well[TARGET_WT].values.astype(np.float32)
    cause = df_well[TARGET_CAUSE].values.astype(np.int64)

    windows = []
    y_rul, y_cr, y_wt, y_cause, y_forecast = [], [], [], [], []

    for t in range(window_size, n, stride):
        windows.append(scaled_features[t - window_size: t])
        y_rul.append(rul[t])
        y_cr.append(cr[t])
        y_wt.append(wt[t])
        y_cause.append(cause[t])

        # 5-year forecast targets
        forecast = []
        for h in FORECAST_HORIZONS:
            future_idx = t + h
            if future_idx < n:
                forecast.append(wt[future_idx])
            else:
                forecast.append(np.nan)
        y_forecast.append(forecast)

    return (
        np.array(windows, dtype=np.float32),
        np.array(y_rul, dtype=np.float32),
        np.array(y_cr, dtype=np.float32),
        np.array(y_wt, dtype=np.float32),
        np.array(y_cause, dtype=np.int64),
        np.array(y_forecast, dtype=np.float32),
    )


def build_dataset(df, well_ids, scaler, cols_to_scale, feature_cols,
                  window_size=WINDOW_SIZE, stride=STRIDE):
    """Build a CasingDataset from a list of well IDs."""
    all_windows, all_rul, all_cr, all_wt, all_cause, all_forecast = (
        [], [], [], [], [], []
    )

    for wid in tqdm(well_ids, desc="    Windows", leave=False,
                    bar_format="{l_bar}{bar:20}{r_bar}"):
        df_well = df[df["Well_ID"] == wid].sort_values("Day").reset_index(drop=True)
        scaled = apply_scaler(df_well, scaler, cols_to_scale, feature_cols)
        result = create_windows_for_well(scaled, df_well, window_size, stride)
        if result is None:
            continue
        w, r, c, t, ca, f = result
        all_windows.append(w)
        all_rul.append(r)
        all_cr.append(c)
        all_wt.append(t)
        all_cause.append(ca)
        all_forecast.append(f)

    return CasingDataset(
        np.concatenate(all_windows),
        np.concatenate(all_rul),
        np.concatenate(all_cr),
        np.concatenate(all_wt),
        np.concatenate(all_cause),
        np.concatenate(all_forecast),
    )


# ============================================================================
# 5. MASTER PIPELINE
# ============================================================================

def prepare_data(feature_option="A", window_size=WINDOW_SIZE, batch_size=BATCH_SIZE,
                 dataset_path=None, num_workers=0):
    """
    End-to-end data preparation.

    Parameters
    ----------
    feature_option : str
        "A" for realistic (14 features), "B" for cheating baseline (15).
    window_size : int
        Sliding window length.
    batch_size : int
        DataLoader batch size.
    dataset_path : str or Path, optional
        Override default CSV path.
    num_workers : int
        DataLoader workers (0 for local, 2 for GPU servers).

    Returns
    -------
    dict with keys:
        'train_loader', 'val_loader', 'test_loader',
        'n_features', 'well_splits', 'scaler', 'feature_cols'
    """
    print(f"Loading dataset from {dataset_path or DATASET_PATH}...")
    df = load_dataset(dataset_path)
    print(f"  Rows: {len(df):,}  |  Wells: {df['Well_ID'].nunique()}")

    feature_cols = FEATURES_A if feature_option == "A" else FEATURES_B

    # Split wells
    print("Splitting wells (stratified by failure + cause)...")
    splits = split_wells(df)
    print(f"  Train: {len(splits['train'])} wells  |  "
          f"Val: {len(splits['val'])} wells  |  "
          f"Test: {len(splits['test'])} wells")

    # Fit scaler on training data
    print("Fitting scaler on training data...")
    df_train = df[df["Well_ID"].isin(splits["train"])]
    scaler, cols_to_scale = fit_scaler(df_train, feature_cols)

    # Build datasets
    print(f"Creating sliding windows (size={window_size}, stride={STRIDE})...")
    train_ds = build_dataset(df, splits["train"], scaler, cols_to_scale,
                             feature_cols, window_size)
    val_ds = build_dataset(df, splits["val"], scaler, cols_to_scale,
                           feature_cols, window_size)
    test_ds = build_dataset(df, splits["test"], scaler, cols_to_scale,
                            feature_cols, window_size)

    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples:   {len(val_ds):,}")
    print(f"  Test samples:  {len(test_ds):,}")

    # DataLoaders (pin_memory only for CUDA, not MPS)
    pin = torch.cuda.is_available()
    persist = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=persist)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin,
                             persistent_workers=persist)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_features": len(feature_cols),
        "well_splits": splits,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
