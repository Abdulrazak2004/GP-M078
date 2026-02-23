"""
Model loading and inference for the dashboard.

Loads the trained BiLSTM-Attention model, scalers, and provides
prediction functions with AI-showcase data:
  - Attention weights (which timesteps the model focuses on)
  - MC Dropout confidence intervals
  - Raw input window features for visualization
  - Prediction vs actual error tracking
"""

import sys
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURES_A, BINARY_FEATURES, WINDOW_SIZE, STRIDE,
    NUM_FORECAST_HORIZONS, RUL_CAP,
    FORECAST_HORIZONS,
)
from src.models import BiLSTMAttention
from src.cfi import compute_cfi, cfi_label
from src.data_loader import engineer_features

# Paths to model artifacts
MODEL_PATH = PROJECT_ROOT / "outputs" / "exp3_bilstm_optA" / "models" / "best_model.pt"
SCALER_PATH = PROJECT_ROOT / "outputs" / "scalers" / "feature_scaler.joblib"
SCALED_COLS_PATH = PROJECT_ROOT / "outputs" / "scalers" / "scaled_columns.joblib"

# MC Dropout config
MC_SAMPLES = 15  # Number of stochastic forward passes

# Key feature indices for the input window heatmap (most interpretable ones)
KEY_FEATURE_NAMES = [
    "Pressure_psi", "Temp_F", "pH", "Water_Cut_pct",
    "Flow_Velocity_fps", "CO2_Partial_Pressure_psi",
    "Current_Thickness_mm", "Inhibitor_Active",
    "Thickness_RollMean_7d", "Thickness_Slope_7d",
    "Cumulative_Damage",
]

# Global state
_model = None
_scaler = None
_scaled_columns = None
_device = None
_key_feature_indices = None


def load_model():
    """Load the trained BiLSTM-Attention model and scalers."""
    global _model, _scaler, _scaled_columns, _device, _key_feature_indices

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    n_features = len(FEATURES_A)
    _model = BiLSTMAttention(n_features)
    checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _model.load_state_dict(checkpoint["model_state_dict"])
    else:
        _model.load_state_dict(checkpoint)

    _model.to(_device)
    _model.eval()

    # Load scalers
    _scaler = joblib.load(SCALER_PATH)
    _scaled_columns = joblib.load(SCALED_COLS_PATH)

    # Pre-compute key feature indices
    _key_feature_indices = [FEATURES_A.index(f) for f in KEY_FEATURE_NAMES if f in FEATURES_A]

    print(f"Model loaded on {_device} | Features: {n_features} | MC samples: {MC_SAMPLES}")


def _apply_scaler(df, feature_cols=FEATURES_A):
    """Apply fitted scaler to a dataframe. Returns scaled feature array."""
    arr = df[feature_cols].values.copy().astype(np.float32)
    scale_idx = [feature_cols.index(c) for c in _scaled_columns]
    arr[:, scale_idx] = _scaler.transform(df[_scaled_columns].values)
    return arr


def _enable_dropout(model):
    """Enable dropout layers for MC Dropout inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def _mc_dropout_predict(window_tensor):
    """
    Run MC Dropout: multiple stochastic forward passes to get
    prediction mean, std, and confidence intervals.

    Returns dict with mean predictions, std, and confidence bounds.
    """
    _model.eval()
    _enable_dropout(_model)

    all_rul, all_cr, all_wt = [], [], []

    for _ in range(MC_SAMPLES):
        with torch.no_grad():
            preds = _model(window_tensor)
        all_rul.append(preds["rul"].cpu().item())
        all_cr.append(preds["cr"].cpu().item())
        all_wt.append(preds["wt"].cpu().item())

    _model.eval()  # Restore full eval mode

    rul_arr = np.array(all_rul)
    cr_arr = np.array(all_cr)
    wt_arr = np.array(all_wt)

    return {
        "rul_mean": float(np.mean(rul_arr)),
        "rul_std": float(np.std(rul_arr)),
        "rul_ci_low": float(np.percentile(rul_arr, 5)),
        "rul_ci_high": float(np.percentile(rul_arr, 95)),
        "cr_mean": float(np.mean(cr_arr)),
        "cr_std": float(np.std(cr_arr)),
        "wt_mean": float(np.mean(wt_arr)),
        "wt_std": float(np.std(wt_arr)),
        "wt_ci_low": float(np.percentile(wt_arr, 5)),
        "wt_ci_high": float(np.percentile(wt_arr, 95)),
    }


def _get_attention_weights(window_tensor):
    """Run a forward pass and extract attention weights."""
    _model.eval()
    with torch.no_grad():
        preds = _model(window_tensor)
    weights = _model.attention_weights  # (1, 30) from BiLSTMAttention
    return weights.cpu().numpy().flatten().tolist()


def _extract_window_features(scaled, day_idx):
    """Extract key features from the input window for visualization."""
    window = scaled[day_idx - WINDOW_SIZE: day_idx]  # (30, 26)
    # Return only the key interpretable features
    key_data = window[:, _key_feature_indices]  # (30, 11)
    return key_data.tolist()


def predict_at_day(df_well, day_idx, feature_cols=FEATURES_A):
    """Run model inference at a specific day index for a well."""
    if day_idx < WINDOW_SIZE:
        day_idx = WINDOW_SIZE

    n = len(df_well)
    if day_idx >= n:
        day_idx = n - 1

    scaled = _apply_scaler(df_well, feature_cols)
    window = scaled[day_idx - WINDOW_SIZE: day_idx]
    window_tensor = torch.from_numpy(window).unsqueeze(0).to(_device)

    actual_row = df_well.iloc[day_idx]
    initial_thickness = actual_row.get("Initial_Thickness_mm", 11.0)
    actual_wt = actual_row.get("Current_Thickness_mm", 11.0)
    thickness_loss_pct = (1.0 - actual_wt / initial_thickness) * 100.0

    with torch.no_grad():
        preds = _model(window_tensor)

    rul = float(preds["rul"].cpu().item())
    cr = float(preds["cr"].cpu().item())
    wt = float(preds["wt"].cpu().item())
    forecast = preds["forecast"].cpu().numpy().flatten().tolist()

    cause_probs = [1.0 / 6] * 6
    cfi_val = float(compute_cfi(rul, cr, thickness_loss_pct, [cause_probs]))

    return {
        "rul": round(rul, 1),
        "cr": round(cr, 2),
        "wt": round(wt, 3),
        "cfi": round(cfi_val, 1),
        "cfi_label": cfi_label(cfi_val),
        "forecast": [round(f, 3) for f in forecast],
        "day": int(df_well.iloc[day_idx]["Day"]),
        "actual_wt": round(float(actual_wt), 3),
        "actual_cr": round(float(actual_row.get("Corrosion_Rate_mpy", 0)), 2),
        "actual_rul": round(float(actual_row.get("RUL_days", 0)), 1),
        "thickness_loss_pct": round(thickness_loss_pct, 2),
    }


def predict_playback(df_well, stride=30):
    """
    Pre-compute enriched predictions for playback animation.

    Each point includes:
      - Model predictions (rul, cr, wt, cfi)
      - Actual values for comparison
      - Attention weights (30 timesteps)
      - MC Dropout confidence intervals
      - Input window key features (30 x 11 matrix)
      - Prediction errors
    """
    n = len(df_well)
    results = []

    # Scale once
    scaled = _apply_scaler(df_well)

    for day_idx in range(WINDOW_SIZE, n, stride):
        window = scaled[day_idx - WINDOW_SIZE: day_idx]
        window_tensor = torch.from_numpy(window).unsqueeze(0).to(_device)

        actual_row = df_well.iloc[day_idx]
        initial_thickness = actual_row.get("Initial_Thickness_mm", 11.0)
        actual_wt = float(actual_row.get("Current_Thickness_mm", 11.0))
        actual_cr = float(actual_row.get("Corrosion_Rate_mpy", 0))
        actual_rul = float(actual_row.get("RUL_days", 0))
        thickness_loss_pct = (1.0 - actual_wt / initial_thickness) * 100.0

        # ── 1. Standard prediction ──
        _model.eval()
        with torch.no_grad():
            preds = _model(window_tensor)

        rul = float(preds["rul"].cpu().item())
        cr = float(preds["cr"].cpu().item())
        wt = float(preds["wt"].cpu().item())
        forecast = preds["forecast"].cpu().numpy().flatten().tolist()

        # ── 2. Attention weights ──
        attn = _model.attention_weights.cpu().numpy().flatten()
        # Normalize to 0-1 for visualization
        attn_min, attn_max = attn.min(), attn.max()
        if attn_max > attn_min:
            attn_norm = ((attn - attn_min) / (attn_max - attn_min)).tolist()
        else:
            attn_norm = [1.0 / WINDOW_SIZE] * WINDOW_SIZE

        # ── 3. MC Dropout confidence ──
        mc = _mc_dropout_predict(window_tensor)

        # ── 4. Input window features ──
        window_features = _extract_window_features(scaled, day_idx)

        # ── 5. Prediction errors ──
        wt_error = abs(wt - actual_wt)
        cr_error = abs(cr - actual_cr)
        rul_error = abs(rul - actual_rul) if actual_rul > 0 else None

        # ── CFI ──
        cause_probs = [1.0 / 6] * 6
        cfi_val = float(compute_cfi(rul, cr, thickness_loss_pct, [cause_probs]))

        results.append({
            "day": int(actual_row["Day"]),
            "day_idx": int(day_idx),

            # Predictions
            "rul": round(rul, 1),
            "cr": round(cr, 2),
            "wt": round(wt, 3),
            "cfi": round(cfi_val, 1),
            "cfi_label": cfi_label(cfi_val),

            # Actuals
            "actual_wt": round(actual_wt, 3),
            "actual_cr": round(actual_cr, 2),
            "actual_rul": round(actual_rul, 1),
            "thickness_loss_pct": round(thickness_loss_pct, 2),

            # AI Showcase: Attention
            "attention": [round(a, 4) for a in attn_norm],

            # AI Showcase: Confidence (MC Dropout)
            "wt_ci_low": round(mc["wt_ci_low"], 3),
            "wt_ci_high": round(mc["wt_ci_high"], 3),
            "wt_std": round(mc["wt_std"], 4),
            "rul_ci_low": round(mc["rul_ci_low"], 1),
            "rul_ci_high": round(mc["rul_ci_high"], 1),

            # AI Showcase: Input window (30 x 11 key features)
            "window_features": window_features,

            # AI Showcase: Errors
            "wt_error": round(wt_error, 4),
            "cr_error": round(cr_error, 4),
            "rul_error": round(rul_error, 1) if rul_error is not None else None,

            # Forecast (first 12 months for bandwidth)
            "forecast": [round(f, 3) for f in forecast[:12]],
        })

    return results


def predict_design_well(params: dict):
    """
    Simulate a custom well configuration and predict its 30-year trajectory.
    """
    from data_generation.well_simulator import simulate_well

    cause_code = _map_cause(params.get("corrosion_cause", "CO2"))
    co2_frac = 0.03 if params.get("corrosion_cause", "CO2") == "CO2" else 0.01
    h2s_frac = 0.02 if params.get("corrosion_cause", "CO2") == "H2S" else 0.0005

    well = {
        "well_id": "DESIGN_001",
        "field_name": "Custom",
        "sub_area": "Custom",
        "latitude": 25.4,
        "longitude": 49.3,
        "reservoir_type": params.get("reservoir_type", "Carbonate"),
        "casing_grade": params.get("casing_grade", "L80"),
        "casing_od_in": params.get("casing_od_in", 9.625),
        "initial_thickness_mm": params.get("initial_thickness_mm", 11.0),
        "seed": 42,
        "initial_pressure_psi": params.get("avg_pressure_psi", 3200),
        "pressure_decline_annual_pct": 1.0,
        "bottomhole_temp_F": params.get("avg_temp_f", 180),
        "co2_mol_frac": co2_frac,
        "h2s_mol_frac": h2s_frac,
        "wc_midpoint_day": 3500,
        "wc_steepness": 0.0012,
        "wc_max": min(params.get("avg_water_cut_pct", 15) * 3 / 100.0, 0.95),
        "initial_flow_bpd": 5000,
        "flow_decline_rate": 0.0001,
        "pipe_diameter_in": 4.5,
        "fluid_density_ppg": 7.1,
        "roughness_m": 0.0003,
        "api_gravity": 34,
        "gor_scf_bbl": 600,
        "inhibitor_efficiency": 0.80,
        "inhibitor_active": True,
        "inhibitor_start_day": 0,
        "inhibitor_reliability": 0.92,
        "pitting_factor": 1.5,
        "pitting_probability": 0.01,
        "ambient_temp_amplitude_F": 18,
        "noise_temp": 0.01,
        "noise_pressure": 0.003,
        "noise_flow": 0.03,
        "noise_ph": 0.01,
        "shutin_frequency": 0.003,
        "shutin_duration_min": 5,
        "shutin_duration_max": 15,
        "choke_change_frequency": 0.01,
        "choke_change_magnitude": 0.2,
        "initial_ph": params.get("avg_ph", 5.2),
        "ph_drift_rate": 0.0001,
        "corrosion_cause": cause_code,
    }

    df_sim = simulate_well(well, n_days=10950)
    df_sim = engineer_features(df_sim)

    trajectory = []
    scaled = _apply_scaler(df_sim)

    for day_idx in range(WINDOW_SIZE, len(df_sim), 90):
        window = scaled[day_idx - WINDOW_SIZE: day_idx]
        window_tensor = torch.from_numpy(window).unsqueeze(0).to(_device)

        actual_row = df_sim.iloc[day_idx]
        initial_thickness = params.get("initial_thickness_mm", 11.0)
        actual_wt = actual_row.get("Current_Thickness_mm", initial_thickness)
        thickness_loss_pct = (1.0 - actual_wt / initial_thickness) * 100.0

        with torch.no_grad():
            preds = _model(window_tensor)

        rul = float(preds["rul"].cpu().item())
        cr = float(preds["cr"].cpu().item())
        wt = float(preds["wt"].cpu().item())

        cause_probs = [1.0 / 6] * 6
        cfi_val = float(compute_cfi(rul, cr, thickness_loss_pct, [cause_probs]))

        trajectory.append({
            "day": int(actual_row["Day"]),
            "year": round(int(actual_row["Day"]) / 365.25, 1),
            "wt": round(wt, 3),
            "actual_wt": round(float(actual_wt), 3),
            "cr": round(cr, 2),
            "rul": round(rul, 1),
            "cfi": round(cfi_val, 1),
            "cfi_label": cfi_label(cfi_val),
        })

    latest = trajectory[-1] if trajectory else {"rul": 0, "cfi": 0}
    risk_timeline = _compute_risk_timeline(trajectory)
    recommendation = _material_recommendation(trajectory, params.get("casing_grade", "L80"))

    return {
        "trajectory": trajectory,
        "rul_years": round(latest["rul"] / 365.25, 1),
        "final_cfi": latest["cfi"],
        "risk_timeline": risk_timeline,
        "recommendation": recommendation,
    }


def _map_cause(cause_str):
    mapping = {"CO2": 0, "H2S": 1, "MIC": 2, "Erosion": 3, "O2": 4, "Combined": 5}
    return mapping.get(cause_str, 0)


def _compute_risk_timeline(trajectory):
    thresholds = [(25, "Green (Safe)"), (50, "Yellow (Watch)"), (75, "Orange (Elevated)"), (100, "Red (Critical)")]
    timeline = []
    crossed = set()
    for point in trajectory:
        for threshold, label in thresholds:
            if threshold not in crossed and point["cfi"] >= threshold:
                crossed.add(threshold)
                timeline.append({"threshold": threshold, "label": label, "year": point["year"], "day": point["day"]})
    return timeline


def _material_recommendation(trajectory, current_grade):
    if not trajectory:
        return "Insufficient data for recommendation."
    critical_year = None
    for point in trajectory:
        if point["cfi"] >= 75:
            critical_year = point["year"]
            break
    grade_hierarchy = ["N80", "L80", "P110"]
    current_idx = grade_hierarchy.index(current_grade) if current_grade in grade_hierarchy else 1
    if critical_year is None or critical_year > 25:
        return f"{current_grade} is adequate for the full 30-year design life."
    elif critical_year > 15:
        return (f"{current_grade} adequate for ~{critical_year:.0f} years. "
                f"For >25-year life, consider {grade_hierarchy[min(current_idx + 1, 2)]}.")
    else:
        upgrade = grade_hierarchy[min(current_idx + 1, 2)]
        return (f"Warning: {current_grade} reaches critical risk at ~{critical_year:.0f} years. "
                f"Strongly recommend {upgrade} or corrosion-resistant alloy (CRA).")
