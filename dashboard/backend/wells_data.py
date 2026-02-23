"""
Pre-load and cache well data for the dashboard.

Loads the synthetic corrosion dataset, assigns geographic coordinates
from the Saudi oil well locations database, and provides fast access
to per-well data and summary statistics.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_PATH
from src.data_loader import engineer_features, load_dataset

# Well data store
_df = None
_well_ids = None
_well_summary = None
_well_groups = {}


def load_wells():
    """Load dataset, engineer features, assign coordinates, and cache."""
    global _df, _well_ids, _well_summary, _well_groups

    print("Loading dataset...")
    _df = load_dataset()
    print(f"  Rows: {len(_df):,}  |  Wells: {_df['Well_ID'].nunique()}")

    print("Engineering features...")
    _df = engineer_features(_df)

    _well_ids = sorted(_df["Well_ID"].unique().tolist())

    # Pre-group by Well_ID
    print("Grouping by Well_ID...")
    for wid in _well_ids:
        _well_groups[wid] = _df[_df["Well_ID"] == wid].sort_values("Day").reset_index(drop=True)

    # Build summary (last-day snapshot for each well)
    print("Building well summary...")
    _build_summary()

    print(f"Wells loaded: {len(_well_ids)}")


def _build_summary():
    """Build summary data for all wells (for map view)."""
    global _well_summary

    summaries = []
    for wid in _well_ids:
        df_w = _well_groups[wid]
        last_row = df_w.iloc[-1]
        first_row = df_w.iloc[0]

        initial_thickness = first_row.get("Initial_Thickness_mm", 11.0)
        current_wt = last_row["Current_Thickness_mm"]
        thickness_loss_pct = (1.0 - current_wt / initial_thickness) * 100.0

        # Compute simple CFI from actual values
        from src.cfi import compute_cfi, cfi_label as get_cfi_label
        cause_probs = [1.0 / 6] * 6
        cfi_arr = compute_cfi(
            last_row.get("RUL_days", 500),
            last_row.get("Corrosion_Rate_mpy", 1.0),
            thickness_loss_pct,
            cause_probs,
        )
        cfi_val = float(np.atleast_1d(cfi_arr).flat[0])

        summaries.append({
            "well_id": wid,
            "field": last_row.get("Field_Name", "Unknown"),
            "sub_area": last_row.get("Sub_Area", "Unknown"),
            "lat": float(last_row.get("Latitude", 25.0)),
            "lon": float(last_row.get("Longitude", 49.0)),
            "reservoir_type": last_row.get("Reservoir_Type", "Carbonate"),
            "casing_grade": last_row.get("Casing_Grade", "L80"),
            "initial_thickness": round(float(initial_thickness), 2),
            "current_wt": round(float(current_wt), 3),
            "rul": round(float(last_row.get("RUL_days", 0)), 0),
            "cr": round(float(last_row.get("Corrosion_Rate_mpy", 0)), 2),
            "cfi": round(cfi_val, 1),
            "cfi_label": get_cfi_label(cfi_val),
            "risk_color": _cfi_to_color(cfi_val),
        })

    _well_summary = summaries


def _cfi_to_color(cfi):
    """Map CFI value to risk color name."""
    if cfi <= 25:
        return "green"
    elif cfi <= 50:
        return "yellow"
    elif cfi <= 75:
        return "orange"
    else:
        return "red"


def get_all_wells_summary():
    """Return summary for all wells (for map markers)."""
    return _well_summary


def get_well_ids():
    """Return list of all well IDs."""
    return _well_ids


def get_well_data(well_id):
    """Return full time series DataFrame for a specific well."""
    return _well_groups.get(well_id)


def get_well_timeseries(well_id):
    """Return time series data as list of dicts for API response."""
    df_w = _well_groups.get(well_id)
    if df_w is None:
        return None

    # Return key columns only to keep response size manageable
    cols = [
        "Day", "Status", "Pressure_psi", "Temp_F", "pH",
        "Water_Cut_pct", "Production_Rate_bpd", "Flow_Velocity_fps",
        "Corrosion_Rate_mpy", "Current_Thickness_mm", "RUL_days",
        "Inhibitor_Active", "Thickness_Loss_Pct",
    ]
    available_cols = [c for c in cols if c in df_w.columns]

    # Downsample to every 30 days for bandwidth
    df_sampled = df_w.iloc[::30]
    records = df_sampled[available_cols].to_dict(orient="records")

    # Round floats
    for r in records:
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 3)

    return records


def get_field_list():
    """Return unique field names."""
    if _well_summary is None:
        return []
    return sorted(set(w["field"] for w in _well_summary))


def get_aggregate_stats():
    """Return aggregate statistics for the overview."""
    if _well_summary is None:
        return {}

    total = len(_well_summary)
    by_risk = {"green": 0, "yellow": 0, "orange": 0, "red": 0}
    fields = set()

    for w in _well_summary:
        by_risk[w["risk_color"]] = by_risk.get(w["risk_color"], 0) + 1
        fields.add(w["field"])

    return {
        "total_wells": total,
        "total_fields": len(fields),
        "by_risk": by_risk,
        "fields": sorted(fields),
    }
