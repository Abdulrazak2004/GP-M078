"""
FastAPI backend for the Corrosion Intelligence Dashboard.

Endpoints:
  GET  /api/wells/summary           — All wells with lat/lon, CFI, risk color
  GET  /api/wells/stats              — Aggregate statistics
  GET  /api/wells/{id}/timeseries    — Full time series for one well
  GET  /api/wells/{id}/predictions   — Model predictions at a specific day
  GET  /api/wells/{id}/playback      — Pre-computed predictions for playback
  POST /api/design-well              — Custom well design → trajectory prediction
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup."""
    from inference import load_model
    from wells_data import load_wells

    print("=" * 60)
    print("  CORROSION INTELLIGENCE PLATFORM — Starting up...")
    print("=" * 60)

    load_wells()
    load_model()

    print("=" * 60)
    print("  Ready. Serving at http://localhost:8000")
    print("=" * 60)

    yield


app = FastAPI(
    title="Corrosion Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/wells/summary")
def wells_summary(
    field: Optional[str] = Query(None),
    risk: Optional[str] = Query(None),
    reservoir_type: Optional[str] = Query(None),
):
    """Return summary for all wells, optionally filtered."""
    from wells_data import get_all_wells_summary

    wells = get_all_wells_summary()
    if wells is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    if field:
        wells = [w for w in wells if w["field"] == field]
    if risk:
        wells = [w for w in wells if w["risk_color"] == risk]
    if reservoir_type:
        wells = [w for w in wells if w["reservoir_type"] == reservoir_type]

    return {"wells": wells, "count": len(wells)}


@app.get("/api/wells/stats")
def wells_stats():
    """Return aggregate statistics."""
    from wells_data import get_aggregate_stats
    return get_aggregate_stats()


@app.get("/api/wells/{well_id}/timeseries")
def well_timeseries(well_id: str):
    """Return downsampled time series for a well."""
    from wells_data import get_well_timeseries

    data = get_well_timeseries(well_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Well {well_id} not found")

    return {"well_id": well_id, "timeseries": data}


@app.get("/api/wells/{well_id}/predictions")
def well_predictions(well_id: str, day: int = Query(default=5000)):
    """Return model predictions at a specific day for a well."""
    from wells_data import get_well_data
    from inference import predict_at_day

    df_well = get_well_data(well_id)
    if df_well is None:
        raise HTTPException(status_code=404, detail=f"Well {well_id} not found")

    # Find the closest day index
    day_idx = (df_well["Day"] - day).abs().idxmin()
    # Convert pandas index to positional index
    day_idx = df_well.index.get_loc(day_idx) if day_idx in df_well.index else int(day_idx)

    result = predict_at_day(df_well, day_idx)
    result["well_id"] = well_id

    return result


@app.get("/api/wells/{well_id}/playback")
def well_playback(well_id: str, stride: int = Query(default=30)):
    """Return pre-computed predictions for playback animation."""
    from wells_data import get_well_data
    from inference import predict_playback

    df_well = get_well_data(well_id)
    if df_well is None:
        raise HTTPException(status_code=404, detail=f"Well {well_id} not found")

    results = predict_playback(df_well, stride=stride)

    # Also include well metadata
    first_row = df_well.iloc[0]
    metadata = {
        "well_id": well_id,
        "field": str(first_row.get("Field_Name", "Unknown")),
        "sub_area": str(first_row.get("Sub_Area", "Unknown")),
        "reservoir_type": str(first_row.get("Reservoir_Type", "Carbonate")),
        "casing_grade": str(first_row.get("Casing_Grade", "L80")),
        "initial_thickness": round(float(first_row.get("Initial_Thickness_mm", 11.0)), 2),
    }

    return {"metadata": metadata, "predictions": results}


class DesignWellRequest(BaseModel):
    reservoir_type: str = "Carbonate"
    casing_grade: str = "L80"
    casing_od_in: float = 9.625
    initial_thickness_mm: float = 11.0
    avg_pressure_psi: float = 3200
    avg_temp_f: float = 180
    avg_ph: float = 5.2
    avg_water_cut_pct: float = 15
    corrosion_cause: str = "CO2"


@app.post("/api/design-well")
def design_well(req: DesignWellRequest):
    """Design a custom well and predict its 30-year trajectory."""
    from inference import predict_design_well

    params = req.model_dump()
    result = predict_design_well(params)

    return result
