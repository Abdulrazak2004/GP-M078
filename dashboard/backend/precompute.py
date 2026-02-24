"""
Precompute predictions for all 500 wells and save to disk.

Run this ONCE on the server after setup. After this, well loading is instant.

Usage:
    python precompute.py
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_model, predict_playback
from wells_data import load_wells, get_well_data, get_all_wells_summary

CACHE_DIR = Path(__file__).resolve().parent / "cache"
STRIDE = 60


def main():
    print("=" * 60)
    print("  PRECOMPUTING ALL WELL PREDICTIONS")
    print("=" * 60)

    # Load everything
    print("\n[1/3] Loading data and model...")
    load_wells()
    load_model()

    wells = get_all_wells_summary()
    total = len(wells)
    print(f"  {total} wells to process\n")

    # Create cache directory
    CACHE_DIR.mkdir(exist_ok=True)

    # Precompute each well
    print("[2/3] Running inference for all wells...")
    start = time.time()
    failed = []

    for i, w in enumerate(wells):
        wid = w["well_id"]
        cache_file = CACHE_DIR / f"{wid}.json"

        # Skip if already cached
        if cache_file.exists():
            elapsed = time.time() - start
            print(f"  [{i+1}/{total}] {wid} — cached (skipped)")
            continue

        try:
            df_well = get_well_data(wid)
            if df_well is None:
                failed.append(wid)
                continue

            t0 = time.time()
            results = predict_playback(df_well, stride=STRIDE)

            # Get metadata
            first_row = df_well.iloc[0]
            metadata = {
                "well_id": wid,
                "field": str(first_row.get("Field_Name", "Unknown")),
                "sub_area": str(first_row.get("Sub_Area", "Unknown")),
                "reservoir_type": str(first_row.get("Reservoir_Type", "Carbonate")),
                "casing_grade": str(first_row.get("Casing_Grade", "L80")),
                "initial_thickness": round(float(first_row.get("Initial_Thickness_mm", 11.0)), 2),
            }

            payload = {"metadata": metadata, "predictions": results}

            with open(cache_file, "w") as f:
                json.dump(payload, f)

            dt = time.time() - t0
            elapsed = time.time() - start
            eta = (elapsed / (i + 1)) * (total - i - 1)
            print(f"  [{i+1}/{total}] {wid} — {dt:.1f}s  (ETA: {eta/60:.1f} min)")

        except Exception as e:
            failed.append(wid)
            print(f"  [{i+1}/{total}] {wid} — FAILED: {e}")

    elapsed = time.time() - start
    print(f"\n[3/3] Done in {elapsed/60:.1f} minutes")
    print(f"  Cached: {total - len(failed)} / {total}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"\n  Restart the server and all wells will load instantly!")


if __name__ == "__main__":
    main()
