"""
Master script: assign wells to fields, run simulations, validate, save CSV.

Usage:
    python -m data_generation.generate_dataset
"""

import numpy as np
import pandas as pd
import time
import os

from data_generation.config_fields import (
    FIELD_CONFIGS,
    WELL_DISTRIBUTION,
    FIELD_CAUSE_DISTRIBUTION,
    CAUSE_LABELS,
)
from data_generation.well_simulator import simulate_well


# ============================================================================
# WELL-ID GENERATOR
# ============================================================================

_FIELD_PREFIXES = {
    "Ghawar": "GHW",
    "Khurais": "KHR",
    "Safaniya": "SFN",
    "Shaybah": "SHB",
    "Manifa": "MNF",
    "Abqaiq": "ABQ",
    "Berri": "BRR",
    "Zuluf": "ZLF",
    "Others": "OTH",
}


def make_well_id(field_name: str, counter: int) -> str:
    prefix = _FIELD_PREFIXES.get(field_name, "UNK")
    return f"{prefix}-{counter:03d}"


# ============================================================================
# CAUSE ASSIGNMENT
# ============================================================================

def assign_causes(n_wells: int, cause_dist: dict) -> list:
    """
    Deterministically distribute corrosion cause codes across a field's wells.

    Guarantees each cause code is assigned at least once if its fraction > 0.
    Returns a shuffled list of length n_wells.
    """
    codes = []
    remaining = n_wells

    # First pass: guarantee at least one well per cause
    for cause, frac in cause_dist.items():
        if frac > 0 and remaining > 0:
            codes.append(cause)
            remaining -= 1

    # Second pass: fill remainder proportionally
    if remaining > 0:
        causes = list(cause_dist.keys())
        fracs = np.array([cause_dist[c] for c in causes])
        fracs = fracs / fracs.sum()
        counts = np.round(fracs * remaining).astype(int)
        # Fix rounding to exactly match remaining
        diff = remaining - counts.sum()
        if diff > 0:
            idx = np.argmax(fracs)
            counts[idx] += diff
        elif diff < 0:
            idx = np.argmax(counts)
            counts[idx] += diff  # diff is negative

        for cause, count in zip(causes, counts):
            codes.extend([cause] * count)

    # Trim or pad to exact size
    codes = codes[:n_wells]
    while len(codes) < n_wells:
        # Add the dominant cause
        dominant = max(cause_dist, key=cause_dist.get)
        codes.append(dominant)

    np.random.shuffle(codes)
    return codes


# ============================================================================
# WELL PROPERTY SAMPLING
# ============================================================================

def _uniform(rng, cfg, key):
    """Sample uniformly from a (min, max) tuple in the config."""
    lo, hi = cfg[key]
    return rng.uniform(lo, hi)


def sample_well_properties(
    field_name: str,
    config: dict,
    well_id: str,
    sub_area: str,
    lat: float,
    lon: float,
    cause: int,
    seed: int,
) -> dict:
    """
    Sample all well parameters from the field config ranges.

    Returns the complete well dict consumed by simulate_well().
    """
    rng = np.random.RandomState(seed)

    initial_pressure = _uniform(rng, config, 'pressure_psi')
    initial_thickness = _uniform(rng, config, 'initial_thickness_mm')

    pipe_diam_range = config['pipe_diameter_in']
    if isinstance(pipe_diam_range, tuple):
        pipe_diam = rng.uniform(*pipe_diam_range)
    else:
        pipe_diam = pipe_diam_range

    # For "Others" with mixed casing, randomly pick
    casing_grade = config['casing_grade']
    if casing_grade == "Mixed":
        casing_grade = rng.choice(["N80", "L80", "P110"])

    well = {
        'well_id': well_id,
        'field_name': field_name,
        'sub_area': sub_area,
        'latitude': lat,
        'longitude': lon,
        'reservoir_type': config['reservoir_type'],
        'casing_grade': casing_grade,
        'casing_od_in': config['casing_od_in'],
        'seed': seed,

        # Reservoir
        'initial_pressure_psi': initial_pressure,
        'pressure_decline_annual_pct': _uniform(rng, config, 'pressure_decline_annual_pct'),
        'bottomhole_temp_F': _uniform(rng, config, 'temperature_F'),
        'co2_mol_frac': _uniform(rng, config, 'co2_mol_frac'),
        'h2s_mol_frac': _uniform(rng, config, 'h2s_mol_frac'),

        # Water cut
        'wc_midpoint_day': int(_uniform(rng, config, 'wc_midpoint_day')),
        'wc_steepness': _uniform(rng, config, 'wc_steepness'),
        'wc_max': _uniform(rng, config, 'wc_max'),

        # Flow
        'initial_flow_bpd': _uniform(rng, config, 'initial_flow_bpd'),
        'flow_decline_rate': _uniform(rng, config, 'flow_decline_rate'),
        'pipe_diameter_in': pipe_diam,
        'fluid_density_ppg': _uniform(rng, config, 'fluid_density_ppg'),
        'roughness_m': _uniform(rng, config, 'roughness_m'),
        'api_gravity': _uniform(rng, config, 'api_gravity'),
        'gor_scf_bbl': _uniform(rng, config, 'gor_scf_bbl'),

        # Casing
        'initial_thickness_mm': initial_thickness,

        # Inhibitor
        'inhibitor_efficiency': _uniform(rng, config, 'inhibitor_efficiency'),
        'inhibitor_active': rng.random() < config['inhibitor_active_prob'],
        'inhibitor_start_day': int(_uniform(rng, config, 'inhibitor_start_day')),
        'inhibitor_reliability': _uniform(rng, config, 'inhibitor_reliability'),

        # Pitting
        'pitting_factor': _uniform(rng, config, 'pitting_factor'),
        'pitting_probability': _uniform(rng, config, 'pitting_probability'),

        # Seasonality
        'ambient_temp_amplitude_F': _uniform(rng, config, 'ambient_temp_amplitude_F'),

        # Sensor noise
        'noise_temp': _uniform(rng, config, 'noise_temp'),
        'noise_pressure': _uniform(rng, config, 'noise_pressure'),
        'noise_flow': _uniform(rng, config, 'noise_flow'),
        'noise_ph': _uniform(rng, config, 'noise_ph'),

        # Shut-ins / choke
        'shutin_frequency': _uniform(rng, config, 'shutin_frequency'),
        'shutin_duration_min': config['shutin_duration_range'][0],
        'shutin_duration_max': config['shutin_duration_range'][1],
        'choke_change_frequency': _uniform(rng, config, 'choke_change_frequency'),
        'choke_change_magnitude': _uniform(rng, config, 'choke_change_magnitude'),

        # pH
        'initial_ph': _uniform(rng, config, 'initial_ph'),
        'ph_drift_rate': _uniform(rng, config, 'ph_drift_rate'),

        # Corrosion cause
        'corrosion_cause': cause,
    }
    return well


# ============================================================================
# VALIDATION
# ============================================================================

def validate_generated_data(df: pd.DataFrame) -> bool:
    """Post-generation sanity checks. Returns True if all pass."""
    errors = []

    # No NaN/Inf
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"NaN found in columns: {null_cols}")
    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        errors.append("Inf values found")

    # Range checks
    if (df['Corrosion_Rate_mpy'] < 0).any():
        errors.append("Negative corrosion rates found")
    if (df['Current_Thickness_mm'] < -0.01).any():
        errors.append("Negative thickness found")
    if (df['RUL_days'] < 0).any():
        errors.append("Negative RUL found")
    if (df['Water_Cut_pct'] < -0.01).any() or (df['Water_Cut_pct'] > 100.5).any():
        errors.append("Water cut out of [0, 100] range")
    if (df['pH'] < 2.5).any() or (df['pH'] > 8.0).any():
        errors.append("pH out of [2.5, 8.0] range")

    # Cause constant per well
    cause_nunique = df.groupby('Well_ID')['Corrosion_Cause'].nunique()
    if (cause_nunique > 1).any():
        bad = cause_nunique[cause_nunique > 1].index.tolist()
        errors.append(f"Corrosion_Cause not constant for wells: {bad}")

    # All 6 causes represented
    unique_causes = df['Corrosion_Cause'].unique()
    for c in range(6):
        if c not in unique_causes:
            errors.append(f"Cause {c} ({CAUSE_LABELS.get(c, '?')}) not represented")

    # Thickness non-increasing per well
    for wid in df['Well_ID'].unique()[:10]:  # spot-check first 10
        wdf = df[df['Well_ID'] == wid]
        active_mask = wdf['Status'] == 1
        thickness = wdf.loc[active_mask, 'Current_Thickness_mm'].values
        if len(thickness) > 1:
            diffs = np.diff(thickness)
            if (diffs > 0.01).any():  # small tolerance for rounding
                errors.append(f"Thickness increasing for well {wid}")

    if errors:
        print("\n  VALIDATION ERRORS:")
        for e in errors:
            print(f"    - {e}")
        return False

    print("  All validation checks PASSED.")
    return True


# ============================================================================
# MASTER GENERATION
# ============================================================================

def generate_full_dataset(n_days: int = 10950, seed: int = 42) -> pd.DataFrame:
    """
    Generate the complete 80-well dataset.

    1. Assign causes per field
    2. Sample well properties
    3. Simulate each well
    4. Concatenate, validate, return
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    print(f"{'=' * 65}")
    print(f"  SYNTHETIC CORROSION DATA GENERATION PIPELINE  (Phase 1)")
    print(f"  80 Wells | 30-Year Horizon | 6 Corrosion Mechanisms")
    print(f"{'=' * 65}")

    all_wells = []
    all_dfs = []
    well_counter = 0

    for field_name, n_wells in WELL_DISTRIBUTION.items():
        config = FIELD_CONFIGS[field_name]
        cause_dist = FIELD_CAUSE_DISTRIBUTION[field_name]
        causes = assign_causes(n_wells, cause_dist)

        sub_area_names = list(config['sub_areas'].keys())

        for i in range(n_wells):
            well_counter += 1
            wid = make_well_id(field_name, i + 1)
            cause = causes[i]

            # Assign sub-area (round-robin)
            sa_name = sub_area_names[i % len(sub_area_names)]
            sa_coords = config['sub_areas'][sa_name]
            lat = sa_coords['lat'] + rng.uniform(-0.02, 0.02)
            lon = sa_coords['lon'] + rng.uniform(-0.02, 0.02)

            # Sample well properties
            well_seed = seed * 1000 + well_counter
            well_dict = sample_well_properties(
                field_name, config, wid, sa_name, lat, lon, cause, well_seed
            )
            all_wells.append(well_dict)

            # Simulate
            t0 = time.time()
            df = simulate_well(well_dict, n_days)
            dt = time.time() - t0
            all_dfs.append(df)

            # Progress
            final_thickness = df['Current_Thickness_mm'].iloc[-1]
            failure_threshold = max(3.0, 0.5 * well_dict['initial_thickness_mm'])
            failed = final_thickness < failure_threshold
            life = len(df)
            cause_name = CAUSE_LABELS.get(cause, '?')
            status_str = f"FAILED day {life}" if failed else f"SURVIVED ({life} d)"

            if well_counter % 10 == 0 or well_counter <= 3 or well_counter == 80:
                print(f"  [{well_counter:3d}/80] {wid:>8s} | {field_name:<10s} | "
                      f"Cause: {cause_name:<9s} | "
                      f"{well_dict['initial_thickness_mm']:.1f} -> "
                      f"{final_thickness:.2f} mm | "
                      f"{status_str} | {dt:.1f}s")

    # Concatenate
    dataset = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'─' * 65}")
    print(f"  Dataset Shape:  {dataset.shape}")
    print(f"  Total Rows:     {len(dataset):,}")
    print(f"  Columns:        {len(dataset.columns)}")
    print(f"  Memory:         {dataset.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Failure stats
    well_final = dataset.groupby('Well_ID').agg({
        'Current_Thickness_mm': 'last',
        'Initial_Thickness_mm': 'first',
    })
    thresholds = np.maximum(3.0, 0.5 * well_final['Initial_Thickness_mm'])
    n_failed = (well_final['Current_Thickness_mm'] < thresholds).sum()
    print(f"  Wells Failed:   {n_failed}/80 ({100 * n_failed / 80:.0f}%)")

    # Cause distribution
    cause_counts = dataset.groupby('Well_ID')['Corrosion_Cause'].first().value_counts().sort_index()
    print(f"\n  Cause Distribution (wells):")
    for c, count in cause_counts.items():
        print(f"    {CAUSE_LABELS.get(c, '?'):<10s}: {count:2d} ({100 * count / 80:.0f}%)")

    print(f"\n{'─' * 65}")

    # Validate
    print("\n  Running validation...")
    validate_generated_data(dataset)

    return dataset


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    dataset = generate_full_dataset()

    csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'synthetic_corrosion_dataset.csv')

    dataset.to_csv(csv_path, index=False)
    file_size_mb = os.path.getsize(csv_path) / 1e6
    print(f"\n  CSV exported: {csv_path}")
    print(f"  File size:    {file_size_mb:.1f} MB")
    print(f"\n  Pipeline complete!")
