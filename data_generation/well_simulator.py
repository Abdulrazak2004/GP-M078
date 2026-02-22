"""
Single-well 30-year simulation engine.

simulate_well(well_dict, n_days=10950) -> DataFrame

Adapted from the existing generate_well_timeseries() with:
  - All outputs in API oilfield units (psi, F, bpd, fps, ppg, mpy)
  - H2S partial pressure
  - Beggs-Robinson viscosity (replaces hardcoded 3 cP)
  - Dissolved oxygen for O2/Combined wells
  - Corrosion cause dispatch to correct physics model
  - Waterflood-supported pressure decline
  - Casing metadata columns in every row
  - Thickness_Loss_Pct and Corrosion_Cause target columns
"""

import numpy as np
import pandas as pd

from data_generation.config_fields import (
    CAUSE_CO2, CAUSE_EROSION, CAUSE_OXYGEN, CAUSE_COMBINED,
)
from data_generation.corrosion_models import (
    compute_combined_corrosion,
    beggs_robinson_viscosity,
    norsok_m506_corrosion_rate,
    BAR_PER_PSI,
    F_TO_C,
    C_TO_F,
    MPY_PER_MMYR,
    BPD_TO_M3D,
    INCH_TO_M,
    PPG_TO_KGM3,
    MS_TO_FPS,
)


def simulate_well(well: dict, n_days: int = 10950) -> pd.DataFrame:
    """
    Generate the full operational time-series for a single well.

    Parameters
    ----------
    well : dict
        Well properties (sampled by generate_dataset.sample_well_properties).
    n_days : int
        Simulation length in days (default 10,950 = 30 years).

    Returns
    -------
    pd.DataFrame with 25 columns.
    """
    w = well
    days = np.arange(n_days)
    rng = np.random.RandomState(w.get('seed', None))

    # ===================================================================
    # 1. SHUT-IN STATUS  (1=online, 0=shut-in)
    # ===================================================================
    status = np.ones(n_days, dtype=int)
    d = 0
    while d < n_days:
        if rng.random() < w['shutin_frequency'] and status[d] == 1:
            duration = rng.randint(w['shutin_duration_min'], w['shutin_duration_max'] + 1)
            end = min(d + duration, n_days)
            status[d:end] = 0
            d = end
        else:
            d += 1
    shutin_mask = (status == 0)

    # ===================================================================
    # 2. RESERVOIR PRESSURE  (psi, with waterflood support after year 10)
    # ===================================================================
    annual_decline = w['pressure_decline_annual_pct'] / 100.0
    daily_decline = annual_decline / 365.0
    decline_rate = np.full(n_days, daily_decline)
    # Waterflood support: decline slows 40% after year 10 (day 3650)
    decline_rate[3650:] *= 0.60
    cum_decline = np.cumsum(decline_rate)
    base_pressure_psi = w['initial_pressure_psi'] * np.exp(-cum_decline)

    # ===================================================================
    # 3. WATER CUT  (sigmoid breakthrough, midpoint year 7-14)
    # ===================================================================
    water_cut = w['wc_max'] / (1.0 + np.exp(
        -w['wc_steepness'] * (days - w['wc_midpoint_day'])
    ))

    # ===================================================================
    # 4. TEMPERATURE  (°F, with seasonal cycling)
    # ===================================================================
    seasonal_F = w['ambient_temp_amplitude_F'] * np.sin(
        2.0 * np.pi * (days - rng.uniform(0, 365)) / 365.0
    )
    surface_blend = 0.15
    temperature_F = w['bottomhole_temp_F'] + surface_blend * seasonal_F

    # During shut-ins, BHT dips ~5 °F (no flow to maintain heat transfer)
    temperature_F[shutin_mask] -= 5.0

    # ===================================================================
    # 5. FLOW RATE  (bbl/d, with decline + choke changes)
    # ===================================================================
    flow_rate_bpd = w['initial_flow_bpd'] * np.exp(-w['flow_decline_rate'] * days)

    for d_idx in range(n_days):
        if rng.random() < w['choke_change_frequency']:
            mag = rng.uniform(-w['choke_change_magnitude'], w['choke_change_magnitude'])
            persist = rng.randint(5, 60)
            end_idx = min(d_idx + persist, n_days)
            flow_rate_bpd[d_idx:end_idx] *= (1.0 + mag)

    flow_rate_bpd = np.maximum(flow_rate_bpd, 0.0)
    flow_rate_bpd[shutin_mask] = 0.0

    # ===================================================================
    # 6. CO2 PARTIAL PRESSURE  (psi)
    # ===================================================================
    co2_enhancement = 1.0 + 0.15 * water_cut
    fco2_psi = base_pressure_psi * w['co2_mol_frac'] * co2_enhancement

    # ===================================================================
    # 7. H2S PARTIAL PRESSURE  (psi)   — NEW
    # ===================================================================
    p_h2s_psi = base_pressure_psi * w['h2s_mol_frac']

    # ===================================================================
    # 8. pH EVOLUTION  (computed after CO2 PP to couple pH-CO2)
    # ===================================================================
    # pH is computed below (section 8b) after CO2 partial pressure conversion

    # ===================================================================
    # 9. VISCOSITY  (cP)   — NEW, Beggs-Robinson
    # ===================================================================
    viscosity_cP = beggs_robinson_viscosity(
        temperature_F, w['api_gravity'], w['gor_scf_bbl'], water_cut
    )

    # ===================================================================
    # 10. FLOW VELOCITY (fps) + SHEAR STRESS (Pa)
    # ===================================================================
    pipe_diameter_m = w['pipe_diameter_in'] * INCH_TO_M
    pipe_area = np.pi * (pipe_diameter_m / 2.0) ** 2

    # Convert bpd -> m³/s for velocity calculation
    flow_rate_m3s = flow_rate_bpd * BPD_TO_M3D / 86400.0
    flow_velocity_ms = flow_rate_m3s / pipe_area
    flow_velocity_ms = np.maximum(flow_velocity_ms, 0.0)

    # Fluid density (ppg -> kg/m³ for physics, increases with water cut)
    density_ppg = w['fluid_density_ppg'] + (8.34 - w['fluid_density_ppg']) * water_cut
    density_kgm3 = density_ppg * PPG_TO_KGM3

    # Viscosity for Reynolds number (cP -> Pa·s)
    mu_Pa_s = viscosity_cP * 1e-3

    Re = np.where(
        flow_velocity_ms > 0,
        density_kgm3 * flow_velocity_ms * pipe_diameter_m / np.maximum(mu_Pa_s, 1e-6),
        0.0
    )

    # Colebrook-White friction factor
    friction = np.where(
        Re > 2300,
        0.25 / (np.log10(
            w['roughness_m'] / (3.7 * pipe_diameter_m) +
            5.74 / np.maximum(Re, 1.0) ** 0.9
        )) ** 2,
        np.where(Re > 0, 64.0 / np.maximum(Re, 1.0), 0.0)
    )

    # Wall shear stress: tau = f/2 * rho * v^2
    shear_stress_Pa = 0.5 * friction * density_kgm3 * flow_velocity_ms ** 2
    shear_stress_Pa = np.maximum(shear_stress_Pa, 0.0)

    # ===================================================================
    # 11. DISSOLVED OXYGEN  (ppb)   — NEW
    # ===================================================================
    # Only meaningful for Oxygen and Combined wells
    if w['corrosion_cause'] in (CAUSE_OXYGEN, CAUSE_COMBINED):
        # Baseline 5-15 ppb with periodic scavenger-failure spikes
        o2_baseline = rng.uniform(5.0, 15.0)
        dissolved_o2 = np.full(n_days, o2_baseline)
        # Scavenger failure spikes: ~2-4 per year, lasting 3-10 days
        spike_prob = 0.008
        for d_idx in range(n_days):
            if rng.random() < spike_prob:
                spike_intensity = rng.uniform(50.0, 300.0)  # ppb
                spike_dur = rng.randint(3, 11)
                end_idx = min(d_idx + spike_dur, n_days)
                dissolved_o2[d_idx:end_idx] = spike_intensity
        # Add noise
        dissolved_o2 += rng.normal(0, 2.0, n_days)
        dissolved_o2 = np.maximum(dissolved_o2, 0.0)
    else:
        dissolved_o2 = np.full(n_days, 5.0)  # nominal background

    # ===================================================================
    # 12. CORROSION RATE  (mm/yr, dispatched to correct mechanism)
    # ===================================================================
    # Convert to metric for corrosion models
    temp_C = F_TO_C(temperature_F)
    fco2_bar = fco2_psi * BAR_PER_PSI
    p_h2s_bar = p_h2s_psi * BAR_PER_PSI

    # 8b. pH EVOLUTION — coupled to CO2 partial pressure
    ph = (w['initial_ph'] + w['ph_drift_rate'] * days
          - 0.15 * water_cut
          - 0.08 * np.maximum(fco2_bar - 1.0, 0.0))
    ph = np.clip(ph, 3.5, 6.5)

    cr_mmyr = compute_combined_corrosion(
        cause=w['corrosion_cause'],
        temperature_C=temp_C,
        fco2_bar=fco2_bar,
        p_h2s_bar=p_h2s_bar,
        ph=ph,
        shear_stress_Pa=shear_stress_Pa,
        velocity_ms=flow_velocity_ms,
        density_kgm3=density_kgm3,
        water_cut=water_cut,
        dissolved_o2_ppb=dissolved_o2,
        days=days,
        rng=rng,
    )

    # --- Oil wetting factor (CO2-based mechanisms only) ---
    # Below ~25% WC the pipe surface is oil-wet → negligible corrosion.
    # Full water-wetting above ~75% WC.  H2S/MIC/O2 already have internal
    # water-phase dependence so this only applies to CO2, Erosion, Combined.
    if w['corrosion_cause'] in (CAUSE_CO2, CAUSE_EROSION, CAUSE_COMBINED):
        wetting = np.clip((water_cut - 0.20) / 0.55, 0.0, 1.0)
        cr_mmyr *= wetting

    # --- FeCO3 protective scale (CO2-based mechanisms only) ---
    # At temperatures > 60 °C, protective iron-carbonate scale builds
    # up over time and reduces the effective corrosion rate.
    if w['corrosion_cause'] in (CAUSE_CO2, CAUSE_EROSION, CAUSE_COMBINED):
        scale_protection = np.where(
            temp_C > 60.0,
            np.maximum(1.0 - 0.80 * (1.0 - np.exp(-days / 1000.0)), 0.15),
            1.0,
        )
        cr_mmyr *= scale_protection

    # Zero corrosion during shut-ins
    cr_mmyr[shutin_mask] = 0.0

    # ===================================================================
    # 13. INHIBITOR EFFECT
    # ===================================================================
    if w['inhibitor_active']:
        inhibitor_mask = (days >= w['inhibitor_start_day'])
        inhibitor_working = rng.random(n_days) < w['inhibitor_reliability']
        active = inhibitor_mask & inhibitor_working & (~shutin_mask)
        cr_mmyr[active] *= (1.0 - w['inhibitor_efficiency'])

    # ===================================================================
    # 14. PITTING FACTOR  (stochastic localized corrosion events)
    # ===================================================================
    pitting_events = rng.random(n_days) < w['pitting_probability']
    pitting_multiplier = np.ones(n_days)
    pitting_multiplier[pitting_events] = w['pitting_factor']
    cr_effective_mmyr = cr_mmyr * pitting_multiplier

    # ===================================================================
    # 15. WALL THICKNESS DEGRADATION
    # ===================================================================
    daily_loss_mm = cr_effective_mmyr / 365.0
    cumulative_loss_mm = np.cumsum(daily_loss_mm)
    thickness_mm = w['initial_thickness_mm'] - cumulative_loss_mm

    # ===================================================================
    # 16. FAILURE DETECTION + RUL
    # ===================================================================
    failure_threshold_mm = max(3.0, 0.5 * w['initial_thickness_mm'])
    failure_day = n_days
    failed_indices = np.where(thickness_mm < failure_threshold_mm)[0]
    if len(failed_indices) > 0:
        failure_day = failed_indices[0]

    rul = np.maximum(failure_day - days, 0).astype(float)
    if failure_day < n_days:
        rul[failure_day:] = 0

    # ===================================================================
    # 17. SENSOR NOISE
    # ===================================================================
    temp_observed_F = temperature_F * (1.0 + rng.normal(0, w['noise_temp'], n_days))
    pressure_observed_psi = base_pressure_psi * (1.0 + rng.normal(0, w['noise_pressure'], n_days))
    flow_observed_bpd = flow_rate_bpd * (1.0 + rng.normal(0, w['noise_flow'], n_days))
    flow_observed_bpd = np.maximum(flow_observed_bpd, 0.0)
    ph_observed = ph + rng.normal(0, w['noise_ph'] * 5, n_days)
    ph_observed = np.clip(ph_observed, 3.0, 7.5)

    # ===================================================================
    # 18. UNIT CONVERSIONS FOR OUTPUT
    # ===================================================================
    flow_velocity_fps = flow_velocity_ms * MS_TO_FPS
    cr_mpy = cr_effective_mmyr * MPY_PER_MMYR
    thickness_loss_pct = (
        (w['initial_thickness_mm'] - thickness_mm) / w['initial_thickness_mm'] * 100.0
    )
    thickness_loss_pct = np.clip(thickness_loss_pct, 0.0, 100.0)

    # ===================================================================
    # 19. BUILD 25-COLUMN DATAFRAME
    # ===================================================================
    df = pd.DataFrame({
        # Metadata (constant per well)
        'Well_ID': w['well_id'],
        'Field_Name': w['field_name'],
        'Sub_Area': w['sub_area'],
        'Latitude': round(w['latitude'], 5),
        'Longitude': round(w['longitude'], 5),
        'Reservoir_Type': w['reservoir_type'],
        'Casing_Grade': w['casing_grade'],
        'Casing_OD_in': round(w['casing_od_in'], 3),
        'Initial_Thickness_mm': round(w['initial_thickness_mm'], 2),
        # Time-series features
        'Day': days,
        'Status': status,
        'Pressure_psi': np.round(pressure_observed_psi, 2),
        'Temp_F': np.round(temp_observed_F, 2),
        'pH': np.round(ph_observed, 3),
        'Water_Cut_pct': np.round(water_cut * 100.0, 2),
        'Production_Rate_bpd': np.round(flow_observed_bpd, 2),
        'Flow_Velocity_fps': np.round(flow_velocity_fps, 4),
        'Shear_Stress_Pa': np.round(shear_stress_Pa, 4),
        'CO2_Partial_Pressure_psi': np.round(fco2_psi, 4),
        'H2S_Partial_Pressure_psi': np.round(p_h2s_psi, 4),
        'Fluid_Density_ppg': np.round(density_ppg, 3),
        'Viscosity_cP': np.round(viscosity_cP, 3),
        'Inhibitor_Active': int(w['inhibitor_active']),
        # Targets
        'Corrosion_Rate_mpy': np.round(cr_mpy, 4),
        'Current_Thickness_mm': np.round(thickness_mm, 4),
        'Thickness_Loss_Pct': np.round(thickness_loss_pct, 2),
        'RUL_days': rul.astype(int),
        'Corrosion_Cause': w['corrosion_cause'],
    })

    # ===================================================================
    # 20. TRUNCATE AFTER FAILURE + 30-DAY BUFFER
    # ===================================================================
    if failure_day < n_days:
        df = df.iloc[:failure_day + 30].copy()
        df.loc[df['Day'] >= failure_day, 'Status'] = 0

    return df
