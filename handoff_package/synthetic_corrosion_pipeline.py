#!/usr/bin/env python3
"""
===============================================================================
Synthetic Corrosion Data Generation Pipeline
Project: Casing Life Prediction (LSTM / Transformer Training Data)
Standard: NORSOK M-506 (CO2 Corrosion Rate Model)
===============================================================================

Generates high-fidelity synthetic operational data for 50 oil wells over
~2000 days each, including:
  - NORSOK M-506 physics kernel (CO2 corrosion)
  - Reservoir decline, seasonality, shut-ins, choke changes
  - Wall thickness degradation & RUL labels
  - Corrosion inhibitor effects, pitting factors, sensor noise

Author: AI Data Engineering Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# SECTION 1: NORSOK M-506 CONSTANTS & LOOKUP TABLES
# =============================================================================

# Temperature-dependent constant K_T (from NORSOK M-506 Table 1)
# Temperatures in °C, K_T values (mm/yr base rate constant)
NORSOK_TEMPS = np.array([5, 15, 20, 40, 60, 80, 90, 120, 150])
NORSOK_KT = np.array([0.42, 0.59, 0.70, 1.59, 4.57, 8.50, 9.40, 6.30, 3.80])

# pH correction factors f(pH) at standard temperatures
# Rows: pH values, Columns: temperatures
NORSOK_PH_VALUES = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
NORSOK_PH_FACTORS = {
    20:  np.array([7.0, 4.6, 2.8, 1.6, 1.0, 0.7, 0.5]),
    40:  np.array([6.8, 4.4, 2.6, 1.5, 1.0, 0.7, 0.4]),
    60:  np.array([6.5, 4.2, 2.5, 1.4, 1.0, 0.6, 0.3]),
    80:  np.array([6.2, 4.0, 2.3, 1.3, 1.0, 0.5, 0.2]),
    120: np.array([5.5, 3.5, 2.0, 1.1, 1.0, 0.4, 0.1]),
    150: np.array([5.0, 3.2, 1.8, 1.0, 1.0, 0.3, 0.08]),
}
NORSOK_PH_TEMP_KEYS = np.array(sorted(NORSOK_PH_FACTORS.keys()))


def interpolate_kt(temperature: np.ndarray) -> np.ndarray:
    """Interpolate K_T from NORSOK table for given temperatures."""
    interp_func = interp1d(NORSOK_TEMPS, NORSOK_KT, kind='cubic',
                           fill_value='extrapolate')
    kt = interp_func(np.clip(temperature, 5, 150))
    return np.maximum(kt, 0.01)  # floor to avoid negative from extrapolation


def interpolate_ph_factor(temperature: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """
    2D interpolation of pH correction factor f(pH)_T.
    Interpolates across both temperature and pH dimensions.
    """
    # Build a 2D grid of pH factors
    ph_factor_matrix = np.array([NORSOK_PH_FACTORS[t] for t in NORSOK_PH_TEMP_KEYS])
    # shape: (n_temps, n_ph)

    result = np.zeros_like(temperature, dtype=float)
    for i in range(len(temperature)):
        t = np.clip(temperature[i], 20, 150)
        p = np.clip(ph[i], 3.5, 6.5)

        # Interpolate along pH axis for each reference temperature
        ph_at_temps = np.array([
            np.interp(p, NORSOK_PH_VALUES, ph_factor_matrix[j])
            for j in range(len(NORSOK_PH_TEMP_KEYS))
        ])
        # Interpolate along temperature axis
        result[i] = np.interp(t, NORSOK_PH_TEMP_KEYS, ph_at_temps)

    return np.maximum(result, 0.01)


def norsok_m506_corrosion_rate(
    temperature: np.ndarray,
    fco2: np.ndarray,
    ph: np.ndarray,
    shear_stress: np.ndarray
) -> np.ndarray:
    """
    NORSOK M-506 CO2 Corrosion Rate (mm/yr).

    CR = K_T × f_CO2^0.62 × (S/19)^(0.146 + 0.0324·log(f_CO2)) × f(pH)_T

    Parameters
    ----------
    temperature : array, °C
    fco2 : array, CO2 fugacity / partial pressure (bar)
    ph : array, pH of formation water
    shear_stress : array, wall shear stress (Pa)

    Returns
    -------
    cr : array, corrosion rate in mm/yr
    """
    kt = interpolate_kt(temperature)

    fco2_safe = np.maximum(fco2, 1e-6)
    shear_safe = np.maximum(shear_stress, 0.1)

    # CO2 fugacity term
    fco2_term = fco2_safe ** 0.62

    # Shear stress term with CO2-dependent exponent
    shear_exponent = 0.146 + 0.0324 * np.log10(fco2_safe)
    shear_term = (shear_safe / 19.0) ** shear_exponent

    # pH correction
    ph_factor = interpolate_ph_factor(temperature, ph)

    cr = kt * fco2_term * shear_term * ph_factor
    return np.maximum(cr, 0.0)


# =============================================================================
# SECTION 2: WELL PARAMETER PROFILES (Per-Well Randomized Properties)
# =============================================================================

def generate_well_properties(n_wells: int = 50) -> list:
    """
    Generate unique physical/chemical properties for each well.
    Models realistic variability in Saudi Arabian carbonate reservoirs.
    """
    wells = []
    for i in range(n_wells):
        well = {
            'well_id': f'WELL-{i+1:03d}',

            # --- Reservoir Properties ---
            'initial_reservoir_pressure_bar': np.random.uniform(180, 350),
            'pressure_decline_rate': np.random.uniform(0.0005, 0.002),  # exp decay
            'bottomhole_temp_C': np.random.uniform(80, 130),

            # --- Formation Water Chemistry ---
            'initial_pH': np.random.uniform(4.8, 6.2),
            'pH_drift_rate': np.random.uniform(-0.0002, 0.0003),  # pH changes w/ water cut
            'co2_mole_fraction': np.random.uniform(0.005, 0.04),  # 0.5-4% CO2 (typical)

            # --- Flow Properties ---
            'initial_flow_rate_m3d': np.random.uniform(150, 600),  # m³/day
            'flow_decline_rate': np.random.uniform(0.0002, 0.0008),
            'fluid_density_kgm3': np.random.uniform(780, 900),  # oil-water mix
            'pipe_diameter_m': np.random.choice([0.1016, 0.1143, 0.1270]),  # 4", 4.5", 5"
            'roughness': np.random.uniform(0.00015, 0.0005),  # pipe roughness (m)

            # --- Water Cut (Sigmoid Breakthrough) ---
            'water_cut_midpoint_day': np.random.randint(400, 1200),
            'water_cut_steepness': np.random.uniform(0.004, 0.012),
            'water_cut_max': np.random.uniform(0.75, 0.98),

            # --- Casing Properties ---
            'initial_wall_thickness_mm': np.random.uniform(10.0, 13.5),
            'failure_threshold_mm': 3.0,

            # --- Corrosion Inhibitor ---
            'inhibitor_efficiency': np.random.uniform(0.70, 0.95),  # 70-95% when active
            'inhibitor_active': np.random.choice([True, False], p=[0.8, 0.2]),
            'inhibitor_start_day': np.random.randint(0, 300),
            'inhibitor_reliability': np.random.uniform(0.85, 0.98),  # uptime fraction

            # --- Pitting Factor (localized corrosion multiplier) ---
            'pitting_factor': np.random.uniform(1.2, 2.5),  # typically 1.5-3x
            'pitting_probability': np.random.uniform(0.005, 0.025),  # less frequent

            # --- Seasonality (Surface Temperature Effect) ---
            'ambient_temp_mean_C': np.random.uniform(32, 40),  # Saudi Arabia
            'ambient_temp_amplitude_C': np.random.uniform(8, 15),  # summer-winter swing
            'season_phase_days': np.random.uniform(0, 365),  # offset

            # --- Operational Chaos ---
            'shutin_frequency': np.random.uniform(0.001, 0.006),  # ~1-2 per year
            'shutin_duration_range': (3, 7),
            'choke_change_frequency': np.random.uniform(0.005, 0.02),
            'choke_change_magnitude': np.random.uniform(0.1, 0.4),  # fraction of flow

            # --- Sensor Noise (% of true value) ---
            'noise_temp': np.random.uniform(0.005, 0.02),
            'noise_pressure': np.random.uniform(0.01, 0.03),
            'noise_flow': np.random.uniform(0.02, 0.05),
            'noise_ph': np.random.uniform(0.005, 0.015),
        }
        wells.append(well)
    return wells


# =============================================================================
# SECTION 3: SCENARIO ENGINE (Generate Time-Series Inputs)
# =============================================================================

def generate_well_timeseries(well: dict, n_days: int = 2000) -> pd.DataFrame:
    """
    Generate the full operational time-series for a single well.
    Combines reservoir physics, seasonality, and operational events.
    """
    days = np.arange(n_days)
    w = well  # shorthand

    # --- Status Array (1=online, 0=shut-in) ---
    status = np.ones(n_days, dtype=int)
    d = 0
    while d < n_days:
        if np.random.random() < w['shutin_frequency'] and status[d] == 1:
            duration = np.random.randint(*w['shutin_duration_range'])
            end = min(d + duration, n_days)
            status[d:end] = 0
            d = end
        else:
            d += 1

    # --- Reservoir Pressure (Exponential Decline) ---
    base_pressure = w['initial_reservoir_pressure_bar'] * np.exp(
        -w['pressure_decline_rate'] * days
    )

    # --- Water Cut (Sigmoid Breakthrough) ---
    water_cut = w['water_cut_max'] / (1 + np.exp(
        -w['water_cut_steepness'] * (days - w['water_cut_midpoint_day'])
    ))

    # --- Temperature (Bottomhole + Seasonal Surface Effect) ---
    # BHT is stable; wellhead temp is affected by ambient
    seasonal_effect = w['ambient_temp_amplitude_C'] * np.sin(
        2 * np.pi * (days - w['season_phase_days']) / 365.0
    )
    # Blend: deeper wells are less affected by surface temp
    surface_blend = 0.15  # 15% surface influence on downhole measurements
    temperature = w['bottomhole_temp_C'] + surface_blend * seasonal_effect

    # During shut-ins, temperature drops toward geothermal gradient (~40°C)
    geothermal_temp = 40.0
    shutin_mask = (status == 0)
    temperature[shutin_mask] = geothermal_temp + 0.3 * (
        temperature[shutin_mask] - geothermal_temp
    )

    # --- Flow Rate with Decline & Choke Changes ---
    flow_rate = w['initial_flow_rate_m3d'] * np.exp(-w['flow_decline_rate'] * days)

    # Apply random choke changes (step changes)
    for d_idx in range(n_days):
        if np.random.random() < w['choke_change_frequency']:
            magnitude = np.random.uniform(
                -w['choke_change_magnitude'],
                w['choke_change_magnitude']
            )
            # Step change persists for a random duration
            persist = np.random.randint(5, 60)
            end_idx = min(d_idx + persist, n_days)
            flow_rate[d_idx:end_idx] *= (1 + magnitude)

    flow_rate = np.maximum(flow_rate, 0)
    flow_rate[shutin_mask] = 0  # No flow during shut-ins

    # --- CO2 Partial Pressure ---
    # P_CO2 = Total_Pressure × CO2_mole_fraction
    # As water cut increases, dissolved CO2 can increase
    co2_enhancement = 1.0 + 0.15 * water_cut  # modest increase with water
    fco2 = base_pressure * w['co2_mole_fraction'] * co2_enhancement

    # --- pH Evolution ---
    # pH tends to decrease as water cut increases (more formation water)
    ph = w['initial_pH'] + w['pH_drift_rate'] * days - 0.15 * water_cut
    ph = np.clip(ph, 3.5, 6.5)

    # --- Flow Velocity & Shear Stress ---
    pipe_area = np.pi * (w['pipe_diameter_m'] / 2) ** 2
    # Convert m³/day to m/s
    flow_velocity = (flow_rate / 86400.0) / pipe_area  # m/s
    flow_velocity = np.maximum(flow_velocity, 0)

    # Fluid density increases with water cut (water ~1020 kg/m³, oil ~850 kg/m³)
    density = w['fluid_density_kgm3'] + 170 * water_cut  # increases toward water density

    # Simplified Fanning friction factor (Moody chart approximation)
    Re = np.where(flow_velocity > 0,
                  density * flow_velocity * w['pipe_diameter_m'] / 0.003,  # μ ≈ 3 cP
                  0)
    # Colebrook-White approximation
    friction = np.where(
        Re > 2300,
        0.25 / (np.log10(w['roughness'] / (3.7 * w['pipe_diameter_m']) +
                         5.74 / np.maximum(Re, 1) ** 0.9)) ** 2,
        np.where(Re > 0, 64.0 / np.maximum(Re, 1), 0)
    )

    # Wall shear stress: τ = f/2 × ρ × v²
    shear_stress = 0.5 * friction * density * flow_velocity ** 2
    shear_stress = np.maximum(shear_stress, 0)

    # --- Corrosion Rate Calculation (NORSOK M-506) ---
    cr = norsok_m506_corrosion_rate(temperature, fco2, ph, shear_stress)

    # Zero corrosion during shut-ins
    cr[shutin_mask] = 0

    # --- Corrosion Inhibitor Effect ---
    if w['inhibitor_active']:
        inhibitor_mask = (days >= w['inhibitor_start_day'])
        # Inhibitor reliability: random daily failures
        inhibitor_working = np.random.random(n_days) < w['inhibitor_reliability']
        active = inhibitor_mask & inhibitor_working & (~shutin_mask)
        cr[active] *= (1.0 - w['inhibitor_efficiency'])

    # --- Pitting Factor (Stochastic Localized Corrosion Events) ---
    pitting_events = np.random.random(n_days) < w['pitting_probability']
    pitting_multiplier = np.ones(n_days)
    pitting_multiplier[pitting_events] = w['pitting_factor']
    cr_effective = cr * pitting_multiplier

    # --- Wall Thickness Degradation ---
    # Convert mm/yr to mm/day
    daily_loss = cr_effective / 365.0
    cumulative_loss = np.cumsum(daily_loss)
    thickness = w['initial_wall_thickness_mm'] - cumulative_loss

    # --- Determine Failure & RUL ---
    failure_day = n_days  # default: survives full period
    failed_indices = np.where(thickness < w['failure_threshold_mm'])[0]
    if len(failed_indices) > 0:
        failure_day = failed_indices[0]

    rul = np.maximum(failure_day - days, 0).astype(float)
    # After failure, RUL = 0
    if failure_day < n_days:
        rul[failure_day:] = 0

    # --- Apply Sensor Noise to "Observed" Features ---
    temp_observed = temperature * (1 + np.random.normal(0, w['noise_temp'], n_days))
    pressure_observed = base_pressure * (1 + np.random.normal(0, w['noise_pressure'], n_days))
    flow_observed = flow_rate * (1 + np.random.normal(0, w['noise_flow'], n_days))
    ph_observed = ph + np.random.normal(0, w['noise_ph'] * 5, n_days)  # absolute noise
    ph_observed = np.clip(ph_observed, 3.0, 7.0)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        'Well_ID': w['well_id'],
        'Day': days,
        'Status': status,
        'Pressure_bar': np.round(pressure_observed, 2),
        'Temp_C': np.round(temp_observed, 2),
        'pH': np.round(ph_observed, 3),
        'Water_Cut': np.round(water_cut, 4),
        'Flow_Rate_m3d': np.round(flow_observed, 2),
        'Flow_Velocity_ms': np.round(flow_velocity, 4),
        'Shear_Stress_Pa': np.round(shear_stress, 4),
        'CO2_Partial_Pressure_bar': np.round(fco2, 4),
        'Fluid_Density_kgm3': np.round(density, 2),
        'Corrosion_Rate_mm_yr': np.round(cr_effective, 6),
        'Current_Thickness_mm': np.round(thickness, 4),
        'Inhibitor_Active': int(w['inhibitor_active']),
        'RUL_days': rul.astype(int),
    })

    # Truncate after failure + small buffer (well doesn't report after death)
    if failure_day < n_days:
        df = df.iloc[:failure_day + 30].copy()  # 30-day post-failure buffer
        df.loc[df['Day'] >= failure_day, 'Status'] = 0

    return df


# =============================================================================
# SECTION 4: FULL DATASET GENERATION
# =============================================================================

def generate_full_dataset(n_wells: int = 50, n_days: int = 2000) -> pd.DataFrame:
    """Generate the complete multi-well dataset."""
    print(f"{'='*60}")
    print(f"  SYNTHETIC CORROSION DATA GENERATION PIPELINE")
    print(f"  NORSOK M-506 | {n_wells} Wells | {n_days} Days Max")
    print(f"{'='*60}")

    wells = generate_well_properties(n_wells)
    all_dfs = []

    for i, well in enumerate(wells):
        df = generate_well_timeseries(well, n_days)
        all_dfs.append(df)

        # Progress
        failed = df['Current_Thickness_mm'].iloc[-1] < well['failure_threshold_mm']
        life = len(df)
        status_str = f"FAILED day {life}" if failed else f"SURVIVED ({life} days)"
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Well {i+1:3d}/{n_wells}: {well['well_id']} | "
                  f"Thickness: {well['initial_wall_thickness_mm']:.1f} → "
                  f"{df['Current_Thickness_mm'].iloc[-1]:.2f} mm | {status_str}")

    dataset = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'─'*60}")
    print(f"  Dataset Shape: {dataset.shape}")
    print(f"  Total Rows:    {len(dataset):,}")
    print(f"  Columns:       {len(dataset.columns)}")
    print(f"  Wells Failed:  {sum(1 for df in all_dfs if df['Current_Thickness_mm'].iloc[-1] < 3.0)}/{n_wells}")
    print(f"  Memory:        {dataset.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"{'─'*60}")

    return dataset


# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================

def plot_well_history(dataset: pd.DataFrame, well_id: str = None):
    """
    Visualize a single well's complete operational history.
    Shows Temperature, Pressure, Corrosion Rate, Thickness, Water Cut, and RUL.
    """
    if well_id is None:
        # Pick a well that failed (more interesting to visualize)
        failed_wells = []
        for wid in dataset['Well_ID'].unique():
            wdf = dataset[dataset['Well_ID'] == wid]
            if wdf['Current_Thickness_mm'].iloc[-1] < 3.5:
                failed_wells.append(wid)
        well_id = failed_wells[0] if failed_wells else dataset['Well_ID'].unique()[0]

    df = dataset[dataset['Well_ID'] == well_id].copy()
    days = df['Day'].values

    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(6, 2, hspace=0.35, wspace=0.25)

    # Color scheme
    colors = {
        'temp': '#E74C3C', 'pressure': '#3498DB', 'thickness': '#2ECC71',
        'cr': '#E67E22', 'water': '#9B59B6', 'rul': '#1ABC9C',
        'flow': '#34495E', 'ph': '#F39C12', 'shutin': '#BDC3C7',
    }

    # Mark shut-in periods
    shutin_mask = df['Status'] == 0

    # --- Plot 1: Temperature ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(days, df['Temp_C'], color=colors['temp'], linewidth=0.5, alpha=0.8)
    ax1.fill_between(days, df['Temp_C'].min(), df['Temp_C'],
                     where=shutin_mask, color=colors['shutin'], alpha=0.3, label='Shut-in')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'{well_id} — Temperature Profile', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Pressure ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(days, df['Pressure_bar'], color=colors['pressure'], linewidth=0.5, alpha=0.8)
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title(f'{well_id} — Reservoir Pressure Decline', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Water Cut ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(days, df['Water_Cut'] * 100, color=colors['water'], linewidth=1.2)
    ax3.set_ylabel('Water Cut (%)')
    ax3.set_title('Water Cut (Sigmoid Breakthrough)', fontweight='bold')
    ax3.set_ylim(-2, 102)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: pH ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(days, df['pH'], color=colors['ph'], linewidth=0.5, alpha=0.8)
    ax4.set_ylabel('pH')
    ax4.set_title('Formation Water pH', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Flow Velocity ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(days, df['Flow_Velocity_ms'], color=colors['flow'], linewidth=0.5, alpha=0.8)
    ax5.set_ylabel('Flow Velocity (m/s)')
    ax5.set_title('Flow Velocity (with Choke Changes)', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # --- Plot 6: CO2 Partial Pressure ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(days, df['CO2_Partial_Pressure_bar'], color='#8E44AD', linewidth=0.5, alpha=0.8)
    ax6.set_ylabel('P_CO2 (bar)')
    ax6.set_title('CO₂ Partial Pressure', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # --- Plot 7: Corrosion Rate ---
    ax7 = fig.add_subplot(gs[3, :])
    ax7.plot(days, df['Corrosion_Rate_mm_yr'], color=colors['cr'],
             linewidth=0.5, alpha=0.8)
    ax7.set_ylabel('Corrosion Rate (mm/yr)')
    ax7.set_title('NORSOK M-506 Corrosion Rate (with Pitting Spikes)', fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # --- Plot 8: Wall Thickness Degradation ---
    ax8 = fig.add_subplot(gs[4, :])
    ax8.plot(days, df['Current_Thickness_mm'], color=colors['thickness'],
             linewidth=1.5, label='Wall Thickness')
    ax8.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5,
                label='Failure Threshold (3mm)')
    ax8.fill_between(days, 0, df['Current_Thickness_mm'],
                     where=df['Current_Thickness_mm'] < 3.0,
                     color='red', alpha=0.2, label='Below Threshold')
    ax8.set_ylabel('Thickness (mm)')
    ax8.set_title('Casing Wall Thickness Degradation', fontweight='bold')
    ax8.legend(loc='upper right')
    ax8.grid(True, alpha=0.3)

    # --- Plot 9: RUL ---
    ax9 = fig.add_subplot(gs[5, :])
    ax9.plot(days, df['RUL_days'], color=colors['rul'], linewidth=1.5)
    ax9.set_ylabel('RUL (days)')
    ax9.set_xlabel('Day')
    ax9.set_title('Remaining Useful Life (Target Variable)', fontweight='bold')
    ax9.grid(True, alpha=0.3)

    plt.suptitle(f'Well Operational History: {well_id}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('/home/claude/well_history_plot.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"\n  Plot saved: well_history_plot.png ({well_id})")


def plot_fleet_summary(dataset: pd.DataFrame):
    """Summary statistics across all wells."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Thickness trajectories for all wells
    ax = axes[0, 0]
    for wid in dataset['Well_ID'].unique():
        wdf = dataset[dataset['Well_ID'] == wid]
        color = '#E74C3C' if wdf['Current_Thickness_mm'].iloc[-1] < 3.0 else '#2ECC71'
        ax.plot(wdf['Day'], wdf['Current_Thickness_mm'], linewidth=0.5,
                alpha=0.4, color=color)
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title('All Wells — Thickness Trajectories', fontweight='bold')
    ax.set_xlabel('Day')
    ax.set_ylabel('Thickness (mm)')
    ax.grid(True, alpha=0.3)

    # 2. Corrosion rate distribution
    ax = axes[0, 1]
    cr_data = dataset[dataset['Status'] == 1]['Corrosion_Rate_mm_yr']
    ax.hist(cr_data[cr_data > 0], bins=100, color='#E67E22', alpha=0.7, edgecolor='none')
    ax.set_title('Corrosion Rate Distribution (Active Days)', fontweight='bold')
    ax.set_xlabel('Corrosion Rate (mm/yr)')
    ax.set_ylabel('Count')
    ax.set_xlim(0, cr_data.quantile(0.99))
    ax.grid(True, alpha=0.3)

    # 3. RUL distribution
    ax = axes[1, 0]
    ax.hist(dataset['RUL_days'], bins=80, color='#1ABC9C', alpha=0.7, edgecolor='none')
    ax.set_title('RUL Distribution', fontweight='bold')
    ax.set_xlabel('RUL (days)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    # 4. Feature correlations heatmap
    ax = axes[1, 1]
    numeric_cols = ['Temp_C', 'Pressure_bar', 'pH', 'Water_Cut',
                    'Flow_Velocity_ms', 'Shear_Stress_Pa',
                    'CO2_Partial_Pressure_bar', 'Corrosion_Rate_mm_yr']
    corr = dataset[numeric_cols].corr()
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    short_labels = ['Temp', 'Press', 'pH', 'WC', 'Vel', 'Shear', 'PCO2', 'CR']
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Fleet-Level Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/fleet_summary_plot.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Fleet summary saved: fleet_summary_plot.png")


# =============================================================================
# SECTION 6: DATA QUALITY REPORT
# =============================================================================

def data_quality_report(dataset: pd.DataFrame):
    """Print a comprehensive data quality and statistical summary."""
    print(f"\n{'='*60}")
    print(f"  DATA QUALITY REPORT")
    print(f"{'='*60}")

    print(f"\n  Shape: {dataset.shape}")
    print(f"  Null values: {dataset.isnull().sum().sum()}")
    print(f"  Duplicate rows: {dataset.duplicated().sum()}")

    n_wells = dataset['Well_ID'].nunique()
    failed = dataset.groupby('Well_ID')['Current_Thickness_mm'].last()
    n_failed = (failed < 3.0).sum()

    print(f"\n  Wells: {n_wells}")
    print(f"  Failed: {n_failed} ({100*n_failed/n_wells:.0f}%)")
    print(f"  Survived: {n_wells - n_failed}")

    print(f"\n  {'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─'*70}")
    for col in ['Temp_C', 'Pressure_bar', 'pH', 'Water_Cut', 'Flow_Velocity_ms',
                'Shear_Stress_Pa', 'CO2_Partial_Pressure_bar',
                'Corrosion_Rate_mm_yr', 'Current_Thickness_mm', 'RUL_days']:
        s = dataset[col]
        print(f"  {col:<30} {s.mean():>10.3f} {s.std():>10.3f} "
              f"{s.min():>10.3f} {s.max():>10.3f}")

    # Corrosion rate sanity check
    active = dataset[dataset['Status'] == 1]
    cr_active = active['Corrosion_Rate_mm_yr']
    print(f"\n  Corrosion Rate (active days only):")
    print(f"    Median: {cr_active.median():.4f} mm/yr")
    print(f"    95th percentile: {cr_active.quantile(0.95):.4f} mm/yr")
    print(f"    99th percentile: {cr_active.quantile(0.99):.4f} mm/yr")
    print(f"    (Typical CO2 corrosion: 0.1 - 10 mm/yr without inhibitor)")

    shutin_pct = (1 - dataset['Status'].mean()) * 100
    print(f"\n  Shut-in time (overall): {shutin_pct:.1f}%")
    print(f"{'='*60}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Generate the full dataset
    dataset = generate_full_dataset(n_wells=80, n_days=2000)

    # Data quality report
    data_quality_report(dataset)

    # Export to CSV
    csv_path = '/home/claude/synthetic_corrosion_dataset.csv'
    dataset.to_csv(csv_path, index=False)
    print(f"\n  CSV exported: {csv_path}")
    print(f"  File size: {pd.io.common.file_exists(csv_path)}")

    # Visualization
    plot_well_history(dataset)
    plot_fleet_summary(dataset)

    print(f"\n  ✓ Pipeline complete!")
