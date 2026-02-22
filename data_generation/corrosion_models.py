"""
Corrosion mechanism models and unit conversions.

Pure functions — no state, no I/O.  Every model takes numpy arrays and returns
corrosion-rate arrays in mm/yr.

Models implemented:
  0. NORSOK M-506 (CO2) — copied verbatim from existing pipeline
  1. H2S (De Waard-Milliams modified)
  2. MIC (stochastic pitting / SRB)
  3. Erosion-Corrosion (API RP 14E)
  4. Oxygen corrosion
  5. Combined (multi-mechanism)

Also: Beggs-Robinson viscosity correlation, unit-conversion constants, and a
master dispatcher that routes to the correct model(s) based on cause code.
"""

import numpy as np
from scipy.interpolate import interp1d

from data_generation.config_fields import (
    CAUSE_CO2, CAUSE_H2S, CAUSE_MIC, CAUSE_EROSION, CAUSE_OXYGEN, CAUSE_COMBINED,
)

# ============================================================================
# UNIT CONVERSION CONSTANTS
# ============================================================================
PSI_PER_BAR = 14.5038
BAR_PER_PSI = 1.0 / PSI_PER_BAR
F_TO_C = lambda f: (f - 32.0) * 5.0 / 9.0      # noqa: E731
C_TO_F = lambda c: c * 9.0 / 5.0 + 32.0         # noqa: E731
MPY_PER_MMYR = 39.3701                           # 1 mm/yr = 39.37 mpy
MMYR_PER_MPY = 1.0 / MPY_PER_MMYR
BPD_TO_M3D = 0.158987                            # 1 bbl = 0.158987 m³
M3D_TO_BPD = 1.0 / BPD_TO_M3D
FPS_TO_MS = 0.3048                               # 1 ft/s = 0.3048 m/s
MS_TO_FPS = 1.0 / FPS_TO_MS
PPG_TO_KGM3 = 119.826                            # 1 ppg = 119.826 kg/m³
KGM3_TO_PPG = 1.0 / PPG_TO_KGM3
INCH_TO_M = 0.0254

# ============================================================================
# SECTION 1: NORSOK M-506  (VERBATIM from existing pipeline lines 30-123)
# ============================================================================

# Temperature-dependent constant K_T (from NORSOK M-506 Table 1)
NORSOK_TEMPS = np.array([5, 15, 20, 40, 60, 80, 90, 120, 150])
NORSOK_KT = np.array([0.42, 0.59, 0.70, 1.59, 4.57, 8.50, 9.40, 6.30, 3.80])

# pH correction factors f(pH) at standard temperatures
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
    return np.maximum(kt, 0.01)


def interpolate_ph_factor(temperature: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """2D interpolation of pH correction factor f(pH)_T."""
    ph_factor_matrix = np.array([NORSOK_PH_FACTORS[t] for t in NORSOK_PH_TEMP_KEYS])

    result = np.zeros_like(temperature, dtype=float)
    for i in range(len(temperature)):
        t = np.clip(temperature[i], 20, 150)
        p = np.clip(ph[i], 3.5, 6.5)
        ph_at_temps = np.array([
            np.interp(p, NORSOK_PH_VALUES, ph_factor_matrix[j])
            for j in range(len(NORSOK_PH_TEMP_KEYS))
        ])
        result[i] = np.interp(t, NORSOK_PH_TEMP_KEYS, ph_at_temps)

    return np.maximum(result, 0.01)


def norsok_m506_corrosion_rate(
    temperature_C: np.ndarray,
    fco2_bar: np.ndarray,
    ph: np.ndarray,
    shear_stress_Pa: np.ndarray,
) -> np.ndarray:
    """
    NORSOK M-506 CO2 Corrosion Rate (mm/yr).

    CR = K_T * f_CO2^0.62 * (S/19)^(0.146+0.0324*log(f_CO2)) * f(pH)_T
    """
    kt = interpolate_kt(temperature_C)

    fco2_safe = np.maximum(fco2_bar, 1e-6)
    shear_safe = np.maximum(shear_stress_Pa, 0.1)

    fco2_term = fco2_safe ** 0.62
    shear_exponent = 0.146 + 0.0324 * np.log10(fco2_safe)
    shear_term = (shear_safe / 19.0) ** shear_exponent
    ph_factor = interpolate_ph_factor(temperature_C, ph)

    cr = kt * fco2_term * shear_term * ph_factor
    return np.maximum(cr, 0.0)

# ============================================================================
# SECTION 2: H2S CORROSION  — De Waard-Milliams modified
# ============================================================================

def h2s_corrosion_rate(
    temperature_C: np.ndarray,
    p_h2s_bar: np.ndarray,
    ph: np.ndarray,
    water_cut: np.ndarray,
) -> np.ndarray:
    """
    H2S corrosion rate (mm/yr).

    CR = A * exp(-B/T_K) * P_H2S^0.33 * f(pH) * scale_factor * wetting

    Arrhenius-type temperature dependence.
    FeS scale factor: protective below 60 C, breaks down above 80 C.
    """
    A = 50.0
    B = 1900.0

    T_K = temperature_C + 273.15
    p_h2s_safe = np.maximum(p_h2s_bar, 1e-8)

    # Arrhenius term
    arrhenius = A * np.exp(-B / T_K)

    # H2S partial pressure (sub-linear exponent 0.33)
    p_term = p_h2s_safe ** 0.33

    # pH factor — lower pH = more corrosion
    ph_factor = np.clip(7.0 - ph, 0.5, 4.0) / 2.0

    # FeS scale factor: protective below 60 C, breaks down above 80 C
    scale = np.where(
        temperature_C < 60,
        0.5 + 0.5 * (temperature_C / 60.0),    # partial protection
        np.where(
            temperature_C < 80,
            1.0,                                 # transition
            1.0 + 0.02 * (temperature_C - 80)   # scale breakdown
        )
    )

    # Water wetting — no corrosion without water contact
    wetting = np.clip(water_cut / 0.3, 0.0, 1.0)

    cr = arrhenius * p_term * ph_factor * scale * wetting
    return np.maximum(cr, 0.0)

# ============================================================================
# SECTION 3: MIC  — Stochastic pitting (SRB-driven)
# ============================================================================

def mic_corrosion_rate(
    temperature_C: np.ndarray,
    water_cut: np.ndarray,
    days: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Microbiologically-influenced corrosion rate (mm/yr).

    Baseline general corrosion 0.4 mm/yr.
    SRB activity: bimodal Gaussian (40 C mesophilic + 65 C thermophilic),
    zero outside 15-120 C.
    SRB need >30% water cut.
    Stochastic pitting events: daily probability * SRB activity, intensity 4.0x.
    """
    n = len(temperature_C)
    baseline = 0.4  # mm/yr general MIC

    # SRB activity — bimodal: mesophilic (40 C) + thermophilic (65 C)
    srb_mesophilic = np.exp(-0.5 * ((temperature_C - 40.0) / 12.0) ** 2)
    srb_thermophilic = np.exp(-0.5 * ((temperature_C - 65.0) / 18.0) ** 2)
    srb_activity = 0.6 * srb_mesophilic + 0.4 * srb_thermophilic
    # Zero outside 15-120 C
    srb_activity[temperature_C < 15] = 0.0
    srb_activity[temperature_C > 120] = 0.0

    # Water wetting requirement: SRB need >30% water cut
    wetting = np.clip((water_cut - 0.30) / 0.20, 0.0, 1.0)

    # General MIC corrosion
    cr = baseline * srb_activity * wetting

    # Stochastic pitting events
    pitting_prob = 0.015 * srb_activity * wetting
    pitting_events = rng.random(n) < pitting_prob
    pitting_intensity = 4.0
    cr[pitting_events] += baseline * pitting_intensity * srb_activity[pitting_events]

    return np.maximum(cr, 0.0)

# ============================================================================
# SECTION 4: EROSION-CORROSION  — API RP 14E
# ============================================================================

def erosion_corrosion_rate(
    base_cr_mmyr: np.ndarray,
    velocity_ms: np.ndarray,
    density_kgm3: np.ndarray,
) -> np.ndarray:
    """
    Erosion-corrosion rate (mm/yr).

    Uses API RP 14E erosional velocity:
        V_critical = C / sqrt(rho)   (C=125 for continuous service)

    When V > V_critical:
        CR = CR_base * (1 + k * (V/V_critical - 1)^2)
    Enhancement capped at 10x.
    """
    C = 100.0
    k = 3.0   # enhancement constant

    density_safe = np.maximum(density_kgm3, 1.0)
    v_critical = C / np.sqrt(density_safe)       # m/s

    v_ratio = velocity_ms / np.maximum(v_critical, 0.01)

    enhancement = np.where(
        v_ratio > 1.0,
        1.0 + k * (v_ratio - 1.0) ** 2,
        1.0
    )
    enhancement = np.minimum(enhancement, 10.0)  # cap at 10x

    return base_cr_mmyr * enhancement

# ============================================================================
# SECTION 5: OXYGEN CORROSION
# ============================================================================

def oxygen_corrosion_rate(
    temperature_C: np.ndarray,
    dissolved_o2_ppb: np.ndarray,
    water_cut: np.ndarray,
) -> np.ndarray:
    """
    Oxygen corrosion rate (mm/yr).

    CR = k_O2 * [O2]_ppb * f(T) * wetting

    Linear in dissolved O2 concentration.
    Temperature factor peaks at 60-80 C.
    """
    k_O2 = 0.0003  # mm/yr per ppb O2 at reference conditions

    # Temperature factor: peaks at 70 C
    f_temp = np.exp(-0.5 * ((temperature_C - 70.0) / 25.0) ** 2)
    # Boost: always at least 0.3 of max
    f_temp = np.maximum(f_temp, 0.3)

    # Wetting factor
    wetting = np.clip(water_cut / 0.2, 0.0, 1.0)

    cr = k_O2 * dissolved_o2_ppb * f_temp * wetting
    return np.maximum(cr, 0.0)

# ============================================================================
# SECTION 6: BEGGS-ROBINSON VISCOSITY CORRELATION
# ============================================================================

def beggs_robinson_viscosity(
    temperature_F: np.ndarray,
    api_gravity: float,
    gor_scf_bbl: float,
    water_cut: np.ndarray,
) -> np.ndarray:
    """
    Dead/live oil + emulsion viscosity (cP) via Beggs-Robinson.

    Step 1: mu_dead = 10^(10^(3.0324 - 0.02023*API) * T_F^(-1.163)) - 1
    Step 2: mu_live = A * mu_dead^B   (GOR correction)
    Step 3: mu_emulsion = mu_live * (1 + 2.5*phi_w + 10.05*phi_w^2)
    """
    T_safe = np.maximum(temperature_F, 100.0)   # avoid extreme extrapolation

    # Step 1 — dead oil viscosity
    x = 10.0 ** (3.0324 - 0.02023 * api_gravity) * T_safe ** (-1.163)
    mu_dead = 10.0 ** x - 1.0
    mu_dead = np.clip(mu_dead, 0.1, 500.0)

    # Step 2 — live oil (GOR correction)
    Rs = max(gor_scf_bbl, 1.0)
    A_gor = 10.715 * (Rs + 100.0) ** (-0.515)
    B_gor = 5.44 * (Rs + 150.0) ** (-0.338)
    mu_live = A_gor * mu_dead ** B_gor
    mu_live = np.clip(mu_live, 0.1, 200.0)

    # Step 3 — emulsion viscosity (water-in-oil)
    phi_w = np.clip(water_cut, 0.0, 0.9)
    mu_emulsion = mu_live * (1.0 + 2.5 * phi_w + 10.05 * phi_w ** 2)

    return np.clip(mu_emulsion, 0.1, 1000.0)

# ============================================================================
# SECTION 7: MASTER DISPATCHER
# ============================================================================

def compute_combined_corrosion(
    cause: int,
    temperature_C: np.ndarray,
    fco2_bar: np.ndarray,
    p_h2s_bar: np.ndarray,
    ph: np.ndarray,
    shear_stress_Pa: np.ndarray,
    velocity_ms: np.ndarray,
    density_kgm3: np.ndarray,
    water_cut: np.ndarray,
    dissolved_o2_ppb: np.ndarray,
    days: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Route to the correct corrosion model(s) based on cause code.

    For COMBINED (5): computes multiple mechanisms, takes the max + 20% of
    the second-highest (synergistic interaction).

    Returns corrosion rate in mm/yr.
    """
    if cause == CAUSE_CO2:
        return norsok_m506_corrosion_rate(temperature_C, fco2_bar, ph, shear_stress_Pa)

    elif cause == CAUSE_H2S:
        return h2s_corrosion_rate(temperature_C, p_h2s_bar, ph, water_cut)

    elif cause == CAUSE_MIC:
        return mic_corrosion_rate(temperature_C, water_cut, days, rng)

    elif cause == CAUSE_EROSION:
        base_co2 = norsok_m506_corrosion_rate(temperature_C, fco2_bar, ph, shear_stress_Pa)
        return erosion_corrosion_rate(base_co2, velocity_ms, density_kgm3)

    elif cause == CAUSE_OXYGEN:
        return oxygen_corrosion_rate(temperature_C, dissolved_o2_ppb, water_cut)

    elif cause == CAUSE_COMBINED:
        # Compute all relevant mechanisms
        cr_co2 = norsok_m506_corrosion_rate(temperature_C, fco2_bar, ph, shear_stress_Pa)
        cr_h2s = h2s_corrosion_rate(temperature_C, p_h2s_bar, ph, water_cut)
        cr_mic = mic_corrosion_rate(temperature_C, water_cut, days, rng)
        cr_o2 = oxygen_corrosion_rate(temperature_C, dissolved_o2_ppb, water_cut)

        # Stack and sort descending per day
        all_cr = np.stack([cr_co2, cr_h2s, cr_mic, cr_o2], axis=0)  # (4, n_days)
        sorted_cr = np.sort(all_cr, axis=0)[::-1]  # descending

        # Max + 20% of second-highest (synergistic)
        cr = sorted_cr[0] + 0.20 * sorted_cr[1]
        return np.maximum(cr, 0.0)

    else:
        raise ValueError(f"Unknown corrosion cause code: {cause}")
