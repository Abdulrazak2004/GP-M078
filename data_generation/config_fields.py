"""
Field-specific parameters for Saudi Arabian oil fields.

Pure data — no logic. Defines all field-specific parameters, well distribution,
corrosion cause assignments, and coordinate sub-areas sourced from
saudi_oil_well_locations.py.
"""

# ---------------------------------------------------------------------------
# Corrosion cause integer codes
# ---------------------------------------------------------------------------
CAUSE_CO2 = 0
CAUSE_H2S = 1
CAUSE_MIC = 2
CAUSE_EROSION = 3
CAUSE_OXYGEN = 4
CAUSE_COMBINED = 5

CAUSE_LABELS = {
    CAUSE_CO2: "CO2",
    CAUSE_H2S: "H2S",
    CAUSE_MIC: "MIC",
    CAUSE_EROSION: "Erosion",
    CAUSE_OXYGEN: "Oxygen",
    CAUSE_COMBINED: "Combined",
}

# ---------------------------------------------------------------------------
# Well distribution across fields (total = 500)
# Proportions preserved from original 80-well layout, scaled 6.25x
# ---------------------------------------------------------------------------
WELL_DISTRIBUTION = {
    "Ghawar": 156,
    "Khurais": 63,
    "Safaniya": 50,
    "Shaybah": 38,
    "Manifa": 38,
    "Abqaiq": 31,
    "Berri": 31,
    "Zuluf": 31,
    "Others": 62,
}

# ---------------------------------------------------------------------------
# Per-field corrosion cause fractions  (must sum to ~1.0 per field)
# ---------------------------------------------------------------------------
FIELD_CAUSE_DISTRIBUTION = {
    "Ghawar": {
        CAUSE_CO2: 0.72,
        CAUSE_MIC: 0.08,
        CAUSE_COMBINED: 0.20,
    },
    "Khurais": {
        CAUSE_CO2: 0.70,
        CAUSE_MIC: 0.10,
        CAUSE_COMBINED: 0.20,
    },
    "Safaniya": {
        CAUSE_EROSION: 0.625,
        CAUSE_CO2: 0.25,
        CAUSE_COMBINED: 0.125,
    },
    "Shaybah": {
        CAUSE_CO2: 0.67,
        CAUSE_COMBINED: 0.33,
    },
    "Manifa": {
        CAUSE_H2S: 0.67,
        CAUSE_COMBINED: 0.33,
    },
    "Abqaiq": {
        CAUSE_MIC: 0.60,
        CAUSE_CO2: 0.20,
        CAUSE_COMBINED: 0.20,
    },
    "Berri": {
        CAUSE_OXYGEN: 0.60,
        CAUSE_CO2: 0.20,
        CAUSE_COMBINED: 0.20,
    },
    "Zuluf": {
        CAUSE_EROSION: 0.60,
        CAUSE_CO2: 0.20,
        CAUSE_COMBINED: 0.20,
    },
    "Others": {
        CAUSE_CO2: 0.20,
        CAUSE_H2S: 0.30,
        CAUSE_MIC: 0.10,
        CAUSE_EROSION: 0.10,
        CAUSE_OXYGEN: 0.20,
        CAUSE_COMBINED: 0.10,
    },
}

# ---------------------------------------------------------------------------
# Field configurations
#
# Each value is a tuple (min, max) unless stated otherwise.
# Pressures in psi, temperatures in degF, casing per API 5CT.
# Sub-areas carry real lat/lon from saudi_oil_well_locations.py.
# ---------------------------------------------------------------------------
FIELD_CONFIGS = {
    # -----------------------------------------------------------------------
    # 1. GHAWAR  (world's largest, 25 wells)
    # -----------------------------------------------------------------------
    "Ghawar": {
        "reservoir_type": "Carbonate",
        # Reservoir
        "pressure_psi": (3200, 3395),
        "temperature_F": (210, 240),
        "co2_mol_frac": (0.01, 0.035),
        "h2s_mol_frac": (0.0, 0.001),          # sweet
        "pressure_decline_annual_pct": (0.8, 1.2),
        # Water cut sigmoid
        "wc_midpoint_day": (2500, 4000),        # year 7-11
        "wc_steepness": (0.0008, 0.0018),
        "wc_max": (0.80, 0.97),
        # Flow
        "initial_flow_bpd": (3000, 8000),
        "flow_decline_rate": (0.00005, 0.00015),
        "pipe_diameter_in": (4.0, 5.0),
        "fluid_density_ppg": (6.8, 7.4),
        "roughness_m": (0.00015, 0.0005),
        "api_gravity": (32, 36),
        "gor_scf_bbl": (500, 800),
        # Casing (API 5CT)
        "casing_grade": "N80",
        "casing_od_in": 7.0,
        "initial_thickness_mm": (10.36, 12.65),  # 7" N80 wall range
        # Inhibitor
        "inhibitor_efficiency": (0.70, 0.95),
        "inhibitor_active_prob": 0.80,
        "inhibitor_start_day": (0, 300),
        "inhibitor_reliability": (0.85, 0.98),
        # Pitting
        "pitting_factor": (1.2, 2.5),
        "pitting_probability": (0.005, 0.025),
        # Seasonality / noise
        "ambient_temp_mean_F": (90, 104),       # 32-40 C
        "ambient_temp_amplitude_F": (14, 27),   # 8-15 C swing
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        # Shut-ins / choke
        "shutin_frequency": (0.001, 0.006),
        "shutin_duration_range": (5, 20),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.4),
        # pH
        "initial_ph": (4.8, 6.0),
        "ph_drift_rate": (-0.0002, 0.0003),
        # Sub-areas with real coordinates
        "sub_areas": {
            "Fazran":      {"lat": 26.10,    "lon": 49.35},
            "Ain_Dar":     {"lat": 25.94917, "lon": 49.42378},
            "Shedgum":     {"lat": 25.6768,  "lon": 49.39743},
            "Uthmaniyah":  {"lat": 25.1938,  "lon": 49.3095},
            "Hawiyah":     {"lat": 24.7983,  "lon": 49.18},
            "Haradh":      {"lat": 24.2417,  "lon": 49.1864},
        },
    },

    # -----------------------------------------------------------------------
    # 2. KHURAIS  (10 wells)
    # -----------------------------------------------------------------------
    "Khurais": {
        "reservoir_type": "Carbonate",
        "pressure_psi": (2800, 3200),
        "temperature_F": (200, 220),
        "co2_mol_frac": (0.01, 0.03),
        "h2s_mol_frac": (0.0, 0.001),
        "pressure_decline_annual_pct": (0.9, 1.3),
        "wc_midpoint_day": (3000, 4500),
        "wc_steepness": (0.0008, 0.0016),
        "wc_max": (0.78, 0.95),
        "initial_flow_bpd": (2500, 6000),
        "flow_decline_rate": (0.00006, 0.00018),
        "pipe_diameter_in": (4.0, 5.0),
        "fluid_density_ppg": (6.8, 7.3),
        "roughness_m": (0.00015, 0.0005),
        "api_gravity": (30, 34),
        "gor_scf_bbl": (400, 700),
        "casing_grade": "N80",
        "casing_od_in": 7.0,
        "initial_thickness_mm": (10.36, 12.65),
        "inhibitor_efficiency": (0.70, 0.95),
        "inhibitor_active_prob": 0.80,
        "inhibitor_start_day": (0, 300),
        "inhibitor_reliability": (0.85, 0.98),
        "pitting_factor": (1.2, 2.5),
        "pitting_probability": (0.005, 0.025),
        "ambient_temp_mean_F": (90, 104),
        "ambient_temp_amplitude_F": (14, 27),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.006),
        "shutin_duration_range": (5, 20),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.4),
        "initial_ph": (4.8, 6.0),
        "ph_drift_rate": (-0.0002, 0.0003),
        "sub_areas": {
            "Khurais_Main":  {"lat": 25.2028, "lon": 48.0153},
            "Abu_Jifan":     {"lat": 25.0631, "lon": 48.0304},
            "Mazalij":       {"lat": 25.0631, "lon": 48.0304},
        },
    },

    # -----------------------------------------------------------------------
    # 3. SAFANIYA  (offshore, 8 wells, erosion-dominant)
    # -----------------------------------------------------------------------
    "Safaniya": {
        "reservoir_type": "Clastic",
        "pressure_psi": (2500, 3000),
        "temperature_F": (160, 200),
        "co2_mol_frac": (0.008, 0.025),
        "h2s_mol_frac": (0.0, 0.005),          # low H2S
        "pressure_decline_annual_pct": (1.0, 1.5),
        "wc_midpoint_day": (2500, 3800),
        "wc_steepness": (0.0010, 0.0020),
        "wc_max": (0.82, 0.97),
        "initial_flow_bpd": (8000, 20000),      # high-rate offshore
        "flow_decline_rate": (0.00005, 0.00012),
        "pipe_diameter_in": (2.875, 3.5),     # production tubing, not casing
        "fluid_density_ppg": (7.0, 7.6),
        "roughness_m": (0.0002, 0.0006),
        "api_gravity": (26, 30),                 # heavy oil
        "gor_scf_bbl": (200, 500),
        "casing_grade": "N80",
        "casing_od_in": 9.625,
        "initial_thickness_mm": (11.05, 13.84),  # 9-5/8" N80
        "inhibitor_efficiency": (0.70, 0.90),
        "inhibitor_active_prob": 0.75,
        "inhibitor_start_day": (0, 400),
        "inhibitor_reliability": (0.80, 0.95),
        "pitting_factor": (1.2, 2.0),
        "pitting_probability": (0.005, 0.020),
        "ambient_temp_mean_F": (82, 95),         # offshore cooler
        "ambient_temp_amplitude_F": (10, 20),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.005),
        "shutin_duration_range": (5, 15),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.35),
        "initial_ph": (5.0, 6.2),
        "ph_drift_rate": (-0.0002, 0.0002),
        "sub_areas": {
            "Safaniya_Central": {"lat": 28.28,   "lon": 48.75},
            "Safaniya_North":   {"lat": 28.2833, "lon": 48.75},
        },
    },

    # -----------------------------------------------------------------------
    # 4. SHAYBAH  (6 wells, CO2-dominant, remote desert)
    # -----------------------------------------------------------------------
    "Shaybah": {
        "reservoir_type": "Carbonate",
        "pressure_psi": (2700, 2900),
        "temperature_F": (180, 200),
        "co2_mol_frac": (0.01, 0.03),
        "h2s_mol_frac": (0.0, 0.001),
        "pressure_decline_annual_pct": (0.8, 1.1),
        "wc_midpoint_day": (3500, 5000),
        "wc_steepness": (0.0007, 0.0015),
        "wc_max": (0.75, 0.92),
        "initial_flow_bpd": (2000, 5000),
        "flow_decline_rate": (0.00005, 0.00015),
        "pipe_diameter_in": (4.0, 5.0),
        "fluid_density_ppg": (6.5, 7.2),
        "roughness_m": (0.00015, 0.0005),
        "api_gravity": (38, 42),                 # light crude
        "gor_scf_bbl": (600, 1000),
        "casing_grade": "N80",
        "casing_od_in": 7.0,
        "initial_thickness_mm": (10.36, 12.65),
        "inhibitor_efficiency": (0.70, 0.95),
        "inhibitor_active_prob": 0.80,
        "inhibitor_start_day": (0, 300),
        "inhibitor_reliability": (0.85, 0.98),
        "pitting_factor": (1.2, 2.5),
        "pitting_probability": (0.005, 0.025),
        "ambient_temp_mean_F": (95, 110),        # desert — hotter
        "ambient_temp_amplitude_F": (18, 30),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.006),
        "shutin_duration_range": (5, 20),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.4),
        "initial_ph": (4.8, 6.0),
        "ph_drift_rate": (-0.0002, 0.0003),
        "sub_areas": {
            "Shaybah_Main":     {"lat": 21.7237, "lon": 53.6572},
            "Shaybah_Facility": {"lat": 22.5106, "lon": 53.9519},
        },
    },

    # -----------------------------------------------------------------------
    # 5. MANIFA  (6 wells, H2S-dominant, shallow offshore, sour crude)
    # -----------------------------------------------------------------------
    "Manifa": {
        "reservoir_type": "Clastic",
        "pressure_psi": (2000, 3000),
        "temperature_F": (160, 190),
        "co2_mol_frac": (0.008, 0.02),
        "h2s_mol_frac": (0.10, 0.14),           # ** 10-14% H2S — sour **
        "pressure_decline_annual_pct": (1.0, 1.5),
        "wc_midpoint_day": (2500, 4000),
        "wc_steepness": (0.0008, 0.0018),
        "wc_max": (0.80, 0.96),
        "initial_flow_bpd": (3000, 8000),
        "flow_decline_rate": (0.00005, 0.00014),
        "pipe_diameter_in": (5.0, 6.0),
        "fluid_density_ppg": (7.2, 7.8),
        "roughness_m": (0.0002, 0.0006),
        "api_gravity": (27, 31),                 # heavy-sour
        "gor_scf_bbl": (200, 500),
        "casing_grade": "L80",                   # sour-service grade
        "casing_od_in": 9.625,
        "initial_thickness_mm": (11.05, 13.84),
        "inhibitor_efficiency": (0.65, 0.90),
        "inhibitor_active_prob": 0.85,
        "inhibitor_start_day": (0, 200),
        "inhibitor_reliability": (0.82, 0.95),
        "pitting_factor": (1.5, 3.0),           # H2S pitting more severe
        "pitting_probability": (0.008, 0.030),
        "ambient_temp_mean_F": (82, 95),
        "ambient_temp_amplitude_F": (10, 20),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.005),
        "shutin_duration_range": (5, 15),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.35),
        "initial_ph": (4.5, 5.8),
        "ph_drift_rate": (-0.0003, 0.0002),
        "sub_areas": {
            "Manifa_Central": {"lat": 27.7159, "lon": 48.9834},
        },
    },

    # -----------------------------------------------------------------------
    # 6. ABQAIQ  (5 wells, MIC-dominant, oldest field)
    # -----------------------------------------------------------------------
    "Abqaiq": {
        "reservoir_type": "Carbonate",
        "pressure_psi": (3300, 3500),
        "temperature_F": (210, 230),
        "co2_mol_frac": (0.01, 0.03),
        "h2s_mol_frac": (0.0, 0.001),
        "pressure_decline_annual_pct": (0.8, 1.2),
        "wc_midpoint_day": (2500, 3500),         # older field, earlier WC
        "wc_steepness": (0.0010, 0.0020),
        "wc_max": (0.85, 0.98),
        "initial_flow_bpd": (2000, 5000),
        "flow_decline_rate": (0.00006, 0.00018),
        "pipe_diameter_in": (4.0, 5.0),
        "fluid_density_ppg": (6.8, 7.4),
        "roughness_m": (0.00015, 0.0005),
        "api_gravity": (34, 38),
        "gor_scf_bbl": (500, 800),
        "casing_grade": "N80",
        "casing_od_in": 7.0,
        "initial_thickness_mm": (10.36, 12.65),
        "inhibitor_efficiency": (0.70, 0.95),
        "inhibitor_active_prob": 0.75,
        "inhibitor_start_day": (0, 300),
        "inhibitor_reliability": (0.85, 0.98),
        "pitting_factor": (1.3, 2.8),
        "pitting_probability": (0.006, 0.028),
        "ambient_temp_mean_F": (90, 104),
        "ambient_temp_amplitude_F": (14, 27),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.006),
        "shutin_duration_range": (5, 20),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.4),
        "initial_ph": (4.8, 6.0),
        "ph_drift_rate": (-0.0002, 0.0003),
        "sub_areas": {
            "Abqaiq_Main":     {"lat": 26.1687, "lon": 49.7841},
            "Abqaiq_Facility": {"lat": 25.9371, "lon": 49.6776},
        },
    },

    # -----------------------------------------------------------------------
    # 7. BERRI  (5 wells, Oxygen-dominant)
    # -----------------------------------------------------------------------
    "Berri": {
        "reservoir_type": "Carbonate",
        "pressure_psi": (3000, 3200),
        "temperature_F": (200, 220),
        "co2_mol_frac": (0.01, 0.03),
        "h2s_mol_frac": (0.0, 0.001),
        "pressure_decline_annual_pct": (0.9, 1.3),
        "wc_midpoint_day": (3000, 4500),
        "wc_steepness": (0.0008, 0.0018),
        "wc_max": (0.78, 0.95),
        "initial_flow_bpd": (2500, 6000),
        "flow_decline_rate": (0.00006, 0.00018),
        "pipe_diameter_in": (4.0, 5.0),
        "fluid_density_ppg": (6.8, 7.3),
        "roughness_m": (0.00015, 0.0005),
        "api_gravity": (32, 36),
        "gor_scf_bbl": (500, 800),
        "casing_grade": "N80",
        "casing_od_in": 7.0,
        "initial_thickness_mm": (10.36, 12.65),
        "inhibitor_efficiency": (0.70, 0.95),
        "inhibitor_active_prob": 0.80,
        "inhibitor_start_day": (0, 300),
        "inhibitor_reliability": (0.85, 0.98),
        "pitting_factor": (1.2, 2.5),
        "pitting_probability": (0.005, 0.025),
        "ambient_temp_mean_F": (88, 100),
        "ambient_temp_amplitude_F": (12, 24),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.006),
        "shutin_duration_range": (5, 20),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.4),
        "initial_ph": (4.8, 6.0),
        "ph_drift_rate": (-0.0002, 0.0003),
        "sub_areas": {
            "Berri_Main": {"lat": 27.1145, "lon": 49.6389},
            "Berri_Alt":  {"lat": 27.2156, "lon": 49.7169},
        },
    },

    # -----------------------------------------------------------------------
    # 8. ZULUF  (5 wells, Erosion-dominant, offshore)
    # -----------------------------------------------------------------------
    "Zuluf": {
        "reservoir_type": "Carbonate",
        "pressure_psi": (2800, 3200),
        "temperature_F": (190, 220),
        "co2_mol_frac": (0.008, 0.025),
        "h2s_mol_frac": (0.0, 0.005),
        "pressure_decline_annual_pct": (1.0, 1.5),
        "wc_midpoint_day": (2800, 4200),
        "wc_steepness": (0.0008, 0.0018),
        "wc_max": (0.80, 0.96),
        "initial_flow_bpd": (8000, 18000),
        "flow_decline_rate": (0.00005, 0.00014),
        "pipe_diameter_in": (2.875, 3.5),     # production tubing, not casing
        "fluid_density_ppg": (7.0, 7.5),
        "roughness_m": (0.0002, 0.0006),
        "api_gravity": (28, 33),
        "gor_scf_bbl": (300, 600),
        "casing_grade": "N80",
        "casing_od_in": 9.625,
        "initial_thickness_mm": (11.05, 13.84),
        "inhibitor_efficiency": (0.70, 0.90),
        "inhibitor_active_prob": 0.75,
        "inhibitor_start_day": (0, 350),
        "inhibitor_reliability": (0.82, 0.96),
        "pitting_factor": (1.2, 2.2),
        "pitting_probability": (0.005, 0.022),
        "ambient_temp_mean_F": (82, 95),
        "ambient_temp_amplitude_F": (10, 20),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.005),
        "shutin_duration_range": (5, 15),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.35),
        "initial_ph": (5.0, 6.2),
        "ph_drift_rate": (-0.0002, 0.0002),
        "sub_areas": {
            "Zuluf_Main": {"lat": 28.611, "lon": 49.059},
        },
    },

    # -----------------------------------------------------------------------
    # 9. OTHERS  (10 wells, mixed causes, mixed fields)
    # -----------------------------------------------------------------------
    "Others": {
        "reservoir_type": "Mixed",
        "pressure_psi": (2500, 3200),
        "temperature_F": (170, 230),
        "co2_mol_frac": (0.008, 0.035),
        "h2s_mol_frac": (0.0, 0.06),            # some sour
        "pressure_decline_annual_pct": (0.8, 1.5),
        "wc_midpoint_day": (2500, 5000),
        "wc_steepness": (0.0007, 0.0020),
        "wc_max": (0.75, 0.98),
        "initial_flow_bpd": (2000, 8000),
        "flow_decline_rate": (0.00005, 0.00018),
        "pipe_diameter_in": (4.0, 6.0),
        "fluid_density_ppg": (6.5, 7.8),
        "roughness_m": (0.00015, 0.0006),
        "api_gravity": (26, 42),
        "gor_scf_bbl": (200, 1000),
        "casing_grade": "Mixed",
        "casing_od_in": 7.0,                     # default
        "initial_thickness_mm": (10.36, 13.84),
        "inhibitor_efficiency": (0.65, 0.95),
        "inhibitor_active_prob": 0.75,
        "inhibitor_start_day": (0, 400),
        "inhibitor_reliability": (0.80, 0.98),
        "pitting_factor": (1.2, 3.0),
        "pitting_probability": (0.005, 0.030),
        "ambient_temp_mean_F": (85, 108),
        "ambient_temp_amplitude_F": (12, 28),
        "noise_temp": (0.005, 0.02),
        "noise_pressure": (0.001, 0.005),
        "noise_flow": (0.02, 0.05),
        "noise_ph": (0.005, 0.015),
        "shutin_frequency": (0.001, 0.006),
        "shutin_duration_range": (5, 20),
        "choke_change_frequency": (0.005, 0.02),
        "choke_change_magnitude": (0.1, 0.4),
        "initial_ph": (4.5, 6.2),
        "ph_drift_rate": (-0.0003, 0.0003),
        "sub_areas": {
            "Marjan":       {"lat": 28.4389, "lon": 49.6746},
            "Qatif":        {"lat": 26.7077, "lon": 49.9393},
            "Khursaniyah":  {"lat": 27.0287, "lon": 48.5437},
            "Dammam":       {"lat": 26.4201, "lon": 50.1045},
            "Abu_Safah":    {"lat": 26.975,  "lon": 50.5522},
        },
    },
}
