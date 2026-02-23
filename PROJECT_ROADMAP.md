# Casing Corrosion RUL Prediction — Complete Project Roadmap

**Project:** LSTM-based Remaining Useful Life & Corrosion Prediction for Oil Well Casings
**Team:** KFUPM Senior Design / Graduation Project
**Date:** February 2026

---

## Table of Contents

1. [Big Picture: What We're Building](#1-big-picture)
2. [Phase 1: Synthetic Data Generation](#2-phase-1-synthetic-data-generation)
3. [Phase 2: Data Validation & EDA](#3-phase-2-data-validation--eda)
4. [Phase 3: Feature Engineering & Preprocessing](#4-phase-3-feature-engineering--preprocessing)
5. [Phase 4: Model Architecture (Multi-Task)](#5-phase-4-model-architecture)
6. [Phase 5: Training Pipeline](#6-phase-5-training-pipeline)
7. [Phase 6: Evaluation & Testing](#7-phase-6-evaluation--testing)
8. [Phase 7: Streamlit Dashboard](#8-phase-7-streamlit-dashboard)
9. [File Structure](#9-file-structure)
10. [Spec Compliance Checklist](#10-spec-compliance-checklist)
11. [Build Order](#11-build-order)

---

## 1. Big Picture

We are building a **multi-task deep learning system** that takes operational sensor data from oil wells and outputs:

| Output | What It Is | Which Spec |
|--------|-----------|------------|
| **RUL (days)** | How many days until casing failure | Core deliverable |
| **Corrosion Rate (mm/yr)** | Current metal loss speed | S2 |
| **Wall Thickness (mm)** | Current predicted casing thickness | IS1 |
| **Corrosion Cause** | Why the casing is corroding (6 classes) | IS2 |
| **5-Year Forecast** | Thickness at +12, +24, +36, +48, +60 months | IS3 |
| **95% Confidence Intervals** | Uncertainty bands on all forecasts | IS3 |
| **Corrosion Failure Index (0-100)** | Composite risk score for field engineers | IS4 |

The system is trained on **synthetic data** grounded in real Saudi Arabian oil field parameters, then deployed as a **Streamlit dashboard** that field engineers use to monitor wells and make maintenance decisions.

### How It All Connects

```
[Phase 1: Data Generation]
    Real Saudi field parameters (locations, reservoir conditions, casing specs)
    + NORSOK M-506 physics model (CO2 corrosion)
    + De Waard-Milliams model (supplementary)
    + H2S / MIC / Erosion mechanisms
    → synthetic_corrosion_dataset.csv (~80 wells, ~30 years each, ~800K+ rows)

[Phase 2: Validation & EDA]
    → Sanity checks (TEST-01 through TEST-10)
    → Visualizations confirming physics makes sense
    → Dataset statistics and distributions

[Phase 3: Preprocessing]
    → RUL capping at 500 days
    → Split by Well_ID (48 train / 16 val / 16 test)
    → Min-Max scaling (fit on train only)
    → Sliding window creation (30-day windows)
    → Multi-step target creation (5-year look-aheads)

[Phase 4: Model]
    → Shared BiLSTM+Attention backbone
    → 5 task-specific output heads
    → MC Dropout for confidence intervals

[Phase 5: Training]
    → Multi-task weighted loss (Huber + CrossEntropy)
    → Early stopping, LR scheduling
    → Best checkpoint saving

[Phase 6: Evaluation]
    → Metrics per task (MAE, R², accuracy, calibration)
    → 7+ evaluation plots
    → Critical tests (TEST-11 through TEST-18+)
    → Comparison table across model variants

[Phase 7: Dashboard]
    → 5-page Streamlit app
    → Fleet overview, well detail, data entry, scenario analysis, model health
    → Plotly charts with gauges, time series, confidence bands
```

---

## 2. Phase 1: Synthetic Data Generation

### 2.1 What We're Generating

A CSV dataset with **80 synthetic oil wells**, each simulated over **up to 30 years (10,950 days)**, producing approximately **800,000+ rows**. Each row is one well on one day.

### 2.2 Why We're Regenerating (Not Using the Old Data)

The existing `synthetic_corrosion_dataset.csv` has these issues:
- Only 2,000 days (~5.5 years) — real casing life is 25-30 years
- No field-level grounding (no lat/long, no field names)
- Wrong units (bar, m³/d, °C instead of API oilfield standard psi, bbl/d, °F)
- Missing columns the mentor and specs require (viscosity, casing metadata, H2S, corrosion cause)
- No multi-mechanism corrosion (only CO2, no H2S/MIC/erosion)

### 2.3 Well Assignment to Real Saudi Fields

Each of the 80 wells gets assigned to a real Saudi oil field from our researched database (`saudi_oil_well_locations.py` — 34 locations across 18 fields). The assignment:

| Field | # Wells | Type | Dominant Corrosion | Why |
|-------|---------|------|-------------------|-----|
| **Ghawar** (6 sub-areas) | 25 | Onshore | CO2 (sweet) | Largest field, most wells, Arab-D carbonate |
| **Khurais** | 10 | Onshore | CO2 (sweet) | Major modern field, Arab-D |
| **Safaniya** | 8 | Offshore | CO2 + erosion | Heavy crude, high flow, sandstone reservoir |
| **Shaybah** | 6 | Onshore (desert) | CO2 (sweet) | Extra-light crude, remote |
| **Manifa** | 6 | Offshore (shallow) | **H2S (sour)** | 14% H2S in gas — very sour |
| **Abqaiq** | 5 | Onshore | CO2 + MIC | Mature field, water injection introduces bacteria |
| **Berri** | 5 | Mixed | CO2 + oxygen | Water injection wells, seawater system |
| **Zuluf** | 5 | Offshore | CO2 + erosion | Deep offshore, high production |
| **Other** (Marjan, Qatif, Khursaniyah, etc.) | 10 | Mixed | Mixed | Variety for generalization |

Each well gets:
- A **real lat/long** from its field (with small random offset so wells aren't stacked)
- **Field-specific reservoir conditions** (initial pressure, temperature, depth, API gravity)
- **Field-appropriate casing specs** (grade, OD, initial wall thickness from API 5CT tables)

### 2.4 Field-Specific Initial Conditions

When we create a well, its starting parameters come from the real reservoir data we researched:

| Parameter | Ghawar | Safaniya | Manifa | Shaybah | Units |
|-----------|--------|----------|--------|---------|-------|
| Depth | 6,200-6,700 | 4,000-7,000 | 4,000-6,000 | ~4,900 | ft |
| Initial Pressure | 3,200-3,395 | 2,500-3,000 | 2,000-3,000 | ~2,800 | psi |
| Temperature | 210-240 | 160-200 | 160-190 | 180-200 | °F |
| API Gravity | 33-34 | 27 | 28-29 | 40-42 | ° |
| GOR | 636-850 | 200-350 | 300-400 | 600-700 | scf/STB |
| H2S Content | Low (<0.05 psi) | Low-Moderate | **High (14%)** | Low | mol% |
| Typical Casing Grade | N80 | N80/L80 | **L80/13Cr** | N80 | API 5CT |
| Initial Wall Thickness | 8.05-10.03 | 8.94-11.99 | 8.94-11.99 | 8.05-10.03 | mm |
| Production Casing OD | 7" | 9-5/8" | 9-5/8" | 7" | inches |

### 2.5 Corrosion Physics Models

We simulate **6 corrosion mechanisms** (one dominant per well, matching IS2 classification requirement):

**Mechanism 0 — CO2 Corrosion (Sweet):** ~35 wells
- Uses NORSOK M-506 model (already implemented in old pipeline)
- CR = K_T × f_CO2^0.62 × (S/19)^(0.146 + 0.0324×log(f_CO2)) × f(pH)_T
- Dominant in Ghawar, Khurais, Shaybah (low H2S environments)
- Corrosion rate range: 0.1–3.0 mm/yr depending on conditions

**Mechanism 1 — H2S Corrosion (Sour):** ~10 wells
- Uses De Waard-Milliams model modified for H2S
- Additional risk of Sulfide Stress Cracking (SSC) at high H2S partial pressure
- Dominant in Manifa (14% H2S), some Khuff gas wells
- More aggressive and less predictable than CO2 corrosion

**Mechanism 2 — MIC (Microbiologically Influenced Corrosion):** ~8 wells
- Pitting corrosion driven by sulfate-reducing bacteria (SRB)
- Occurs in water injection wells where seawater introduces bacteria
- Dominant in Abqaiq and Berri water injection systems
- Highly localized — modeled as sporadic pitting events on top of baseline

**Mechanism 3 — Erosion-Corrosion:** ~8 wells
- Synergistic effect of high flow velocity + corrosion
- Dominant in Safaniya (heavy crude, high flow) and Zuluf (deep offshore)
- Depends heavily on flow velocity, sand content, and fluid density
- Removes protective corrosion product layers

**Mechanism 4 — Oxygen Corrosion:** ~6 wells
- Occurs in water injection strings where dissolved O2 is present
- Dominant in Berri and other fields with seawater injection
- Controlled by O2 scavenging — models inhibitor on/off effect

**Mechanism 5 — Combined/Multi-mechanism:** ~13 wells
- Two or more mechanisms active simultaneously
- Represents the "real world" where conditions aren't clean
- Examples: CO2 + MIC in aging Ghawar injectors, H2S + erosion in Manifa

### 2.6 Time Evolution Per Well (30-Year Simulation)

Each well simulates these time-varying processes:

1. **Reservoir Pressure Decline:** Starts at field-specific initial pressure, declines ~1-2% per year (Saudi fields use water injection to maintain pressure, so decline is slow)
2. **Water Cut Increase:** Sigmoid curve from ~5% to 80-95% over 15-25 years (the biggest corrosion driver — water wetting of the casing wall)
3. **Temperature Cycling:** Seasonal variation ±5-10°F around field-specific base temperature
4. **Shut-ins:** Random operational shutdowns (10-30 days each, 2-5 per year). During shut-ins: flow=0, corrosion rate drops to near-zero, RUL pauses
5. **Choke Changes:** Sudden flow rate changes (2-4 per year), affecting flow velocity and shear stress
6. **Inhibitor Events:** Corrosion inhibitor active 70-90% of the time, with random failures (3-15 day gaps). Inhibitor efficiency: 70-95% when active
7. **Wall Thickness Degradation:** Initial thickness (from API 5CT spec) decreases daily by corrosion_rate/365. Failure when thickness < 3mm (or 50% of nominal)
8. **Viscosity:** Computed from API gravity and temperature using Beggs-Robinson correlation. Changes with water cut (oil-water emulsion)

### 2.7 Output Columns (25 columns)

The generated CSV will have these columns:

**Metadata (per well, constant):**

| Column | Example | Notes |
|--------|---------|-------|
| `Well_ID` | "GHAWAR-AIN-DAR-003" | Unique identifier |
| `Field_Name` | "Ghawar" | Real Saudi field |
| `Sub_Area` | "Ain Dar" | Sub-area within field |
| `Latitude` | 25.9492 | Real coordinates + small offset |
| `Longitude` | 49.4238 | Real coordinates + small offset |
| `Reservoir_Type` | "Carbonate" | Carbonate or Sandstone |
| `Casing_Grade` | "N80" | API 5CT grade |
| `Casing_OD_in` | 7.0 | Outer diameter in inches |
| `Initial_Thickness_mm` | 9.19 | Nominal wall thickness from API spec |

**Time-series features (change daily):**

| Column | Unit | Range | Notes |
|--------|------|-------|-------|
| `Day` | int | 0-10,950 | Simulation day |
| `Status` | int | 0 or 1 | 1=operating, 0=shut-in |
| `Pressure_psi` | psi | 1,500-3,500 | Reservoir/flowing pressure |
| `Temp_F` | °F | 140-260 | Operating temperature |
| `pH` | - | 3.5-7.0 | Formation water pH |
| `Water_Cut_pct` | % | 0-100 | Water percentage |
| `Production_Rate_bpd` | bbl/d | 0-15,000 | Production rate |
| `Flow_Velocity_fps` | ft/s | 0-25 | Flow velocity |
| `Shear_Stress_Pa` | Pa | 0-500 | Wall shear stress |
| `CO2_Partial_Pressure_psi` | psi | 0.1-75 | CO2 partial pressure |
| `H2S_Partial_Pressure_psi` | psi | 0-100 | H2S partial pressure (0 for sweet wells) |
| `Fluid_Density_ppg` | ppg | 7.0-11.0 | Fluid density in pounds per gallon |
| `Viscosity_cP` | cP | 0.5-50 | Dynamic viscosity |
| `Inhibitor_Active` | int | 0 or 1 | Corrosion inhibitor status |

**Target / Label columns:**

| Column | Unit | Notes |
|--------|------|-------|
| `Corrosion_Rate_mpy` | mils/yr | Instantaneous corrosion rate (1 mil = 0.0254 mm) |
| `Current_Thickness_mm` | mm | Current wall thickness |
| `Thickness_Loss_Pct` | % | (initial - current) / initial × 100 |
| `RUL_days` | days | Days until thickness < 3mm (or 50% of nominal) |
| `Corrosion_Cause` | int 0-5 | Dominant corrosion mechanism label |

### 2.8 Dataset Size Estimate

- 80 wells × ~8,000 days average (some fail early at year 5-15, some survive 30 years) = **~640,000 rows**
- 25 columns per row
- CSV size: ~150-200 MB
- Satisfies S3 (>8,000 unique simulation points) and S4 (>50 well logs)

### 2.9 Files to Create

| File | Description |
|------|-------------|
| `data_generation/config_fields.py` | Field-specific parameters (pressure, temp, casing specs per field) |
| `data_generation/corrosion_models.py` | NORSOK M-506 + H2S + MIC + erosion + O2 physics models |
| `data_generation/well_simulator.py` | Single-well 30-year simulation engine |
| `data_generation/generate_dataset.py` | Master script: create 80 wells, assign to fields, run simulations, save CSV |
| `data_generation/saudi_well_locations.py` | Already exists — real coordinates database |

---

## 3. Phase 2: Data Validation & EDA

### 3.1 Sanity Checks (Automated — TEST-01 through TEST-10)

These run automatically after data generation to verify physics are correct:

| Test | Check | Pass Condition |
|------|-------|----------------|
| TEST-01 | No NaN/Inf values in any column | Zero NaN/Inf |
| TEST-02 | Day column monotonically increases per well | All wells pass |
| TEST-03 | RUL_days monotonically decreasing per well (except shut-in plateaus) | All wells pass |
| TEST-04 | Current_Thickness_mm monotonically non-increasing per well | All wells pass |
| TEST-05 | When Status=0 (shut-in), Flow_Velocity ≈ 0 and Corrosion_Rate ≈ 0 | All shut-in rows pass |
| TEST-06 | Corrosion_Rate_mpy always ≥ 0 | No negative rates |
| TEST-07 | Water_Cut in [0,100], pH in [3.5,7.0], Temp_F in [100,300] | All rows in range |
| TEST-08 | For failed wells, final row has RUL_days = 0 | All failed wells pass |
| TEST-09 | Current_Thickness_mm never goes negative | Min thickness ≥ 0 |
| TEST-10 | Pressure correlates with CO2_Partial_Pressure | Pearson r > 0.3 |

### 3.2 EDA Visualizations

| Plot | What It Shows |
|------|--------------|
| Per-well trajectories (5 sample wells) | Thickness decay, RUL curve, feature evolution over 30 years |
| Feature distributions | Histograms of all features across entire dataset |
| Correlation heatmap | Verify physics relationships (pressure↔CO2, watercut↔corrosion) |
| RUL distribution | Check for imbalance at low RUL values |
| Corrosion cause distribution | Verify class balance across 6 categories |
| Field comparison boxplots | Compare corrosion rates and lifespans across fields |
| Failed vs. survived well comparison | Do failed wells have distinct feature patterns? |

### 3.3 File to Create

| File | Description |
|------|-------------|
| `src/data_validation.py` | Runs TEST-01 through TEST-10, prints pass/fail |
| `notebooks/01_EDA.ipynb` | Optional notebook for exploration visualizations |

---

## 4. Phase 3: Feature Engineering & Preprocessing

### 4.1 RUL Capping

```python
RUL_capped = min(RUL_days, 500)
```

**Why:** The model doesn't need to distinguish between 1,500 and 2,000 days remaining — both mean "far from failure." Capping at 500 concentrates the model's learning on the critical low-RUL zone where predictions matter for maintenance decisions. This is standard practice (NASA C-MAPSS benchmarks use the same technique).

### 4.2 Feature Selection

**Model input features (what the LSTM sees):**

For the **realistic model (Option A)** — drops Corrosion_Rate because in production you don't have the true instantaneous rate:

```
Status, Pressure_psi, Temp_F, pH, Water_Cut_pct, Production_Rate_bpd,
Flow_Velocity_fps, Shear_Stress_Pa, CO2_Partial_Pressure_psi,
H2S_Partial_Pressure_psi, Fluid_Density_ppg, Viscosity_cP,
Current_Thickness_mm, Inhibitor_Active
```
→ **14 features**

For the **cheating baseline (Option B)** — keeps Corrosion_Rate to verify pipeline works:

```
Same as Option A + Corrosion_Rate_mpy
```
→ **15 features**

**Excluded from model input (metadata only):**
- Well_ID, Field_Name, Sub_Area, Latitude, Longitude (used for splitting/grouping, not as features)
- Reservoir_Type, Casing_Grade, Casing_OD_in, Initial_Thickness_mm (static per well — could be added as static features later)
- Day (replaced by sequence position in the window)
- Corrosion_Cause (this is a TARGET, not an input)
- Thickness_Loss_Pct (derived from thickness, redundant)

### 4.3 Train / Val / Test Split

**CRITICAL: Split by Well_ID, NOT by row.** Row-level splitting causes temporal leakage.

```
Total wells: 80
Train: 48 wells (60%) → ~384,000 rows
Val:   16 wells (20%) → ~128,000 rows
Test:  16 wells (20%) → ~128,000 rows
```

**Stratification (two axes):**
1. **Failed vs. survived** — each split gets proportional numbers
2. **Corrosion cause** — each split has representation of all 6 classes
3. **Field** — each split has a mix of fields (not all Ghawar in train and all Manifa in test)

### 4.4 Normalization

- **Method:** Min-Max scaling to [0, 1]
- **Fit on training set ONLY** — then transform val/test with the same scaler
- **Save scaler** with joblib for use during inference in the dashboard
- Binary features (Status, Inhibitor_Active) are NOT scaled

### 4.5 Sliding Window Creation

Convert each well's time series into overlapping windows for the LSTM:

```
WINDOW_SIZE = 30 days (primary — also try 15, 60)
STRIDE = 1 (every day is a prediction point)

For each well:
  For t in range(WINDOW_SIZE, len(well)):
    X = scaled_features[t-WINDOW_SIZE : t]   → shape (30, 14)
    y_rul = RUL_capped[t]                     → scalar
    y_cr  = Corrosion_Rate_mpy[t]             → scalar
    y_wt  = Current_Thickness_mm[t]           → scalar
    y_cause = Corrosion_Cause[t]              → int (0-5)
    y_forecast = [thickness_at_t+365, t+730, t+1095, t+1460, t+1825]  → 5 scalars
```

Wells shorter than WINDOW_SIZE are dropped (shouldn't happen with 30-day window and minimum ~1,800 day wells).

### 4.6 Multi-Step Forecast Targets (IS3)

For IS3's 5-year forecast, we create look-ahead targets at each timestep:

```python
for t in range(WINDOW_SIZE, len(well) - 5*365):
    y_forecast = [
        well_thickness[t + 365],   # +1 year
        well_thickness[t + 730],   # +2 years
        well_thickness[t + 1095],  # +3 years
        well_thickness[t + 1460],  # +4 years
        well_thickness[t + 1825],  # +5 years
    ]
```

**Note:** The last 5 years of each well cannot be used as input for forecast training (no look-ahead target available). A 30-year well gives 25 years of valid forecast training samples — this is more than enough.

### 4.7 Files to Create

| File | Description |
|------|-------------|
| `src/config.py` | All hyperparameters, feature lists, split ratios, paths |
| `src/data_loader.py` | Load CSV, split, scale, window, create DataLoaders |

---

## 5. Phase 4: Model Architecture

### 5.1 The Multi-Task Design

All models share a common backbone that extracts temporal features from the input window, then branch into task-specific heads:

```
Input: (batch, 30, 14)     ← 30 timesteps, 14 features
         |
   [Shared Backbone]       ← BiLSTM, CNN-LSTM, or Transformer
         |
   [Temporal Features]     ← shape (batch, hidden_dim)
         |
   +--------+--------+--------+--------+--------+
   |        |        |        |        |        |
 Head 1   Head 2   Head 3   Head 4   Head 5
  RUL     Corr.    Wall     Cause    5-Year
 (days)   Rate     Thick    (6-cls)  Forecast
  relu    relu     relu     softmax  relu×5
```

### 5.2 Backbone Variants (we build and compare all 4)

**Variant 1: Naive Baseline (not neural)**
- Linear extrapolation from the slope of `Current_Thickness_mm` over the window
- Estimates when thickness hits 3mm
- This is the **"must beat" baseline** — if the LSTM can't beat this, something is wrong

**Variant 2: Simple LSTM**
```
LSTM(64, return_sequences=True) → Dropout(0.2)
→ LSTM(32, return_sequences=False) → Dropout(0.2)
→ shared_features (dim=32)
```

**Variant 3: BiLSTM + Attention (primary model)**
```
Bidirectional LSTM(64, return_sequences=True) → Dropout(0.3)
→ Bidirectional LSTM(32, return_sequences=True) → Dropout(0.2)
→ Temporal Attention (learnable weighted sum over timesteps)
→ shared_features (dim=64)
```

The **Temporal Attention Layer** learns which timesteps in the 30-day window matter most. For example, recent days with a sudden thickness drop should get higher attention weight than a stable period 25 days ago.

**Variant 4: CNN-LSTM Hybrid**
```
Conv1D(32, kernel=3, relu) → Conv1D(64, kernel=3, relu)
→ MaxPool1D(2) → LSTM(64, return_sequences=False)
→ shared_features (dim=64)
```

The CNN layers extract local patterns (short-term trends), then the LSTM captures longer dependencies.

### 5.3 Output Heads (same for all backbone variants)

| Head | Architecture | Activation | Loss | Target |
|------|-------------|------------|------|--------|
| **Head 1: RUL** | Dense(16, relu) → Dense(1) | ReLU (RUL ≥ 0) | Huber (δ=100) | RUL_capped |
| **Head 2: Corrosion Rate** | Dense(16, relu) → Dense(1) | ReLU (rate ≥ 0) | Huber (δ=50) | Corrosion_Rate_mpy |
| **Head 3: Wall Thickness** | Dense(16, relu) → Dense(1) | ReLU (thickness ≥ 0) | MSE | Current_Thickness_mm |
| **Head 4: Corrosion Cause** | Dense(32, relu) → Dense(6) | Softmax | CrossEntropy | Corrosion_Cause (0-5) |
| **Head 5: 5-Year Forecast** | Dense(32, relu) → Dense(5) | ReLU (thickness ≥ 0) | Huber (δ=50) | [WT+1yr, WT+2yr, WT+3yr, WT+4yr, WT+5yr] |

### 5.4 Multi-Task Loss

```python
L_total = w1*L_rul + w2*L_corr_rate + w3*L_wall_thick + w4*L_cause + w5*L_forecast

# Starting weights (tune on validation):
w1 = 1.0    # RUL — primary objective
w2 = 0.5    # Corrosion rate
w3 = 0.5    # Wall thickness
w4 = 0.3    # Cause classification
w5 = 0.5    # 5-year forecast
```

### 5.5 Confidence Intervals (IS3)

**Method: MC Dropout**

During inference, keep dropout layers active. Run N=50 forward passes for the same input. Each pass gives slightly different outputs due to random dropout. Take the 2.5th and 97.5th percentiles of the 50 predictions as the 95% confidence interval.

```python
model.train()  # keeps dropout active
predictions = [model(x) for _ in range(50)]
mean_pred = torch.stack(predictions).mean(dim=0)
lower_95 = torch.stack(predictions).quantile(0.025, dim=0)
upper_95 = torch.stack(predictions).quantile(0.975, dim=0)
```

Inference time: 50 × ~2ms = ~100ms per well. With 10 wells: ~1 second total. Meets S1 (0.5s per individual prediction) and IS5 (10 wells on 16GB GPU).

### 5.6 Corrosion Failure Index (IS4)

The CFI is a **post-processing formula**, not a model output:

```python
def compute_cfi(rul_days, corr_rate_mpy, thickness_loss_pct, cause_probs):
    # Wall Thickness Loss Score (0-100): 50% loss = 100
    wt_score = min(thickness_loss_pct / 50.0 * 100, 100)

    # Corrosion Rate Score (0-100): NACE SP0775 scale
    #   < 1 mpy → 0-20, 1-5 mpy → 20-50, 5-10 mpy → 50-75, >10 mpy → 75-100
    cr_score = nace_severity_scale(corr_rate_mpy)

    # Remaining Life Score (0-100): 0 days = 100, 5+ years = 0
    rul_score = max(0, 100 - (rul_days / (5 * 365)) * 100)

    # Cause Severity Score (0-100): weighted by predicted class probabilities
    cause_weights = [40, 70, 60, 55, 50, 80]  # CO2, H2S, MIC, erosion, O2, combined
    cause_score = sum(p * w for p, w in zip(cause_probs, cause_weights))

    # Final CFI
    CFI = 0.35 * wt_score + 0.25 * cr_score + 0.25 * rul_score + 0.15 * cause_score
    return round(CFI, 1)
```

**CFI Interpretation:**
- 0-25: Green (Safe) — standard monitoring
- 25-50: Yellow (Watch) — increased monitoring frequency
- 50-75: Orange (Elevated) — schedule inspection within 60 days
- 75-100: Red (Critical) — immediate inspection, consider replacement

### 5.7 Files to Create

| File | Description |
|------|-------------|
| `src/models.py` | NaiveBaseline, SimpleLSTM, BiLSTMAttention, CNNLSTM (all multi-task) |
| `src/attention.py` | Temporal Attention Layer implementation |

---

## 6. Phase 5: Training Pipeline

### 6.1 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | lr=1e-3 |
| Loss | Multi-task weighted (see 5.4) | Huber + CrossEntropy |
| Batch Size | 256 | Fits in 16GB VRAM |
| Epochs | 100 max | Early stopping will typically stop at 30-50 |
| Early Stopping | patience=15 on val_loss | Prevents overfitting |
| LR Scheduler | ReduceLROnPlateau | factor=0.5, patience=5 |
| Gradient Clipping | max_norm=1.0 | Prevents exploding gradients in LSTMs |
| Dropout | 0.2 (LSTM), 0.3 (BiLSTM) | Also used for MC Dropout at inference |

### 6.2 Training Loop (Per Epoch)

```
For each batch in train_loader:
    1. Forward pass → get all 5 head outputs
    2. Compute L_total (weighted sum of all task losses)
    3. Backward pass
    4. Clip gradients
    5. Optimizer step

After each epoch:
    1. Evaluate on val_loader → compute val_loss, val_MAE_rul, val_MAE_cr, val_accuracy_cause
    2. Log metrics (TensorBoard or print)
    3. LR scheduler step (based on val_loss)
    4. If val_loss improved → save model checkpoint
    5. If patience exhausted → early stop
```

### 6.3 Models to Train (6 experiments)

| Experiment | Backbone | Features | Purpose |
|-----------|----------|----------|---------|
| 1 | SimpleLSTM | Option B (with CR) | Pipeline verification ("cheating baseline") |
| 2 | SimpleLSTM | Option A (no CR) | Realistic baseline |
| 3 | BiLSTM+Attention | Option A | **Primary model** (expected best) |
| 4 | CNN-LSTM | Option A | Comparison |
| 5 | BiLSTM+Attention | Option A, window=15 | Window size ablation |
| 6 | BiLSTM+Attention | Option A, window=60 | Window size ablation |

### 6.4 Files to Create

| File | Description |
|------|-------------|
| `src/train.py` | Training loop, early stopping, LR scheduling, checkpoint saving |
| `src/losses.py` | Multi-task weighted loss, asymmetric loss option |

---

## 7. Phase 6: Evaluation & Testing

### 7.1 Metrics (Per Task)

**RUL Prediction (Head 1):**

| Metric | Formula | Target |
|--------|---------|--------|
| MAE | mean(\|actual - predicted\|) | < 60 days |
| RMSE | sqrt(mean((actual - predicted)²)) | < 80 days |
| R² | 1 - SS_res/SS_tot | > 0.92 |
| Late Prediction Rate | % where predicted > actual by >50 days | < 10% (safety) |

**Corrosion Rate (Head 2 — S2):**

| Metric | Formula | Target |
|--------|---------|--------|
| MAE | mean(\|actual - predicted\|) | < 5% of actual (S2 spec) |
| MAPE | mean(\|actual - predicted\| / actual) × 100 | < 5% (S2 compliance) |

**Wall Thickness Detection (Head 3 — IS1):**

| Metric | Formula | Target |
|--------|---------|--------|
| Detection Accuracy | % of samples where 10% loss correctly detected | > 90% (IS1 spec) |
| Precision | TP / (TP + FP) | > 85% |
| Recall | TP / (TP + FN) | > 90% |

**Corrosion Cause Classification (Head 4 — IS2):**

| Metric | Formula | Target |
|--------|---------|--------|
| Classification Accuracy | correct / total | > 70% (IS2 spec) |
| Macro F1 | average F1 across 6 classes | > 0.65 |
| Confusion Matrix | per-class breakdown | No class below 50% recall |

**5-Year Forecast (Head 5 — IS3):**

| Metric | Formula | Target |
|--------|---------|--------|
| MAE per horizon | MAE at each 12-month interval | MAE increases with horizon (expected) |
| CI Calibration | % of actual values falling within 95% CI | 90-98% (target: ~95%) |
| CI Width | average width of 95% CI | Narrow enough to be actionable |

### 7.2 Evaluation Plots

| Plot | What It Shows | Which Spec |
|------|--------------|------------|
| PLOT-01 | Predicted vs. Actual RUL scatter (45° line = perfect) | Core |
| PLOT-02 | RUL error distribution histogram | Core |
| PLOT-03 | MAE by RUL bucket (0-100, 100-200, 200-500, 500+) | Core |
| PLOT-04 | 4 test wells — predicted vs actual RUL trajectory overlay | Core |
| PLOT-05 | Training loss curves (train + val per epoch) | Core |
| PLOT-06 | Feature importance (permutation importance) | Core |
| PLOT-07 | Residual vs. predicted RUL (heteroscedasticity check) | Core |
| PLOT-08 | Corrosion rate: predicted vs actual scatter | S2 |
| PLOT-09 | Confusion matrix for corrosion cause classification | IS2 |
| PLOT-10 | 5-year forecast with confidence bands for 4 test wells | IS3 |
| PLOT-11 | CI calibration plot (expected vs observed coverage) | IS3 |
| PLOT-12 | CFI distribution across test wells | IS4 |
| PLOT-13 | Model comparison bar chart (MAE by model variant) | Comparison |

### 7.3 Critical Tests (Automated — TEST-11 through TEST-22)

| Test | Check | Pass Condition | Spec |
|------|-------|----------------|------|
| TEST-11 | All RUL predictions ≥ 0 | Zero negatives | Arch. |
| TEST-12 | Test MAE beats naive baseline | LSTM MAE < Baseline MAE | Core |
| TEST-13 | No single test well has MAE > 200 days | All wells below 200 | Core |
| TEST-14 | Late prediction rate < 15% | Safety requirement | Core |
| TEST-15 | Val and test MAE within 20% of each other | No overfitting | Core |
| TEST-16 | Shut-in wells don't have significantly worse MAE | Robust to shut-ins | Core |
| TEST-17 | No single feature removal causes >50% MAE increase | Not over-dependent | Core |
| TEST-18 | Window=30 outperforms window=15 | Architecture validation | Core |
| TEST-19 | Corrosion rate MAPE < 5% | S2 compliance | S2 |
| TEST-20 | 10% thickness detection accuracy > 90% | IS1 compliance | IS1 |
| TEST-21 | Cause classification accuracy > 70% | IS2 compliance | IS2 |
| TEST-22 | 95% CI coverage between 90-98% | IS3 compliance | IS3 |

### 7.4 Files to Create

| File | Description |
|------|-------------|
| `src/evaluate.py` | All metrics, all plots, all critical tests |
| `src/cfi.py` | Corrosion Failure Index computation |

---

## 8. Phase 7: Streamlit Dashboard

### 8.1 Overview

A 5-page Streamlit application that loads the trained model and lets field engineers monitor wells, enter data, and explore forecasts. Uses Plotly for all visualizations.

### 8.2 Page Structure

```
Sidebar (always visible):
├── App logo + title ("CasingGuard ML")
├── Active wells count
├── Last model run timestamp
└── Global alert badge

Pages:
├── Page 1: Fleet Overview (landing page)
├── Page 2: Well Detail (deep dive)
├── Page 3: Well Data Entry (configurable input → prediction)
├── Page 4: Forecast Explorer (what-if scenarios)
└── Page 5: Model Health & Diagnostics
```

### 8.3 Page 1 — Fleet Overview

The command center. Shows all monitored wells at a glance.

**Top row — 4 KPI cards (`st.metric`):**
- Active Wells: "10 / 10"
- Critical Alerts: count of wells with CFI > 80
- Fleet Average CFI: e.g., "34.2" with 30-day delta
- Next Predicted Failure: "WELL-007 in 42 days"

**Middle — Well Health Grid (2×5 cards):**
Each card shows:
- Well ID
- CFI gauge (Plotly `go.Indicator` — green/amber/red arc)
- Three mini-metrics: RUL, Corrosion Rate, Wall Thickness
- Status badge (Operating / Shut-In / Critical)
- Clickable → navigates to Well Detail

**Bottom — Fleet Comparison Charts:**
- Left: CFI bar chart for all wells (sorted descending)
- Right: RUL trend lines for all wells over past 90 days

**Alert Banner:** If any well has CFI > 75, a prominent warning banner at the top.

### 8.4 Page 2 — Well Detail

Deep dive into a single well's full history and predictions.

**Controls:** Well selector dropdown, date range picker.

**Section 1 — Well Header:** 5 KPI cards (CFI, RUL, Corrosion Rate, Wall Thickness, Days Since Inspection)

**Section 2 — Primary Time Series (Plotly subplots, shared x-axis):**
- Subplot A: Wall thickness over time (actual + predicted + 5-year forecast with 95% CI bands + failure threshold line + shut-in bands)
- Subplot B: Corrosion rate over time (actual + predicted, inhibitor periods as green bands)
- Subplot C: RUL trajectory (actual + predicted)

**Section 3 — Forecast Table:** Monthly breakdown of predicted thickness with upper/lower 95% CI bounds.

**Section 4 — Operating Conditions (expandable):** Last 30 days of sensor data, model input window.

**Section 5 — Inspection History (tabs):** Log of past inspections and maintenance actions.

### 8.5 Page 3 — Well Data Entry (Configurable Input)

This is what the mentor specifically asked for. A field engineer fills in well parameters and gets instant predictions.

**Layout:** Two columns — left is the input form, right is the results panel.

**Left Column — `st.form` with 5 sections:**

| Section | Fields |
|---------|--------|
| A: Well ID | Well selector or "New Well", Field Location |
| B: Casing | Grade (dropdown: J55/K55/N80/L80/P110), OD (inches), Initial Thickness (mm) |
| C: Reservoir | Pressure (psi), Temperature (°F), CO2 pp (psi), H2S pp (psi), Water Cut (%) |
| D: Production | Production Rate (bpd), Flow Velocity (fps), Viscosity (cP) |
| E: Chemistry | pH, Fluid Density (ppg), Inhibitor Status (Active/Failed/None) |
| F: History | Well Age (days), Current Measured Thickness (mm), Days Since Last Inspection |
| G: Forecast Config | Horizon (1yr/2yr/5yr), Model variant selector |

**"Run Prediction" button** at the bottom.

**Right Column — Results (appears after submit):**
- Large CFI gauge with color coding
- 3 metric cards: RUL ± CI, Corrosion Rate, Time to Critical
- Forecast chart (thickness over selected horizon with confidence bands)
- Recommendation box (rule-based: "Schedule inspection within 30 days")
- Download button (CSV or report)

**Smart defaults:** When a known well is selected, all fields auto-populate from the last recorded values.

### 8.6 Page 4 — Forecast Explorer (What-If)

Side-by-side scenario comparison.

**Layout:** Two panels (Scenario A baseline vs Scenario B modified).
Each panel has parameter sliders and a forecast chart below.

**Difference section below:** Shows delta between scenarios — "Scenario B extends RUL by 87 days due to improved pH and inhibitor status."

**Sensitivity tornado chart:** Which parameters, if changed ±10%, produce the largest RUL change.

### 8.7 Page 5 — Model Health

For data scientists / model maintainers.

- Model performance comparison table (MAE, R², accuracy per model variant)
- Predicted vs. Actual scatter plot (from test set)
- Error distribution histogram
- Data freshness monitoring per well
- Feature importance bar chart

### 8.8 Files to Create

```
dashboard/
├── app.py                    # Main entry point, page registration
├── pages/
│   ├── 1_fleet_overview.py   # Fleet command center
│   ├── 2_well_detail.py      # Single well deep dive
│   ├── 3_data_entry.py       # Configurable well input
│   ├── 4_forecast_explorer.py # What-if scenarios
│   └── 5_model_health.py     # Diagnostics
├── components/
│   ├── cfi_gauge.py          # Reusable CFI gauge component
│   ├── well_card.py          # Reusable well summary card
│   └── forecast_chart.py     # Reusable forecast + CI chart
├── utils/
│   ├── model_loader.py       # Load trained .pt model + scaler
│   ├── inference.py          # MC Dropout inference, CFI computation
│   └── data_utils.py         # Data formatting helpers
└── assets/
    └── logo.png              # App branding
```

---

## 9. Complete File Structure

```
casing_rul_prediction/
│
├── data/
│   ├── synthetic_corrosion_dataset.csv      # Generated dataset (~800K rows)
│   └── saudi_well_locations.py              # Real coordinates database
│
├── data_generation/
│   ├── config_fields.py                     # Field-specific parameters
│   ├── corrosion_models.py                  # NORSOK + H2S + MIC + erosion physics
│   ├── well_simulator.py                    # Single-well simulation engine
│   └── generate_dataset.py                  # Master generation script
│
├── src/
│   ├── config.py                            # All hyperparameters
│   ├── data_loader.py                       # Load, validate, split, window, scale
│   ├── data_validation.py                   # TEST-01 through TEST-10
│   ├── models.py                            # All 4 model architectures (multi-task)
│   ├── attention.py                         # Temporal Attention layer
│   ├── losses.py                            # Multi-task weighted loss
│   ├── train.py                             # Training loop
│   ├── evaluate.py                          # Metrics + plots + critical tests
│   ├── cfi.py                               # Corrosion Failure Index formula
│   └── utils.py                             # Helpers (scaler save/load, etc.)
│
├── dashboard/
│   ├── app.py                               # Streamlit main entry
│   ├── pages/
│   │   ├── 1_fleet_overview.py
│   │   ├── 2_well_detail.py
│   │   ├── 3_data_entry.py
│   │   ├── 4_forecast_explorer.py
│   │   └── 5_model_health.py
│   ├── components/
│   │   ├── cfi_gauge.py
│   │   ├── well_card.py
│   │   └── forecast_chart.py
│   └── utils/
│       ├── model_loader.py
│       ├── inference.py
│       └── data_utils.py
│
├── outputs/
│   ├── models/                              # Saved .pt checkpoints
│   ├── plots/                               # All evaluation figures
│   ├── metrics/                             # JSON results per experiment
│   └── scalers/                             # Saved MinMaxScaler objects
│
├── notebooks/
│   └── 01_EDA.ipynb                         # Exploration notebook
│
├── run_experiment.py                         # Master training + eval script
├── requirements.txt                          # Dependencies
└── PROJECT_ROADMAP.md                        # This file
```

---

## 10. Spec Compliance Checklist

| Spec | Requirement | How We Meet It | Status |
|------|------------|----------------|--------|
| **C3** | Comply with Aramco data format standards | API oilfield units (psi, bbl/d, °F, mpy), real field names | Designed |
| **C6** | Min 500 data points or 50 well logs | 80 wells, ~640K+ rows | Met |
| **C8** | Dedicated GPU for training/inference | PyTorch with CUDA support, works on KFUPM GPUs | Ready |
| **S1** | Prediction < 0.5 seconds | Single LSTM forward pass ~2-5ms. MC Dropout (50 passes) ~100ms | Will meet |
| **S2** | Corrosion rate MAE < 5% | Head 2 output, trained with Huber loss | Will test |
| **S3** | ≥ 8,000 unique simulation points | ~640,000 rows from 80 wells | Met |
| **S4** | ≥ 500 data points or 50 well logs | 80 wells >> 50 | Met |
| **S5** | Lab validation with 1 API 5CT grade | Calibrate NORSOK model against published L80/N80 lab data | Planned |
| **S6** | ~30% reduction in monitoring costs | CFI-based scheduling extends inspection intervals | Dashboard |
| **S7** | ~25% reduction in high-risk inversions | Continuous monitoring eliminates unnecessary inspections | Dashboard |
| **IS1** | Detect 10% wall loss with 90% accuracy | Threshold on Head 3 (wall thickness) output | Will test |
| **IS2** | Corrosion cause classification 70% accuracy | Head 4 (6-class softmax) | Will test |
| **IS3** | 5-year forecast with 95% CI | Head 5 (5-step output) + MC Dropout | Designed |
| **IS4** | Corrosion Failure Index 0-100 | Post-processing formula from all head outputs | Designed |
| **IS5** | 10 wells on 16GB VRAM | Batch inference of 10 windows, LSTM is lightweight | Will meet |

---

## 11. Build Order

### Step 1: Data Generation (~2-3 hours)
1. `data_generation/config_fields.py` — field parameters
2. `data_generation/corrosion_models.py` — physics models
3. `data_generation/well_simulator.py` — simulation engine
4. `data_generation/generate_dataset.py` — run generation
5. Verify output CSV: correct columns, correct ranges, correct size

### Step 2: Validation & Config (~1 hour)
6. `src/config.py` — all hyperparameters
7. `src/data_validation.py` — run TEST-01 through TEST-10
8. (Optional) `notebooks/01_EDA.ipynb` — exploration

### Step 3: Data Pipeline (~1.5 hours)
9. `src/data_loader.py` — load, split, scale, window, DataLoaders

### Step 4: Models (~1.5 hours)
10. `src/attention.py` — Temporal Attention layer
11. `src/models.py` — all 4 architectures with 5 output heads
12. `src/losses.py` — multi-task weighted loss
13. `src/cfi.py` — CFI formula

### Step 5: Training (~1 hour code + N hours GPU time)
14. `src/train.py` — training loop
15. `run_experiment.py` — master script
16. Train all 6 experiments → save checkpoints

### Step 6: Evaluation (~1.5 hours)
17. `src/evaluate.py` — all metrics, plots, critical tests
18. Run evaluation → verify all specs pass
19. Generate comparison tables and all 13 plots

### Step 7: Dashboard (~3-4 hours)
20. `dashboard/app.py` + page structure
21. `dashboard/utils/` — model loader, inference, data utils
22. `dashboard/components/` — reusable components
23. `dashboard/pages/1_fleet_overview.py` — fleet command center
24. `dashboard/pages/2_well_detail.py` — well deep dive
25. `dashboard/pages/3_data_entry.py` — configurable input (mentor requirement)
26. `dashboard/pages/4_forecast_explorer.py` — what-if scenarios
27. `dashboard/pages/5_model_health.py` — diagnostics

### Step 8: Integration Testing (~1 hour)
28. End-to-end test: generate data → train → evaluate → dashboard
29. Verify all specs in compliance checklist
30. Final cleanup and documentation

---

**Total estimated work: ~15-20 hours of Claude Code time + GPU training time**

**START WITH STEP 1.**
