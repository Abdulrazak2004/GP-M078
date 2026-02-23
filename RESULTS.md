# Experiment Results — Casing RUL Prediction Pipeline

## Latest Run: `exp3_bilstm_optA` (Physics-Informed BiLSTM + Attention)

**Date:** February 23, 2026
**Hardware:** 4x NVIDIA RTX 4090 (vast.ai), parallel fold training
**Architecture:** BiLSTM + Temporal Attention, physics-informed CR head, 26 features

### Test Set Performance (100 wells held out)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| RUL MAE | 7.3 days | < 60 days | PASS |
| RUL R² | 0.671 | > 0.50 | PASS |
| RUL RMSE | 44.1 days | < 80 days | PASS |
| WT MAE | 0.242 mm | < 0.5 mm | PASS |
| WT Loss Detection | 96.9% | >= 90% | PASS |
| Forecast MAE | 0.698 mm | < 1.5 mm | PASS |
| CI Coverage (95%) | 96.2% | >= 85% | PASS |
| RUL vs Baseline | 7.3 < 315.9 | Model < Naive | PASS |
| Training Converged | epoch 63 | best epoch > 5 | PASS |
| Inference Speed | 0.024 ms/sample | < 500 ms | PASS |
| **CR NMAE** | **100%** | **< 5%** | **FAIL** |
| **CR R²** | **-0.148** | **> 0.30** | **FAIL** |

**Result: 10/12 critical tests passing**

### Additional Metrics (not in critical tests)

| Metric | Value |
|--------|-------|
| WT R² | 0.972 |
| WT RMSE | 0.307 mm |
| WT Loss Detection Recall | 99.9% |
| CR MAE | 9.10 mpy |
| CI Mean Width | 5.0 days |

### 5-Fold Cross-Validation Summary

| Fold | Best Epoch | Val MAE RUL | Val MAE CR |
|------|-----------|-------------|------------|
| 1 | 31 | 7.6 days | 8.97 mpy |
| 2 | 81 | 7.4 days | 9.80 mpy |
| 3 | 19 | 7.6 days | 9.31 mpy |
| 4 | 15 | 8.1 days | 9.22 mpy |
| 5 | 80 | 7.3 days | 9.18 mpy |
| **Average** | — | **7.6 +/- 0.3 days** | **9.30 +/- 0.28 mpy** |

CV time: ~2.1 hours (parallel across 4 GPUs)

### Comparison Across All Experiments (Pre-Physics Run)

These results are from the earlier run (before physics-informed changes). They show relative model comparisons:

| Experiment | RUL MAE | RUL R² | CR MAE | CR R² | WT MAE | WT R² | Forecast MAE |
|------------|---------|--------|--------|-------|--------|-------|-------------|
| exp1_lstm_optB | 19.1 | 0.393 | 11.8 | -0.058 | 1.18 | 0.096 | 1.79 |
| exp2_lstm_optA | 21.8 | 0.395 | 12.0 | -0.054 | 1.14 | 0.252 | 2.02 |
| exp3_bilstm_optA | 28.6 | 0.324 | 10.2 | 0.209 | 1.11 | 0.347 | 2.35 |
| exp4_cnnlstm_optA | 20.9 | 0.227 | 10.7 | 0.199 | 0.62 | 0.790 | 1.74 |
| exp5_bilstm_w15 | 24.6 | 0.382 | 11.2 | 0.047 | 1.18 | 0.281 | 2.55 |
| exp6_bilstm_w60 | 30.8 | 0.316 | 11.1 | 0.018 | 1.19 | 0.237 | 2.73 |

### Key Improvements from Physics-Informed Architecture

Comparing exp3_bilstm_optA before vs. after physics changes:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| RUL MAE | 28.6 days | 7.3 days | -74% |
| RUL R² | 0.324 | 0.671 | +107% |
| WT MAE | 1.112 mm | 0.242 mm | -78% |
| WT Detection | ~85% | 96.9% | +12pp |
| Forecast MAE | 2.35 mm | 0.698 mm | -70% |
| CR R² | 0.209 | -0.148 | -171% (regression) |

---

## CR Failure Analysis

### What Failed

The Corrosion Rate (CR) prediction failed both critical tests:
- **CR NMAE = 100%** (target: < 5%) — predictions are off by the full scale of the target
- **CR R² = -0.148** (target: > 0.50) — model is worse than predicting the mean

### Root Cause: Physics-Informed CR Head Backfired

The CR head was redesigned based on a correct theoretical insight but a flawed practical assumption:

**The Theory (Correct):**
Corrosion rate is instantaneous — the NORSOK CO2 equation takes current conditions (temperature, pressure, pH, CO2 partial pressure) and outputs CR directly. There's zero temporal dependency. So feeding CR through an LSTM is wasteful.

**The Implementation:**
Changed the CR head from using temporal LSTM features → to a standalone MLP that takes only the raw features from the last timestep: `x[:, -1, :]` → Linear(26→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) → ReLU

**Why It Failed:**

1. **The dataset has 6 different corrosion equations** (CO2, H2S, MIC, Erosion, O2, Combined), each with completely different math. A shallow 3-layer MLP with 26 inputs can't simultaneously learn 6 different non-linear equations.

2. **The physics loss is too weak to constrain CR.** The WT-CR consistency loss computes:
   ```
   WT(t) ≈ WT(t-1) - CR(t) * MPY_TO_MMDAY
   ```
   But `MPY_TO_MMDAY = 0.0254 / 365.0 ≈ 6.96e-5`. This means a CR error of 100 mpy only causes a WT error of 0.007 mm/day — the physics loss can be trivially satisfied even with wildly wrong CR predictions.

3. **Before the change, CR had access to temporal LSTM features** which gave it some signal (R² = 0.209). Removing temporal features made it strictly worse.

### Recommendations to Fix CR

1. **Give the CR head both raw AND temporal features** — concatenate `[raw_last, temporal_features]` as input to the CR head. This preserves the physics intuition while giving the head access to temporal context.

2. **Add a mechanism-routing layer** — the model needs to know which of the 6 corrosion equations applies. The one-hot Reservoir_Type features help but aren't sufficient. A gating mechanism or mixture-of-experts approach could help.

3. **Larger CR MLP** — even with perfect input features, a 26→64→32→1 MLP may be too small to learn 6 distinct equations. Consider 26→128→128→64→1.

4. **Scale the physics loss** — multiply CR by a larger constant or normalize CR to [0,1] before the physics constraint so that CR errors have meaningful gradient signal.

5. **Direct CR supervision with per-equation loss** — if corrosion cause labels are known during training, weight the CR loss by equation type.

---

## What's Missing

### Tests Not Yet Passing (2/12)

| Test | Current | Target | Gap |
|------|---------|--------|-----|
| CR NMAE | 100% | < 5% | Need fundamental CR head redesign |
| CR R² | -0.148 | > 0.50 | Need access to temporal features + routing |

### Experiments Not Re-Run with Physics Architecture

The physics-informed architecture was only fully trained for `exp3_bilstm_optA`. The comparison table above shows old results for other experiments. Consider re-running:
- `exp4_cnnlstm_optA` — had best WT R² (0.790) before physics changes
- `exp6_transformer_optA` — never run with current architecture

### Cause Classification

Cause classification was removed from the multi-task model and trained separately. The confusion matrix shows the model predicts only class 0 (CO2) for all samples — this is a majority-class collapse. Not part of the 12 critical tests but listed in evaluation output.

---

## File Locations

| What | Path |
|------|------|
| Best model checkpoint | `outputs/exp3_bilstm_optA/models/best_model.pt` |
| Test metrics JSON | `outputs/exp3_bilstm_optA/metrics/test_metrics.json` |
| Critical tests JSON | `outputs/exp3_bilstm_optA/metrics/critical_tests.json` |
| CV results | `outputs/exp3_bilstm_optA/metrics/cv_results.json` |
| Training history | `outputs/exp3_bilstm_optA/metrics/training_history.json` |
| Loss curves plot | `outputs/exp3_bilstm_optA/plots/loss_curves.png` |
| All evaluation plots | `outputs/exp3_bilstm_optA/plots/plot01-plot12*.png` |
| Model comparison table | `outputs/comparison/comparison_table.csv` |
| Model comparison plot | `outputs/comparison/plot13_model_comparison.png` |
| Training logs | `outputs/logs/exp3_bilstm_optA.log` |
