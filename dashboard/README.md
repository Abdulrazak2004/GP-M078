# Corrosion Intelligence Dashboard

Palantir-style interactive dashboard for the GP defense, showcasing the trained BiLSTM-Attention model on 500 synthetic Saudi oil wells.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19 + Vite + Tailwind CSS v4 |
| Backend | FastAPI + Uvicorn |
| Model | PyTorch BiLSTM-Attention (multi-task) |
| Charts | Recharts |
| Map | react-leaflet + CartoDB Dark Matter tiles |
| Icons | lucide-react |

## Quick Start

```bash
# From project root
chmod +x dashboard/start.sh
./dashboard/start.sh
```

Or manually:

```bash
# Terminal 1 — Backend (takes ~45-60s to load 3.65M rows)
cd dashboard/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd dashboard/frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173` | Backend API: `http://localhost:8000/docs`

---

## Pages

### 1. Operations Overview (MapView)

Full-screen Leaflet dark map centered on Saudi Arabia showing all 500 wells as colored markers by CFI risk level.

- **Marker colors**: Green (CFI 0-25), Yellow (25-50), Orange (50-75), Red (75-100)
- **Critical wells** shown with larger radius + thick stroke for visibility
- **Click a well** → glassmorphic detail panel with CFI gauge, RUL, WT, CR, and link to monitor
- **Floating stats panel** with total counts per risk level
- **Filter bar**: by field, risk level, reservoir type

### 2. Well Surveillance (WellMonitor)

Day-by-day AI prediction playback with time accelerator. The core showcase page.

**Layout — Split Screen: Reality vs AI**

| Left Panel | Right Panel |
|-----------|-------------|
| Actual sensor data (wall thickness) | AI predictions with 90% confidence band (MC Dropout) |

**5 KPI Cards**:
- Wall Thickness (actual) — color-coded by severity
- AI Predicted WT (cyan) — model's prediction
- Corrosion Rate (actual)
- Thickness Loss % — cumulative degradation
- CFI — Corrosion Failure Index with colored glow

**AI Showcase Components**:

| Component | What It Shows |
|-----------|--------------|
| **Attention Heatmap** | Which of the 30 input days the model focuses on (Bahdanau attention weights) |
| **Accuracy Tracker** | Running MAPE-based accuracy (typically 97-99%), live pred vs actual comparison, MAE stats |
| **Input Window Viz** | 30-day × 11-feature heatmap — the exact data matrix fed into the BiLSTM |
| **CFI Risk Evolution** | Area chart showing CFI climbing through risk zones over time |
| **Forecast Chart** | 60-month wall thickness forecast from current playback position |

**Playback Controls**: Play/Pause, step forward/back, speed (1x / 10x / 100x / 1000x), draggable slider.

### 3. Well Designer (WellDesigner)

"Design Your Well" tool — configure hypothetical well parameters and get a 30-year AI prediction.

**Input Parameters**:
- Reservoir type (Carbonate / Clastic / Mixed)
- Casing grade (J55 / K55 / N80 / L80 / P110)
- Initial wall thickness, pressure, temperature, pH, water cut
- Corrosion cause (CO2 / H2S / MIC / Erosion)

**Output**:
- Predicted RUL with confidence interval
- 30-year wall thickness trajectory chart
- CFI evolution chart
- Horizontal risk timeline (when each CFI zone is reached)
- Material recommendation text
- Ghost line comparison when re-predicting with different params

---

## Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/wells/summary` | GET | All 500 wells with id, lat, lon, field, CFI, risk. Supports `?field=`, `?risk=`, `?reservoir_type=` filters |
| `/api/wells/stats` | GET | Aggregate counts per risk level |
| `/api/wells/{id}/timeseries` | GET | Full time series for one well |
| `/api/wells/{id}/predictions?day=` | GET | Model prediction at a specific day |
| `/api/wells/{id}/playback?stride=30` | GET | Pre-computed predictions at every Nth day for playback. Returns attention weights, MC Dropout CI, window features, errors |
| `/api/design-well` | POST | Simulate + predict a hypothetical well configuration |

### Playback Response (per prediction point)

```json
{
  "day": 1500,
  "wt": 8.42,
  "cr": 3.21,
  "rul": 2847,
  "cfi": 34.2,
  "forecast": [8.3, 8.1, ...],
  "actual_wt": 8.45,
  "actual_cr": 3.18,
  "actual_rul": 2900,
  "wt_error": 0.03,
  "cr_error": 0.03,
  "rul_error": 53,
  "wt_ci_low": 8.38,
  "wt_ci_high": 8.46,
  "attention": [0.02, 0.03, ..., 0.08],
  "window_features": [[...], ...],
  "thickness_loss_pct": 12.3
}
```

---

## Design System

Palantir Blueprint-inspired dark theme.

### Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `bg-primary` | `#111418` | App background |
| `bg-surface` | `#1C2127` | Cards, panels |
| `bg-elevated` | `#252A31` | Hover states |
| `bg-border` | `#2F343C` | Borders, dividers |
| `accent-blue` | `#4C90F0` | Primary actions |
| `accent-green` | `#32A467` | Safe / healthy |
| `accent-orange` | `#EC9A3C` | Warning |
| `accent-red` | `#E76A6E` | Critical |
| `accent-cyan` | `#68D4F8` | AI highlights, glow |

### Typography
- **Font**: Inter (sans-serif)
- **KPI numbers**: `font-mono tabular-nums`
- **Labels**: `text-xs uppercase tracking-widest`

### Visual Patterns
- Flat cards with subtle borders (no shadows)
- Glow effects on critical metrics (`box-shadow` with accent color)
- Glassmorphism on floating panels (`backdrop-blur-md`)
- Fade-in / slide-in animations

---

## Model Details

The dashboard serves the **BiLSTM-Attention** model from `outputs/exp3_bilstm_optA/`:

- **Architecture**: Bidirectional LSTM → Bahdanau Temporal Attention → Multi-task heads
- **Input**: 30-day sliding window × 26 engineered features
- **Outputs**: Wall Thickness, Corrosion Rate, RUL, 60-month Forecast
- **Uncertainty**: MC Dropout (15 stochastic forward passes) → 90% confidence intervals
- **CFI**: Composite index = 0.35×WT + 0.25×CR + 0.25×RUL + 0.15×Cause

---

## Key Decisions & Fixes

| Issue | Solution |
|-------|----------|
| Map markers "floating" every second | Removed CSS pulse animation from Leaflet CircleMarkers; used size/stroke differentiation instead |
| Accuracy showing 0% | Switched from range-based metric to MAPE (Mean Absolute Percentage Error); actual accuracy ~98% |
| CR predictions always 0.00 for healthy wells | Model behavior — ReLU clamps near-zero for low-corrosion wells. Not a bug. |
| KPIs never changing during playback | Default well was too healthy; now defaults to a well with CFI 55-80 showing visible degradation |
| FitBounds re-running on every render | Added `useRef` guard to only fit map bounds once on initial load |

---

## File Structure

```
dashboard/
├── backend/
│   ├── main.py              # FastAPI app, CORS, 6 endpoints
│   ├── inference.py          # Model loading, MC Dropout, attention extraction
│   ├── wells_data.py         # Load 500 wells, assign geo coords, build summary
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # React Router (3 pages)
│   │   ├── index.css         # Tailwind v4 theme + custom animations
│   │   ├── pages/
│   │   │   ├── MapView.jsx       # Leaflet map + well markers
│   │   │   ├── WellMonitor.jsx   # AI playback showcase
│   │   │   └── WellDesigner.jsx  # Design-your-well tool
│   │   ├── components/
│   │   │   ├── Layout.jsx            # Header + sidebar shell
│   │   │   ├── MetricCard.jsx        # Glowing KPI card
│   │   │   ├── TimeSlider.jsx        # Playback controls
│   │   │   ├── AttentionHeatmap.jsx  # Attention weight strip
│   │   │   ├── InputWindowViz.jsx    # 30×11 feature heatmap
│   │   │   ├── AccuracyTracker.jsx   # MAPE accuracy scorecard
│   │   │   ├── CfiGauge.jsx          # Semi-circular CFI gauge
│   │   │   ├── RiskTimeline.jsx      # Horizontal risk bar
│   │   │   └── WellDetailPanel.jsx   # Map flyout panel
│   │   └── utils/
│   │       ├── api.js        # API client functions
│   │       └── colors.js     # Palantir palette + helpers
│   ├── index.html
│   ├── package.json
│   └── vite.config.js        # Tailwind v4 plugin + API proxy
└── start.sh                  # Launch both backend + frontend
```
