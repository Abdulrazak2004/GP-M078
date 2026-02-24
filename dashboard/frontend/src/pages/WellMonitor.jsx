import { useEffect, useState, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer,
  ReferenceLine, Area, ComposedChart, Tooltip as RTooltip, AreaChart,
} from 'recharts';
import { getWellsSummary, getWellPlayback } from '../utils/api';
import { CHART_COLORS, getCfiColor, COLORS } from '../utils/colors';
import MetricCard from '../components/MetricCard';
import TimeSlider from '../components/TimeSlider';
import AttentionHeatmap from '../components/AttentionHeatmap';
import InputWindowViz from '../components/InputWindowViz';
import AccuracyTracker from '../components/AccuracyTracker';

// Exponential moving average for smoothing noisy model predictions
function ema(data, key, alpha = 0.3) {
  if (!data.length) return data;
  const result = [...data];
  let prev = result[0][key];
  for (let i = 0; i < result.length; i++) {
    const raw = result[i][key];
    if (raw != null) {
      prev = alpha * raw + (1 - alpha) * (prev ?? raw);
      result[i] = { ...result[i], [key]: +prev.toFixed(3) };
    }
  }
  return result;
}

function SetPageTitle() {
  useEffect(() => {
    const el = document.getElementById('page-title');
    if (el) el.textContent = 'Well Surveillance';
  }, []);
  return null;
}

export default function WellMonitor() {
  const [searchParams] = useSearchParams();
  const [wellList, setWellList] = useState([]);
  const [selectedWell, setSelectedWell] = useState(searchParams.get('well') || '');
  const [playbackData, setPlaybackData] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);

  // Load well list — default to a well with visible degradation
  useEffect(() => {
    getWellsSummary().then(data => {
      setWellList(data.wells);
      if (!selectedWell && data.wells.length > 0) {
        // Pick a well with interesting degradation (CFI 55-80) for demo
        const interesting = data.wells.find(w => w.cfi >= 55 && w.cfi <= 80);
        setSelectedWell(interesting?.well_id || data.wells[0].well_id);
      }
    });
  }, []);

  // Load playback data when well changes
  useEffect(() => {
    if (!selectedWell) return;
    setLoading(true);
    setPlaying(false);
    setCurrentStep(0);
    getWellPlayback(selectedWell, 30)
      .then(data => {
        setPlaybackData(data.predictions);
        setMetadata(data.metadata);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load playback:', err);
        setLoading(false);
      });
  }, [selectedWell]);

  const current = playbackData?.[currentStep];

  // Chart data: side-by-side actual vs predicted up to current step
  const historyData = useMemo(() => {
    if (!playbackData) return [];
    const raw = playbackData.slice(0, currentStep + 1).map(p => ({
      day: p.day,
      year: +(p.day / 365.25).toFixed(1),
      actual_wt: p.actual_wt,
      predicted_wt: p.wt,
      wt_ci_low: p.wt_ci_low,
      wt_ci_high: p.wt_ci_high,
      actual_cr: p.actual_cr,
      predicted_cr: p.cr,
      cfi: p.cfi,
      actual_rul: p.actual_rul,
      predicted_rul: p.rul,
    }));
    // Smooth the noisy model predictions (EMA alpha=0.3)
    return ema(ema(ema(raw, 'predicted_wt', 0.3), 'predicted_cr', 0.3), 'cfi', 0.3);
  }, [playbackData, currentStep]);

  // Full timeline (faded, for context)
  const fullWtData = useMemo(() => {
    if (!playbackData) return [];
    return playbackData.map(p => ({
      day: p.day,
      year: +(p.day / 365.25).toFixed(1),
      actual_wt: p.actual_wt,
    }));
  }, [playbackData]);

  // Forecast from current point
  const forecastData = useMemo(() => {
    if (!current?.forecast) return [];
    return current.forecast.map((wt, i) => ({
      month: i + 1,
      wt,
    }));
  }, [current]);

  const handleStepChange = useCallback((valOrFn) => {
    setCurrentStep(prev => {
      const next = typeof valOrFn === 'function' ? valOrFn(prev) : valOrFn;
      return Math.max(0, Math.min(next, (playbackData?.length || 1) - 1));
    });
  }, [playbackData]);

  const handlePlayPause = useCallback((val) => setPlaying(val), []);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <SetPageTitle />
        <div className="text-center animate-fade-in">
          <div className="w-10 h-10 border-2 border-accent-blue border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <div className="text-sm text-text-muted tracking-widest uppercase">
            Loading AI Predictions...
          </div>
          <div className="text-xs text-text-muted mt-2">
            Running MC Dropout inference ({'>'}5,000 forward passes)
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <SetPageTitle />

      {/* Top bar: well selector + metadata */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-bg-border shrink-0">
        <label className="text-xs uppercase tracking-widest text-text-muted">Well</label>
        <select
          value={selectedWell}
          onChange={e => setSelectedWell(e.target.value)}
          className="bg-bg-surface text-text-primary text-sm px-3 py-1.5 rounded-lg border border-bg-border focus:outline-none focus:border-accent-blue"
        >
          {wellList.map(w => (
            <option key={w.well_id} value={w.well_id}>
              {w.well_id} — {w.field} ({w.sub_area})
            </option>
          ))}
        </select>
        {metadata && (
          <div className="ml-auto flex items-center gap-3 text-xs text-text-muted">
            <span>{metadata.reservoir_type}</span>
            <span className="w-px h-3 bg-bg-border" />
            <span>{metadata.casing_grade}</span>
            <span className="w-px h-3 bg-bg-border" />
            <span>WT₀: {metadata.initial_thickness} mm</span>
          </div>
        )}
      </div>

      {/* Playback controls */}
      {playbackData && (
        <div className="px-4 py-2 shrink-0">
          <TimeSlider
            totalSteps={playbackData.length}
            currentStep={currentStep}
            onStepChange={handleStepChange}
            playing={playing}
            onPlayPause={handlePlayPause}
          />
        </div>
      )}

      {/* Main content: split layout */}
      {current && (
        <div className="flex-1 min-h-0 overflow-y-auto px-4 pb-4">
          {/* KPI Cards row — show actual + AI predicted side-by-side */}
          <div className="grid grid-cols-5 gap-3 mb-3 animate-fade-in">
            <MetricCard
              label="Wall Thickness"
              value={current.actual_wt?.toFixed(2)}
              unit="mm"
              color={current.actual_wt < 5 ? 'red' : current.actual_wt < 7 ? 'orange' : 'blue'}
            />
            <MetricCard
              label="AI Predicted WT"
              value={current.wt?.toFixed(2)}
              unit="mm"
              color="cyan"
            />
            <MetricCard
              label="Corrosion Rate"
              value={current.actual_cr?.toFixed(2)}
              unit="mpy"
              color={current.actual_cr > 10 ? 'red' : current.actual_cr > 5 ? 'orange' : 'blue'}
            />
            <MetricCard
              label="Thickness Loss"
              value={current.thickness_loss_pct?.toFixed(1)}
              unit="%"
              color={current.thickness_loss_pct > 30 ? 'red' : current.thickness_loss_pct > 15 ? 'orange' : 'green'}
            />
            <MetricCard label="CFI" value={Math.round(current.cfi)} cfi={current.cfi} />
          </div>

          {/* ═══════════ SPLIT SCREEN: REALITY vs AI ═══════════ */}
          <div className="grid grid-cols-2 gap-3 mb-3">
            {/* LEFT: Reality (Actual sensor data) */}
            <div className="bg-bg-surface rounded-lg border border-bg-border p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 rounded-full bg-text-secondary" />
                <div className="text-xs uppercase tracking-widest text-text-secondary font-semibold">
                  Reality — Actual Sensor Data
                </div>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <ComposedChart data={historyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="year" stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} />
                  <YAxis stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                  <RTooltip
                    contentStyle={{ background: CHART_COLORS.tooltip, border: `1px solid ${CHART_COLORS.grid}`, borderRadius: 8, fontSize: 11 }}
                    formatter={(v) => [v?.toFixed(3), '']}
                  />
                  {/* Faded full timeline for context */}
                  <Line data={fullWtData} dataKey="actual_wt" stroke={CHART_COLORS.actual} strokeOpacity={0.2} dot={false} strokeWidth={1} />
                  {/* Revealed actual data */}
                  <Line type="monotone" dataKey="actual_wt" stroke={CHART_COLORS.actual} dot={false} strokeWidth={2} name="Actual WT (mm)" />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="text-[10px] text-text-muted mt-1 text-center">
                What's actually happening to the casing wall
              </div>
            </div>

            {/* RIGHT: AI Predictions (with confidence band) */}
            <div className="bg-bg-surface rounded-lg border border-accent-cyan/20 p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 rounded-full bg-accent-cyan" />
                <div className="text-xs uppercase tracking-widest text-accent-cyan font-semibold">
                  AI Prediction — BiLSTM-Attention
                </div>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <ComposedChart data={historyData}>
                  <defs>
                    <linearGradient id="confBand" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={COLORS.accentCyan} stopOpacity={0.2} />
                      <stop offset="100%" stopColor={COLORS.accentCyan} stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="year" stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} />
                  <YAxis stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                  <RTooltip
                    contentStyle={{ background: CHART_COLORS.tooltip, border: `1px solid ${CHART_COLORS.grid}`, borderRadius: 8, fontSize: 11 }}
                    formatter={(v) => [v?.toFixed(3), '']}
                  />
                  {/* Confidence band (MC Dropout) */}
                  <Area type="monotone" dataKey="wt_ci_high" stroke="none" fill="url(#confBand)" name="95% CI Upper" />
                  <Area type="monotone" dataKey="wt_ci_low" stroke="none" fill="transparent" name="95% CI Lower" />
                  {/* Predicted line */}
                  <Line type="monotone" dataKey="predicted_wt" stroke={COLORS.accentCyan} dot={false} strokeWidth={2} name="Predicted WT (mm)" />
                  {/* Ghost: actual for comparison */}
                  <Line type="monotone" dataKey="actual_wt" stroke={CHART_COLORS.actual} strokeOpacity={0.3} dot={false} strokeWidth={1} strokeDasharray="4 4" name="Actual (ref)" />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center gap-4 text-[10px] text-text-muted mt-1">
                <span><span className="text-accent-cyan">━━</span> AI prediction</span>
                <span className="text-accent-cyan/30">░ 90% confidence</span>
                <span><span className="text-text-muted">╌╌</span> actual (reference)</span>
              </div>
            </div>
          </div>

          {/* ═══════════ AI SHOWCASE ROW ═══════════ */}
          <div className="grid grid-cols-2 gap-3 mb-3">
            {/* Attention Heatmap */}
            <AttentionHeatmap weights={current.attention} />

            {/* Accuracy Tracker */}
            <AccuracyTracker playbackData={playbackData} currentStep={currentStep} />
          </div>

          {/* ═══════════ INPUT WINDOW VISUALIZATION ═══════════ */}
          <div className="mb-3">
            <InputWindowViz
              windowFeatures={current.window_features}
              attention={current.attention}
            />
          </div>

          {/* ═══════════ BOTTOM: CFI + FORECAST ═══════════ */}
          <div className="grid grid-cols-2 gap-3">
            {/* CFI Risk Evolution */}
            <div className="bg-bg-surface rounded-lg border border-bg-border p-4">
              <div className="text-xs uppercase tracking-widest text-text-muted mb-3">
                CFI Risk Evolution
              </div>
              <ResponsiveContainer width="100%" height={140}>
                <AreaChart data={historyData}>
                  <defs>
                    <linearGradient id="cfiGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.cfi} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={CHART_COLORS.cfi} stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="year" stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} />
                  <YAxis stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} />
                  <RTooltip contentStyle={{ background: CHART_COLORS.tooltip, border: `1px solid ${CHART_COLORS.grid}`, borderRadius: 8, fontSize: 11 }} />
                  <ReferenceLine y={25} stroke={COLORS.cfiGreen} strokeDasharray="2 4" strokeOpacity={0.4} />
                  <ReferenceLine y={50} stroke={COLORS.cfiYellow} strokeDasharray="2 4" strokeOpacity={0.4} />
                  <ReferenceLine y={75} stroke={COLORS.cfiOrange} strokeDasharray="2 4" strokeOpacity={0.4} />
                  <Area type="monotone" dataKey="cfi" stroke={CHART_COLORS.cfi} fill="url(#cfiGrad)" strokeWidth={2} dot={false} name="CFI" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Forecast from current point */}
            <div className="bg-bg-surface rounded-lg border border-bg-border p-4">
              <div className="text-xs uppercase tracking-widest text-text-muted mb-3">
                AI Forecast from Current Day ({forecastData.length} months ahead)
              </div>
              <ResponsiveContainer width="100%" height={140}>
                <LineChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="month" stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }}
                    label={{ value: 'Months', position: 'bottom', fontSize: 10, fill: CHART_COLORS.axis }}
                  />
                  <YAxis stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                  <RTooltip contentStyle={{ background: CHART_COLORS.tooltip, border: `1px solid ${CHART_COLORS.grid}`, borderRadius: 8, fontSize: 11 }} />
                  <Line type="monotone" dataKey="wt" stroke={CHART_COLORS.forecast} strokeWidth={2} dot={false} strokeDasharray="5 3" name="Predicted WT" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!current && !loading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-text-muted">
            <div className="text-sm">Select a well to start AI-powered surveillance</div>
          </div>
        </div>
      )}
    </div>
  );
}
