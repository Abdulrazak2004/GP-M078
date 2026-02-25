import { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer,
  Tooltip as RTooltip, AreaChart, Area,
} from 'recharts';
import { postDesignWell } from '../utils/api';
import { CHART_COLORS, COLORS, getCfiColor } from '../utils/colors';
import { useUnits } from '../contexts/UnitContext';
import { convert } from '../utils/units';
import MetricCard from '../components/MetricCard';
import RiskTimeline from '../components/RiskTimeline';

function SetPageTitle() {
  useEffect(() => {
    const el = document.getElementById('page-title');
    if (el) el.textContent = 'Well Designer';
  }, []);
  return null;
}

const DEFAULTS = {
  reservoir_type: 'Carbonate',
  casing_grade: 'L80',
  casing_od_in: 9.625,
  initial_thickness_mm: 11.0,
  avg_pressure_psi: 3200,
  avg_temp_f: 180,
  avg_ph: 5.2,
  avg_water_cut_pct: 15,
  corrosion_cause: 'CO2',
};

export default function WellDesigner() {
  const [params, setParams] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [prevResult, setPrevResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { fmt, unitLabel } = useUnits();

  const update = (key, value) => setParams(p => ({ ...p, [key]: value }));

  // Convert trajectory WT values to current unit system for chart display
  const convertedTrajectory = useMemo(() => {
    if (!result?.trajectory) return [];
    const wtUnit = unitLabel('wt');
    return result.trajectory.map(p => ({
      ...p,
      actual_wt: convert('wt', p.actual_wt, wtUnit),
      wt: p.wt != null ? convert('wt', p.wt, wtUnit) : p.wt,
    }));
  }, [result, unitLabel]);

  const convertedPrevTrajectory = useMemo(() => {
    if (!prevResult?.trajectory) return [];
    const wtUnit = unitLabel('wt');
    return prevResult.trajectory.map(p => ({
      ...p,
      actual_wt: convert('wt', p.actual_wt, wtUnit),
    }));
  }, [prevResult, unitLabel]);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    if (result) setPrevResult(result);
    try {
      const data = await postDesignWell(params);
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="h-full flex p-4 gap-4 overflow-hidden">
      <SetPageTitle />

      {/* Left: Parameter Form */}
      <div className="w-80 shrink-0 bg-bg-surface rounded-lg border border-bg-border p-4 overflow-y-auto">
        <div className="text-xs uppercase tracking-widest text-text-muted mb-4">Well Configuration</div>

        {/* Reservoir Type */}
        <FormSection label="Reservoir Type">
          <RadioGroup
            options={['Carbonate', 'Clastic', 'Mixed']}
            value={params.reservoir_type}
            onChange={v => update('reservoir_type', v)}
          />
        </FormSection>

        {/* Casing Grade */}
        <FormSection label="Casing Grade">
          <select
            value={params.casing_grade}
            onChange={e => update('casing_grade', e.target.value)}
            className="w-full bg-bg-elevated text-text-primary text-sm px-3 py-1.5 rounded border border-bg-border focus:outline-none focus:border-accent-blue"
          >
            <option value="N80">N80</option>
            <option value="L80">L80</option>
            <option value="P110">P110</option>
          </select>
        </FormSection>

        {/* Sliders — always send internal units (mm, psi, °F) to backend */}
        <SliderField
          label="Initial Wall Thickness"
          value={params.initial_thickness_mm}
          onChange={v => update('initial_thickness_mm', v)}
          min={7} max={16} step={0.5} unit="mm"
        />
        <SliderField
          label="Avg Pressure"
          value={params.avg_pressure_psi}
          onChange={v => update('avg_pressure_psi', v)}
          min={1500} max={6000} step={100} unit="psi"
        />
        <SliderField
          label="Avg Temperature"
          value={params.avg_temp_f}
          onChange={v => update('avg_temp_f', v)}
          min={100} max={350} step={5} unit="°F"
        />
        <SliderField
          label="pH"
          value={params.avg_ph}
          onChange={v => update('avg_ph', v)}
          min={3.0} max={8.0} step={0.1} unit=""
        />
        <SliderField
          label="Water Cut"
          value={params.avg_water_cut_pct}
          onChange={v => update('avg_water_cut_pct', v)}
          min={0} max={95} step={1} unit="%"
        />

        {/* Corrosion Cause */}
        <FormSection label="Corrosion Cause">
          <RadioGroup
            options={['CO2', 'H2S', 'MIC', 'Erosion']}
            value={params.corrosion_cause}
            onChange={v => update('corrosion_cause', v)}
          />
        </FormSection>

        {/* Predict Button */}
        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full mt-4 py-2.5 bg-accent-blue text-white font-semibold rounded-lg
                     hover:bg-accent-blue/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                     flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Predicting...
            </>
          ) : (
            'Predict Trajectory'
          )}
        </button>

        {error && (
          <div className="mt-3 text-xs text-accent-red bg-accent-red/10 rounded p-2">{error}</div>
        )}
      </div>

      {/* Right: Results */}
      <div className="flex-1 overflow-y-auto space-y-4">
        {!result && !loading && (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-text-muted max-w-xs">
              <div className="text-4xl mb-4 opacity-20">⚙</div>
              <div className="text-sm">
                Configure your well parameters and click <strong className="text-text-primary">Predict</strong> to see the 30-year trajectory.
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="space-y-4 animate-fade-in">
            {/* Summary KPIs */}
            <div className="grid grid-cols-3 gap-3">
              <MetricCard
                label="Estimated RUL"
                value={fmt('rul', result.rul_years * 365.25)}
                unit={unitLabel('rul')}
                color={result.rul_years < 5 ? 'red' : result.rul_years < 15 ? 'orange' : 'green'}
              />
              <MetricCard
                label="Final CFI"
                value={Math.round(result.final_cfi)}
                cfi={result.final_cfi}
              />
              <div className="bg-bg-surface rounded-lg border border-bg-border p-3">
                <div className="text-xs uppercase tracking-widest text-text-muted mb-1">Material</div>
                <div className="text-sm text-text-primary">{result.recommendation}</div>
              </div>
            </div>

            {/* Risk Timeline */}
            <div className="bg-bg-surface rounded-lg border border-bg-border p-4">
              <RiskTimeline trajectory={result.trajectory} maxYears={30} />
            </div>

            {/* WT Trajectory Chart */}
            <div className="bg-bg-surface rounded-lg border border-bg-border p-4">
              <div className="text-xs uppercase tracking-widest text-text-muted mb-3">
                Wall Thickness Trajectory
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={convertedTrajectory}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis
                    dataKey="year"
                    stroke={CHART_COLORS.axis}
                    tick={{ fontSize: 10 }}
                    label={{ value: 'Years', position: 'bottom', fontSize: 10, fill: CHART_COLORS.axis }}
                  />
                  <YAxis
                    stroke={CHART_COLORS.axis}
                    tick={{ fontSize: 10 }}
                    domain={['auto', 'auto']}
                    label={{ value: unitLabel('wt'), angle: -90, position: 'insideLeft', fontSize: 10, fill: CHART_COLORS.axis }}
                  />
                  <RTooltip
                    contentStyle={{
                      background: CHART_COLORS.tooltip,
                      border: `1px solid ${CHART_COLORS.grid}`,
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                  />
                  {/* Ghost line from previous prediction */}
                  {prevResult && (
                    <Line
                      data={convertedPrevTrajectory}
                      type="monotone"
                      dataKey="actual_wt"
                      stroke={CHART_COLORS.actual}
                      strokeOpacity={0.3}
                      dot={false}
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      name="Previous"
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="actual_wt"
                    stroke={CHART_COLORS.wt}
                    dot={false}
                    strokeWidth={2}
                    name={`Simulated WT (${unitLabel('wt')})`}
                  />
                  <Line
                    type="monotone"
                    dataKey="wt"
                    stroke={CHART_COLORS.forecast}
                    dot={false}
                    strokeWidth={1.5}
                    strokeDasharray="5 3"
                    name={`Predicted WT (${unitLabel('wt')})`}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* CFI Over Time */}
            <div className="bg-bg-surface rounded-lg border border-bg-border p-4">
              <div className="text-xs uppercase tracking-widest text-text-muted mb-3">
                CFI Over Time
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={result.trajectory}>
                  <defs>
                    <linearGradient id="cfiDesignGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.cfi} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={CHART_COLORS.cfi} stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="year" stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} />
                  <YAxis
                    stroke={CHART_COLORS.axis}
                    tick={{ fontSize: 10 }}
                    domain={[0, 100]}
                    ticks={[0, 25, 50, 75, 100]}
                  />
                  <RTooltip
                    contentStyle={{
                      background: CHART_COLORS.tooltip,
                      border: `1px solid ${CHART_COLORS.grid}`,
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="cfi"
                    stroke={CHART_COLORS.cfi}
                    fill="url(#cfiDesignGradient)"
                    strokeWidth={2}
                    dot={false}
                    name="CFI"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="text-xs text-text-muted text-center pb-4">
              Try different parameters to compare well designs
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Form Components ──

function FormSection({ label, children }) {
  return (
    <div className="mb-4">
      <div className="text-xs text-text-secondary mb-1.5">{label}</div>
      {children}
    </div>
  );
}

function RadioGroup({ options, value, onChange }) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map(opt => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`px-3 py-1 text-xs rounded-full border transition-colors ${
            value === opt
              ? 'border-accent-blue bg-accent-blue/15 text-accent-blue'
              : 'border-bg-border bg-bg-elevated text-text-muted hover:text-text-secondary'
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

function SliderField({ label, value, onChange, min, max, step, unit }) {
  return (
    <FormSection label={`${label}: ${value}${unit ? ` ${unit}` : ''}`}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1 bg-bg-border rounded-full appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                   [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-accent-blue [&::-webkit-slider-thumb]:cursor-pointer"
      />
      <div className="flex justify-between mt-0.5 text-[10px] text-text-muted">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </FormSection>
  );
}
