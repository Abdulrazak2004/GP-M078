import { useUnits } from '../contexts/UnitContext';
import { COLORS } from '../utils/colors';

/**
 * Displays the current day's raw sensor feature values extracted from the
 * model's 30-day input window (last timestep = "right now").
 *
 * These are the scaled values the model actually sees — shown as a compact
 * grid of labeled readings, styled like an instrument panel.
 */

// Maps the 11 key features to display config
// Index matches KEY_FEATURE_NAMES in inference.py
const FEATURES = [
  { key: 0,  label: 'Pressure',      shortUnit: 'psi',    unitMetric: 'pressure', icon: '⏲', decimals: 0 },
  { key: 1,  label: 'Temperature',    shortUnit: '°F',     unitMetric: 'temp',     icon: '🌡', decimals: 0 },
  { key: 2,  label: 'pH',            shortUnit: '',        unitMetric: null,        icon: '⚗',  decimals: 2 },
  { key: 3,  label: 'Water Cut',     shortUnit: '%',       unitMetric: null,        icon: '💧', decimals: 1 },
  { key: 4,  label: 'Flow Velocity', shortUnit: 'ft/s',    unitMetric: null,        icon: '↗',  decimals: 2 },
  { key: 5,  label: 'CO₂ Partial P', shortUnit: 'psi',     unitMetric: 'pressure', icon: '☁',  decimals: 1 },
  { key: 6,  label: 'Wall Thickness', shortUnit: 'mm',     unitMetric: 'wt',       icon: '◎',  decimals: 2 },
  { key: 7,  label: 'Inhibitor',     shortUnit: '',        unitMetric: null,        icon: '🛡', decimals: 0, isBoolean: true },
  { key: 8,  label: 'WT Roll Mean',  shortUnit: 'mm',      unitMetric: 'wt',       icon: '〰', decimals: 3 },
  { key: 9,  label: 'WT Slope',      shortUnit: 'mm/d',    unitMetric: null,        icon: '📉', decimals: 4 },
  { key: 10, label: 'Cumul. Damage', shortUnit: '',        unitMetric: null,        icon: '⚠',  decimals: 3 },
];

export default function FeatureReadings({ windowFeatures }) {
  const { fmt, unitLabel } = useUnits();

  if (!windowFeatures || windowFeatures.length === 0) return null;

  // Last timestep in the 30-day window = current day's values
  const currentValues = windowFeatures[windowFeatures.length - 1];
  if (!currentValues) return null;

  return (
    <div className="bg-bg-surface rounded-lg border border-bg-border p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-widest text-text-muted">
          Sensor Readings — Current Day
        </div>
        <div className="text-[10px] text-text-muted">
          {FEATURES.length} features (scaled values from model input)
        </div>
      </div>

      <div className="grid grid-cols-4 lg:grid-cols-6 gap-2">
        {FEATURES.map((feat) => {
          const raw = currentValues[feat.key];
          if (raw == null) return null;

          // Display value
          let displayVal;
          let displayUnit = feat.shortUnit;

          if (feat.isBoolean) {
            displayVal = raw > 0.5 ? 'Active' : 'Inactive';
            displayUnit = '';
          } else {
            displayVal = raw.toFixed(feat.decimals);
          }

          return (
            <div
              key={feat.key}
              className="bg-bg-elevated rounded-md px-2.5 py-2 border border-bg-border/50"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <span className="text-xs opacity-60">{feat.icon}</span>
                <span className="text-[10px] text-text-muted truncate">{feat.label}</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-sm font-mono tabular-nums text-text-primary font-medium">
                  {displayVal}
                </span>
                {displayUnit && (
                  <span className="text-[10px] text-text-muted">{displayUnit}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="text-[10px] text-text-muted mt-2">
        These are the actual sensor measurements at the current playback day — the raw inputs the BiLSTM model uses to make its predictions.
      </div>
    </div>
  );
}
