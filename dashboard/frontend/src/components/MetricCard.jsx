import { getCfiColor, getGlowClass, getRiskColorName } from '../utils/colors';

const accentColors = {
  blue: 'bg-accent-blue',
  green: 'bg-accent-green',
  orange: 'bg-accent-orange',
  red: 'bg-accent-red',
  cyan: 'bg-accent-cyan',
};

export default function MetricCard({ label, value, unit, color = 'blue', cfi, trend }) {
  // If cfi is provided, derive color from it
  const effectiveColor = cfi !== undefined ? getRiskColorName(cfi) : color;
  const glowClass = getGlowClass(effectiveColor);

  return (
    <div className={`bg-bg-surface rounded-lg border border-bg-border flex overflow-hidden ${glowClass}`}>
      {/* Accent stripe */}
      <div className={`w-1 shrink-0 ${accentColors[effectiveColor] || 'bg-accent-blue'}`} />

      <div className="p-3 flex-1 min-w-0">
        <div className="text-xs uppercase tracking-widest text-text-muted mb-1">{label}</div>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-mono tabular-nums font-semibold text-text-primary">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </span>
          {unit && <span className="text-xs text-text-muted">{unit}</span>}
        </div>
        {trend !== undefined && (
          <div className={`text-xs mt-1 ${trend > 0 ? 'text-accent-red' : 'text-accent-green'}`}>
            {trend > 0 ? '▲' : '▼'} {Math.abs(trend).toFixed(1)}
          </div>
        )}
      </div>
    </div>
  );
}
