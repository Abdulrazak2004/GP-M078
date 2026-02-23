import { COLORS } from '../utils/colors';

/**
 * "What the Model Sees" — visualizes the 30-timestep × 11-feature
 * input window as a heatmap grid. The audience literally sees the
 * data matrix feeding into the neural network.
 */

const FEATURE_LABELS = [
  "Pressure", "Temp", "pH", "Water Cut",
  "Flow Vel", "CO₂ psi",
  "Thickness", "Inhibitor",
  "WT RollMean", "WT Slope",
  "Damage",
];

export default function InputWindowViz({ windowFeatures, attention }) {
  if (!windowFeatures || windowFeatures.length === 0) return null;

  // windowFeatures is 30 x 11 (timesteps x features)
  const nTimesteps = windowFeatures.length;
  const nFeatures = windowFeatures[0]?.length || 0;

  // Compute min/max per feature for normalization
  const featureMins = [];
  const featureMaxs = [];
  for (let f = 0; f < nFeatures; f++) {
    let mn = Infinity, mx = -Infinity;
    for (let t = 0; t < nTimesteps; t++) {
      const v = windowFeatures[t][f];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    featureMins.push(mn);
    featureMaxs.push(mx === mn ? mn + 1 : mx);
  }

  return (
    <div className="bg-bg-surface rounded-lg border border-bg-border p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs uppercase tracking-widest text-text-muted">
          What The Model Sees
        </div>
        <div className="text-[10px] text-text-muted">
          30-day window × {nFeatures} features
        </div>
      </div>

      {/* Heatmap grid */}
      <div className="overflow-hidden rounded border border-bg-border">
        {/* Header row: timestep numbers */}
        <div className="flex">
          <div className="w-16 shrink-0" />
          <div className="flex-1 flex">
            {Array.from({ length: nTimesteps }, (_, i) => (
              <div
                key={i}
                className="flex-1 text-center text-[7px] text-text-muted py-0.5"
                style={{
                  backgroundColor: attention
                    ? `rgba(104, 212, 248, ${(attention[i] || 0) * 0.3})`
                    : 'transparent',
                }}
              >
                {i % 5 === 0 ? `${i + 1}` : ''}
              </div>
            ))}
          </div>
        </div>

        {/* Feature rows */}
        {FEATURE_LABELS.slice(0, nFeatures).map((label, f) => (
          <div key={f} className="flex">
            {/* Feature label */}
            <div className="w-16 shrink-0 text-[8px] text-text-muted px-1 py-0.5 flex items-center truncate border-r border-bg-border">
              {label}
            </div>
            {/* Cells */}
            <div className="flex-1 flex">
              {windowFeatures.map((timestep, t) => {
                const raw = timestep[f];
                const norm = (raw - featureMins[f]) / (featureMaxs[f] - featureMins[f]);
                return (
                  <div
                    key={t}
                    className="flex-1 group relative"
                    style={{
                      backgroundColor: heatColor(norm),
                      minHeight: '10px',
                    }}
                    title={`${label} at t-${30 - t}: ${raw.toFixed(3)}`}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between mt-1.5">
        <span className="text-[9px] text-text-muted">Low</span>
        <div className="flex h-2 flex-1 mx-2 rounded overflow-hidden">
          {Array.from({ length: 20 }, (_, i) => (
            <div key={i} className="flex-1" style={{ backgroundColor: heatColor(i / 19) }} />
          ))}
        </div>
        <span className="text-[9px] text-text-muted">High</span>
      </div>

      <div className="text-[10px] text-text-muted mt-1">
        Each row is a sensor feature. Each column is one day.
        This is the exact <span className="text-accent-cyan">26-feature × 30-day matrix</span> fed into the BiLSTM.
      </div>
    </div>
  );
}

function heatColor(value) {
  // Cool-warm: dark blue → teal → yellow → orange
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.33) {
    const t = v / 0.33;
    return `rgb(${Math.round(17 + t * 15)}, ${Math.round(20 + t * 80)}, ${Math.round(60 + t * 60)})`;
  } else if (v < 0.66) {
    const t = (v - 0.33) / 0.33;
    return `rgb(${Math.round(32 + t * 180)}, ${Math.round(100 + t * 64)}, ${Math.round(120 - t * 60)})`;
  } else {
    const t = (v - 0.66) / 0.34;
    return `rgb(${Math.round(212 + t * 44)}, ${Math.round(164 - t * 60)}, ${Math.round(60 - t * 20)})`;
  }
}
