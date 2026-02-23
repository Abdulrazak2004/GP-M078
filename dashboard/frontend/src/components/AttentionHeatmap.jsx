import { COLORS } from '../utils/colors';

/**
 * Visualizes the BiLSTM-Attention weights as a colored strip.
 * Shows which of the 30 timesteps in the input window the model
 * is "paying attention to" for its prediction.
 */
export default function AttentionHeatmap({ weights }) {
  if (!weights || weights.length === 0) return null;

  // Find peak attention timestep
  const maxIdx = weights.indexOf(Math.max(...weights));

  return (
    <div className="bg-bg-surface rounded-lg border border-bg-border p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs uppercase tracking-widest text-text-muted">
          Model Attention
        </div>
        <div className="text-[10px] text-text-muted">
          Peak focus: Day {maxIdx + 1} of 30
        </div>
      </div>

      {/* Attention bar */}
      <div className="flex h-8 rounded overflow-hidden gap-px">
        {weights.map((w, i) => (
          <div
            key={i}
            className="flex-1 relative group"
            style={{
              backgroundColor: interpolateColor(w),
              opacity: 0.3 + w * 0.7,
            }}
          >
            {/* Tooltip on hover */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10">
              <div className="bg-bg-elevated text-text-primary text-[9px] px-1.5 py-0.5 rounded border border-bg-border whitespace-nowrap">
                t-{30 - i}: {(w * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Labels */}
      <div className="flex justify-between mt-1 text-[9px] text-text-muted">
        <span>t-30 (oldest)</span>
        <span className="text-accent-cyan">← Model reads this window →</span>
        <span>t-1 (newest)</span>
      </div>

      {/* Explanation */}
      <div className="mt-2 text-[10px] text-text-muted leading-relaxed">
        <span className="text-accent-cyan">Brighter</span> = model pays more attention.
        The BiLSTM learns which recent sensor readings matter most for predicting corrosion.
      </div>
    </div>
  );
}

function interpolateColor(value) {
  // 0 = dark blue, 1 = bright cyan
  const r = Math.round(20 + value * 84);
  const g = Math.round(30 + value * 182);
  const b = Math.round(60 + value * 188);
  return `rgb(${r}, ${g}, ${b})`;
}
