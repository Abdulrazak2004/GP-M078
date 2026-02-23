import { useMemo } from 'react';
import { COLORS } from '../utils/colors';

/**
 * Running accuracy scorecard using MAPE (Mean Absolute Percentage Error).
 * Shows prediction vs actual comparisons and accumulates accuracy
 * over the playback timeline.
 */
export default function AccuracyTracker({ playbackData, currentStep }) {
  const stats = useMemo(() => {
    if (!playbackData || currentStep < 1) return null;

    const slice = playbackData.slice(0, currentStep + 1);
    let wtMapes = [], crMapes = [], rulMapes = [];
    let wtErrors = [], crErrors = [], rulErrors = [];

    for (const p of slice) {
      wtErrors.push(p.wt_error);
      crErrors.push(p.cr_error);

      // MAPE: |pred - actual| / actual * 100 (skip if actual ≈ 0)
      if (p.actual_wt > 0.1) {
        wtMapes.push(Math.abs(p.wt - p.actual_wt) / p.actual_wt * 100);
      }
      if (p.actual_cr > 0.1) {
        crMapes.push(Math.abs(p.cr - p.actual_cr) / p.actual_cr * 100);
      }
      if (p.rul_error !== null && p.rul_error !== undefined) {
        rulErrors.push(p.rul_error);
        if (p.actual_rul > 10) {
          rulMapes.push(Math.abs(p.rul - p.actual_rul) / p.actual_rul * 100);
        }
      }
    }

    const wtMAE = mean(wtErrors);
    const crMAE = mean(crErrors);
    const rulMAE = rulErrors.length > 0 ? mean(rulErrors) : null;

    // MAPE → Accuracy = 100 - MAPE (clamped to [0, 100])
    const wtMAPE = wtMapes.length > 0 ? mean(wtMapes) : 0;
    const wtAccuracy = Math.max(0, Math.min(100, 100 - wtMAPE));

    const crMAPE = crMapes.length > 0 ? mean(crMapes) : null;
    const rulMAPE = rulMapes.length > 0 ? mean(rulMapes) : null;

    const current = slice[slice.length - 1];

    return {
      wtMAE, crMAE, rulMAE,
      wtAccuracy,
      wtMAPE,
      crMAPE,
      rulMAPE,
      samples: slice.length,
      currentWtError: current.wt_error,
      currentCrError: current.cr_error,
      currentRulError: current.rul_error,
      predWt: current.wt,
      actualWt: current.actual_wt,
      predCr: current.cr,
      actualCr: current.actual_cr,
      predRul: current.rul,
      actualRul: current.actual_rul,
    };
  }, [playbackData, currentStep]);

  if (!stats) return null;

  const accuracy = stats.wtAccuracy;
  const accColor = accuracy >= 95 ? COLORS.accentGreen
    : accuracy >= 85 ? COLORS.cfiYellow
    : COLORS.accentRed;

  return (
    <div className="bg-bg-surface rounded-lg border border-bg-border p-3">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-widest text-text-muted">
          AI Accuracy Tracker
        </div>
        <div className="text-[10px] text-text-muted">
          {stats.samples} predictions scored
        </div>
      </div>

      {/* Big accuracy number */}
      <div className="flex items-center gap-4 mb-3">
        <div className="text-center min-w-[80px]">
          <div className="text-3xl font-mono tabular-nums font-bold" style={{ color: accColor }}>
            {accuracy.toFixed(1)}%
          </div>
          <div className="text-[9px] uppercase tracking-widest text-text-muted">
            WT Accuracy
          </div>
        </div>

        {/* Current prediction comparison */}
        <div className="flex-1 space-y-1.5">
          <CompareRow
            label="Wall Thickness"
            predicted={stats.predWt}
            actual={stats.actualWt}
            error={stats.currentWtError}
            unit="mm"
            decimals={3}
          />
          <CompareRow
            label="Corr. Rate"
            predicted={stats.predCr}
            actual={stats.actualCr}
            error={stats.currentCrError}
            unit="mpy"
            decimals={2}
          />
          <CompareRow
            label="RUL"
            predicted={stats.predRul}
            actual={stats.actualRul}
            error={stats.currentRulError}
            unit="days"
            decimals={0}
          />
        </div>
      </div>

      {/* Running metrics */}
      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-bg-border">
        <MiniStat label="WT MAE" value={stats.wtMAE?.toFixed(3)} unit="mm" />
        <MiniStat label="WT MAPE" value={stats.wtMAPE?.toFixed(2) + '%'} unit="" good />
        <MiniStat
          label="RUL MAE"
          value={stats.rulMAE !== null ? stats.rulMAE?.toFixed(1) : '—'}
          unit="days"
        />
      </div>
    </div>
  );
}

function CompareRow({ label, predicted, actual, error, unit, decimals = 2 }) {
  const predStr = predicted != null ? predicted.toFixed(decimals) : '—';
  const actStr = actual != null ? actual.toFixed(decimals) : '—';
  const errStr = error != null ? error.toFixed(decimals > 0 ? decimals : 1) : null;

  return (
    <div className="flex items-center gap-1.5 text-[10px]">
      <span className="text-text-muted w-20 truncate">{label}</span>
      <span className="text-accent-cyan font-mono tabular-nums">{predStr}</span>
      <span className="text-text-muted">vs</span>
      <span className="text-text-primary font-mono tabular-nums">{actStr}</span>
      <span className="text-text-muted text-[9px]">{unit}</span>
      {errStr && (
        <span className="ml-auto text-accent-orange font-mono tabular-nums text-[9px]">
          ±{errStr}
        </span>
      )}
    </div>
  );
}

function MiniStat({ label, value, unit, good }) {
  return (
    <div className="text-center">
      <div className={`text-sm font-mono tabular-nums ${good ? 'text-accent-green' : 'text-text-primary'}`}>
        {value}
      </div>
      <div className="text-[8px] uppercase tracking-widest text-text-muted">
        {label}{unit ? ` (${unit})` : ''}
      </div>
    </div>
  );
}

function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
