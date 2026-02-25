import { X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import CfiGauge from './CfiGauge';
import { getCfiColor } from '../utils/colors';
import { useUnits } from '../contexts/UnitContext';

export default function WellDetailPanel({ well, onClose }) {
  const navigate = useNavigate();
  const { fmt, unitLabel } = useUnits();
  if (!well) return null;

  return (
    <div className="absolute right-4 top-4 bottom-4 w-72 glass rounded-lg border border-bg-border animate-slide-in z-[1000] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-bg-border">
        <div>
          <div className="text-sm font-semibold text-text-primary">{well.well_id}</div>
          <div className="text-xs text-text-secondary">{well.field} ({well.sub_area})</div>
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 flex items-center justify-center rounded hover:bg-bg-elevated text-text-muted hover:text-text-primary transition-colors"
        >
          <X size={14} />
        </button>
      </div>

      {/* CFI Gauge */}
      <div className="p-4 flex justify-center">
        <CfiGauge value={well.cfi} size={140} />
      </div>

      {/* Metrics */}
      <div className="px-4 space-y-3">
        <MetricRow label="RUL" value={fmt('rul', well.rul)} unit={unitLabel('rul')} />
        <MetricRow label="Wall Thickness" value={fmt('wt', well.current_wt)} unit={unitLabel('wt')} />
        <MetricRow label="Corrosion Rate" value={fmt('cr', well.cr)} unit={unitLabel('cr')} />
        <MetricRow label="Reservoir" value={well.reservoir_type} />
        <MetricRow label="Casing" value={well.casing_grade} />
      </div>

      {/* Actions */}
      <div className="mt-auto p-4 space-y-2">
        <button
          onClick={() => navigate(`/monitor?well=${well.well_id}`)}
          className="w-full py-2 px-3 bg-accent-blue/15 text-accent-blue text-sm font-medium rounded-lg
                     hover:bg-accent-blue/25 transition-colors text-left"
        >
          Monitor Well →
        </button>
        <button
          onClick={() => navigate(`/designer`)}
          className="w-full py-2 px-3 bg-bg-elevated text-text-secondary text-sm font-medium rounded-lg
                     hover:bg-bg-muted hover:text-text-primary transition-colors text-left"
        >
          Design Similar →
        </button>
      </div>
    </div>
  );
}

function MetricRow({ label, value, unit }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-text-muted">{label}</span>
      <span className="text-sm font-mono tabular-nums text-text-primary">
        {value}
        {unit && <span className="text-xs text-text-muted ml-1">{unit}</span>}
      </span>
    </div>
  );
}
