import { COLORS } from '../utils/colors';

const ZONES = [
  { max: 25, color: COLORS.cfiGreen, label: 'Safe' },
  { max: 50, color: COLORS.cfiYellow, label: 'Watch' },
  { max: 75, color: COLORS.cfiOrange, label: 'Elevated' },
  { max: 100, color: COLORS.cfiRed, label: 'Critical' },
];

export default function RiskTimeline({ trajectory, maxYears = 30 }) {
  if (!trajectory || trajectory.length === 0) return null;

  // Find year boundaries where CFI crosses thresholds
  const segments = [];
  let currentZoneIdx = 0;

  for (const point of trajectory) {
    const newZoneIdx = point.cfi <= 25 ? 0 : point.cfi <= 50 ? 1 : point.cfi <= 75 ? 2 : 3;
    if (newZoneIdx !== currentZoneIdx || segments.length === 0) {
      segments.push({
        startYear: point.year,
        endYear: point.year,
        zone: ZONES[newZoneIdx],
        zoneIdx: newZoneIdx,
      });
      currentZoneIdx = newZoneIdx;
    } else {
      segments[segments.length - 1].endYear = point.year;
    }
  }

  // Extend last segment to maxYears
  if (segments.length > 0) {
    segments[segments.length - 1].endYear = maxYears;
  }

  return (
    <div>
      <div className="text-xs uppercase tracking-widest text-text-muted mb-2">Risk Timeline</div>
      <div className="flex h-6 rounded overflow-hidden border border-bg-border">
        {segments.map((seg, i) => {
          const widthPct = ((seg.endYear - seg.startYear) / maxYears) * 100;
          if (widthPct <= 0) return null;
          return (
            <div
              key={i}
              className="flex items-center justify-center text-[9px] font-semibold text-bg-primary"
              style={{
                width: `${widthPct}%`,
                backgroundColor: seg.zone.color,
                minWidth: widthPct > 5 ? undefined : '4px',
              }}
              title={`${seg.zone.label}: Year ${seg.startYear.toFixed(0)}-${seg.endYear.toFixed(0)}`}
            >
              {widthPct > 12 && `${seg.zone.label}`}
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-1 text-[10px] text-text-muted">
        <span>0 yr</span>
        <span>{maxYears} yr</span>
      </div>
    </div>
  );
}
