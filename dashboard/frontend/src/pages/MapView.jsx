import { useEffect, useState, useMemo, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import { getWellsSummary, getWellsStats } from '../utils/api';
import { getCfiColor, COLORS } from '../utils/colors';
import WellDetailPanel from '../components/WellDetailPanel';

const DARK_TILES = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const SAUDI_CENTER = [25.5, 49.5];

function SetPageTitle() {
  useEffect(() => {
    const el = document.getElementById('page-title');
    if (el) el.textContent = 'Operations Overview';
  }, []);
  return null;
}

function FitBounds({ wells }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (fitted.current || wells.length === 0) return;
    const lats = wells.map(w => w.lat);
    const lons = wells.map(w => w.lon);
    const bounds = [
      [Math.min(...lats) - 0.5, Math.min(...lons) - 0.5],
      [Math.max(...lats) + 0.5, Math.max(...lons) + 0.5],
    ];
    map.fitBounds(bounds, { padding: [40, 40] });
    fitted.current = true;
  }, [wells, map]);
  return null;
}

export default function MapView() {
  const [wells, setWells] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedWell, setSelectedWell] = useState(null);
  const [filters, setFilters] = useState({ field: '', risk: '', reservoir_type: '' });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getWellsSummary(), getWellsStats()])
      .then(([wellsData, statsData]) => {
        setWells(wellsData.wells);
        setStats(statsData);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load wells:', err);
        setLoading(false);
      });
  }, []);

  const filteredWells = useMemo(() => {
    return wells.filter(w => {
      if (filters.field && w.field !== filters.field) return false;
      if (filters.risk && w.risk_color !== filters.risk) return false;
      if (filters.reservoir_type && w.reservoir_type !== filters.reservoir_type) return false;
      return true;
    });
  }, [wells, filters]);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center animate-fade-in">
          <div className="w-10 h-10 border-2 border-accent-blue border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <div className="text-sm text-text-muted tracking-widest uppercase">Loading Wells...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full relative">
      <SetPageTitle />

      <MapContainer
        center={SAUDI_CENTER}
        zoom={7}
        className="h-full w-full"
        zoomControl={true}
        attributionControl={true}
      >
        <TileLayer url={DARK_TILES} attribution="CartoDB" />
        <FitBounds wells={filteredWells} />

        {filteredWells.map(well => (
          <WellMarker
            key={well.well_id}
            well={well}
            onClick={() => setSelectedWell(well)}
          />
        ))}
      </MapContainer>

      {/* Aggregate Stats - bottom left */}
      {stats && (
        <div className="absolute bottom-16 left-4 glass rounded-lg border border-bg-border p-3 z-[1000] animate-fade-in">
          <div className="flex items-center gap-4 text-sm">
            <div>
              <span className="font-mono tabular-nums font-semibold text-text-primary">{stats.total_wells}</span>
              <span className="text-text-muted ml-1">Wells</span>
            </div>
            <div className="w-px h-4 bg-bg-border" />
            <div>
              <span className="font-mono tabular-nums font-semibold text-text-primary">{stats.total_fields}</span>
              <span className="text-text-muted ml-1">Fields</span>
            </div>
            <div className="w-px h-4 bg-bg-border" />
            <div className="flex items-center gap-2">
              <RiskDot color={COLORS.cfiGreen} count={stats.by_risk.green} />
              <RiskDot color={COLORS.cfiYellow} count={stats.by_risk.yellow} />
              <RiskDot color={COLORS.cfiOrange} count={stats.by_risk.orange} />
              <RiskDot color={COLORS.cfiRed} count={stats.by_risk.red} />
            </div>
          </div>
        </div>
      )}

      {/* Filter Bar - bottom */}
      <div className="absolute bottom-0 left-0 right-0 bg-bg-surface/90 backdrop-blur-sm border-t border-bg-border px-4 py-2 z-[1000] flex items-center gap-3">
        <span className="text-xs text-text-muted uppercase tracking-widest">Filter</span>
        <FilterSelect
          value={filters.field}
          onChange={v => setFilters(f => ({ ...f, field: v }))}
          options={stats?.fields || []}
          placeholder="All Fields"
        />
        <FilterSelect
          value={filters.risk}
          onChange={v => setFilters(f => ({ ...f, risk: v }))}
          options={['green', 'yellow', 'orange', 'red']}
          labels={['Safe', 'Watch', 'Elevated', 'Critical']}
          placeholder="All Risk"
        />
        <FilterSelect
          value={filters.reservoir_type}
          onChange={v => setFilters(f => ({ ...f, reservoir_type: v }))}
          options={['Carbonate', 'Clastic', 'Mixed']}
          placeholder="All Types"
        />
        <div className="ml-auto text-xs text-text-muted">
          Showing {filteredWells.length} of {wells.length} wells
        </div>
      </div>

      {/* Well Detail Panel */}
      {selectedWell && (
        <WellDetailPanel well={selectedWell} onClose={() => setSelectedWell(null)} />
      )}
    </div>
  );
}

function WellMarker({ well, onClick }) {
  const color = getCfiColor(well.cfi);
  const isCritical = well.risk_color === 'red';
  const radius = isCritical ? 7 : 5;

  return (
    <CircleMarker
      center={[well.lat, well.lon]}
      radius={radius}
      pathOptions={{
        color: isCritical ? color : 'transparent',
        fillColor: color,
        fillOpacity: isCritical ? 0.9 : 0.75,
        weight: isCritical ? 2 : 0,
      }}
      eventHandlers={{ click: onClick }}
    >
      <Tooltip>
        <div className="text-xs">
          <div className="font-semibold">{well.well_id}</div>
          <div>{well.field} — {well.sub_area}</div>
          <div>CFI: {well.cfi} ({well.cfi_label})</div>
        </div>
      </Tooltip>
    </CircleMarker>
  );
}

function RiskDot({ color, count }) {
  return (
    <div className="flex items-center gap-1">
      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-xs font-mono tabular-nums text-text-secondary">{count}</span>
    </div>
  );
}

function FilterSelect({ value, onChange, options, labels, placeholder }) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="bg-bg-elevated text-text-primary text-xs px-2 py-1 rounded border border-bg-border
                 focus:outline-none focus:border-accent-blue appearance-none cursor-pointer"
    >
      <option value="">{placeholder}</option>
      {options.map((opt, i) => (
        <option key={opt} value={opt}>{labels ? labels[i] : opt}</option>
      ))}
    </select>
  );
}
