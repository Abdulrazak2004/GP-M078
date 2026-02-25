import { createContext, useContext, useState, useCallback } from 'react';
import { UNIT_SYSTEMS, convert, getUnit, formatValue } from '../utils/units';

const UnitContext = createContext(null);

export function UnitProvider({ children }) {
  const [system, setSystem] = useState(() => {
    try { return localStorage.getItem('unitSystem') || 'mixed'; } catch { return 'mixed'; }
  });

  const changeSystem = useCallback((s) => {
    setSystem(s);
    try { localStorage.setItem('unitSystem', s); } catch {}
  }, []);

  const units = UNIT_SYSTEMS[system];

  // Shorthand: convertMetric('wt', 11.0) → uses current system's wt unit
  const cv = useCallback((metric, value) => {
    return convert(metric, value, getUnit(system, metric));
  }, [system]);

  // Shorthand: formatMetric('wt', 11.0) → "11.00" (or "0.433" in field mode)
  const fmt = useCallback((metric, value, decimals) => {
    return formatValue(metric, value, getUnit(system, metric), decimals);
  }, [system]);

  // Get unit label: unitLabel('wt') → "mm" or "in"
  const unitLabel = useCallback((metric) => {
    return getUnit(system, metric);
  }, [system]);

  return (
    <UnitContext.Provider value={{ system, setSystem: changeSystem, units, cv, fmt, unitLabel }}>
      {children}
    </UnitContext.Provider>
  );
}

export function useUnits() {
  const ctx = useContext(UnitContext);
  if (!ctx) throw new Error('useUnits must be used within UnitProvider');
  return ctx;
}
