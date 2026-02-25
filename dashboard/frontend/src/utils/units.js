/**
 * Unit conversion system for petroleum engineering metrics.
 *
 * Three presets:
 *   - "field"  → API / Saudi Aramco convention (inches, mpy, psi, °F)
 *   - "si"     → ISO / NORSOK / metric  (mm, mm/yr, MPa, °C)
 *   - "mixed"  → Current dashboard default (mm, mpy, psi, °F)
 */

// ─── Unit definitions per metric ───────────────────────────────────────────────

export const UNIT_SYSTEMS = {
  mixed: {
    label: 'Mixed (Default)',
    desc: 'mm + mpy + psi + °F',
    wt: 'mm',
    cr: 'mpy',
    pressure: 'psi',
    temp: '°F',
    rul: 'days',
    od: 'in',
  },
  field: {
    label: 'API / Field',
    desc: 'in + mpy + psi + °F',
    wt: 'in',
    cr: 'mpy',
    pressure: 'psi',
    temp: '°F',
    rul: 'days',
    od: 'in',
  },
  si: {
    label: 'SI / Metric',
    desc: 'mm + mm/yr + MPa + °C',
    wt: 'mm',
    cr: 'mm/yr',
    pressure: 'MPa',
    temp: '°C',
    rul: 'days',
    od: 'mm',
  },
};

// ─── Conversion functions (from internal units) ────────────────────────────────
// Internal storage: wt=mm, cr=mpy, pressure=psi, temp=°F, od=in, rul=days

const converters = {
  wt: {
    mm:   v => v,
    in:   v => v * 0.03937,
    mils: v => v * 39.37,
  },
  cr: {
    mpy:    v => v,
    'mm/yr': v => v * 0.0254,
    'μm/yr': v => v * 25.4,
  },
  pressure: {
    psi: v => v,
    bar: v => v * 0.06895,
    MPa: v => v * 0.006895,
    kPa: v => v * 6.895,
  },
  temp: {
    '°F': v => v,
    '°C': v => (v - 32) * 5 / 9,
    'K':  v => (v - 32) * 5 / 9 + 273.15,
  },
  rul: {
    days:   v => v,
    years:  v => v / 365.25,
    months: v => v / 30.44,
  },
  od: {
    in: v => v,
    mm: v => v * 25.4,
  },
};

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Convert a value from internal units to the target unit.
 * @param {string} metric - one of: wt, cr, pressure, temp, rul, od
 * @param {number} value  - the value in internal units
 * @param {string} toUnit - the target unit string (e.g. "mm", "in", "mpy")
 * @returns {number}
 */
export function convert(metric, value, toUnit) {
  if (value == null || isNaN(value)) return value;
  const fn = converters[metric]?.[toUnit];
  return fn ? fn(value) : value;
}

/**
 * Get the display unit string for a metric given a unit system key.
 * @param {string} system - "mixed", "field", or "si"
 * @param {string} metric - one of: wt, cr, pressure, temp, rul, od
 * @returns {string}
 */
export function getUnit(system, metric) {
  return UNIT_SYSTEMS[system]?.[metric] || '';
}

/**
 * Format a converted value for display.
 * @param {string} metric  - wt, cr, pressure, temp, rul, od
 * @param {number} value   - raw value in internal units
 * @param {string} toUnit  - target unit
 * @param {number} [decimals] - decimal places (auto if not provided)
 * @returns {string}
 */
export function formatValue(metric, value, toUnit, decimals) {
  if (value == null || isNaN(value)) return '—';
  const converted = convert(metric, value, toUnit);
  if (decimals !== undefined) return converted.toFixed(decimals);

  // Auto-decimals based on metric
  switch (metric) {
    case 'wt':
      return toUnit === 'in' ? converted.toFixed(3) : converted.toFixed(2);
    case 'cr':
      return toUnit === 'mm/yr' ? converted.toFixed(4) : converted.toFixed(2);
    case 'pressure':
      return toUnit === 'psi' ? Math.round(converted).toLocaleString() : converted.toFixed(1);
    case 'temp':
      return Math.round(converted).toString();
    case 'rul':
      return toUnit === 'days' ? Math.round(converted).toLocaleString() : converted.toFixed(1);
    case 'od':
      return toUnit === 'mm' ? converted.toFixed(1) : converted.toFixed(3);
    default:
      return converted.toFixed(2);
  }
}
