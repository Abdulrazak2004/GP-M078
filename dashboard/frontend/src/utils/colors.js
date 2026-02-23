// Palantir Blueprint color palette constants

export const COLORS = {
  bgPrimary: '#111418',
  bgSurface: '#1C2127',
  bgElevated: '#252A31',
  bgBorder: '#2F343C',
  bgMuted: '#383E47',

  textPrimary: '#F6F7F9',
  textSecondary: '#ABB3BF',
  textMuted: '#5F6B7C',

  accentBlue: '#4C90F0',
  accentGreen: '#32A467',
  accentOrange: '#EC9A3C',
  accentRed: '#E76A6E',
  accentCyan: '#68D4F8',

  cfiGreen: '#32A467',
  cfiYellow: '#FBD065',
  cfiOrange: '#EC9A3C',
  cfiRed: '#E76A6E',
};

export function getCfiColor(cfi) {
  if (cfi <= 25) return COLORS.cfiGreen;
  if (cfi <= 50) return COLORS.cfiYellow;
  if (cfi <= 75) return COLORS.cfiOrange;
  return COLORS.cfiRed;
}

export function getRiskColorName(cfi) {
  if (cfi <= 25) return 'green';
  if (cfi <= 50) return 'yellow';
  if (cfi <= 75) return 'orange';
  return 'red';
}

export function getGlowClass(riskColor) {
  const map = {
    green: 'glow-green',
    yellow: 'glow-yellow',
    orange: 'glow-orange',
    red: 'glow-red',
    cyan: 'glow-blue',
    blue: '',
  };
  return map[riskColor] || '';
}

// For Recharts
export const CHART_COLORS = {
  grid: '#2F343C',
  axis: '#5F6B7C',
  tooltip: '#1C2127',
  wt: '#4C90F0',
  cr: '#EC9A3C',
  rul: '#32A467',
  cfi: '#E76A6E',
  forecast: '#68D4F8',
  actual: '#ABB3BF',
  confidence: 'rgba(76, 144, 240, 0.15)',
};
