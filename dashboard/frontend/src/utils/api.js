const BASE = '/api';

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function getWellsSummary(filters = {}) {
  const params = new URLSearchParams();
  if (filters.field) params.set('field', filters.field);
  if (filters.risk) params.set('risk', filters.risk);
  if (filters.reservoir_type) params.set('reservoir_type', filters.reservoir_type);
  const qs = params.toString();
  return fetchJSON(`${BASE}/wells/summary${qs ? '?' + qs : ''}`);
}

export async function getWellsStats() {
  return fetchJSON(`${BASE}/wells/stats`);
}

export async function getWellTimeseries(wellId) {
  return fetchJSON(`${BASE}/wells/${encodeURIComponent(wellId)}/timeseries`);
}

export async function getWellPredictions(wellId, day) {
  return fetchJSON(`${BASE}/wells/${encodeURIComponent(wellId)}/predictions?day=${day}`);
}

export async function getWellPlayback(wellId, stride = 30) {
  return fetchJSON(`${BASE}/wells/${encodeURIComponent(wellId)}/playback?stride=${stride}`);
}

export async function postDesignWell(params) {
  const res = await fetch(`${BASE}/design-well`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
