const API_BASE = 'http://localhost:8000';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchDatasetStats() {
  const res = await fetch(`${API_BASE}/dataset-stats`);
  return res.json();
}

export async function predictBatch(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predictFromParams(params) {
  const res = await fetch(`${API_BASE}/predict-params`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function optimizeBatch(params) {
  const res = await fetch(`${API_BASE}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function explainBatch(batchFeatures) {
  const res = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ batch_features: batchFeatures }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchGoldenSignature() {
  const res = await fetch(`${API_BASE}/golden-signature`);
  return res.json();
}

export async function checkAnomaly(params) {
  const res = await fetch(`${API_BASE}/anomaly-check`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function checkCarbon(energyKwh, hour = null) {
  const body = { predicted_energy_kwh: energyKwh };
  if (hour !== null) body.hour = hour;
  const res = await fetch(`${API_BASE}/carbon`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}
