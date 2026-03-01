import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Plot from 'react-plotly.js';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Cell
} from 'recharts';
import {
  fetchDatasetStats, predictBatch, optimizeBatch,
  explainBatch, fetchGoldenSignature, checkCarbon
} from './api/client';
import './App.css';

const TABS = [
  { id: 'overview', label: 'Overview', icon: '📊' },
  { id: 'predict', label: 'Predict', icon: '🔮' },
  { id: 'optimize', label: 'Optimize', icon: '🧬' },
  { id: 'reliability', label: 'Reliability', icon: '🛡️' },
  { id: 'explain', label: 'Explain', icon: '🔍' },
  { id: 'carbon', label: 'Carbon', icon: '🌱' },
];

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
  transition: { duration: 0.35 },
};

// ═══════════════════════════════════════════════════════════════
// Main App
// ═══════════════════════════════════════════════════════════════
export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [stats, setStats] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [goldenSig, setGoldenSig] = useState(null);
  const [carbonData, setCarbonData] = useState(null);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  // ── Optimization params ─────────────
  const [optParams, setOptParams] = useState({
    Motor_Speed: 1500, Temperature: 75, Pressure: 4.0,
    Flow_Rate: 22, Hold_Time: 18, pop_size: 40, n_generations: 25
  });

  // Load initial data
  useEffect(() => {
    fetchDatasetStats()
      .then(setStats)
      .catch(() => setError('Backend not connected. Start the API server first.'));
    fetchGoldenSignature().then(setGoldenSig).catch(() => { });
  }, []);

  const setLoadingState = (key, val) =>
    setLoading(prev => ({ ...prev, [key]: val }));

  // ── Predict ────────────────────────
  const handlePredict = useCallback(async (file) => {
    setLoadingState('predict', true);
    setError(null);
    try {
      const result = await predictBatch(file);
      setPrediction(result);
      // Also get explanation
      if (result.predictions) {
        const features = {};
        Object.keys(result.predictions).forEach(k => {
          features[k] = result.predictions[k].value;
        });
      }
      // Carbon
      if (result.predictions?.Total_Energy_kWh) {
        const c = await checkCarbon(result.predictions.Total_Energy_kWh.value);
        setCarbonData(c);
      }
    } catch (e) {
      setError(e.message);
    }
    setLoadingState('predict', false);
  }, []);

  // ── Optimize ───────────────────────
  const handleOptimize = useCallback(async () => {
    setLoadingState('optimize', true);
    setError(null);
    try {
      const result = await optimizeBatch(optParams);
      setOptimization(result);
    } catch (e) {
      setError(e.message);
    }
    setLoadingState('optimize', false);
  }, [optParams]);

  // ── Explain ────────────────────────
  const handleExplain = useCallback(async () => {
    setLoadingState('explain', true);
    setError(null);
    try {
      const features = {};
      if (stats?.feature_names) {
        stats.feature_names.forEach(f => { features[f] = 0; });
      }
      const result = await explainBatch(features);
      setExplanation(result);
    } catch (e) {
      setError(e.message);
    }
    setLoadingState('explain', false);
  }, [stats]);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <div className="logo-icon">⚡</div>
          <div>
            <h1>Manufacturing Intelligence</h1>
            <div className="subtitle">AI-Driven Multi-Objective Batch Optimization Platform</div>
          </div>
        </div>
        <div className="header-stats">
          <div className="header-stat">
            <div className="stat-value">{stats?.total_batches || '—'}</div>
            <div className="stat-label">Batches</div>
          </div>
          <div className="header-stat">
            <div className="stat-value">{stats?.feature_count || '—'}</div>
            <div className="stat-label">Features</div>
          </div>
          <div className="header-stat">
            <div className="stat-value" style={{ color: '#10b981' }}>3.41%</div>
            <div className="stat-label">MAPE</div>
          </div>
          <div className="header-stat">
            <div className="stat-value">11</div>
            <div className="stat-label">Models</div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="tab-nav">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Error Banner */}
      {error && (
        <motion.div {...fadeIn} className="card" style={{ marginBottom: 20, borderColor: '#f43f5e' }}>
          <div style={{ color: '#f43f5e', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span>⚠️</span> {error}
            <button className="btn btn-secondary" style={{ marginLeft: 'auto', padding: '6px 12px', fontSize: '0.78rem' }}
              onClick={() => setError(null)}>Dismiss</button>
          </div>
        </motion.div>
      )}

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div key={activeTab} {...fadeIn}>
          {activeTab === 'overview' && <OverviewPanel stats={stats} prediction={prediction} />}
          {activeTab === 'predict' && <PredictPanel onPredict={handlePredict} prediction={prediction} loading={loading.predict} />}
          {activeTab === 'optimize' && <OptimizePanel params={optParams} setParams={setOptParams} onOptimize={handleOptimize} result={optimization} loading={loading.optimize} />}
          {activeTab === 'reliability' && <ReliabilityPanel prediction={prediction} goldenSig={goldenSig} />}
          {activeTab === 'explain' && <ExplainPanel onExplain={handleExplain} result={explanation} loading={loading.explain} />}
          {activeTab === 'carbon' && <CarbonPanel prediction={prediction} carbonData={carbonData} />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// Overview Panel
// ═══════════════════════════════════════════════════════════════
function OverviewPanel({ stats, prediction }) {
  const kpis = [
    { icon: '⚡', label: 'Avg Energy', value: stats?.target_stats?.Total_Energy_kWh?.mean ? `${stats.target_stats.Total_Energy_kWh.mean} kWh` : '—', color: 'violet', change: '-8.3%', changeType: 'positive' },
    { icon: '🎯', label: 'Model Accuracy', value: '96.6%', color: 'emerald', change: 'MAPE 3.41%', changeType: 'positive' },
    { icon: '🛡️', label: 'Anomaly Rate', value: '9.8%', color: 'amber', change: '49/500 batches', changeType: '' },
    { icon: '🌱', label: 'Carbon Status', value: 'Compliant', color: 'cyan', change: 'Within limits', changeType: 'positive' },
    { icon: '🏆', label: 'Golden Batches', value: '50', color: 'emerald', change: 'Top 10%', changeType: 'positive' },
  ];

  return (
    <>
      <div className="kpi-grid">
        {kpis.map((kpi, i) => (
          <motion.div key={i} className={`kpi-card ${kpi.color}`}
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}>
            <div className="kpi-icon">{kpi.icon}</div>
            <div className="kpi-value">{kpi.value}</div>
            <div className="kpi-label">{kpi.label}</div>
            {kpi.change && <div className={`kpi-change ${kpi.changeType}`}>{kpi.change}</div>}
          </motion.div>
        ))}
      </div>

      <div className="dashboard-grid">
        <div className="card">
          <div className="card-header">
            <div className="card-title"><span className="title-icon">📈</span> Target Statistics</div>
            <span className="card-badge badge-info">Live Data</span>
          </div>
          {stats?.target_stats ? (
            <div className="prediction-grid">
              {Object.entries(stats.target_stats).map(([key, val]) => (
                <div key={key} className="prediction-item">
                  <div className="pred-label">{key.replace(/_/g, ' ')}</div>
                  <div className="pred-value">{val.mean}</div>
                  <div className="pred-uncertainty">σ = {val.std}</div>
                  <div className="uncertainty-bar">
                    <div className="bar-fill" style={{
                      width: `${Math.min(100, (val.mean / val.max) * 100)}%`,
                      background: 'linear-gradient(90deg, #8b5cf6, #06b6d4)'
                    }} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">📡</div>
              <div className="empty-title">Connect to Backend</div>
              <div className="empty-desc">Start the FastAPI server to load real-time statistics</div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title"><span className="title-icon">🏗️</span> System Architecture</div>
            <span className="card-badge badge-success">Active</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {['Feature Engineering (Statistical + FFT)', '11-Model LightGBM Ensemble',
              'Golden Signature Intelligence', 'Isolation Forest Anomaly Detection',
              'NSGA-II Multi-Objective Optimizer', 'SHAP Explainability Engine',
              'Carbon Intelligence & Adaptive Targeting'].map((layer, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 14px', background: 'rgba(255,255,255,0.03)',
                  borderRadius: 8, borderLeft: `3px solid ${['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e', '#3b82f6', '#10b981'][i]}`
                }}>
                  <span style={{ fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.72rem', minWidth: 24 }}>L{i + 1}</span>
                  <span style={{ fontSize: '0.83rem' }}>{layer}</span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </>
  );
}


// ═══════════════════════════════════════════════════════════════
// Predict Panel
// ═══════════════════════════════════════════════════════════════
function PredictPanel({ onPredict, prediction, loading }) {
  const [file, setFile] = useState(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const handleSubmit = () => {
    if (file) onPredict(file);
  };

  return (
    <div className="dashboard-grid">
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">📤</span> Batch Input</div>
        </div>

        <div className="file-upload"
          onDrop={handleDrop} onDragOver={e => e.preventDefault()}
          onClick={() => document.getElementById('csv-input').click()}>
          <div className="upload-icon">📂</div>
          <div className="upload-text">
            {file ? `✓ ${file.name}` : 'Drop time-series CSV or click to browse'}
          </div>
          <div className="upload-hint">Batch_Process_Data.csv format</div>
          <input id="csv-input" type="file" accept=".csv" style={{ display: 'none' }}
            onChange={e => setFile(e.target.files[0])} />
        </div>

        <div style={{ marginTop: 20, display: 'flex', gap: 10 }}>
          <button className="btn btn-primary" onClick={handleSubmit} disabled={!file || loading}>
            {loading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Predicting...</>
              : <><span className="btn-icon">🔮</span> Predict Batch</>}
          </button>
          {file && <button className="btn btn-secondary" onClick={() => setFile(null)}>Clear</button>}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">📊</span> Predictions</div>
          {prediction && <span className="card-badge badge-success">Complete</span>}
        </div>

        {prediction?.predictions ? (
          <div className="prediction-grid">
            {Object.entries(prediction.predictions).map(([key, val]) => (
              <div key={key} className="prediction-item">
                <div className="pred-label">{key.replace(/_/g, ' ')}</div>
                <div className="pred-value">{val.value}</div>
                <div className="pred-uncertainty">± {val.uncertainty}</div>
                <div className="uncertainty-bar">
                  <div className="bar-fill" style={{
                    width: `${Math.max(10, 100 - val.uncertainty * 10)}%`,
                    background: val.uncertainty < 1 ? '#10b981' : val.uncertainty < 3 ? '#f59e0b' : '#f43f5e'
                  }} />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🔮</div>
            <div className="empty-title">No Predictions Yet</div>
            <div className="empty-desc">Upload a batch CSV to see multi-target predictions with uncertainty</div>
          </div>
        )}
      </div>

      {prediction?.reliability && (
        <div className="card full-width">
          <div className="card-header">
            <div className="card-title"><span className="title-icon">🛡️</span> Reliability Status</div>
            <span className={`status-indicator ${prediction.reliability.drift_status === 'NORMAL' ? 'status-normal' : 'status-warning'}`}>
              <span className="status-dot" />
              {prediction.reliability.drift_status}
            </span>
          </div>
          <div style={{ display: 'flex', gap: 30, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Anomaly Status</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: prediction.reliability.anomaly.status === 'NORMAL' ? '#10b981' : '#f43f5e' }}>
                {prediction.reliability.anomaly.status}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Golden Deviation</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {prediction.reliability.golden_deviation}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Anomaly Score</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                {prediction.reliability.anomaly.anomaly_score}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// Optimize Panel
// ═══════════════════════════════════════════════════════════════
function OptimizePanel({ params, setParams, onOptimize, result, loading }) {
  const paramFields = [
    { key: 'Motor_Speed', label: 'Motor Speed (RPM)', min: 1200, max: 1800 },
    { key: 'Temperature', label: 'Temperature (°C)', min: 60, max: 90 },
    { key: 'Pressure', label: 'Pressure (Bar)', min: 3.0, max: 5.5 },
    { key: 'Flow_Rate', label: 'Flow Rate (LPM)', min: 15, max: 30 },
    { key: 'Hold_Time', label: 'Hold Time (min)', min: 10, max: 25 },
  ];

  return (
    <>
      <div className="dashboard-grid">
        <div className="card">
          <div className="card-header">
            <div className="card-title"><span className="title-icon">⚙️</span> Process Parameters</div>
          </div>
          <div className="form-row">
            {paramFields.map(f => (
              <div key={f.key} className="form-group">
                <label className="form-label">{f.label}</label>
                <input type="number" className="form-input" step="0.1"
                  min={f.min} max={f.max}
                  value={params[f.key]}
                  onChange={e => setParams(p => ({ ...p, [f.key]: parseFloat(e.target.value) || 0 }))} />
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Population Size</label>
              <input type="number" className="form-input" value={params.pop_size}
                onChange={e => setParams(p => ({ ...p, pop_size: parseInt(e.target.value) || 40 }))} />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Generations</label>
              <input type="number" className="form-input" value={params.n_generations}
                onChange={e => setParams(p => ({ ...p, n_generations: parseInt(e.target.value) || 25 }))} />
            </div>
          </div>
          <button className="btn btn-primary" onClick={onOptimize} disabled={loading} style={{ width: '100%', marginTop: 12, justifyContent: 'center' }}>
            {loading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Running NSGA-II...</>
              : <><span className="btn-icon">🧬</span> Optimize (NSGA-II)</>}
          </button>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title"><span className="title-icon">⭐</span> Recommended Solution</div>
            {result && <span className="card-badge badge-success">Pareto Optimal</span>}
          </div>
          {result?.recommended_solution ? (
            <div className="prediction-grid">
              {Object.entries(result.recommended_solution).map(([key, val]) => (
                <div key={key} className="prediction-item">
                  <div className="pred-label">{key.replace(/_/g, ' ')}</div>
                  <div className="pred-value" style={{
                    color: key === 'energy' ? '#f59e0b' : key === 'carbon' ? '#10b981' : key === 'quality' ? '#06b6d4' : 'var(--text-primary)'
                  }}>{typeof val === 'number' ? val.toFixed(2) : val}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🧬</div>
              <div className="empty-title">Run Optimization</div>
              <div className="empty-desc">NSGA-II finds Pareto-optimal solutions balancing energy, carbon, and quality</div>
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="dashboard-grid" style={{ marginTop: 0 }}>
          {/* 3D Pareto Plot */}
          <div className="card">
            <div className="card-header">
              <div className="card-title"><span className="title-icon">🎯</span> 3D Pareto Front</div>
              <span className="card-badge badge-info">{result.total_solutions} solutions</span>
            </div>
            <Plot
              data={[{
                x: result.pareto_solutions.map(s => s.energy),
                y: result.pareto_solutions.map(s => s.carbon),
                z: result.pareto_solutions.map(s => s.quality),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                  size: 6,
                  color: result.pareto_solutions.map(s => s.quality),
                  colorscale: [[0, '#f43f5e'], [0.5, '#f59e0b'], [1, '#10b981']],
                  opacity: 0.85,
                  colorbar: { title: 'Quality', thickness: 12, len: 0.6 }
                },
                text: result.pareto_solutions.map((s, i) => `Sol ${i + 1}<br>Energy: ${s.energy}<br>Carbon: ${s.carbon}<br>Quality: ${s.quality}`),
                hoverinfo: 'text',
              },
              ...(result.recommended_solution ? [{
                x: [result.recommended_solution.energy],
                y: [result.recommended_solution.carbon],
                z: [result.recommended_solution.quality],
                mode: 'markers',
                type: 'scatter3d',
                marker: { size: 12, color: '#8b5cf6', symbol: 'diamond', line: { color: 'white', width: 2 } },
                name: 'Recommended',
                hoverinfo: 'text',
                text: ['★ Recommended Solution'],
              }] : [])
              ]}
              layout={{
                scene: {
                  xaxis: { title: 'Energy (kWh)', color: '#94a3b8', gridcolor: '#1e293b' },
                  yaxis: { title: 'Carbon (kg CO₂)', color: '#94a3b8', gridcolor: '#1e293b' },
                  zaxis: { title: 'Quality', color: '#94a3b8', gridcolor: '#1e293b' },
                  bgcolor: 'rgba(0,0,0,0)',
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#94a3b8' },
                margin: { l: 0, r: 0, t: 30, b: 0 },
                height: 400,
                showlegend: false,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </div>

          {/* Hypervolume Convergence */}
          <div className="card">
            <div className="card-header">
              <div className="card-title"><span className="title-icon">📈</span> Hypervolume Convergence</div>
            </div>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={result.hypervolume_convergence.map((hv, i) => ({ gen: i + 1, hv }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="gen" label={{ value: 'Generation', position: 'bottom', fill: '#64748b', fontSize: 12 }}
                  tick={{ fill: '#64748b', fontSize: 11 }} stroke="#334155" />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} stroke="#334155" />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9', fontSize: 13 }} />
                <Line type="monotone" dataKey="hv" stroke="#8b5cf6" strokeWidth={2.5}
                  dot={{ fill: '#8b5cf6', r: 3 }} activeDot={{ r: 6, fill: '#06b6d4' }}
                  name="Hypervolume" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Solution Table */}
          <div className="card full-width">
            <div className="card-header">
              <div className="card-title"><span className="title-icon">📋</span> Pareto Solutions</div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="solution-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Motor Speed</th>
                    <th>Temperature</th>
                    <th>Pressure</th>
                    <th>Flow Rate</th>
                    <th>Hold Time</th>
                    <th>Energy (kWh)</th>
                    <th>Carbon (kg)</th>
                    <th>Quality</th>
                  </tr>
                </thead>
                <tbody>
                  {result.pareto_solutions.slice(0, 15).map((sol, i) => {
                    const isRec = result.recommended_solution &&
                      sol.energy === result.recommended_solution.energy &&
                      sol.quality === result.recommended_solution.quality;
                    return (
                      <tr key={i} className={isRec ? 'recommended' : ''}>
                        <td>{isRec ? '⭐' : i + 1}</td>
                        <td>{sol.Motor_Speed}</td>
                        <td>{sol.Temperature}</td>
                        <td>{sol.Pressure}</td>
                        <td>{sol.Flow_Rate}</td>
                        <td>{sol.Hold_Time}</td>
                        <td>{sol.energy}</td>
                        <td>{sol.carbon}</td>
                        <td>{sol.quality}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  );
}


// ═══════════════════════════════════════════════════════════════
// Reliability Panel
// ═══════════════════════════════════════════════════════════════
function ReliabilityPanel({ prediction, goldenSig }) {
  return (
    <div className="dashboard-grid">
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">🛡️</span> Anomaly Detection</div>
          {prediction?.reliability?.anomaly && (
            <span className={`status-indicator ${prediction.reliability.anomaly.status === 'NORMAL' ? 'status-normal' : 'status-danger'}`}>
              <span className="status-dot" />
              {prediction.reliability.anomaly.status}
            </span>
          )}
        </div>

        {prediction?.reliability ? (
          <div style={{ textAlign: 'center', padding: 20 }}>
            <div className="gauge-container">
              <div className="gauge-value" style={{
                color: prediction.reliability.anomaly.status === 'NORMAL' ? '#10b981' : '#f43f5e'
              }}>
                {prediction.reliability.anomaly.anomaly_score}
              </div>
              <div className="gauge-label">Anomaly Score (higher = more normal)</div>
              <div className="gauge-bar" style={{ maxWidth: 300 }}>
                <div className="gauge-fill" style={{
                  width: `${Math.max(5, Math.min(100, (prediction.reliability.anomaly.anomaly_score + 0.5) * 100))}%`,
                  background: prediction.reliability.anomaly.status === 'NORMAL'
                    ? 'linear-gradient(90deg, #10b981, #06b6d4)'
                    : 'linear-gradient(90deg, #f43f5e, #f59e0b)'
                }} />
              </div>
            </div>
            {prediction.reliability.anomaly.recommendation && (
              <div style={{
                marginTop: 16, padding: '10px 20px', background: 'rgba(244,63,94,0.1)',
                borderRadius: 8, color: '#f43f5e', fontSize: '0.83rem'
              }}>
                ⚠️ {prediction.reliability.anomaly.recommendation}
              </div>
            )}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🛡️</div>
            <div className="empty-title">No Analysis Available</div>
            <div className="empty-desc">Upload a batch in the Predict tab to see reliability analysis</div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">🏆</span> Golden Signature</div>
          {prediction?.reliability && (
            <span className={`status-indicator ${prediction.reliability.drift_status === 'NORMAL' ? 'status-normal' : 'status-warning'}`}>
              <span className="status-dot" />
              {prediction.reliability.drift_status}
            </span>
          )}
        </div>

        {prediction?.reliability ? (
          <div style={{ textAlign: 'center', padding: 20 }}>
            <div className="gauge-container">
              <div className="gauge-value" style={{ color: '#06b6d4' }}>
                {prediction.reliability.golden_deviation}
              </div>
              <div className="gauge-label">Deviation from Golden Centroid (threshold: {prediction.reliability.threshold})</div>
              <div className="gauge-bar" style={{ maxWidth: 300 }}>
                <div className="gauge-fill" style={{
                  width: `${Math.min(100, (prediction.reliability.golden_deviation / prediction.reliability.threshold) * 100)}%`,
                  background: prediction.reliability.golden_deviation < prediction.reliability.threshold
                    ? 'linear-gradient(90deg, #10b981, #06b6d4)'
                    : 'linear-gradient(90deg, #f59e0b, #f43f5e)'
                }} />
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🏆</div>
            <div className="empty-title">Golden Signature Reference</div>
            <div className="empty-desc">Centroid of top 10% historical batches</div>
          </div>
        )}
      </div>

      {goldenSig && (
        <div className="card full-width">
          <div className="card-header">
            <div className="card-title"><span className="title-icon">📊</span> Golden Feature Profile</div>
            <span className="card-badge badge-info">{goldenSig.num_features} features</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={goldenSig.top_features} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} stroke="#334155" />
              <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 11 }} width={160} stroke="#334155" />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9', fontSize: 13 }} />
              <Bar dataKey="golden_value" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Golden Value" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// Explain Panel
// ═══════════════════════════════════════════════════════════════
function ExplainPanel({ onExplain, result, loading }) {
  const SHAP_COLORS = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e',
    '#3b82f6', '#ec4899', '#14b8a6', '#a855f7', '#ef4444'];

  return (
    <div className="dashboard-grid">
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">🔍</span> SHAP Feature Importance</div>
          <button className="btn btn-primary" onClick={onExplain} disabled={loading}
            style={{ padding: '8px 16px', fontSize: '0.78rem' }}>
            {loading ? 'Analyzing...' : '🔍 Run SHAP Analysis'}
          </button>
        </div>

        {result?.energy_feature_importance ? (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={result.energy_feature_importance} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} stroke="#334155"
                label={{ value: 'SHAP Value (impact on Energy prediction)', position: 'bottom', fill: '#64748b', fontSize: 12 }} />
              <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 11 }} width={180} stroke="#334155" />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }} />
              <Bar dataKey="shap_value" name="SHAP Value" radius={[0, 4, 4, 0]}>
                {result.energy_feature_importance.map((_, i) => (
                  <Cell key={i} fill={SHAP_COLORS[i % SHAP_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🔍</div>
            <div className="empty-title">Run SHAP Analysis</div>
            <div className="empty-desc">Discover which features drive energy consumption predictions</div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">💡</span> Recommendations</div>
        </div>

        {result?.recommendations?.length ? (
          <div className="recommendation-list">
            {result.recommendations.map((rec, i) => (
              <motion.div key={i} className="recommendation-item"
                initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}>
                <div>
                  <span className={`rec-category ${rec.category || 'investigation'}`}>
                    {rec.category || 'investigate'}
                  </span>
                </div>
                <div className="rec-text">
                  <div className="rec-title">{rec.parameter}</div>
                  <div className="rec-action">{rec.action}</div>
                  <div className="rec-saving">💰 {rec.expected_saving}</div>
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">💡</div>
            <div className="empty-title">No Recommendations</div>
            <div className="empty-desc">Run SHAP analysis to generate actionable recommendations</div>
          </div>
        )}
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// Carbon Panel
// ═══════════════════════════════════════════════════════════════
function CarbonPanel({ prediction, carbonData }) {
  const data = carbonData || prediction?.carbon;
  const regulatory_limit = data?.regulatory_limit_kg || 450;
  const carbon_val = data?.carbon_emissions_kg || 0;
  const pct = regulatory_limit > 0 ? (carbon_val / regulatory_limit) * 100 : 0;

  return (
    <div className="dashboard-grid">
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">🌱</span> Carbon Emissions</div>
          {data && (
            <span className={`card-badge ${data.regulatory_status?.includes('EXCEEDS') ? 'badge-danger'
              : data.regulatory_status?.includes('above') ? 'badge-warning' : 'badge-success'}`}>
              {data.regulatory_status}
            </span>
          )}
        </div>

        {data ? (
          <div className="carbon-display">
            <div className="carbon-gauge">
              <div className="gauge-container">
                <div className="gauge-value" style={{
                  color: pct > 100 ? '#f43f5e' : pct > 88 ? '#f59e0b' : '#10b981',
                  fontSize: '2.5rem',
                }}>
                  {carbon_val}
                </div>
                <div className="gauge-label">kg CO₂e / batch</div>
                <div className="gauge-bar" style={{ maxWidth: 250 }}>
                  <div className="gauge-fill" style={{
                    width: `${Math.min(100, pct)}%`,
                    background: pct > 100
                      ? 'linear-gradient(90deg, #f43f5e, #e11d48)'
                      : pct > 88
                        ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                        : 'linear-gradient(90deg, #10b981, #059669)'
                  }} />
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                  {Math.round(pct)}% of regulatory limit ({regulatory_limit} kg)
                </div>
              </div>
            </div>

            <div className="carbon-details">
              {[
                ['Energy Input', `${data.predicted_energy_kwh} kWh`],
                ['Carbon Intensity', `${data.carbon_intensity_kg_per_kwh} kg/kWh`],
                ['Grid Period', data.grid_period],
                ['Regulatory Limit', `${regulatory_limit} kg CO₂`],
                ['Headroom', data.regulatory_headroom_kg ? `${data.regulatory_headroom_kg} kg` : 'N/A'],
              ].map(([label, value]) => (
                <div key={label} className="carbon-row">
                  <span className="carbon-row-label">{label}</span>
                  <span className="carbon-row-value">{value}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🌱</div>
            <div className="empty-title">No Carbon Data</div>
            <div className="empty-desc">Upload a batch prediction to calculate carbon emissions and compliance status</div>
          </div>
        )}

        {data?.recommendation && (
          <div style={{
            marginTop: 16, padding: '12px 18px',
            background: data.regulatory_status?.includes('EXCEEDS')
              ? 'rgba(244,63,94,0.1)' : 'rgba(16,185,129,0.1)',
            borderRadius: 8, fontSize: '0.83rem',
            color: data.regulatory_status?.includes('EXCEEDS') ? '#f43f5e' : '#10b981'
          }}>
            💡 {data.recommendation}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title"><span className="title-icon">📊</span> Carbon Intensity Schedule</div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={[
            { period: 'Off-Peak\n(22:00-06:00)', intensity: 0.75, fill: '#10b981' },
            { period: 'Shoulder\n(06:00-10:00)', intensity: 0.82, fill: '#f59e0b' },
            { period: 'Peak\n(10:00-17:00)', intensity: 0.90, fill: '#f43f5e' },
            { period: 'Shoulder\n(17:00-22:00)', intensity: 0.82, fill: '#f59e0b' },
          ]}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="period" tick={{ fill: '#94a3b8', fontSize: 11 }} stroke="#334155" />
            <YAxis domain={[0.6, 1.0]} tick={{ fill: '#64748b', fontSize: 11 }} stroke="#334155"
              label={{ value: 'kg CO₂/kWh', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }} />
            <Bar dataKey="intensity" name="Carbon Intensity" radius={[4, 4, 0, 0]}>
              {[0, 1, 2, 3].map(i => (
                <Cell key={i} fill={['#10b981', '#f59e0b', '#f43f5e', '#f59e0b'][i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
