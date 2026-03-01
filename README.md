# AI-Driven Manufacturing Intelligence

> Adaptive Multi-Objective Optimization of Industrial Batch Process and Energy Pattern Analytics for Asset Reliability, Process Optimization, and Carbon Management.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-green)

## 🏗️ Architecture — 7-Layer Intelligence Stack

| Layer | Component | Description |
|-------|-----------|-------------|
| L1 | **Feature Engineering** | Statistical + Phase-wise + FFT frequency-domain features |
| L2 | **11-Model Ensemble** | LightGBM multi-output with uncertainty quantification |
| L3 | **Golden Signature** | Top-10% batch centroid for process drift detection |
| L4 | **Anomaly Detection** | Isolation Forest trained on golden batches |
| L5 | **NSGA-II Optimizer** | Manual multi-objective evolutionary optimization |
| L6 | **SHAP Explainability** | TreeExplainer with actionable recommendations |
| L7 | **Carbon Intelligence** | Dynamic carbon intensity & regulatory compliance |

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Model Accuracy (MAPE) | **3.41%** |
| Synthetic Batches | 500 (46,277 time-series rows) |
| Engineered Features | 52 |
| Ensemble Models | 11 |
| NSGA-II Runtime | ~6 seconds |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Run feature engineering
python feature_engineering.py

# Train all models
python train_models.py
python golden_signature.py
python anomaly_detector.py

# Start API server
python -m uvicorn main:app --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## 🖥️ Dashboard Panels

1. **Overview** — KPI cards, system architecture, live target statistics
2. **Predict** — Upload batch CSV → predictions with uncertainty bars
3. **Optimize** — NSGA-II with 3D Pareto plot & hypervolume convergence
4. **Reliability** — Anomaly score gauge, golden signature deviation
5. **Explain** — SHAP feature importance, actionable recommendations
6. **Carbon** — Emission compliance gauge, carbon intensity schedule

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/dataset-stats` | GET | Dataset overview |
| `/predict` | POST | Batch prediction (CSV upload) |
| `/predict-params` | POST | Prediction from parameters |
| `/optimize` | POST | NSGA-II Pareto optimization |
| `/explain` | POST | SHAP feature importance |
| `/golden-signature` | GET | Golden signature stats |
| `/anomaly-check` | POST | Anomaly detection |
| `/carbon` | POST | Carbon emission check |

## 🔬 Innovation Highlights

- **FFT-based energy pattern intelligence** for predictive maintenance
- **Uncertainty quantification** (±std across 11 ensemble models)
- **Manual NSGA-II** (no external optimization library) with hypervolume tracking
- **SHAP explainability** with actionable parameter recommendations
- **Dynamic carbon intensity** by time-of-day with regulatory compliance

## 👥 Team Pramana

Built for the AVEVA Hackathon — AI-Driven Manufacturing Intelligence track.

---

*Version 1.0*
