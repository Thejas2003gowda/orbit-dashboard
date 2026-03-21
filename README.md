# 🛰 Orbit Determination Dashboard

Autonomous orbit determination pipeline wrapped in a professional Streamlit dashboard.

## What it does

Upload an optical observation file (`.ini`) → click **Determine Orbit** → get:

- **3D animated orbit** rotating around Earth
- **Orbital elements table** — Gauss vs DC+J2 comparison
- **Residuals plot** — RA/Dec fit quality per observation
- **Triplet sweep chart** — all 4000+ triplet combinations visualised
- **Pipeline log** — full step-by-step execution output

## Pipeline

1. Parse observation file (RA/Dec pairs + observer location)
2. Systematic triplet sweep (all step sizes 1→N/2, ~4489 combinations)
3. Auto-select best triplet (lowest local SMA variance)
4. Gauss angles-only preliminary orbit determination
5. Differential correction with RK4 + J2 perturbation

## Deploy to Streamlit Cloud

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy** — you'll get a permanent public URL

## Run locally

```bash
pip install streamlit numpy matplotlib plotly
streamlit run app.py
```

## Files

```
app.py            # Full Streamlit dashboard
requirements.txt  # Dependencies for Streamlit Cloud
README.md         # This file
```
