# 🛰 Orbit Determination — AI Agent Dashboard

Claude acts as a ReAct agent, autonomously orchestrating the full orbit determination pipeline via tool calls. Every step is reasoned through by the LLM — it decides what to call, reads the results, interprets the physics, and drives the pipeline to completion.

## Architecture

```
app.py          — Streamlit UI + Claude agent loop (tool call orchestration)
pipeline.py     — All orbital mechanics (Gauss, DC+J2, RK4, animation)
requirements.txt
```

**What Claude does:**
- Receives the observation file metadata
- Calls `step1` → `step2` → `step3` → `step4` → `step5` → `step6` in sequence
- After each tool result, reasons about what the numbers mean physically
- Gives a final satellite classification (LEO? Sun-sync? ISS-like?)

**What the UI shows:**
- Claude's thinking streamed live as it runs
- Tool calls highlighted in green
- Tool results shown in purple
- 5 result tabs: Animation · Elements · Residuals · Sweep · Log

## Deploy to Streamlit Cloud

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "orbit agent dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/orbit-dashboard.git
git push -u origin main
```

### 2. Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app** → select repo → main file: `app.py`
4. Click **Advanced settings** → **Secrets**
5. Add your API key:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
6. Click **Deploy**

### 3. Use
- Open the URL
- Upload your `.ini` observation file
- Click **⬡ Determine Orbit**
- Watch Claude reason through the pipeline live

## Run Locally
```bash
pip install streamlit numpy matplotlib plotly anthropic
# Create .streamlit/secrets.toml with:
# ANTHROPIC_API_KEY = "sk-ant-..."
streamlit run app.py
```
