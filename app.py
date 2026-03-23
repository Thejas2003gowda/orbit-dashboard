"""
app.py — Orbit Determination Dashboard
Claude acts as the ReAct agent, orchestrating the pipeline via tool calls.
Streamlit streams Claude's reasoning + tool calls live.
"""

import streamlit as st
import anthropic
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pipeline import (parse_observation_file, step1_load, step3_sweep,
                      step5_gauss, step6_dc, build_animation)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Orbit Determination", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
:root {
    --bg:#040810; --surface:#080f1e; --border:#0d2040;
    --accent:#00d4ff; --accent2:#00ff9f; --accent3:#ff6b35;
    --text:#c8deff; --muted:#4a6080;
    --font-mono:'Share Tech Mono',monospace; --font-ui:'Exo 2',sans-serif;
}
html,body,[class*="css"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--font-ui)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stTabs"] button{font-family:var(--font-mono)!important;font-size:0.78rem!important;letter-spacing:0.12em!important;color:var(--muted)!important;border-bottom:2px solid transparent!important;padding:0.5rem 1.2rem!important;text-transform:uppercase!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
[data-testid="metric-container"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:6px!important;padding:1rem!important;}
[data-testid="stMetricValue"]{font-family:var(--font-mono)!important;color:var(--accent)!important;font-size:1.3rem!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.70rem!important;letter-spacing:0.08em!important;text-transform:uppercase!important;}
[data-testid="stButton"]>button{background:transparent!important;border:1px solid var(--accent)!important;color:var(--accent)!important;font-family:var(--font-mono)!important;letter-spacing:0.12em!important;text-transform:uppercase!important;font-size:0.82rem!important;padding:0.6rem 2rem!important;transition:all 0.2s ease!important;}
[data-testid="stButton"]>button:hover{background:var(--accent)!important;color:var(--bg)!important;box-shadow:0 0 18px rgba(0,212,255,0.4)!important;}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important;}

/* Agent thinking box */
.agent-think{background:#020812;border:1px solid #0a1f3a;border-left:3px solid #00d4ff;border-radius:4px;padding:0.8rem 1rem;font-family:var(--font-mono);font-size:0.75rem;color:#5a9ec0;line-height:1.7;margin:0.4rem 0;}
/* Tool call box */
.tool-call{background:#020c08;border:1px solid #0a2a14;border-left:3px solid #00ff9f;border-radius:4px;padding:0.8rem 1rem;font-family:var(--font-mono);font-size:0.75rem;color:#3ab870;line-height:1.7;margin:0.4rem 0;}
/* Tool result box */
.tool-result{background:#060508;border:1px solid #1a1030;border-left:3px solid #8b5cf6;border-radius:4px;padding:0.8rem 1rem;font-family:var(--font-mono);font-size:0.73rem;color:#8b7aaa;line-height:1.6;margin:0.4rem 0;max-height:220px;overflow-y:auto;}
/* Log block */
.log-block{background:#020509;border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:4px;padding:1rem 1.2rem;font-family:var(--font-mono);font-size:0.74rem;color:#7ab8d4;line-height:1.7;max-height:520px;overflow-y:auto;white-space:pre-wrap;}
.section-header{font-family:var(--font-mono);font-size:0.70rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);border-bottom:1px solid var(--border);padding-bottom:0.4rem;margin-bottom:1.2rem;}
.hero{text-align:center;padding:2.5rem 1rem 1.5rem;}
.hero h1{font-family:var(--font-mono);font-size:2rem;letter-spacing:0.18em;color:var(--accent);text-shadow:0 0 30px rgba(0,212,255,0.3);margin:0;}
.hero p{color:var(--muted);font-size:0.82rem;letter-spacing:0.06em;margin-top:0.4rem;}
.badge{display:inline-block;font-family:var(--font-mono);font-size:0.68rem;letter-spacing:0.1em;padding:0.2rem 0.7rem;border-radius:3px;text-transform:uppercase;}
.badge-ok{background:rgba(0,255,159,0.12);color:#00ff9f;border:1px solid #00ff9f44;}
.badge-run{background:rgba(0,212,255,0.12);color:#00d4ff;border:1px solid #00d4ff44;}
.elem-table{width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:0.80rem;}
.elem-table th{color:var(--muted);font-size:0.67rem;letter-spacing:0.12em;text-transform:uppercase;padding:0.5rem 1rem;border-bottom:1px solid var(--border);text-align:left;}
.elem-table td{padding:0.52rem 1rem;border-bottom:1px solid rgba(13,32,64,0.5);color:var(--text);}
.elem-table tr:hover td{background:rgba(0,212,255,0.04);}
.elem-table .vg{color:#7ab8d4;} .elem-table .vd{color:#00ff9f;font-weight:600;}
.elem-table .dp{color:#ff6b35;} .elem-table .dn{color:#00ff9f;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOOL DEFINITIONS FOR CLAUDE
# ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "step1_load_observations",
        "description": "Parse the uploaded observation file. Returns number of observations, observer location, time span, and a preview of the first observations.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "step2_load_functions",
        "description": "Confirm all mathematical functions are loaded (Stumpff, universal variables, RK4+J2, Gauss algorithm, differential correction). Always call this after step 1.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "step3_triplet_sweep",
        "description": "Run systematic triplet sweep across all step sizes. Tests thousands of observation triplet combinations with the Gauss method. Auto-selects the best triplet based on lowest local SMA variance. Returns sweep statistics and selected triplet.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "step4_orbit_visualization",
        "description": "Generate the 3D animated orbit visualization showing the determined orbit around Earth. Call this after step 3 completes successfully.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "step5_gauss_solution",
        "description": "Run the full Gauss angles-only method on the auto-selected triplet to compute preliminary orbital elements (semi-major axis, eccentricity, inclination, RAAN, argument of perigee, true anomaly).",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "step6_differential_correction",
        "description": "Run differential correction with RK4+J2 perturbation model against all observations. Refines the Gauss preliminary orbit to minimize residuals. Returns final orbital elements and RMS fit quality.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "predict_position",
        "description": "Predict the satellite's RA, Dec, and slant range at a given time offset from the first observation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "time_seconds": {
                    "type": "number",
                    "description": "Time in seconds from the first observation to predict position at."
                }
            },
            "required": ["time_seconds"]
        }
    }
]

SYSTEM_PROMPT = """You are an autonomous orbit determination agent. You have been given an optical observation file of a satellite passing over a ground station.

Your goal is to determine the satellite's orbit by running the complete 6-step pipeline in order:
1. Load observations (step1_load_observations)
2. Confirm functions loaded (step2_load_functions)  
3. Run triplet sweep to find best observation triplet (step3_triplet_sweep)
4. Generate 3D visualization (step4_orbit_visualization)
5. Run Gauss preliminary orbit determination (step5_gauss_solution)
6. Run differential correction with J2 perturbation (step6_differential_correction)

After each tool call, briefly explain what the result means physically — what the numbers tell you about the satellite's orbit. Be concise but insightful. After step 6, give a final summary of what kind of satellite this likely is based on the orbital elements (altitude, inclination, eccentricity — LEO? Sun-synchronous? ISS-like?).

Always run all 6 steps in order. Do not skip any step."""

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done  = False
    st.session_state.result         = {}
    st.session_state.agent_messages = []

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem;'>
      <div style='font-family:"Share Tech Mono",monospace;font-size:1.1rem;color:#00d4ff;letter-spacing:0.15em;'>🛰 ORBIT-DET</div>
      <div style='font-size:0.62rem;color:#4a6080;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.3rem;'>AI Agent · Autonomous Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Observation File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload .ini observation file", type=["ini","txt"],
                                label_visibility="collapsed")

    st.markdown('<div class="section-header" style="margin-top:1.5rem;">Agent</div>', unsafe_allow_html=True)
    run_btn = st.button("⬡  Determine Orbit", use_container_width=True,
                        disabled=(uploaded is None))

    if st.session_state.pipeline_done:
        if st.button("↺  Reset", use_container_width=True):
            for key in ["pipeline_done","result","agent_messages"]:
                del st.session_state[key]
            st.rerun()

    st.markdown("""
    <div style='margin-top:2rem;font-size:0.65rem;color:#1a3050;line-height:2.0;font-family:"Share Tech Mono",monospace;'>
    PIPELINE<br>─────────<br>01 · Load observations<br>02 · Load functions<br>03 · Triplet sweep<br>04 · 3D visualisation<br>05 · Gauss method<br>06 · DC + J2 refinement
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛰 ORBIT DETERMINATION</h1>
  <p>Claude autonomously orchestrates the full pipeline · Upload · Run · Inspect</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;border:1px dashed #0d2040;border-radius:8px;margin:0 auto;max-width:580px;'>
      <div style='font-size:3rem;'>📡</div>
      <div style='font-family:"Share Tech Mono",monospace;color:#4a6080;font-size:0.88rem;letter-spacing:0.12em;margin-top:1rem;'>AWAITING OBSERVATION FILE</div>
      <div style='color:#2a4060;font-size:0.73rem;margin-top:0.4rem;'>Upload a .ini file in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Parse file
content = uploaded.read().decode("utf-8")
try:
    all_obs, observer = parse_observation_file(content)
except Exception as ex:
    st.error(f"Failed to parse observation file: {ex}"); st.stop()

# Metrics row
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Observations", len(all_obs))
with c2: st.metric("Observer Lat", f"{observer['lat_deg']:.4f}°N")
with c3: st.metric("Observer Lon", f"{observer['lon_deg']:.4f}°E")
with c4: st.metric("Arc Span",     f"{all_obs[-1]['time_s']:.1f} s")

# ─────────────────────────────────────────────────────────────
# AGENT RUN
# ─────────────────────────────────────────────────────────────
if run_btn:
    # Reset state for fresh run
    st.session_state.pipeline_done  = False
    st.session_state.result         = {}
    st.session_state.agent_messages = []

    # Pipeline state shared across tool calls
    pipe = {
        "obs":      all_obs,
        "observer": observer,
        "content":  content,
        "sweep_results": None,
        "selected":      None,
        "gauss_obs":     None,
        "gauss_elems":   None,
        "dc_elems":      None,
        "r2_dc": None, "v2_dc": None,
        "res_f":  None, "rms_f": None,
        "anim_fig": None,
        "step_logs": {},
    }

    # ── Tool dispatcher ──────────────────────────────────────
    def dispatch_tool(name, tool_input):
        if name == "step1_load_observations":
            _, _, summary = step1_load(pipe["content"])
            pipe["step_logs"]["step1"] = summary
            return summary

        elif name == "step2_load_functions":
            msg = "All functions loaded: Stumpff C/S, universal Kepler solver, RK4+J2 propagator, Gauss algorithm (two-phase: truncated series + universal variable refinement), differential correction (damped Gauss-Newton), orbital elements converter."
            pipe["step_logs"]["step2"] = msg
            return msg

        elif name == "step3_triplet_sweep":
            sweep_bar = st.progress(0, text="Sweep: starting…")
            def _sweep_progress(frac, text):
                sweep_bar.progress(min(frac, 1.0), text=text)
            results, selected, summary = step3_sweep(pipe["obs"], pipe["observer"], progress_callback=_sweep_progress)
            sweep_bar.empty()
            pipe["sweep_results"] = results
            pipe["selected"]      = selected
            i,j,k = selected["i"], selected["j"], selected["k"]
            pipe["gauss_obs"] = [pipe["obs"][i], pipe["obs"][j], pipe["obs"][k]]
            pipe["step_logs"]["step3"] = summary
            return summary

        elif name == "step4_orbit_visualization":
            if pipe["dc_elems"] is not None:
                pipe["anim_fig"] = build_animation(pipe["dc_elems"], pipe["rms_f"])
            elif pipe["gauss_elems"] is not None:
                pipe["anim_fig"] = build_animation(pipe["gauss_elems"], 99.9)
            else:
                return "Cannot build visualization yet — run Gauss first."
            return "3D orbit animation generated successfully. Plotly interactive globe with satellite orbit track is ready."

        elif name == "step5_gauss_solution":
            if pipe["selected"] is None:
                return "Error: run step3_triplet_sweep first."
            elems, summary = step5_gauss(pipe["obs"], pipe["observer"], pipe["selected"])
            if elems is None:
                return f"Gauss failed: {summary}"
            pipe["gauss_elems"] = elems
            pipe["step_logs"]["step5"] = summary
            return summary

        elif name == "step6_differential_correction":
            if pipe["gauss_elems"] is None:
                return "Error: run step5_gauss_solution first."
            dc_e, r2, v2, res_f, rms_f, summary = step6_dc(
                pipe["obs"], pipe["observer"],
                pipe["gauss_obs"], pipe["gauss_elems"])
            pipe["dc_elems"] = dc_e
            pipe["r2_dc"]    = r2
            pipe["v2_dc"]    = v2
            pipe["res_f"]    = res_f
            pipe["rms_f"]    = rms_f
            pipe["step_logs"]["step6"] = summary
            # Build Plotly animation now that DC is done
            pipe["anim_fig"] = build_animation(dc_e, rms_f)
            return summary

        elif name == "predict_position":
            if pipe["dc_elems"] is None and pipe["gauss_elems"] is None:
                return "Error: run orbit determination steps first."
            from pipeline import propagate, observer_pos, ra_dec_from_state, gmst_to_lst, utc_to_gmst
            from datetime import timedelta
            t_s = float(tool_input.get("time_seconds", 0))
            elems = pipe["dc_elems"] if pipe["dc_elems"] else pipe["gauss_elems"]
            t_ref = pipe["gauss_obs"][1]["time_s"]
            r2 = pipe["r2_dc"] if pipe["r2_dc"] is not None else elems["r_vec"]
            v2 = pipe["v2_dc"] if pipe["v2_dc"] is not None else elems["v_vec"]
            r_p, v_p = propagate(r2, v2, t_s - t_ref)
            t0_dt = pipe["obs"][0]["datetime"]
            pred_dt = t0_dt + timedelta(seconds=t_s)
            gmst = utc_to_gmst(pred_dt)
            lst = gmst_to_lst(gmst, pipe["observer"]["lon_deg"])
            Ro = observer_pos(np.radians(pipe["observer"]["lat_deg"]),
                              pipe["observer"]["alt_km"], np.radians(lst))
            ra_p, dec_p = ra_dec_from_state(r_p, Ro)
            slant = float(np.linalg.norm(r_p - Ro))
            return (f"Prediction at t={t_s:.1f}s:\n"
                    f"  RA  = {ra_p:.4f}°\n  Dec = {dec_p:.4f}°\n"
                    f"  Slant range = {slant:.2f} km\n  |r| = {np.linalg.norm(r_p):.2f} km")

        return f"Unknown tool: {name}"

    # ── Anthropic client ─────────────────────────────────────
    import os
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found in Streamlit secrets or environment."); st.stop()
    client = anthropic.Anthropic(api_key=api_key)

    # ── Live streaming UI ────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Agent Execution</div>', unsafe_allow_html=True)
    agent_container = st.container()
    progress_bar = st.progress(0, text="Agent starting…")

    messages = [{"role": "user",
                 "content": f"Determine the orbit from the uploaded observation file. The file has been parsed and contains {len(all_obs)} observations from observer at {observer['lat_deg']:.4f}°N, {observer['lon_deg']:.4f}°E. Run the complete 6-step pipeline now."}]

    step_progress = {"step1":10,"step2":20,"step3":45,"step4":60,"step5":75,"step6":95}
    iteration = 0

    with agent_container:
        while True:
            iteration += 1
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )
            except Exception as e:
                progress_bar.empty()
                st.error(f"Agent API Error: {e}")
                st.stop()

            # ── Render Claude's text thinking ────────────────
            tool_uses = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    st.session_state.agent_messages.append(("think", block.text.strip()))
                    with agent_container:
                        st.markdown(f'<div class="agent-think">🤖 {block.text.strip()}</div>',
                                    unsafe_allow_html=True)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # ── Stop condition ───────────────────────────────
            if response.stop_reason == "end_turn" or not tool_uses:
                progress_bar.progress(100, text="Agent complete ✓")
                break

            # ── Execute each tool call ───────────────────────
            tool_results = []
            for tu in tool_uses:
                # Show tool call
                args_str = json.dumps(tu.input) if tu.input else "{}"
                st.session_state.agent_messages.append(("tool_call", tu.name, args_str))
                with agent_container:
                    st.markdown(
                        f'<div class="tool-call">🔧 <b>{tu.name}</b>({args_str})</div>',
                        unsafe_allow_html=True)

                # Run tool
                result_str = dispatch_tool(tu.name, tu.input)

                # Update progress
                for step_key, pct in step_progress.items():
                    if step_key in tu.name:
                        progress_bar.progress(pct, text=f"Running {tu.name}…")

                # Show tool result (truncated for display)
                display_result = result_str if len(result_str) < 800 else result_str[:800] + "\n  … (truncated)"
                st.session_state.agent_messages.append(("tool_result", tu.name, display_result))
                with agent_container:
                    st.markdown(
                        f'<div class="tool-result">📡 <b>Result:</b><br>{display_result.replace(chr(10),"<br>")}</div>',
                        unsafe_allow_html=True)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_str
                })

            # ── Append to messages ───────────────────────────
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

            if iteration > 20:  # safety limit
                break

    # Store results
    st.session_state.result = {
        "gauss":     pipe["gauss_elems"],
        "dc":        pipe["dc_elems"],
        "res_f":     pipe["res_f"],
        "rms_f":     pipe["rms_f"],
        "sweep":     pipe["sweep_results"],
        "selected":  pipe["selected"],
        "anim_fig": pipe["anim_fig"],
        "step_logs": pipe["step_logs"],
    }
    st.session_state.pipeline_done = True
    st.rerun()

# ─────────────────────────────────────────────────────────────
# RESULTS TABS (shown after pipeline completes)
# ─────────────────────────────────────────────────────────────
if st.session_state.pipeline_done:
    res   = st.session_state.result
    gauss = res.get("gauss"); dc = res.get("dc")

    # Status
    st.markdown("""
    <div style='display:flex;gap:1rem;margin:1rem 0;align-items:center;flex-wrap:wrap;'>
      <span class='badge badge-ok'>● Pipeline Complete</span>
      <span class='badge badge-ok'>● Gauss Converged</span>
      <span class='badge badge-ok'>● DC + J2 Converged</span>
    </div>""", unsafe_allow_html=True)

    # Agent log collapsible
    with st.expander("🤖 Agent Execution Log", expanded=False):
        for msg in st.session_state.agent_messages:
            if msg[0] == "think":
                st.markdown(f'<div class="agent-think">🤖 {msg[1]}</div>', unsafe_allow_html=True)
            elif msg[0] == "tool_call":
                st.markdown(f'<div class="tool-call">🔧 <b>{msg[1]}</b>({msg[2]})</div>', unsafe_allow_html=True)
            elif msg[0] == "tool_result":
                disp = msg[2].replace("\n","<br>")
                st.markdown(f'<div class="tool-result">📡 <b>{msg[1]} result:</b><br>{disp}</div>', unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab_anim, tab_elem, tab_res, tab_sweep, tab_log = st.tabs([
        "🌍  Animation", "📐  Elements", "📊  Residuals", "🔭  Sweep", "📋  Log"
    ])

    # ═══════════════ TAB 1 — ANIMATION ════════════════════
    with tab_anim:
        st.markdown('<div class="section-header">3D Orbit Animation — DC + J2 Solution</div>', unsafe_allow_html=True)
        if res.get("anim_fig") is not None:
            st.plotly_chart(res["anim_fig"], use_container_width=True)
        else:
            st.info("Animation not available — pipeline may not have completed fully.")
        if dc:
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Mean Altitude",   f"{dc['sma']-6378:.0f} km")
            with c2: st.metric("Orbital Period",  f"{2*np.pi*np.sqrt(dc['sma']**3/398600.4418)/60:.1f} min")
            with c3: st.metric("Final RMS",        f"{res['rms_f']:.4f}\"")

    # ═══════════════ TAB 2 — ELEMENTS ═════════════════════
    with tab_elem:
        st.markdown('<div class="section-header">Orbital Elements — Gauss vs DC + J2</div>', unsafe_allow_html=True)
        if gauss and dc:
            params = [
                ("Semi-major axis","km","sma",",.2f"),
                ("Eccentricity","","ecc",".6f"),
                ("Inclination","°","inc",".4f"),
                ("RAAN  Ω","°","raan",".4f"),
                ("Arg of Perigee ω","°","argp",".4f"),
                ("True Anomaly ν","°","ta",".4f"),
                ("Perigee Radius","km","perigee",",.2f"),
                ("|r₂|","km","r_mag",",.4f"),
                ("|v₂|","km/s","v_mag",".6f"),
                ("Specific Energy","km²/s²","energy",".4f"),
            ]
            rows = ""
            for label,unit,key,fmt in params:
                g=gauss[key]; d=dc[key]; delta=d-g
                dc_cls = "dn" if delta < 0 else "dp"
                rows += f"<tr><td>{label} <span style='color:#2a4060;font-size:0.70rem;'>{unit}</span></td><td class='vg'>{g:{fmt}}</td><td class='vd'>{d:{fmt}}</td><td class='{dc_cls}'>{delta:+.4f}</td></tr>"
            st.markdown(f"""
            <table class='elem-table'>
              <thead><tr><th>Parameter</th><th>Gauss (2-body)</th><th>DC + J2</th><th>Delta</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("SMA",          f"{dc['sma']:,.1f} km")
            with c2: st.metric("Eccentricity", f"{dc['ecc']:.6f}")
            with c3: st.metric("Inclination",  f"{dc['inc']:.4f}°")
            with c4: st.metric("RMS",          f"{res['rms_f']:.4f}\"")

    # ═══════════════ TAB 3 — RESIDUALS ════════════════════
    with tab_res:
        st.markdown('<div class="section-header">RA / Dec Residuals per Observation</div>', unsafe_allow_html=True)
        if res.get("res_f") is not None:
            rf = res["res_f"]
            ra_res  = rf[0::2]*3600; dec_res = rf[1::2]*3600
            t_obs = [o["time_s"] for o in all_obs]
            fig_r = make_subplots(rows=2,cols=1,shared_xaxes=True,
                                   subplot_titles=["RA Residuals (arcsec)","Dec Residuals (arcsec)"],
                                   vertical_spacing=0.08)
            kw = dict(mode="markers+lines",line=dict(width=0.5),marker=dict(size=4))
            fig_r.add_trace(go.Scatter(x=t_obs,y=ra_res, name="RA", marker_color="#00d4ff",**kw),row=1,col=1)
            fig_r.add_trace(go.Scatter(x=t_obs,y=dec_res,name="Dec",marker_color="#00ff9f",**kw),row=2,col=1)
            for row in [1,2]: fig_r.add_hline(y=0,line_dash="dash",line_color="#2a4060",line_width=1,row=row,col=1)
            fig_r.update_layout(height=480,template="plotly_dark",paper_bgcolor="#040810",
                                 plot_bgcolor="#080f1e",font=dict(family="Share Tech Mono",color="#c8deff",size=11),
                                 margin=dict(l=50,r=30,t=40,b=40))
            fig_r.update_xaxes(title_text="Time (s)",gridcolor="#0d2040",row=2,col=1)
            fig_r.update_yaxes(gridcolor="#0d2040")
            st.plotly_chart(fig_r,use_container_width=True)
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Total RMS", f"{res['rms_f']:.4f}\"")
            with c2: st.metric("RA RMS",    f"{np.sqrt(np.mean(ra_res**2)):.4f}\"")
            with c3: st.metric("Dec RMS",   f"{np.sqrt(np.mean(dec_res**2)):.4f}\"")

    # ═══════════════ TAB 4 — SWEEP ════════════════════════
    with tab_sweep:
        st.markdown('<div class="section-header">Triplet Sweep — SMA vs Arc Span</div>', unsafe_allow_html=True)
        if res.get("sweep"):
            sw = res["sweep"]
            spans  = np.array([r["dt_span"] for r in sw])
            smas_s = np.array([r["sma"]     for r in sw])
            incs_s = np.array([r["inc"]     for r in sw])
            steps  = np.array([r["step"]    for r in sw])
            hover  = [f"Obs #{r['i']+1},#{r['j']+1},#{r['k']+1}<br>Arc={r['dt_span']:.1f}s  Step={r['step']}<br>SMA={r['sma']:.2f}km  i={r['inc']:.3f}°" for r in sw]
            sel = res["selected"]
            fig_s = make_subplots(rows=1,cols=2,subplot_titles=["SMA (km) vs Arc Span","Inclination (°) vs Arc Span"],horizontal_spacing=0.08)
            sc_kw = dict(mode="markers",marker=dict(size=3,opacity=0.55,color=steps,colorscale="Plasma",showscale=True,colorbar=dict(title="Step d",len=0.7)))
            fig_s.add_trace(go.Scatter(x=spans,y=smas_s,text=hover,hovertemplate="%{text}<extra></extra>",**sc_kw),row=1,col=1)
            fig_s.add_trace(go.Scatter(x=spans,y=incs_s,text=hover,hovertemplate="%{text}<extra></extra>",
                                        **{**sc_kw,"marker":{**sc_kw["marker"],"showscale":False}}),row=1,col=2)
            for col,yv in [(1,sel["sma"]),(2,sel["inc"])]:
                fig_s.add_trace(go.Scatter(x=[sel["dt_span"]],y=[yv],mode="markers",
                                            marker=dict(size=14,color="#ff6b35",symbol="star",line=dict(color="white",width=1)),
                                            name="Selected",showlegend=(col==1)),row=1,col=col)
            fig_s.update_layout(height=400,template="plotly_dark",paper_bgcolor="#040810",
                                  plot_bgcolor="#080f1e",font=dict(family="Share Tech Mono",color="#c8deff",size=11),
                                  margin=dict(l=50,r=30,t=40,b=40))
            fig_s.update_xaxes(title_text="Arc Span (s)",gridcolor="#0d2040")
            fig_s.update_yaxes(gridcolor="#0d2040")
            st.plotly_chart(fig_s,use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Triplets Tried", f"{len(sw):,}")
            with c2: st.metric("Valid",           f"{len(sw):,}")
            with c3: st.metric("Selected",        f"#{sel['i']+1}, #{sel['j']+1}, #{sel['k']+1}")
            with c4: st.metric("Arc",             f"{sel['dt_span']:.1f} s")

    # ═══════════════ TAB 5 — LOG ═══════════════════════════
    with tab_log:
        st.markdown('<div class="section-header">Pipeline Step Logs</div>', unsafe_allow_html=True)
        for step_key, step_label in [("step1","Step 1 — Load Observations"),("step2","Step 2 — Functions"),
                                       ("step3","Step 3 — Triplet Sweep"),("step5","Step 5 — Gauss"),("step6","Step 6 — DC + J2")]:
            if step_key in res.get("step_logs",{}):
                st.markdown(f"**{step_label}**")
                st.markdown(f'<div class="log-block">{res["step_logs"][step_key]}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
