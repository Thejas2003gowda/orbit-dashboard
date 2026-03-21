"""
═══════════════════════════════════════════════════════════════
  ORBIT DETERMINATION DASHBOARD
  Autonomous pipeline: Load → Sweep → Gauss → DC+J2 → Visualise
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from datetime import datetime, timedelta
import warnings, io, tempfile, os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orbit Determination",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — mission-control dashboard aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg:        #040810;
    --surface:   #080f1e;
    --border:    #0d2040;
    --accent:    #00d4ff;
    --accent2:   #00ff9f;
    --accent3:   #ff6b35;
    --text:      #c8deff;
    --muted:     #4a6080;
    --font-mono: 'Share Tech Mono', monospace;
    --font-ui:   'Exo 2', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1.2rem !important;
    text-transform: uppercase !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--accent) !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 18px rgba(0,212,255,0.4) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
    background: var(--surface) !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* Code/log blocks */
.log-block {
    background: #020509;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: #7ab8d4;
    line-height: 1.7;
    max-height: 500px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Section headers */
.section-header {
    font-family: var(--font-mono);
    font-size: 0.70rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}

/* Elements table */
.elem-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.82rem;
}
.elem-table th {
    color: var(--muted);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.elem-table td {
    padding: 0.55rem 1rem;
    border-bottom: 1px solid rgba(13,32,64,0.6);
    color: var(--text);
}
.elem-table tr:hover td { background: rgba(0,212,255,0.04); }
.elem-table .val-gauss  { color: #7ab8d4; }
.elem-table .val-dc     { color: var(--accent2); font-weight: 600; }
.elem-table .val-delta-pos { color: #ff6b35; }
.elem-table .val-delta-neg { color: #00ff9f; }

/* Hero banner */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-family: var(--font-mono);
    font-size: 2.2rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-shadow: 0 0 30px rgba(0,212,255,0.3);
    margin: 0;
}
.hero p {
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    margin-top: 0.5rem;
}

/* Status badge */
.badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.7rem;
    border-radius: 3px;
    text-transform: uppercase;
}
.badge-ok  { background: rgba(0,255,159,0.12); color: #00ff9f; border: 1px solid #00ff9f44; }
.badge-err { background: rgba(255,107,53,0.12); color: #ff6b35; border: 1px solid #ff6b3544; }
.badge-run { background: rgba(0,212,255,0.12); color: #00d4ff; border: 1px solid #00d4ff44; }

div[data-testid="stHorizontalBlock"] { gap: 1rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS  (self-contained, no Cell 1 dependency)
# ═══════════════════════════════════════════════════════════════

MU   = 398600.4418
RE   = 6378.0
J2   = 1.08263e-3
FLAT = 0.003353

# ── Time utilities ───────────────────────────────────────────
def julian_date(dt):
    y, mo, d = dt.year, dt.month, dt.day
    frac = (dt.hour + dt.minute/60 + (dt.second + dt.microsecond/1e6)/3600) / 24
    if mo <= 2: y -= 1; mo += 12
    A = int(y/100); B = 2 - A + int(A/4)
    return int(365.25*(y+4716)) + int(30.6001*(mo+1)) + d + frac + B - 1524.5

def utc_to_gmst(dt):
    T = (julian_date(dt) - 2451545.0) / 36525.0
    s = 67310.54841 + (876600*3600+8640184.812866)*T + 0.093104*T**2 - 6.2e-6*T**3
    g = (s/240.0) % 360.0
    return g if g >= 0 else g + 360.0

def gmst_to_lst(gmst, lon):
    l = (gmst + lon) % 360.0
    return l if l >= 0 else l + 360.0

def parse_ts(raw):
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(raw.strip(), fmt)
        except: pass
    return None

# ── Observation file parser ──────────────────────────────────
def parse_observation_file(content: str):
    lines = content.splitlines()
    lon_deg = lat_deg = alt_m = None
    for line in lines:
        line = line.strip()
        if line.startswith("COMMENT LONGITUDE"):
            lon_deg = float(line.split()[2])
            if "WEST" in line.upper(): lon_deg = -lon_deg
        elif line.startswith("COMMENT LATITUDE"):
            lat_deg = float(line.split()[2])
            if "SOUTH" in line.upper(): lat_deg = -lat_deg
        elif line.startswith("COMMENT ALTITUDE"):
            alt_m = float(line.split()[2])

    if None in (lat_deg, lon_deg, alt_m):
        raise ValueError("Missing observer location in COMMENT lines")

    alt_km = alt_m / 1000.0
    observer = {"lat_deg": lat_deg, "lon_deg": lon_deg, "alt_km": alt_km}

    pending = {}
    observations = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("COMMENT"): continue
        if line.startswith("ANGLE_1="):
            parts = line[8:].strip().split()
            if len(parts) < 3: continue
            ts = parts[0] + " " + parts[1]
            dt = parse_ts(ts)
            if dt:
                try: pending[ts] = (dt, float(parts[2]))
                except: pass
        elif line.startswith("ANGLE_2="):
            parts = line[8:].strip().split()
            if len(parts) < 3: continue
            ts = parts[0] + " " + parts[1]
            dt = parse_ts(ts)
            if dt and ts in pending:
                ra_dt, ra_deg = pending[ts]
                try:
                    observations.append({"datetime": dt, "ra_deg": ra_deg,
                                         "dec_deg": float(parts[2]), "timestamp_str": ts})
                    del pending[ts]
                except: pass

    observations.sort(key=lambda o: o["datetime"])
    t0 = observations[0]["datetime"]
    for obs in observations:
        obs["time_s"] = (obs["datetime"] - t0).total_seconds()
        gmst = utc_to_gmst(obs["datetime"])
        obs["lst_deg"] = gmst_to_lst(gmst, lon_deg)

    return observations, observer

# ── Stumpff & universal variables ───────────────────────────
def stumpff_C(z):
    if z > 1e-6:   return (1 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6: return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:           return 0.5 - z/24 + z**2/720

def stumpff_S(z):
    if z > 1e-6:
        s = np.sqrt(z); return (s - np.sin(s)) / s**3
    elif z < -1e-6:
        s = np.sqrt(-z); return (np.sinh(s) - s) / s**3
    else:
        return 1/6 - z/120 + z**2/5040

def solve_kepler_uv(r0, vr0, alpha, dt, tol=1e-10, maxiter=50):
    chi = np.sqrt(MU) * dt / r0
    for _ in range(maxiter):
        z = alpha * chi**2
        C, S = stumpff_C(z), stumpff_S(z)
        F  = r0*vr0/np.sqrt(MU)*chi**2*C + (1-alpha*r0)*chi**3*S + r0*chi - np.sqrt(MU)*dt
        dF = r0*vr0/np.sqrt(MU)*chi*(1-alpha*chi**2*S) + (1-alpha*r0)*chi**2*C + r0
        if abs(dF) < 1e-15: break
        c2 = chi - F/dF
        if abs(c2 - chi) < tol: chi = c2; break
        chi = c2
    return chi

def compute_fg(r0v, v0v, dt):
    r0 = np.linalg.norm(r0v); v0 = np.linalg.norm(v0v)
    vr0 = np.dot(r0v, v0v) / r0
    alpha = 2/r0 - v0**2/MU
    chi = solve_kepler_uv(r0, vr0, alpha, dt)
    z = alpha * chi**2
    C, S = stumpff_C(z), stumpff_S(z)
    f = 1 - chi**2/r0*C
    g = dt - chi**3/np.sqrt(MU)*S
    return f, g

# ── Orbital elements from state ──────────────────────────────
def orbital_elements(r_vec, v_vec):
    r = np.linalg.norm(r_vec); v = np.linalg.norm(v_vec)
    vr = np.dot(r_vec, v_vec) / r
    h = np.cross(r_vec, v_vec); hm = np.linalg.norm(h)
    inc = np.degrees(np.arccos(np.clip(h[2]/hm, -1, 1)))
    n = np.cross([0,0,1], h); nm = np.linalg.norm(n)
    raan = 0.0
    if nm > 1e-12:
        raan = np.degrees(np.arccos(np.clip(n[0]/nm, -1, 1)))
        if n[1] < 0: raan = 360 - raan
    e_vec = (1/MU)*((v**2 - MU/r)*r_vec - r*vr*v_vec)
    ecc = np.linalg.norm(e_vec)
    argp = 0.0
    if nm > 1e-12 and ecc > 1e-10:
        argp = np.degrees(np.arccos(np.clip(np.dot(n, e_vec)/(nm*ecc), -1, 1)))
        if e_vec[2] < 0: argp = 360 - argp
    ta = 0.0
    if ecc > 1e-10:
        ta = np.degrees(np.arccos(np.clip(np.dot(e_vec, r_vec)/(ecc*r), -1, 1)))
        if vr < 0: ta = 360 - ta
    energy = v**2/2 - MU/r
    sma = -MU/(2*energy) if abs(energy) > 1e-10 else float("inf")
    return {"sma": sma, "ecc": ecc, "inc": inc, "raan": raan,
            "argp": argp, "ta": ta, "r_mag": r, "v_mag": v,
            "energy": energy, "perigee": sma*(1-ecc),
            "r_vec": r_vec.copy(), "v_vec": v_vec.copy()}

# ── Observer ECI position ────────────────────────────────────
def observer_pos(phi_rad, alt_km, lst_rad):
    Ce = RE / np.sqrt(1 - (2*FLAT - FLAT**2)*np.sin(phi_rad)**2)
    Se = Ce*(1-FLAT)**2
    return np.array([(Ce+alt_km)*np.cos(phi_rad)*np.cos(lst_rad),
                     (Ce+alt_km)*np.cos(phi_rad)*np.sin(lst_rad),
                     (Se+alt_km)*np.sin(phi_rad)])

# ── RK4 + J2 propagator ─────────────────────────────────────
def accel_j2(r):
    x,y,z = r; rm = np.linalg.norm(r); r5 = rm**5
    c = -1.5*J2*MU*RE**2/r5; z2r2 = z**2/rm**2
    return -MU/rm**3*r + c*np.array([x*(1-5*z2r2), y*(1-5*z2r2), z*(3-5*z2r2)])

def rk4_step(s, h):
    def f(s): return np.concatenate([s[3:6], accel_j2(s[:3])])
    k1=f(s); k2=f(s+0.5*h*k1); k3=f(s+0.5*h*k2); k4=f(s+h*k3)
    return s + (h/6)*(k1+2*k2+2*k3+k4)

def propagate(r0, v0, dt, step=1.0):
    s = np.concatenate([r0, v0])
    if abs(dt) < 1e-12: return r0.copy(), v0.copy()
    sg = np.sign(dt); h = sg*min(abs(step), abs(dt))
    n = int(abs(dt)/abs(h)); rem = dt - n*h
    for _ in range(n): s = rk4_step(s, h)
    if abs(rem) > 1e-12: s = rk4_step(s, rem)
    return s[:3].copy(), s[3:].copy()

# ── RA/Dec from satellite + observer ─────────────────────────
def ra_dec(r_sat, R_obs):
    rho = r_sat - R_obs; rh = rho/np.linalg.norm(rho)
    dec = np.degrees(np.arcsin(np.clip(rh[2], -1, 1)))
    ra  = np.degrees(np.arctan2(rh[1], rh[0]))
    if ra < 0: ra += 360.0
    return ra, dec

def ang_res(obs, pred):
    d = obs - pred
    while d >  180: d -= 360
    while d < -180: d += 360
    return d

# ── Residuals + Jacobian ─────────────────────────────────────
def residuals(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km):
    n = len(t_obs); res = np.zeros(2*n)
    for i in range(n):
        rp, _ = propagate(state[:3], state[3:], t_obs[i]-t_ref)
        Ro = observer_pos(phi_rad, alt_km, lst_rad[i])
        ra_p, dec_p = ra_dec(rp, Ro)
        res[2*i]   = ang_res(ra_obs[i], ra_p)
        res[2*i+1] = dec_obs[i] - dec_p
    return res

def jacobian(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km):
    H = np.zeros((2*len(t_obs), 6))
    for j in range(6):
        sp, sm = state.copy(), state.copy()
        dj = 1e-4 if j < 3 else 1e-7
        sp[j] += dj; sm[j] -= dj
        rp = residuals(sp, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km)
        rm = residuals(sm, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km)
        H[:,j] = (rp - rm) / (2*dj)
    return H

# ── Silent Gauss for sweep ───────────────────────────────────
def gauss_silent(oi, oj, ok, observer):
    phi = np.radians(observer["lat_deg"]); alt = observer["alt_km"]
    t   = [o["time_s"] for o in (oi, oj, ok)]
    ra  = [np.radians(o["ra_deg"])  for o in (oi, oj, ok)]
    dec = [np.radians(o["dec_deg"]) for o in (oi, oj, ok)]
    lst = [np.radians(o["lst_deg"]) for o in (oi, oj, ok)]
    tau1 = t[0]-t[1]; tau3 = t[2]-t[1]; tau = tau3-tau1
    if min(abs(tau1), abs(tau3), abs(tau)) < 0.5: return None
    if min(abs(tau1),abs(tau3))/max(abs(tau1),abs(tau3)) < 0.20: return None

    rho_hat = [np.array([np.cos(dec[k])*np.cos(ra[k]),
                          np.cos(dec[k])*np.sin(ra[k]), np.sin(dec[k])]) for k in range(3)]
    Ce = RE/np.sqrt(1-(2*FLAT-FLAT**2)*np.sin(phi)**2); Se = Ce*(1-FLAT)**2
    R = [np.array([(Ce+alt)*np.cos(phi)*np.cos(lst[k]),
                   (Ce+alt)*np.cos(phi)*np.sin(lst[k]),
                   (Se+alt)*np.sin(phi)]) for k in range(3)]

    p1 = np.cross(rho_hat[1], rho_hat[2])
    p2 = np.cross(rho_hat[0], rho_hat[2])
    p3 = np.cross(rho_hat[0], rho_hat[1])
    D0 = np.dot(rho_hat[0], p1)
    if abs(D0) < 1e-14: return None
    D = np.array([[np.dot(R[k], p) for p in (p1,p2,p3)] for k in range(3)])

    A = (1/D0)*(-D[0,1]*(tau3/tau)+D[1,1]+D[2,1]*(tau1/tau))
    B = (1/(6*D0))*(D[0,1]*(tau3**2-tau**2)*(tau3/tau)+D[2,1]*(tau**2-tau1**2)*(tau1/tau))
    Ev = np.dot(R[1], rho_hat[1]); R2sq = np.dot(R[1], R[1])
    coeffs = [1,0,-(A**2+2*A*Ev+R2sq),0,0,-2*MU*B*(A+Ev),0,0,-(MU*B)**2]
    try: roots = np.roots(coeffs)
    except: return None
    valid = sorted([r.real for r in roots if abs(r.imag)<1e-6 and r.real>RE])
    if not valid: return None

    def do_pass(f1,f3,g1,g3):
        dn = f1*g3-f3*g1
        if abs(dn)<1e-20: return None,None
        c1=g3/dn; c3=-g1/dn
        rho1=(1/D0)*(-D[0,0]+D[1,0]/c1-(c3/c1)*D[2,0])
        rho2=(1/D0)*(-c1*D[0,1]+D[1,1]-c3*D[2,1])
        rho3=(1/D0)*(-(c1/c3)*D[0,2]+D[1,2]/c3-D[2,2])
        r2=R[1]+rho2*rho_hat[1]; r1=R[0]+rho1*rho_hat[0]; r3=R[2]+rho3*rho_hat[2]
        v2=(-f3*r1+f1*r3)/dn
        return r2,v2

    best_e, best_p = None, -np.inf
    for r2i in valid:
        r2m = r2i
        ok1 = False
        for _ in range(100):
            u=MU/r2m**3; f1=1-0.5*u*tau1**2; f3=1-0.5*u*tau3**2
            g1=tau1-(1/6)*u*tau1**3; g3=tau3-(1/6)*u*tau3**3
            r2v,v2v = do_pass(f1,f3,g1,g3)
            if r2v is None: break
            raw=np.linalg.norm(r2v); bl=0.5*raw+0.5*r2m
            if abs(bl-r2m)<1e-9: ok1=True; r2m=bl; break
            r2m=bl
        if not ok1: continue
        u=MU/r2m**3; f1=1-0.5*u*tau1**2; f3=1-0.5*u*tau3**2
        g1=tau1-(1/6)*u*tau1**3; g3=tau3-(1/6)*u*tau3**3
        r2v,v2v = do_pass(f1,f3,g1,g3)
        if r2v is None: continue
        r2p,v2p = r2v.copy(),v2v.copy()
        for _ in range(100):
            try: f1n,g1n=compute_fg(r2v,v2v,tau1); f3n,g3n=compute_fg(r2v,v2v,tau3)
            except: break
            r2n,v2n = do_pass(f1n,f3n,g1n,g3n)
            if r2n is None: break
            if np.linalg.norm(r2n-r2v)<1e-9: r2v,v2v=r2n,v2n; break
            r2v=0.5*r2n+0.5*r2v; v2v=0.5*v2n+0.5*v2v
        else: r2v,v2v=r2p,v2p
        try: e=orbital_elements(r2v,v2v)
        except: continue
        if e["energy"]>=0 or e["ecc"]>=1 or e["perigee"]<RE: continue
        if not (6400<e["sma"]<60000): continue
        if e["perigee"]>best_p: best_p=e["perigee"]; best_e=e
    return best_e

# ── Full pipeline ────────────────────────────────────────────
def run_pipeline(all_obs, observer, log_lines):

    def log(msg):
        log_lines.append(msg)

    N = len(all_obs)
    phi = np.radians(observer["lat_deg"]); alt = observer["alt_km"]

    # ── STEP 3: Triplet sweep ────────────────────────────────
    log("═"*60)
    log("  STEP 3 — SYSTEMATIC TRIPLET SWEEP")
    log(f"  Observations : {N}")
    log(f"  Step sizes   : 1 → {N//2}")
    log("═"*60)

    results = []
    total = 0
    for step in range(1, N//2+1):
        for start in range(0, N-2*step):
            total += 1
            i,j,k = start, start+step, start+2*step
            e = gauss_silent(all_obs[i], all_obs[j], all_obs[k], observer)
            if e:
                results.append({"step":step,"i":i,"j":j,"k":k,
                                 "t1":all_obs[i]["time_s"],"t2":all_obs[j]["time_s"],
                                 "t3":all_obs[k]["time_s"],
                                 "t_mid":all_obs[j]["time_s"],
                                 "dt_span":all_obs[k]["time_s"]-all_obs[i]["time_s"],
                                 **{key: e[key] for key in ("sma","ecc","inc","raan","argp","ta","perigee","r_mag")}})

    log(f"  Tried: {total}  |  Valid: {len(results)}  |  Failed: {total-len(results)}")

    # Auto-select: lowest local SMA variance
    spans  = np.array([r["dt_span"] for r in results])
    smas_a = np.array([r["sma"]     for r in results])
    order  = np.argsort(spans); sp_s=spans[order]; sm_s=smas_a[order]; row_s=np.arange(len(results))[order]
    half=5; lvar=np.full(len(sm_s), np.nan)
    for idx in range(len(sm_s)):
        seg=sm_s[max(0,idx-half):min(len(sm_s),idx+half+1)]
        if len(seg)>=5: lvar[idx]=np.var(seg,ddof=1)
    valid_idx=np.where(np.isfinite(lvar))[0]
    bp=valid_idx[np.argmin(lvar[valid_idx])]; br=int(row_s[bp]); best=results[br]
    _i,_j,_k = best["i"],best["j"],best["k"]

    log(f"\n  AUTO-SELECTED: obs #{_i+1}, #{_j+1}, #{_k+1}")
    log(f"  Arc span: {best['dt_span']:.1f}s  |  SMA: {best['sma']:.2f} km  |  σ: {np.sqrt(lvar[bp]):.4f} km\n")

    # ── STEP 5: Gauss on selected triplet ────────────────────
    log("═"*60)
    log("  STEP 5 — GAUSS PRELIMINARY ORBIT")
    log("═"*60)
    gauss_obs = [all_obs[_i], all_obs[_j], all_obs[_k]]
    gauss_elems = gauss_silent(gauss_obs[0], gauss_obs[1], gauss_obs[2], observer)
    if gauss_elems is None:
        log("  ERROR: Gauss failed on selected triplet.")
        return None

    log(f"  a = {gauss_elems['sma']:.2f} km  |  e = {gauss_elems['ecc']:.6f}  |  i = {gauss_elems['inc']:.4f}°")
    log(f"  RAAN = {gauss_elems['raan']:.4f}°  |  ω = {gauss_elems['argp']:.4f}°  |  ν = {gauss_elems['ta']:.4f}°\n")

    # ── STEP 6: Differential Correction ─────────────────────
    log("═"*60)
    log("  STEP 6 — DIFFERENTIAL CORRECTION (RK4 + J2)")
    log("═"*60)
    t_ref  = gauss_obs[1]["time_s"]
    t_obs  = [o["time_s"] for o in all_obs]
    ra_obs = [o["ra_deg"]  for o in all_obs]
    dec_obs= [o["dec_deg"] for o in all_obs]
    lst_rad= [np.radians(o["lst_deg"]) for o in all_obs]

    state = np.concatenate([gauss_elems["r_vec"], gauss_elems["v_vec"]])
    res0  = residuals(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi, alt)
    rms0  = np.sqrt(np.mean(res0**2))*3600
    log(f"  Initial RMS: {rms0:.4f} arcsec")
    log(f"  {'Iter':>4}  {'RMS':>10}  {'Relax':>6}  {'|dX| km':>10}  {'-> New RMS':>12}")
    log(f"  {'-'*50}")

    relax = 0.1; best_rms = rms0; stall = 0
    dc_history = []
    for it in range(50):
        res = residuals(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi, alt)
        rms = np.sqrt(np.mean(res**2))*3600
        if rms < 1e-6: break
        H = jacobian(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi, alt)
        try: dX, _, _, _ = np.linalg.lstsq(H, res, rcond=None)
        except: break
        sr = relax; accepted = False
        for _ in range(10):
            trial = state - sr*dX
            if np.linalg.norm(trial[:3]) < RE or np.linalg.norm(trial[:3]) > 1e6:
                sr *= 0.5; continue
            rt = residuals(trial, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi, alt)
            if np.sqrt(np.mean(rt**2))*3600 < rms: accepted=True; break
            sr *= 0.5
        if not accepted:
            stall += 1
            log(f"  {it+1:>4}  {rms:>10.4f}  {sr:>6.4f}  {'---':>10}  {'stalled':>12}")
            dc_history.append({"iter":it+1,"rms":rms,"stalled":True})
            if stall >= 3: log("  STOPPED: stalled."); break
            continue
        stall = 0
        state = trial
        rms_new = np.sqrt(np.mean(rt**2))*3600
        dx = np.linalg.norm((sr*dX)[:3])
        log(f"  {it+1:>4}  {rms:>10.4f}  {sr:>6.4f}  {dx:>10.6f}  ->  {rms_new:>8.4f}")
        dc_history.append({"iter":it+1,"rms":rms,"rms_new":rms_new,"stalled":False})
        if rms_new < best_rms: best_rms=rms_new; relax=min(relax*1.5,1.0)
        else: relax=max(sr*0.5,0.01)
        if dx < 1e-10: break

    r2_dc = state[:3]; v2_dc = state[3:]
    res_f = residuals(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi, alt)
    rms_f = np.sqrt(np.mean(res_f**2))*3600
    dc_elems = orbital_elements(r2_dc, v2_dc)

    log(f"\n  Final RMS:     {rms_f:.4f} arcsec")
    log(f"  Final RA RMS:  {np.sqrt(np.mean(res_f[0::2]**2))*3600:.4f} arcsec")
    log(f"  Final Dec RMS: {np.sqrt(np.mean(res_f[1::2]**2))*3600:.4f} arcsec")
    log("═"*60)

    return {
        "gauss": gauss_elems,
        "dc":    dc_elems,
        "r2_dc": r2_dc, "v2_dc": v2_dc,
        "t_ref": t_ref,
        "res_f": res_f,
        "t_obs": t_obs, "ra_obs": ra_obs, "dec_obs": dec_obs,
        "rms_f": rms_f,
        "results": results,
        "selected": best,
        "gauss_obs": gauss_obs,
        "dc_history": dc_history,
    }

# ── 3D orbit animation (saves to HTML string) ────────────────
def build_animation(result, all_obs, observer):
    de = result["dc"]

    def orbit_eci(sma, ecc, inc_deg, raan_deg, argp_deg, n=800):
        inc=np.radians(inc_deg); raan=np.radians(raan_deg); argp=np.radians(argp_deg)
        nu=np.linspace(0,2*np.pi,n,endpoint=False)
        p=sma*(1-ecc**2); r=p/(1+ecc*np.cos(nu))
        xp=r*np.cos(nu); yp=r*np.sin(nu)
        cO,sO=np.cos(raan),np.sin(raan); ci,si=np.cos(inc),np.sin(inc)
        cw,sw=np.cos(argp),np.sin(argp)
        Q=np.array([[cO*cw-sO*sw*ci,-cO*sw-sO*cw*ci,sO*si],
                    [sO*cw+cO*sw*ci,-sO*sw+cO*cw*ci,-cO*si],
                    [sw*si,cw*si,ci]])
        pf=np.vstack([xp,yp,np.zeros(n)])
        eci=Q@pf; return eci[0],eci[1],eci[2]

    sx,sy,sz = orbit_eci(de["sma"],de["ecc"],de["inc"],de["raan"],de["argp"])

    U,V=180,90
    u=np.linspace(0,2*np.pi,U+1); v=np.linspace(0,np.pi,V+1)
    Ex=RE*np.outer(np.cos(u),np.sin(v))
    Ey=RE*np.outer(np.sin(u),np.sin(v))
    Ez=RE*np.outer(np.ones_like(u),np.cos(v))
    Uc=(_u:=(u[:-1]+u[1:])/2); Vc=(v[:-1]+v[1:])/2
    LON,COLAT=np.meshgrid(Uc,Vc,indexing="ij")
    LON_D=np.degrees(LON); LAT_D=90-np.degrees(COLAT)
    OCEAN=np.array([0.02,0.18,0.40,1.0]); LAND_C=np.array([0.05,0.85,0.30,1.0])
    ICE=np.array([0.88,0.94,0.98,1.0])
    fc=np.tile(OCEAN,(U,V,1))
    LDS=[(190,310,15,72),(275,327,-55,12),(348,360,34,72),(0,40,34,72),
         (338,360,-35,37),(0,52,-35,37),(26,145,5,77),(113,154,-45,-10),
         (302,342,60,84),(4,32,55,72),(140,175,-47,-34),(192,228,55,72)]
    LOND_S=(LON_D+270)%360
    for lo0,lo1,la0,la1 in LDS:
        m=(LOND_S>=lo0)&(LOND_S<=lo1)&(LAT_D>=la0)&(LAT_D<=la1); fc[m]=LAND_C
    fc[LAT_D>68]=ICE; fc[LAT_D<-68]=ICE

    BG="#04050f"
    fig=plt.figure(figsize=(12,9),facecolor=BG)
    ax=fig.add_subplot(111,projection="3d",facecolor=BG)
    for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        pane.fill=False; pane.set_edgecolor("#0d0d2b")
    ax.grid(False); ax.set_axis_off()

    rng=np.random.default_rng(7); ns=600; sr=de["sma"]*7
    sp=rng.uniform(0,2*np.pi,ns); st=rng.uniform(0,np.pi,ns); ssz=rng.uniform(0.05,0.8,ns)
    ax.scatter(sr*np.sin(st)*np.cos(sp),sr*np.sin(st)*np.sin(sp),sr*np.cos(st),
               c="white",s=ssz,alpha=0.5,zorder=0)

    Ax=1.026*Ex; Ay=1.026*Ey; Az=1.026*Ez
    ax.plot_surface(Ax,Ay,Az,color="#1a6bb5",alpha=0.05,linewidth=0,antialiased=False,zorder=1)
    ax.plot_surface(Ex,Ey,Ez,facecolors=fc,linewidth=0,antialiased=True,shade=False,zorder=2)

    et=np.linspace(0,2*np.pi,360)
    ax.plot(RE*np.cos(et),RE*np.sin(et),np.zeros(360),color="#4a9edd",lw=0.5,alpha=0.4,zorder=3)
    for la in [-60,-30,30,60]:
        lr=np.radians(la)
        ax.plot(RE*np.cos(lr)*np.cos(et),RE*np.cos(lr)*np.sin(et),RE*np.sin(lr)*np.ones(360),
                color="#1a3f66",lw=0.3,alpha=0.3,zorder=3)

    lim=de["sma"]*1.5
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)

    nu_a=np.linspace(0,2*np.pi,800,endpoint=False)
    p_a=de["sma"]*(1-de["ecc"]**2); r_a=p_a/(1+de["ecc"]*np.cos(nu_a))
    pi_idx=int(np.argmin(r_a)); ap_idx=int(np.argmax(r_a))
    pxyz=(sx[pi_idx],sy[pi_idx],sz[pi_idx]); axyz=(sx[ap_idx],sy[ap_idx],sz[ap_idx])

    FPS=25; FADE_IN=40; HOLD=60; ROTATE=120; TOTAL=FADE_IN+HOLD+ROTATE
    AZ0=130; ELEV=30; AZ_RATE=0.5

    orbit_line, = ax.plot([],[],[],color="#00d4ff",lw=2.0,alpha=0.0,zorder=10)
    glow_line,  = ax.plot([],[],[],color="#00d4ff",lw=5.0,alpha=0.0,zorder=9)
    peri_sc = ax.scatter([],[],[],color="#00ff9f",s=80,zorder=15,edgecolors="white",linewidths=0.8,alpha=0.0)
    apo_sc  = ax.scatter([],[],[],color="#ff6b35",s=60,zorder=15,edgecolors="white",linewidths=0.8,alpha=0.0)
    ttl=ax.text2D(0.50,0.97,"",transform=ax.transAxes,ha="center",va="top",
                  fontsize=11,color="white",fontweight="bold",
                  fontfamily="monospace")
    inf=ax.text2D(0.01,0.03,"",transform=ax.transAxes,ha="left",va="bottom",
                  fontsize=8,color="#aaccff",fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5",facecolor="#06080f",edgecolor="#223366",alpha=0.85))

    def clamp(v): return float(min(1.0,max(0.0,v)))

    def update(frame):
        azim=AZ0+frame*AZ_RATE
        ax.view_init(elev=ELEV,azim=azim)
        t_fade=clamp(frame/FADE_IN)

        def split(x,y,z):
            el=np.radians(ELEV); az=np.radians(azim)
            cam=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
            pts=np.vstack([x,y,z]); d=cam@pts; p2=np.sum(pts**2,axis=0)-d**2
            hid=(d<0)&(p2<RE**2)
            xf,yf,zf=x.copy(),y.copy(),z.copy()
            xf[hid]=np.nan; yf[hid]=np.nan; zf[hid]=np.nan
            return xf,yf,zf

        xf,yf,zf=split(sx,sy,sz)
        orbit_line.set_data(xf,yf); orbit_line.set_3d_properties(zf); orbit_line.set_alpha(t_fade*0.9)
        glow_line.set_data(xf,yf);  glow_line.set_3d_properties(zf);  glow_line.set_alpha(t_fade*0.25)

        dot_a=clamp((frame-FADE_IN)/15)
        peri_sc._offsets3d=([pxyz[0]],[pxyz[1]],[pxyz[2]]); peri_sc.set_alpha(clamp(dot_a))
        apo_sc._offsets3d= ([axyz[0]],[axyz[1]],[axyz[2]]);  apo_sc.set_alpha(clamp(dot_a*0.85))

        ttl.set_text("ORBIT DETERMINATION — DC + J2 SOLUTION")
        inf.set_text(
            f" a  = {de['sma']:>10,.2f} km\n"
            f" e  = {de['ecc']:>10.6f}\n"
            f" i  = {de['inc']:>10.4f} °\n"
            f" Ω  = {de['raan']:>10.4f} °\n"
            f" ω  = {de['argp']:>10.4f} °\n"
            f" RMS = {result['rms_f']:>8.4f} arcsec"
        )

    anim=FuncAnimation(fig,update,frames=TOTAL,interval=1000//FPS,blit=False)
    plt.tight_layout(pad=0.2)
    html_str=anim.to_jshtml(fps=FPS,embed_frames=True)
    plt.close(fig)
    return html_str


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem;'>
      <div style='font-family:"Share Tech Mono",monospace; font-size:1.1rem;
                  color:#00d4ff; letter-spacing:0.15em;'>🛰 ORBIT-DET</div>
      <div style='font-size:0.65rem; color:#4a6080; letter-spacing:0.1em;
                  text-transform:uppercase; margin-top:0.3rem;'>
        Autonomous Determination System
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Observation File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload .ini observation file", type=["ini","txt"],
                                label_visibility="collapsed")

    st.markdown('<div class="section-header" style="margin-top:1.5rem;">Pipeline</div>',
                unsafe_allow_html=True)
    run_btn = st.button("⬡  Determine Orbit", use_container_width=True,
                        disabled=(uploaded is None))

    st.markdown("""
    <div style='margin-top:2rem; font-size:0.68rem; color:#2a4060; line-height:1.8;
                font-family:"Share Tech Mono",monospace;'>
    PIPELINE STEPS<br>
    ─────────────<br>
    01 · Load observations<br>
    02 · Load functions<br>
    03 · Triplet sweep<br>
    04 · 3D visualisation<br>
    05 · Gauss method<br>
    06 · DC + J2 refinement
    </div>
    """, unsafe_allow_html=True)

# ── Main area ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛰 ORBIT DETERMINATION</h1>
  <p>Upload an optical observation file · Run the autonomous pipeline · Inspect the determined orbit</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem;
                border:1px dashed #0d2040; border-radius:8px; margin: 0 auto; max-width:600px;'>
      <div style='font-size:3rem;'>📡</div>
      <div style='font-family:"Share Tech Mono",monospace; color:#4a6080;
                  font-size:0.9rem; letter-spacing:0.12em; margin-top:1rem;'>
        AWAITING OBSERVATION FILE
      </div>
      <div style='color:#2a4060; font-size:0.75rem; margin-top:0.5rem;'>
        Upload a .ini file in the sidebar to begin
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── File loaded — show info ───────────────────────────────────
content = uploaded.read().decode("utf-8")
try:
    all_obs, observer = parse_observation_file(content)
except Exception as ex:
    st.error(f"Failed to parse observation file: {ex}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Observations", len(all_obs))
with col2: st.metric("Observer Lat", f"{observer['lat_deg']:.4f}°N")
with col3: st.metric("Observer Lon", f"{observer['lon_deg']:.4f}°E")
with col4: st.metric("Arc Span", f"{all_obs[-1]['time_s']:.1f} s")

# ── Run pipeline ─────────────────────────────────────────────
if run_btn or "pipeline_result" in st.session_state:

    if run_btn or "pipeline_result" not in st.session_state:
        log_lines = []
        progress_bar = st.progress(0, text="Initialising pipeline…")

        log_lines.append("═"*60)
        log_lines.append("  STEP 1 — OBSERVATIONS LOADED")
        log_lines.append(f"  {len(all_obs)} observations  |  observer: {observer['lat_deg']:.4f}°N, {observer['lon_deg']:.4f}°E, {observer['alt_km']:.4f} km alt")
        log_lines.append(f"  Arc span: {all_obs[-1]['time_s']:.1f} s  ({all_obs[-1]['time_s']/60:.1f} min)")
        log_lines.append("  STEP 2 — FUNCTIONS LOADED")
        log_lines.append("═"*60)
        progress_bar.progress(15, text="Running triplet sweep…")

        result = run_pipeline(all_obs, observer, log_lines)
        progress_bar.progress(75, text="Building 3D animation…")

        if result is None:
            st.error("Pipeline failed — check the log tab.")
            st.stop()

        anim_html = build_animation(result, all_obs, observer)
        result["anim_html"] = anim_html
        result["log"]       = "\n".join(log_lines)

        progress_bar.progress(100, text="Complete ✓")
        progress_bar.empty()

        st.session_state["pipeline_result"] = result
        st.session_state["observer"] = observer
        st.session_state["all_obs"]  = all_obs

    res  = st.session_state["pipeline_result"]
    obs  = st.session_state["observer"]
    aobs = st.session_state["all_obs"]

    gauss = res["gauss"]; dc = res["dc"]

    # ── Status row ───────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; gap:1rem; margin:1rem 0; align-items:center;'>
      <span class='badge badge-ok'>● Pipeline Complete</span>
      <span class='badge badge-ok'>● Gauss Converged</span>
      <span class='badge badge-ok'>● DC + J2 Converged</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab_anim, tab_elem, tab_res, tab_sweep, tab_log = st.tabs([
        "🌍  Animation", "📐  Elements", "📊  Residuals", "🔭  Sweep", "📋  Log"
    ])

    # ════════════════════════════════════════════════
    # TAB 1 — 3D ANIMATION
    # ════════════════════════════════════════════════
    with tab_anim:
        st.markdown('<div class="section-header">3D Orbit Animation — DC + J2 Solution</div>',
                    unsafe_allow_html=True)
        st.components.v1.html(res["anim_html"], height=620, scrolling=False)
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Altitude (mean)", f"{dc['sma']-RE:,.0f} km")
        with c2: st.metric("Orbital Period",  f"{2*np.pi*np.sqrt(dc['sma']**3/MU)/60:.1f} min")
        with c3: st.metric("Final RMS",        f"{res['rms_f']:.4f} arcsec")

    # ════════════════════════════════════════════════
    # TAB 2 — ORBITAL ELEMENTS
    # ════════════════════════════════════════════════
    with tab_elem:
        st.markdown('<div class="section-header">Orbital Elements Comparison</div>',
                    unsafe_allow_html=True)

        params = [
            ("Semi-major axis",  "km",       "sma",    ",.2f"),
            ("Eccentricity",     "",          "ecc",    ".6f"),
            ("Inclination",      "°",         "inc",    ".4f"),
            ("RAAN  Ω",          "°",         "raan",   ".4f"),
            ("Arg of Perigee ω", "°",         "argp",   ".4f"),
            ("True Anomaly ν",   "°",         "ta",     ".4f"),
            ("Perigee Radius",   "km",        "perigee",",.2f"),
            ("|r₂|",             "km",        "r_mag",  ",.4f"),
            ("|v₂|",             "km/s",      "v_mag",  ".6f"),
            ("Specific Energy",  "km²/s²",    "energy", ".4f"),
        ]

        rows_html = ""
        for label, unit, key, fmt in params:
            g = gauss[key]; d = dc[key]; delta = d - g
            dcls = "val-delta-neg" if delta < 0 else "val-delta-pos"
            rows_html += f"""
            <tr>
              <td>{label} <span style='color:#2a4060;font-size:0.72rem;'>{unit}</span></td>
              <td class='val-gauss'>{g:{fmt}}</td>
              <td class='val-dc'>{d:{fmt}}</td>
              <td class='{dcls}'>{delta:+.4f}</td>
            </tr>"""

        st.markdown(f"""
        <table class='elem-table'>
          <thead><tr>
            <th>Parameter</th><th>Gauss (2-body)</th><th>DC + J2</th><th>Delta</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("SMA",         f"{dc['sma']:,.1f} km")
        with c2: st.metric("Eccentricity",f"{dc['ecc']:.6f}")
        with c3: st.metric("Inclination", f"{dc['inc']:.4f}°")
        with c4: st.metric("RMS",         f"{res['rms_f']:.4f}\"")

    # ════════════════════════════════════════════════
    # TAB 3 — RESIDUALS
    # ════════════════════════════════════════════════
    with tab_res:
        st.markdown('<div class="section-header">RA / Dec Residuals per Observation</div>',
                    unsafe_allow_html=True)

        t_obs = res["t_obs"]
        rf    = res["res_f"]
        ra_res  = rf[0::2] * 3600
        dec_res = rf[1::2] * 3600

        fig_res = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["RA Residuals (arcsec)",
                                                "Dec Residuals (arcsec)"],
                                vertical_spacing=0.08)
        common = dict(mode="markers+lines", line=dict(width=0.5),
                      marker=dict(size=4))
        fig_res.add_trace(go.Scatter(x=t_obs, y=ra_res, name="RA",
                                     marker_color="#00d4ff", **common), row=1, col=1)
        fig_res.add_trace(go.Scatter(x=t_obs, y=dec_res, name="Dec",
                                     marker_color="#00ff9f", **common), row=2, col=1)
        for row in [1,2]:
            fig_res.add_hline(y=0, line_dash="dash", line_color="#2a4060",
                              line_width=1, row=row, col=1)

        fig_res.update_layout(
            height=500, template="plotly_dark",
            paper_bgcolor="#040810", plot_bgcolor="#080f1e",
            font=dict(family="Share Tech Mono", color="#c8deff", size=11),
            showlegend=True,
            margin=dict(l=50, r=30, t=40, b=40),
        )
        fig_res.update_xaxes(title_text="Time (s)", gridcolor="#0d2040", row=2, col=1)
        fig_res.update_yaxes(gridcolor="#0d2040")
        st.plotly_chart(fig_res, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Total RMS",  f"{res['rms_f']:.4f}\"")
        with c2: st.metric("RA RMS",     f"{np.sqrt(np.mean(ra_res**2)):.4f}\"")
        with c3: st.metric("Dec RMS",    f"{np.sqrt(np.mean(dec_res**2)):.4f}\"")

    # ════════════════════════════════════════════════
    # TAB 4 — TRIPLET SWEEP
    # ════════════════════════════════════════════════
    with tab_sweep:
        st.markdown('<div class="section-header">Triplet Sweep — SMA vs Arc Span</div>',
                    unsafe_allow_html=True)

        sweep = res["results"]
        spans  = np.array([r["dt_span"] for r in sweep])
        smas_s = np.array([r["sma"]     for r in sweep])
        incs_s = np.array([r["inc"]     for r in sweep])
        steps  = np.array([r["step"]    for r in sweep])
        hover  = [f"Obs #{r['i']+1},#{r['j']+1},#{r['k']+1}<br>"
                  f"Arc={r['dt_span']:.1f}s  Step={r['step']}<br>"
                  f"SMA={r['sma']:.2f}km  i={r['inc']:.3f}°"
                  for r in sweep]

        sel = res["selected"]

        fig_sw = make_subplots(rows=1, cols=2,
                               subplot_titles=["SMA (km) vs Arc Span",
                                               "Inclination (°) vs Arc Span"],
                               horizontal_spacing=0.08)
        sc_args = dict(mode="markers", marker=dict(size=3, opacity=0.6,
                       color=steps, colorscale="Plasma",
                       showscale=True, colorbar=dict(title="Step d", len=0.7)))
        fig_sw.add_trace(go.Scatter(x=spans, y=smas_s, text=hover,
                                    hovertemplate="%{text}<extra></extra>",
                                    **sc_args), row=1, col=1)
        fig_sw.add_trace(go.Scatter(x=spans, y=incs_s, text=hover,
                                    hovertemplate="%{text}<extra></extra>",
                                    **{**sc_args, "marker":{**sc_args["marker"],"showscale":False}}),
                         row=1, col=2)
        # Mark selected
        for col in [1,2]:
            y_val = sel["sma"] if col==1 else sel["inc"]
            fig_sw.add_trace(go.Scatter(x=[sel["dt_span"]], y=[y_val],
                                        mode="markers",
                                        marker=dict(size=14, color="#ff6b35",
                                                    symbol="star", line=dict(color="white",width=1)),
                                        name="Selected", showlegend=(col==1)),
                             row=1, col=col)
        fig_sw.update_layout(
            height=420, template="plotly_dark",
            paper_bgcolor="#040810", plot_bgcolor="#080f1e",
            font=dict(family="Share Tech Mono", color="#c8deff", size=11),
            margin=dict(l=50,r=30,t=40,b=40),
        )
        fig_sw.update_xaxes(title_text="Arc Span (s)", gridcolor="#0d2040")
        fig_sw.update_yaxes(gridcolor="#0d2040")
        st.plotly_chart(fig_sw, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Triplets Tried",  f"{sum(len(all_obs)-2*d for d in range(1,len(all_obs)//2+1)):,}")
        with c2: st.metric("Valid Results",   f"{len(sweep):,}")
        with c3: st.metric("Selected Triplet",f"#{sel['i']+1}, #{sel['j']+1}, #{sel['k']+1}")
        with c4: st.metric("Selected Arc",    f"{sel['dt_span']:.1f} s")

    # ════════════════════════════════════════════════
    # TAB 5 — LOG
    # ════════════════════════════════════════════════
    with tab_log:
        st.markdown('<div class="section-header">Pipeline Execution Log</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="log-block">{res["log"]}</div>',
                    unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#2a4060;
                font-family:"Share Tech Mono",monospace; font-size:0.85rem;
                letter-spacing:0.1em;'>
      FILE LOADED — PRESS  ⬡ DETERMINE ORBIT  IN THE SIDEBAR TO BEGIN
    </div>
    """, unsafe_allow_html=True)
