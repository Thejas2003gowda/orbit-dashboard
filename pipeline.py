"""
pipeline.py — All orbital mechanics functions.
Called by agent tools in app.py.
"""

import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

MU   = 398600.4418
RE   = 6378.0
J2   = 1.08263e-3
FLAT = 0.003353

# ─────────────────────────────────────────────────────────────
# TIME UTILITIES
# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# OBSERVATION FILE PARSER
# ─────────────────────────────────────────────────────────────

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
    pending = {}; observations = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("COMMENT"): continue
        if line.startswith("ANGLE_1="):
            parts = line[8:].strip().split()
            if len(parts) < 3: continue
            ts = parts[0] + " " + parts[1]; dt = parse_ts(ts)
            if dt:
                try: pending[ts] = (dt, float(parts[2]))
                except: pass
        elif line.startswith("ANGLE_2="):
            parts = line[8:].strip().split()
            if len(parts) < 3: continue
            ts = parts[0] + " " + parts[1]; dt = parse_ts(ts)
            if dt and ts in pending:
                ra_dt, ra_deg = pending[ts]
                try:
                    observations.append({"datetime": dt, "ra_deg": ra_deg,
                                         "dec_deg": float(parts[2])})
                    del pending[ts]
                except: pass
    observations.sort(key=lambda o: o["datetime"])
    t0 = observations[0]["datetime"]
    for obs in observations:
        obs["time_s"] = (obs["datetime"] - t0).total_seconds()
        obs["lst_deg"] = gmst_to_lst(utc_to_gmst(obs["datetime"]), lon_deg)
    return observations, observer

# ─────────────────────────────────────────────────────────────
# STUMPFF & UNIVERSAL VARIABLES
# ─────────────────────────────────────────────────────────────

def stumpff_C(z):
    if z > 1e-6:    return (1 - np.cos(np.sqrt(z))) / z
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
        z = alpha * chi**2; C = stumpff_C(z); S = stumpff_S(z)
        F  = r0*vr0/np.sqrt(MU)*chi**2*C + (1-alpha*r0)*chi**3*S + r0*chi - np.sqrt(MU)*dt
        dF = r0*vr0/np.sqrt(MU)*chi*(1-alpha*chi**2*S) + (1-alpha*r0)*chi**2*C + r0
        if abs(dF) < 1e-15: break
        c2 = chi - F/dF
        if abs(c2-chi) < tol: chi = c2; break
        chi = c2
    return chi

def compute_fg(r0v, v0v, dt):
    r0 = np.linalg.norm(r0v); v0 = np.linalg.norm(v0v)
    vr0 = np.dot(r0v, v0v)/r0; alpha = 2/r0 - v0**2/MU
    chi = solve_kepler_uv(r0, vr0, alpha, dt)
    z = alpha*chi**2; C = stumpff_C(z); S = stumpff_S(z)
    return 1 - chi**2/r0*C, dt - chi**3/np.sqrt(MU)*S

# ─────────────────────────────────────────────────────────────
# ORBITAL ELEMENTS
# ─────────────────────────────────────────────────────────────

def orbital_elements(r_vec, v_vec):
    r = np.linalg.norm(r_vec); v = np.linalg.norm(v_vec)
    vr = np.dot(r_vec, v_vec)/r
    h = np.cross(r_vec, v_vec); hm = np.linalg.norm(h)
    inc = np.degrees(np.arccos(np.clip(h[2]/hm, -1, 1)))
    n = np.cross([0,0,1], h); nm = np.linalg.norm(n)
    raan = 0.0
    if nm > 1e-12:
        raan = np.degrees(np.arccos(np.clip(n[0]/nm, -1, 1)))
        if n[1] < 0: raan = 360 - raan
    e_vec = (1/MU)*((v**2 - MU/r)*r_vec - r*vr*v_vec); ecc = np.linalg.norm(e_vec)
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
    return {"sma": sma, "ecc": ecc, "inc": inc, "raan": raan, "argp": argp,
            "ta": ta, "r_mag": r, "v_mag": v, "energy": energy,
            "perigee": sma*(1-ecc), "r_vec": r_vec.copy(), "v_vec": v_vec.copy()}

# ─────────────────────────────────────────────────────────────
# RK4 + J2 PROPAGATOR
# ─────────────────────────────────────────────────────────────

def observer_pos(phi_rad, alt_km, lst_rad):
    Ce = RE/np.sqrt(1-(2*FLAT-FLAT**2)*np.sin(phi_rad)**2); Se = Ce*(1-FLAT)**2
    return np.array([(Ce+alt_km)*np.cos(phi_rad)*np.cos(lst_rad),
                     (Ce+alt_km)*np.cos(phi_rad)*np.sin(lst_rad),
                     (Se+alt_km)*np.sin(phi_rad)])

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
    sg = np.sign(dt); h = sg*min(abs(step), abs(dt)); n = int(abs(dt)/abs(h)); rem = dt - n*h
    for _ in range(n): s = rk4_step(s, h)
    if abs(rem) > 1e-12: s = rk4_step(s, rem)
    return s[:3].copy(), s[3:].copy()

def ra_dec_from_state(r_sat, R_obs):
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

# ─────────────────────────────────────────────────────────────
# RESIDUALS + JACOBIAN
# ─────────────────────────────────────────────────────────────

def residuals(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km):
    n = len(t_obs); res = np.zeros(2*n)
    for i in range(n):
        rp, _ = propagate(state[:3], state[3:], t_obs[i]-t_ref)
        Ro = observer_pos(phi_rad, alt_km, lst_rad[i])
        ra_p, dec_p = ra_dec_from_state(rp, Ro)
        res[2*i] = ang_res(ra_obs[i], ra_p); res[2*i+1] = dec_obs[i] - dec_p
    return res

def jacobian(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km):
    H = np.zeros((2*len(t_obs), 6))
    for j in range(6):
        sp, sm = state.copy(), state.copy()
        dj = 1e-4 if j < 3 else 1e-7
        sp[j] += dj; sm[j] -= dj
        rp = residuals(sp, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km)
        rm = residuals(sm, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt_km)
        H[:,j] = (rp - rm)/(2*dj)
    return H

# ─────────────────────────────────────────────────────────────
# SILENT GAUSS (for sweep)
# ─────────────────────────────────────────────────────────────

def gauss_silent(oi, oj, ok, observer):
    phi = np.radians(observer["lat_deg"]); alt = observer["alt_km"]
    t   = [o["time_s"] for o in (oi,oj,ok)]
    ra  = [np.radians(o["ra_deg"])  for o in (oi,oj,ok)]
    dec = [np.radians(o["dec_deg"]) for o in (oi,oj,ok)]
    lst = [np.radians(o["lst_deg"]) for o in (oi,oj,ok)]
    tau1=t[0]-t[1]; tau3=t[2]-t[1]; tau=tau3-tau1
    if min(abs(tau1),abs(tau3),abs(tau)) < 0.5: return None
    if min(abs(tau1),abs(tau3))/max(abs(tau1),abs(tau3)) < 0.20: return None
    rho_hat = [np.array([np.cos(dec[k])*np.cos(ra[k]),
                          np.cos(dec[k])*np.sin(ra[k]), np.sin(dec[k])]) for k in range(3)]
    Ce=RE/np.sqrt(1-(2*FLAT-FLAT**2)*np.sin(phi)**2); Se=Ce*(1-FLAT)**2
    R = [np.array([(Ce+alt)*np.cos(phi)*np.cos(lst[k]),
                   (Ce+alt)*np.cos(phi)*np.sin(lst[k]),
                   (Se+alt)*np.sin(phi)]) for k in range(3)]
    p1=np.cross(rho_hat[1],rho_hat[2]); p2=np.cross(rho_hat[0],rho_hat[2]); p3=np.cross(rho_hat[0],rho_hat[1])
    D0=np.dot(rho_hat[0],p1)
    if abs(D0)<1e-14: return None
    D=np.array([[np.dot(R[k],p) for p in (p1,p2,p3)] for k in range(3)])
    A=(1/D0)*(-D[0,1]*(tau3/tau)+D[1,1]+D[2,1]*(tau1/tau))
    B=(1/(6*D0))*(D[0,1]*(tau3**2-tau**2)*(tau3/tau)+D[2,1]*(tau**2-tau1**2)*(tau1/tau))
    Ev=np.dot(R[1],rho_hat[1]); R2sq=np.dot(R[1],R[1])
    coeffs=[1,0,-(A**2+2*A*Ev+R2sq),0,0,-2*MU*B*(A+Ev),0,0,-(MU*B)**2]
    try: roots=np.roots(coeffs)
    except: return None
    valid=sorted([r.real for r in roots if abs(r.imag)<1e-6 and r.real>RE])
    if not valid: return None

    def do_pass(f1,f3,g1,g3):
        dn=f1*g3-f3*g1
        if abs(dn)<1e-20: return None,None
        c1=g3/dn; c3=-g1/dn
        rho2=(1/D0)*(-c1*D[0,1]+D[1,1]-c3*D[2,1])
        rho1=(1/D0)*(-D[0,0]+D[1,0]/c1-(c3/c1)*D[2,0])
        rho3=(1/D0)*(-(c1/c3)*D[0,2]+D[1,2]/c3-D[2,2])
        r2=R[1]+rho2*rho_hat[1]; r1=R[0]+rho1*rho_hat[0]; r3=R[2]+rho3*rho_hat[2]
        return r2, (-f3*r1+f1*r3)/dn

    best_e, best_p = None, -np.inf
    for r2i in valid:
        r2m=r2i; ok1=False
        for _ in range(100):
            u=MU/r2m**3; f1=1-0.5*u*tau1**2; f3=1-0.5*u*tau3**2
            g1=tau1-(1/6)*u*tau1**3; g3=tau3-(1/6)*u*tau3**3
            r2v,v2v=do_pass(f1,f3,g1,g3)
            if r2v is None: break
            raw=np.linalg.norm(r2v); bl=0.5*raw+0.5*r2m
            if abs(bl-r2m)<1e-9: ok1=True; r2m=bl; break
            r2m=bl
        if not ok1: continue
        u=MU/r2m**3; f1=1-0.5*u*tau1**2; f3=1-0.5*u*tau3**2
        g1=tau1-(1/6)*u*tau1**3; g3=tau3-(1/6)*u*tau3**3
        r2v,v2v=do_pass(f1,f3,g1,g3)
        if r2v is None: continue
        r2p,v2p=r2v.copy(),v2v.copy()
        for _ in range(100):
            try: f1n,g1n=compute_fg(r2v,v2v,tau1); f3n,g3n=compute_fg(r2v,v2v,tau3)
            except: break
            r2n,v2n=do_pass(f1n,f3n,g1n,g3n)
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

# ─────────────────────────────────────────────────────────────
# STEP FUNCTIONS (called by agent tools)
# ─────────────────────────────────────────────────────────────

def step1_load(content: str):
    """Parse observation file, return summary dict."""
    obs, observer = parse_observation_file(content)
    rows = []
    for i, o in enumerate(obs[:10]):
        rows.append(f"  #{i+1:>3}  t={o['time_s']:>8.2f}s  RA={o['ra_deg']:>10.4f}°  Dec={o['dec_deg']:>10.4f}°")
    summary = (
        f"Parsed {len(obs)} observations.\n"
        f"Observer: {observer['lat_deg']:.6f}°N, {observer['lon_deg']:.6f}°E, {observer['alt_km']:.4f} km alt\n"
        f"Time span: {obs[-1]['time_s']:.1f}s ({obs[-1]['time_s']/60:.1f} min)\n\n"
        f"First 10 observations:\n" + "\n".join(rows) +
        (f"\n  ... ({len(obs)-10} more)" if len(obs) > 10 else "")
    )
    return obs, observer, summary


MAX_STEP = 20   # cap: ~1300 combos instead of 4489, ~3x faster

def step3_sweep(obs, observer, progress_callback=None):
    """Run triplet sweep up to MAX_STEP, auto-select best triplet.
    progress_callback(fraction, text) called after each step if provided.
    """
    N = len(obs); results = []; total = 0
    max_step = min(MAX_STEP, N//2)
    total_combos = sum(N - 2*d for d in range(1, max_step+1))
    done = 0
    for step in range(1, max_step+1):
        for start in range(0, N-2*step):
            total += 1; done += 1
            i,j,k = start, start+step, start+2*step
            e = gauss_silent(obs[i], obs[j], obs[k], observer)
            if e:
                results.append({"step":step,"i":i,"j":j,"k":k,
                                 "t1":obs[i]["time_s"],"t2":obs[j]["time_s"],"t3":obs[k]["time_s"],
                                 "t_mid":obs[j]["time_s"],
                                 "dt_span":obs[k]["time_s"]-obs[i]["time_s"],
                                 **{key: e[key] for key in ("sma","ecc","inc","raan","argp","ta","perigee","r_mag")}})
        if progress_callback:
            progress_callback(done/total_combos, f"Sweep: step {step}/{max_step} — {len(results)} valid so far")
    # Auto-select lowest local SMA variance
    spans=np.array([r["dt_span"] for r in results]); smas_a=np.array([r["sma"] for r in results])
    order=np.argsort(spans); sp_s=spans[order]; sm_s=smas_a[order]; row_s=np.arange(len(results))[order]
    half=5; lvar=np.full(len(sm_s),np.nan)
    for idx in range(len(sm_s)):
        seg=sm_s[max(0,idx-half):min(len(sm_s),idx+half+1)]
        if len(seg)>=5: lvar[idx]=np.var(seg,ddof=1)
    vi=np.where(np.isfinite(lvar))[0]; bp=vi[np.argmin(lvar[vi])]; br=int(row_s[bp]); best=results[br]
    summary = (
        f"Triplet sweep complete.\n"
        f"Tried: {total} | Valid: {len(results)} | Failed: {total-len(results)}\n\n"
        f"SMA stats: mean={np.mean(smas_a):.2f} km, std={np.std(smas_a):.2f} km\n"
        f"Inc stats: mean={np.mean([r['inc'] for r in results]):.4f}°\n\n"
        f"AUTO-SELECTED TRIPLET (lowest local SMA variance):\n"
        f"  Observations: #{best['i']+1}, #{best['j']+1}, #{best['k']+1}\n"
        f"  Times: t1={best['t1']:.2f}s  t2={best['t2']:.2f}s  t3={best['t3']:.2f}s\n"
        f"  Arc span: {best['dt_span']:.2f}s ({best['dt_span']/60:.2f} min)\n"
        f"  SMA: {best['sma']:.4f} km  |  σ: {np.sqrt(lvar[bp]):.4f} km"
    )
    return results, best, summary


def step5_gauss(obs, observer, selected):
    """Run full Gauss on selected triplet, return elements."""
    oi = obs[selected["i"]]; oj = obs[selected["j"]]; ok = obs[selected["k"]]
    e = gauss_silent(oi, oj, ok, observer)
    if e is None:
        return None, "Gauss failed on selected triplet."
    summary = (
        f"Gauss method converged.\n\n"
        f"  Semi-major axis : {e['sma']:>12,.4f} km\n"
        f"  Eccentricity    : {e['ecc']:>12.6f}\n"
        f"  Inclination     : {e['inc']:>12.4f} °\n"
        f"  RAAN  Ω         : {e['raan']:>12.4f} °\n"
        f"  Arg Perigee ω   : {e['argp']:>12.4f} °\n"
        f"  True Anomaly ν  : {e['ta']:>12.4f} °\n"
        f"  Perigee radius  : {e['perigee']:>12,.4f} km\n"
        f"  Speed |v2|      : {e['v_mag']:>12.6f} km/s\n"
        f"  Specific energy : {e['energy']:>12.4f} km²/s²"
    )
    return e, summary


def step6_dc(obs, observer, gauss_obs_list, gauss_elems):
    """Run differential correction with RK4+J2."""
    phi_rad = np.radians(observer["lat_deg"]); alt = observer["alt_km"]
    t_ref   = gauss_obs_list[1]["time_s"]
    t_obs   = [o["time_s"]  for o in obs]
    ra_obs  = [o["ra_deg"]  for o in obs]
    dec_obs = [o["dec_deg"] for o in obs]
    lst_rad = [np.radians(o["lst_deg"]) for o in obs]

    state = np.concatenate([gauss_elems["r_vec"], gauss_elems["v_vec"]])
    res0  = residuals(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt)
    rms0  = np.sqrt(np.mean(res0**2))*3600

    iter_log = []; relax=0.1; stall=0
    for it in range(50):
        res = residuals(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt)
        rms = np.sqrt(np.mean(res**2))*3600
        if rms < 1e-6: break
        H = jacobian(state, t_ref, t_obs, ra_obs, dec_obs, lst_rad, phi_rad, alt)
        try: dX,_,_,_ = np.linalg.lstsq(H, res, rcond=None)
        except: break
        sr=relax; accepted=False
        for _ in range(10):
            trial=state-sr*dX
            if np.linalg.norm(trial[:3])<RE or np.linalg.norm(trial[:3])>1e6: sr*=0.5; continue
            rt=residuals(trial,t_ref,t_obs,ra_obs,dec_obs,lst_rad,phi_rad,alt)
            if np.sqrt(np.mean(rt**2))*3600 < rms: accepted=True; break
            sr*=0.5
        if not accepted:
            stall+=1
            iter_log.append(f"  iter {it+1:>2}: RMS={rms:.4f}\"  stalled")
            if stall>=3: break
            continue
        stall=0; state=trial
        rms_new=np.sqrt(np.mean(rt**2))*3600; dx=np.linalg.norm((sr*dX)[:3])
        iter_log.append(f"  iter {it+1:>2}: RMS={rms:.4f}\"  relax={sr:.4f}  |dX|={dx:.6f} km  → {rms_new:.4f}\"")
        relax = min(relax*1.5,1.0) if rms_new < rms else max(sr*0.5,0.01)
        if dx < 1e-10: break

    r2_dc=state[:3]; v2_dc=state[3:]
    res_f=residuals(state,t_ref,t_obs,ra_obs,dec_obs,lst_rad,phi_rad,alt)
    rms_f=np.sqrt(np.mean(res_f**2))*3600
    dc_elems=orbital_elements(r2_dc,v2_dc)

    summary = (
        f"Differential correction complete.\n"
        f"Initial RMS: {rms0:.4f}\" → Final RMS: {rms_f:.4f}\"\n\n"
        f"Iteration log:\n" + "\n".join(iter_log) + "\n\n"
        f"FINAL ORBITAL ELEMENTS (DC + J2):\n"
        f"  Semi-major axis : {dc_elems['sma']:>12,.4f} km\n"
        f"  Eccentricity    : {dc_elems['ecc']:>12.6f}\n"
        f"  Inclination     : {dc_elems['inc']:>12.4f} °\n"
        f"  RAAN  Ω         : {dc_elems['raan']:>12.4f} °\n"
        f"  Arg Perigee ω   : {dc_elems['argp']:>12.4f} °\n"
        f"  True Anomaly ν  : {dc_elems['ta']:>12.4f} °\n"
        f"  Altitude (mean) : {dc_elems['sma']-RE:>12,.2f} km\n"
        f"  Perigee radius  : {dc_elems['perigee']:>12,.4f} km\n"
        f"  Final RMS       : {rms_f:>12.4f} arcsec"
    )
    return dc_elems, r2_dc, v2_dc, res_f, rms_f, summary


# ─────────────────────────────────────────────────────────────
# 3D ANIMATION  (Plotly — works natively in Streamlit)
# ─────────────────────────────────────────────────────────────

def build_animation(dc_elems, rms_f):
    """Build a Plotly 3D animated orbit figure — plays natively in Streamlit."""
    import plotly.graph_objects as go

    de = dc_elems

    # ── Orbit curve in ECI ───────────────────────────────────
    def orbit_eci(sma, ecc, inc_deg, raan_deg, argp_deg, n=300):
        inc=np.radians(inc_deg); raan=np.radians(raan_deg); argp=np.radians(argp_deg)
        nu=np.linspace(0, 2*np.pi, n, endpoint=False)
        p=sma*(1-ecc**2); r=p/(1+ecc*np.cos(nu))
        xp=r*np.cos(nu); yp=r*np.sin(nu)
        cO,sO=np.cos(raan),np.sin(raan)
        ci,si=np.cos(inc),np.sin(inc)
        cw,sw=np.cos(argp),np.sin(argp)
        Q=np.array([[cO*cw-sO*sw*ci,-cO*sw-sO*cw*ci,sO*si],
                    [sO*cw+cO*sw*ci,-sO*sw+cO*cw*ci,-cO*si],
                    [sw*si,cw*si,ci]])
        eci=Q@np.vstack([xp,yp,np.zeros(n)])
        return eci[0],eci[1],eci[2]

    sx,sy,sz = orbit_eci(de["sma"],de["ecc"],de["inc"],de["raan"],de["argp"])

    # ── Earth sphere ─────────────────────────────────────────
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0,   np.pi, 40)
    Ex = RE*np.outer(np.cos(u), np.sin(v))
    Ey = RE*np.outer(np.sin(u), np.sin(v))
    Ez = RE*np.outer(np.ones_like(u), np.cos(v))

    # Simple land mask using lat/lon bounding boxes
    LON_D = np.degrees(np.arctan2(Ey, Ex)) % 360
    LAT_D = np.degrees(np.arcsin(np.clip(Ez/RE, -1, 1)))
    land_color = np.zeros(Ex.shape)   # 0 = ocean, 1 = land
    for lo0,lo1,la0,la1 in [(190,310,15,72),(275,327,-55,12),(348,360,34,72),
                              (0,40,34,72),(338,360,-35,37),(0,52,-35,37),
                              (26,145,5,77),(113,154,-45,-10),(4,32,55,72),
                              (192,228,55,72),(140,175,-47,-34)]:
        m=((LON_D>=lo0)&(LON_D<=lo1)&(LAT_D>=la0)&(LAT_D<=la1))
        land_color[m]=1
    land_color[LAT_D>70]=0.5   # ice

    # ── Animation frames — satellite dot travels orbit ────────
    N_FRAMES = 60
    # satellite position at each frame
    nu_frames = np.linspace(0, 2*np.pi, N_FRAMES, endpoint=False)
    p_orb = de["sma"]*(1-de["ecc"]**2)
    r_frames = p_orb/(1+de["ecc"]*np.cos(nu_frames))
    inc=np.radians(de["inc"]); raan=np.radians(de["raan"]); argp=np.radians(de["argp"])
    cO,sO=np.cos(raan),np.sin(raan); ci,si=np.cos(inc),np.sin(inc); cw,sw=np.cos(argp),np.sin(argp)
    Q=np.array([[cO*cw-sO*sw*ci,-cO*sw-sO*cw*ci,sO*si],
                [sO*cw+cO*sw*ci,-sO*sw+cO*cw*ci,-cO*si],
                [sw*si,cw*si,ci]])
    sat_pos = Q @ np.vstack([r_frames*np.cos(nu_frames),
                              r_frames*np.sin(nu_frames),
                              np.zeros(N_FRAMES)])
    sat_x, sat_y, sat_z = sat_pos

    # Perigee / apogee
    r_all = p_orb/(1+de["ecc"]*np.cos(np.linspace(0,2*np.pi,300)))
    pi_nu = np.linspace(0,2*np.pi,300)[np.argmin(r_all)]
    ap_nu = np.linspace(0,2*np.pi,300)[np.argmax(r_all)]
    def eci_pt(nu):
        r=p_orb/(1+de["ecc"]*np.cos(nu))
        return Q @ np.array([r*np.cos(nu), r*np.sin(nu), 0])
    peri_pt = eci_pt(pi_nu); apo_pt = eci_pt(ap_nu)

    # ── Build Plotly figure ───────────────────────────────────
    lim = de["sma"] * 1.55

    # Base traces (static)
    base_traces = [
        # Earth surface
        go.Surface(x=Ex, y=Ey, z=Ez,
                   surfacecolor=land_color,
                   colorscale=[[0,"#0a2a5e"],[0.5,"#c8e6c8"],[1,"#e8f4e8"]],
                   showscale=False, opacity=1.0,
                   lighting=dict(ambient=0.7, diffuse=0.5),
                   name="Earth"),
        # Full orbit track
        go.Scatter3d(x=sx, y=sy, z=sz,
                     mode="lines",
                     line=dict(color="#00d4ff", width=3),
                     opacity=0.85, name="Orbit"),
        # Perigee
        go.Scatter3d(x=[peri_pt[0]], y=[peri_pt[1]], z=[peri_pt[2]],
                     mode="markers+text",
                     marker=dict(size=6, color="#00ff9f", symbol="diamond"),
                     text=["⬇ Perigee"], textposition="top center",
                     textfont=dict(color="#00ff9f", size=10),
                     name="Perigee"),
        # Apogee
        go.Scatter3d(x=[apo_pt[0]], y=[apo_pt[1]], z=[apo_pt[2]],
                     mode="markers+text",
                     marker=dict(size=6, color="#ff6b35", symbol="diamond"),
                     text=["⬆ Apogee"], textposition="top center",
                     textfont=dict(color="#ff6b35", size=10),
                     name="Apogee"),
    ]

    # Animation frames — satellite dot moves
    frames = []
    for k in range(N_FRAMES):
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[sat_x[k]], y=[sat_y[k]], z=[sat_z[k]],
                             mode="markers",
                             marker=dict(size=10, color="#ffffff",
                                         symbol="circle",
                                         line=dict(color="#00d4ff", width=2)),
                             name="Satellite")
            ],
            traces=[4],   # index of satellite trace
            name=str(k)
        ))

    # Initial satellite trace (frame 0)
    sat_trace = go.Scatter3d(
        x=[sat_x[0]], y=[sat_y[0]], z=[sat_z[0]],
        mode="markers",
        marker=dict(size=10, color="#ffffff", symbol="circle",
                    line=dict(color="#00d4ff", width=2)),
        name="Satellite"
    )

    fig = go.Figure(
        data=base_traces + [sat_trace],
        frames=frames
    )

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#04050f",
        scene=dict(
            bgcolor="#04050f",
            xaxis=dict(visible=False, range=[-lim,lim]),
            yaxis=dict(visible=False, range=[-lim,lim]),
            zaxis=dict(visible=False, range=[-lim,lim]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        margin=dict(l=0,r=0,t=40,b=0),
        height=580,
        title=dict(
            text=(f"<b>Orbit Determination — DC + J2</b><br>"
                  f"<sup>a={de['sma']:,.1f} km  |  e={de['ecc']:.6f}  |  "
                  f"i={de['inc']:.4f}°  |  RMS={rms_f:.4f}\"</sup>"),
            font=dict(color="#c8deff", size=13, family="monospace"),
            x=0.5
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0.02, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸  Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
            font=dict(color="#00d4ff", family="monospace"),
            bgcolor="#080f1e", bordercolor="#0d2040",
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(k)],
                        dict(mode="immediate", frame=dict(duration=80, redraw=True))],
                        label="") for k in range(N_FRAMES)],
            transition=dict(duration=0),
            x=0.05, y=0, len=0.90,
            currentvalue=dict(visible=False),
            bgcolor="#0d2040", bordercolor="#0d2040",
            tickcolor="#0d2040",
        )],
        legend=dict(font=dict(color="#c8deff", size=10), bgcolor="#080f1e",
                    bordercolor="#0d2040"),
        font=dict(color="#c8deff"),
    )

    return fig
