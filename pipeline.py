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


def step3_sweep(obs, observer):
    """Run full triplet sweep, auto-select best triplet."""
    N = len(obs); results = []; total = 0
    for step in range(1, N//2+1):
        for start in range(0, N-2*step):
            total += 1
            i,j,k = start, start+step, start+2*step
            e = gauss_silent(obs[i], obs[j], obs[k], observer)
            if e:
                results.append({"step":step,"i":i,"j":j,"k":k,
                                 "t1":obs[i]["time_s"],"t2":obs[j]["time_s"],"t3":obs[k]["time_s"],
                                 "t_mid":obs[j]["time_s"],
                                 "dt_span":obs[k]["time_s"]-obs[i]["time_s"],
                                 **{key: e[key] for key in ("sma","ecc","inc","raan","argp","ta","perigee","r_mag")}})
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
# 3D ANIMATION
# ─────────────────────────────────────────────────────────────

def build_animation(dc_elems, rms_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    de = dc_elems

    def orbit_eci(sma, ecc, inc_deg, raan_deg, argp_deg, n=800):
        inc=np.radians(inc_deg); raan=np.radians(raan_deg); argp=np.radians(argp_deg)
        nu=np.linspace(0,2*np.pi,n,endpoint=False)
        p=sma*(1-ecc**2); r=p/(1+ecc*np.cos(nu))
        xp=r*np.cos(nu); yp=r*np.sin(nu)
        cO,sO=np.cos(raan),np.sin(raan); ci,si=np.cos(inc),np.sin(inc); cw,sw=np.cos(argp),np.sin(argp)
        Q=np.array([[cO*cw-sO*sw*ci,-cO*sw-sO*cw*ci,sO*si],
                    [sO*cw+cO*sw*ci,-sO*sw+cO*cw*ci,-cO*si],[sw*si,cw*si,ci]])
        eci=Q@np.vstack([xp,yp,np.zeros(n)]); return eci[0],eci[1],eci[2]

    sx,sy,sz=orbit_eci(de["sma"],de["ecc"],de["inc"],de["raan"],de["argp"])
    U,V=120,60
    u=np.linspace(0,2*np.pi,U+1); v=np.linspace(0,np.pi,V+1)
    Ex=RE*np.outer(np.cos(u),np.sin(v)); Ey=RE*np.outer(np.sin(u),np.sin(v))
    Ez=RE*np.outer(np.ones_like(u),np.cos(v))
    Uc=(u[:-1]+u[1:])/2; Vc=(v[:-1]+v[1:])/2
    LON,COLAT=np.meshgrid(Uc,Vc,indexing="ij")
    LON_D=np.degrees(LON); LAT_D=90-np.degrees(COLAT)
    OCEAN=np.array([0.02,0.18,0.40,1.0]); LAND_C=np.array([0.05,0.75,0.30,1.0]); ICE=np.array([0.88,0.94,0.98,1.0])
    fc=np.tile(OCEAN,(U,V,1))
    for lo0,lo1,la0,la1 in [(190,310,15,72),(275,327,-55,12),(348,360,34,72),(0,40,34,72),
                             (338,360,-35,37),(0,52,-35,37),(26,145,5,77),(113,154,-45,-10),
                             (302,342,60,84),(4,32,55,72),(140,175,-47,-34),(192,228,55,72)]:
        m=((LON_D+270)%360>=lo0)&((LON_D+270)%360<=lo1)&(LAT_D>=la0)&(LAT_D<=la1); fc[m]=LAND_C
    fc[LAT_D>68]=ICE; fc[LAT_D<-68]=ICE

    BG="#04050f"
    fig=plt.figure(figsize=(11,8),facecolor=BG)
    ax=fig.add_subplot(111,projection="3d",facecolor=BG)
    for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        pane.fill=False; pane.set_edgecolor("#0d0d2b")
    ax.grid(False); ax.set_axis_off()

    rng=np.random.default_rng(7); ns=500; sr=de["sma"]*7
    sp=rng.uniform(0,2*np.pi,ns); st_=rng.uniform(0,np.pi,ns); ssz=rng.uniform(0.05,0.8,ns)
    ax.scatter(sr*np.sin(st_)*np.cos(sp),sr*np.sin(st_)*np.sin(sp),sr*np.cos(st_),c="white",s=ssz,alpha=0.45,zorder=0)

    ax.plot_surface(1.026*Ex,1.026*Ey,1.026*Ez,color="#1a6bb5",alpha=0.05,linewidth=0,antialiased=False,zorder=1)
    ax.plot_surface(Ex,Ey,Ez,facecolors=fc,linewidth=0,antialiased=True,shade=False,zorder=2)
    et=np.linspace(0,2*np.pi,360)
    ax.plot(RE*np.cos(et),RE*np.sin(et),np.zeros(360),color="#4a9edd",lw=0.5,alpha=0.4,zorder=3)

    lim=de["sma"]*1.5; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)

    nu_a=np.linspace(0,2*np.pi,800,endpoint=False)
    r_a=de["sma"]*(1-de["ecc"]**2)/(1+de["ecc"]*np.cos(nu_a))
    pi_idx=int(np.argmin(r_a)); ap_idx=int(np.argmax(r_a))
    pxyz=(sx[pi_idx],sy[pi_idx],sz[pi_idx]); axyz=(sx[ap_idx],sy[ap_idx],sz[ap_idx])

    FPS=25; FADE=40; HOLD=50; ROT=100; TOTAL=FADE+HOLD+ROT
    orbit_line,=ax.plot([],[],[],color="#00d4ff",lw=2.0,alpha=0.0,zorder=10)
    glow_line, =ax.plot([],[],[],color="#00d4ff",lw=5.0,alpha=0.0,zorder=9)
    peri_sc=ax.scatter([],[],[],color="#00ff9f",s=80,zorder=15,edgecolors="white",linewidths=0.8,alpha=0.0)
    apo_sc =ax.scatter([],[],[],color="#ff6b35",s=60,zorder=15,edgecolors="white",linewidths=0.8,alpha=0.0)
    ttl=ax.text2D(0.50,0.97,"",transform=ax.transAxes,ha="center",va="top",fontsize=11,color="white",fontweight="bold",fontfamily="monospace")
    inf=ax.text2D(0.01,0.03,"",transform=ax.transAxes,ha="left",va="bottom",fontsize=8,color="#aaccff",fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5",facecolor="#06080f",edgecolor="#223366",alpha=0.85))

    def clamp(v): return float(min(1.0,max(0.0,v)))
    def split(x,y,z,elev,azim):
        el=np.radians(elev); az=np.radians(azim)
        cam=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        pts=np.vstack([x,y,z]); d=cam@pts; hid=(d<0)&(np.sum(pts**2,axis=0)-d**2<RE**2)
        xf,yf,zf=x.copy(),y.copy(),z.copy(); xf[hid]=np.nan; yf[hid]=np.nan; zf[hid]=np.nan
        return xf,yf,zf

    def update(frame):
        azim=130+frame*0.55; ax.view_init(elev=30,azim=azim)
        t=clamp(frame/FADE)
        xf,yf,zf=split(sx,sy,sz,30,azim)
        orbit_line.set_data(xf,yf); orbit_line.set_3d_properties(zf); orbit_line.set_alpha(t*0.9)
        glow_line.set_data(xf,yf);  glow_line.set_3d_properties(zf);  glow_line.set_alpha(t*0.22)
        dot_a=clamp((frame-FADE)/15)
        peri_sc._offsets3d=([pxyz[0]],[pxyz[1]],[pxyz[2]]); peri_sc.set_alpha(clamp(dot_a))
        apo_sc._offsets3d= ([axyz[0]],[axyz[1]],[axyz[2]]);  apo_sc.set_alpha(clamp(dot_a*0.85))
        ttl.set_text("AUTONOMOUS ORBIT DETERMINATION  ·  DC + J2")
        inf.set_text(f" a  = {de['sma']:>10,.2f} km\n e  = {de['ecc']:>10.6f}\n i  = {de['inc']:>10.4f} °\n"
                     f" Ω  = {de['raan']:>10.4f} °\n ω  = {de['argp']:>10.4f} °\n RMS = {rms_f:>8.4f} \"")

    anim=FuncAnimation(fig,update,frames=TOTAL,interval=1000//FPS,blit=False)
    plt.tight_layout(pad=0.2)
    html=anim.to_jshtml(fps=FPS,embed_frames=True)
    plt.close(fig)
    return html
