# streamlit_app.py
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================================
# App config (no logo per request)
# ================================
st.set_page_config(page_title="Race Edge — Sectional Analysis", layout="wide")
st.title("🏇 Race Edge — Sectional Analysis")
st.caption("Pure sectionals: F200 • tsSPI • Acceleration • Grind • PI v2.4-B++ (CB + CRS).")

# -------------------
# Sidebar controls
# -------------------
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input type", ["Upload CSV", "Manual entry"], index=0)
    distance_m_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=10)
    # round-up to nearest 200 for manual grid & countdown labels
    def round_up_200(x): 
        x = int(round(float(x)))
        return int(np.ceil(x / 200.0) * 200)
    distance_m_grid = round_up_200(distance_m_input)

    st.caption(f"Manual grid rounds **up** to nearest 200m → **{distance_m_grid}m** (for segment rows).")

    st.divider()
    DEBUG = st.toggle("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# ================================
# Helpers: parsing & safe math
# ================================
def parse_time(val):
    """Parse seconds or 'M:SS.sss' into seconds."""
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts)==3:
                m,sec,ms = parts
                return int(m)*60 + int(sec) + int(ms)/100.0
            if len(parts)==2:
                m,sec = parts
                return int(m)*60 + float(sec)
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def safe_div(num, den):
    if isinstance(num, pd.Series):
        return num.divide(den).where((den!=0) & den.notna())
    return np.nan if (den==0 or pd.isna(den)) else (num/den)

# ================================
# Build metrics from a table
# (supports 100m inputs; falls back to 200m proxies)
# ================================
TIME_COL_RE = re.compile(r"^(\d+)(?:_Time)?$")  # matches '100' or '100_Time'

def _collect_time_cols(df):
    """Return {distance_m: column_name} for columns like '100_Time' or '100' (seconds for that 100m segment)."""
    mapping = {}
    for c in df.columns:
        m = TIME_COL_RE.match(str(c).strip())
        if m:
            try:
                d = int(m.group(1))
                mapping[d] = c
            except:
                pass
    return mapping  # per-100m segment times

def _collect_200m_cols(df):
    """Support tables with 200m segment columns named '200m','400m',... (times for each 200m segment)."""
    mapping = {}
    for c in df.columns:
        s = str(c).strip().lower().replace(" ", "").replace("m","")
        if s.isdigit():
            d = int(s)
            if d % 200 == 0:
                mapping[d] = c
    return mapping

def build_metrics(df_in: pd.DataFrame, distance_m: int):
    """Compute RaceTime, Race_AvgSpeed, F200%, tsSPI%, Accel%, Grind% from a runner table.

    Expected columns:
      - Horse (name)
      - Finish_Pos (optional)
      - Either 100m segment times: '100_Time','200_Time',... 'Finish_Time' (last 100m), or numeric '100','200',... columns with seconds
      - Fallback: 200m segment columns '200','400',... with times per 200m chunk
      - If Race Time present, we'll parse, else we'll sum segments.
    """
    w = df_in.copy()
    # Normalize horse/finish headers
    horse_col = None
    for c in ["Horse","Runner","Name"]:
        if c in w.columns: horse_col = c; break
    if horse_col is None:
        w.insert(0, "Horse", [f"Runner {i+1}" for i in range(len(w))])
        horse_col = "Horse"

    if "Finish_Pos" not in w.columns:
        # will infer later by time if possible
        w["Finish_Pos"] = np.nan

    # 100m or 200m columns
    cols_100 = _collect_time_cols(w)   # dict 100->colname
    cols_200 = _collect_200m_cols(w)   # dict 200->colname

    # Build RaceTime if missing
    if "Race Time" in w.columns:
        w["RaceTime_s"] = w["Race Time"].apply(parse_time)
    else:
        # sum available segments
        if cols_100:
            seg_cols = [cols_100.get(d) for d in sorted(cols_100.keys())]
            w["RaceTime_s"] = w[seg_cols].sum(axis=1, skipna=True)
        elif cols_200:
            seg_cols = [cols_200.get(d) for d in sorted(cols_200.keys())]
            w["RaceTime_s"] = w[seg_cols].sum(axis=1, skipna=True)
        else:
            w["RaceTime_s"] = np.nan

    # If Finish_Pos empty, infer from time
    if w["Finish_Pos"].isna().all() and w["RaceTime_s"].notna().any():
        w["Finish_Pos"] = w["RaceTime_s"].rank(method="min").astype("Int64")

    # Average race speed
    w["Race_AvgSpeed"] = safe_div(distance_m, w["RaceTime_s"])

    # -------- F200% (first 200m speed vs avg) --------
    # If 100_Time exists for first 100 and second 100 → sum; else if 200 segment exists → use; else NaN
    if cols_100:
        t100 = w[cols_100.get(100)] if 100 in cols_100 else np.nan
        t200 = w[cols_100.get(200)] if 200 in cols_100 else np.nan
        f200_time = t100.add(t200, fill_value=np.nan) if isinstance(t100, pd.Series) and isinstance(t200, pd.Series) else np.nan
    elif cols_200 and 200 in cols_200:
        f200_time = w[cols_200[200]]
    else:
        f200_time = np.nan
    f200_speed = safe_div(200.0, f200_time) if isinstance(f200_time, pd.Series) else np.nan
    w["F200%"] = safe_div(f200_speed, w["Race_AvgSpeed"]) * 100 if isinstance(f200_speed, pd.Series) else np.nan

    # -------- Grind% (last 200m vs avg; if 100m available, use last 100 only as "Grind100%") --------
    # Prefer true last-100m as grind indicator; else use last 200m.
    last_100_col = cols_100.get(distance_m) if cols_100 else None
    prev_100_col = cols_100.get(distance_m - 100) if cols_100 else None
    if last_100_col is not None:
        g100_speed = safe_div(100.0, w[last_100_col])
        w["Grind%"] = safe_div(g100_speed, w["Race_AvgSpeed"]) * 100
    else:
        # fallback to last 200m if exists
        last_200_col = cols_200.get(distance_m) if cols_200 else None
        if last_200_col is not None:
            g200_speed = safe_div(200.0, w[last_200_col])
            w["Grind%"] = safe_div(g200_speed, w["Race_AvgSpeed"]) * 100
        else:
            w["Grind%"] = np.nan

    # -------- tsSPI% (mid-race; exclude first 200 and last 400) --------
    # If 100m data: include 300..(distance-500) by 100s. If only 200m data: include 400..(distance-600) by 200s.
    if cols_100:
        mids = []
        for d in sorted(cols_100.keys()):
            if 300 <= d <= (distance_m - 500):
                mids.append(cols_100[d])
        if mids:
            mid_time = w[mids].sum(axis=1, skipna=True)
            mid_dist = 100.0 * len(mids)
            mid_speed = safe_div(mid_dist, mid_time)
            w["tsSPI%"] = safe_div(mid_speed, w["Race_AvgSpeed"]) * 100
        else:
            w["tsSPI%"] = np.nan
    elif cols_200:
        mids = []
        for d in sorted(cols_200.keys()):
            if 400 <= d <= (distance_m - 600):
                mids.append(cols_200[d])
        if mids:
            mid_time = w[mids].sum(axis=1, skipna=True)
            mid_dist = 200.0 * len(mids)
            mid_speed = safe_div(mid_dist, mid_time)
            w["tsSPI%"] = safe_div(mid_speed, w["Race_AvgSpeed"]) * 100
        else:
            w["tsSPI%"] = np.nan
    else:
        w["tsSPI%"] = np.nan

    # -------- Accel% (kick: 200→100 vs mid speed) --------
    # If 100_Time for (distance-100): use true last-100 accel vs mid; else proxy: Final200 vs mid.
    if cols_100 and prev_100_col is not None:
        a100_speed = safe_div(100.0, w[prev_100_col])  # 200->100 segment
        # need mid_speed for ratio → recompute as above
        if "tsSPI%" in w and w["tsSPI%"].notna().any():
            # recover mid_speed from tsSPI% * Race_AvgSpeed
            mid_speed = (w["tsSPI%"] / 100.0) * w["Race_AvgSpeed"]
            w["Accel%"] = safe_div(a100_speed, mid_speed) * 100
        else:
            w["Accel%"] = np.nan
    else:
        # proxy by last 200 vs mid_speed
        if cols_200 and distance_m in cols_200:
            fin200_speed = safe_div(200.0, w[cols_200[distance_m]])
            if "tsSPI%" in w and w["tsSPI%"].notna().any():
                mid_speed = (w["tsSPI%"] / 100.0) * w["Race_AvgSpeed"]
                w["Accel%"] = safe_div(fin200_speed, mid_speed) * 100
            else:
                w["Accel%"] = np.nan
        else:
            w["Accel%"] = np.nan

    return w

# ================================
# PI v2.4-B++ + CB + CRS
# ================================
def pi_weights(dist_m: float):
    d = float(dist_m or 1400)
    if d <= 1200:   return dict(F200=0.08, tsSPI=0.30, Accel=0.20, Grind=0.42)
    if d <= 1600:   return dict(F200=0.10, tsSPI=0.32, Accel=0.24, Grind=0.34)
    if d <= 2000:   return dict(F200=0.08, tsSPI=0.34, Accel=0.24, Grind=0.34)
    return            dict(F200=0.06, tsSPI=0.34, Accel=0.22, Grind=0.38)

def pct_rank(s: pd.Series):
    r = s.rank(pct=True)
    mx = r.max()
    if pd.isna(mx) or mx == 0: return r
    return r / mx

def compute_pi_core(w: pd.DataFrame, distance_m: int):
    W = pi_weights(distance_m)
    sF = pct_rank(w["F200%"])
    sM = pct_rank(w["tsSPI%"])
    sA = pct_rank(w["Accel%"])
    sG = pct_rank(w["Grind%"])
    PI = (sF*W["F200"] + sM*W["tsSPI"] + sA*W["Accel"] + sG*W["Grind"])
    return PI, sF, sM, sA, sG

def consistency_bonus(sF, sM, sA, sG, spread_thresh=0.25, cap=0.010, field_size=None):
    S = pd.concat([sF, sM, sA, sG], axis=1)
    S.columns = ["F200","tsSPI","Accel","Grind"]
    rng = (S.max(axis=1) - S.min(axis=1))
    S_min = S.min(axis=1)
    cb_raw = np.maximum(0.0, S_min - 0.55)
    cb = np.minimum(cap, cb_raw)
    cb[rng > spread_thresh] = 0.0
    if field_size is not None and field_size<=6:
        cb *= 0.7
    return cb.fillna(0.0)

def pace_trust_flag(w: pd.DataFrame):
    T = 1.0
    N = len(w)
    if N<=6: T *= 0.8
    med_accel = np.nanmedian(w["Accel%"])
    med_tsmid = np.nanmedian(w["tsSPI%"])
    if pd.notna(med_accel) and pd.notna(med_tsmid):
        if (med_accel >= 101.5) and (med_tsmid <= 99.0):
            T *= 0.7  # sprint-home suspicion
    rt = w["RaceTime_s"]
    if rt.notna().sum()>=4:
        comp = np.nanstd(rt) / np.nanmean(rt)
        if comp >= 0.06:
            T *= 0.85
    return T

def apply_crs(w: pd.DataFrame, distance_m: int, PI_core_cb: pd.Series):
    """Class/Result Safeguard: Result Anchor (winner/close placers) + soft floor for winner."""
    # Winner by Finish_Pos if present else by time
    if "Finish_Pos" in w.columns and w["Finish_Pos"].notna().any():
        idx_win = w["Finish_Pos"].idxmin()
    else:
        idx_win = w["RaceTime_s"].idxmin()
    t_win = w.loc[idx_win, "RaceTime_s"]

    # Convert time-behind to lengths (distance-aware)
    sec_per_len = 0.17 if distance_m<=1200 else (0.18 if distance_m<=1600 else (0.20 if distance_m<=2000 else 0.21))
    Lbehind = (w["RaceTime_s"] - t_win) / sec_per_len

    # Pace trust
    T = pace_trust_flag(w)

    # RA params by trip
    if distance_m<=1200: cap, scale = 0.010, 0.002
    elif distance_m<=1800: cap, scale = 0.012, 0.0025
    else: cap, scale = 0.015, 0.003

    # Winner margin (nearest positive L)
    lb_win = np.nanmin(Lbehind[Lbehind>0]) if np.any(Lbehind>0) else 0.0
    ra_win = min(cap, max(0.0, scale * lb_win)) * T

    # Build RA vector (winner + close placers within 1.5L)
    RA = pd.Series(0.0, index=w.index)
    RA.loc[idx_win] = ra_win

    # Order by finish
    if "Finish_Pos" in w.columns and w["Finish_Pos"].notna().any():
        order = w.sort_values("Finish_Pos").index.tolist()
    else:
        order = w.sort_values("RaceTime_s").index.tolist()
    for pos_i in order[1:3]:  # 2nd, 3rd
        lbh = Lbehind.loc[pos_i]
        ra_p = max(0.0, 0.004 - 0.002 * lbh) * T if pd.notna(lbh) else 0.0
        RA.loc[pos_i] = min(0.006, ra_p)

    # Winner soft floor (distance-aware)
    import math
    if distance_m<=1200:
        PI_floor = max(0.60, 0.56 + 0.010 * math.log1p(lb_win))
    elif distance_m<=1800:
        PI_floor = max(0.62, 0.57 + 0.012 * math.log1p(lb_win))
    else:
        PI_floor = max(0.64, 0.58 + 0.014 * math.log1p(lb_win))

    PI_final = (PI_core_cb + RA).copy()
    # Soft lift winner (≤ +0.015)
    PI_final.loc[idx_win] = max(PI_final.loc[idx_win], min(PI_floor, PI_final.loc[idx_win] + 0.015))
    return PI_final, RA, T, lb_win

# ================================
# Runner narratives
# ================================
def short_note(row, med_tsspi, med_accel, med_grind):
    name = str(row["Horse"])
    f200 = row.get("F200%", np.nan)
    tsm  = row.get("tsSPI%", np.nan)
    acc  = row.get("Accel%", np.nan)
    grd  = row.get("Grind%", np.nan)
    pi   = row.get("PI_final", np.nan)
    fin  = row.get("Finish_Pos", np.nan)

    tags = []
    if pd.notna(acc) and acc > np.nanpercentile(metrics["Accel%"], 75): tags.append("Kick")
    if pd.notna(grd) and grd > np.nanpercentile(metrics["Grind%"], 75): tags.append("Grind")
    if pd.notna(tsm) and tsm > np.nanpercentile(metrics["tsSPI%"], 75): tags.append("Cruise")
    if pd.notna(f200) and f200 > np.nanpercentile(metrics["F200%"], 75): tags.append("Gate")

    shape_bits = []
    if pd.notna(tsm) and tsm > med_tsspi + 1.0: shape_bits.append("strong mid-race engine")
    if pd.notna(acc) and acc > med_accel + 1.0: shape_bits.append("sharp late kick")
    if pd.notna(grd) and grd > med_grind + 1.0: shape_bits.append("kept grinding late")

    if not shape_bits:
        shape = "balanced profile"
    else:
        shape = ", ".join(shape_bits)

    tag_str = f" **[{', '.join(tags)}]**" if tags else ""
    fpos = int(fin) if pd.notna(fin) else "—"
    return f"**{name}** — Finish **{fpos}**, PI **{pi:.3f}**{tag_str}. Sectional shape: {shape}."

# ================================
# INPUTS
# ================================
df_raw = None

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (with per-100m or 200m segment times)", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV, or switch to Manual entry.")
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded)
        st.success("File loaded.")
    except Exception as e:
        st.error("Could not read CSV.")
        _dbg("CSV error", e)
        st.stop()

else:
    st.subheader("Manual Entry")
    st.caption("Enter horse names and segment times. Grid counts **down** from the rounded up race distance, in 200m steps.")
    n_horses = st.number_input("Number of horses", min_value=2, max_value=24, value=12, step=1)

    # Build countdown labels at 200m steps from distance_m_grid to 200
    steps = list(range(distance_m_grid, 0, -200))
    # We will store per-200m segment times as seconds under columns like '200','400',... counting down.
    # Race Time will be the sum; Finish_Pos inferred by time.

    # Build an empty data template
    cols = ["Horse"] + [f"{d}" for d in steps]  # times for each 200m chunk (seconds)
    data = []
    for i in range(int(n_horses)):
        row = {"Horse": f"Runner {i+1}"}
        for c in cols[1:]:
            row[c] = np.nan
        data.append(row)
    df_edit = pd.DataFrame(data, columns=cols)

    st.write("**Enter 200m split times (seconds) for each runner:**")
    df_filled = st.data_editor(
        df_edit,
        use_container_width=True,
        num_rows="dynamic"
    )

    # Convert to analysis schema
    # Sum 200m splits to RaceTime, map per-200m to 200m columns that build_metrics understands
    if df_filled is not None and len(df_filled) > 0:
        work = df_filled.copy()
        # ensure numeric for time cols
        for c in cols[1:]:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work["Race Time"] = work[cols[1:]].sum(axis=1, skipna=True)

        # rename 200m columns into numeric headers (e.g., '200','400',...) that _collect_200m_cols will detect
        # Note: our labels are already numeric strings counting down; OK.
        df_raw = work.copy()
        st.info("Manual grid converted. Finish order will be inferred by total time.")
    else:
        st.stop()

# ================================
# Show raw
# ================================
st.subheader("Raw / Converted Table")
st.dataframe(df_raw.head(12), use_container_width=True)

# ================================
# Compute metrics & PI
# ================================
try:
    metrics = build_metrics(df_raw, distance_m=int(distance_m_input))
except Exception as e:
    st.error("Metric computation failed.")
    _dbg("metrics error", e)
    st.stop()

# Infer Finish_Pos if still missing
if ("Finish_Pos" not in metrics.columns) or metrics["Finish_Pos"].isna().all():
    if metrics["RaceTime_s"].notna().any():
        metrics["Finish_Pos"] = metrics["RaceTime_s"].rank(method="min").astype("Int64")

# PI core
PI_core, sF, sM, sA, sG = compute_pi_core(metrics, int(distance_m_input))
CB = consistency_bonus(sF, sM, sA, sG, field_size=len(metrics))
PI_core_cb = PI_core + CB
PI_final, RA_vec, T_factor, win_margin_L = apply_crs(metrics, int(distance_m_input), PI_core_cb)

metrics["PI_core"] = PI_core
metrics["CB"] = CB
metrics["RA"] = RA_vec
metrics["PI_final"] = PI_final

# ================================
# Sectional Metrics table (new system only)
# ================================
st.subheader("Sectional Metrics (new system)")
disp = metrics[["Horse","Finish_Pos","RaceTime_s","F200%","tsSPI%","Accel%","Grind%","PI_final"]].copy()
# round for display
for c in ["RaceTime_s","F200%","tsSPI%","Accel%","Grind%","PI_final"]:
    if c in disp.columns:
        disp[c] = disp[c].astype(float).round(3)
st.dataframe(disp.sort_values("PI_final", ascending=False), use_container_width=True)

# ================================
# Visual 1: Kick vs Grind Scatter
# ================================
st.subheader("Sectional Shape Map — Kick vs Grind")
fig, ax = plt.subplots()
x = metrics["Accel%"]
y = metrics["Grind%"]
size = 200 * (metrics["PI_final"] - metrics["PI_final"].min()) / (metrics["PI_final"].ptp() + 1e-9) + 60
ax.scatter(x, y, s=size, alpha=0.7, edgecolors="k", linewidths=0.5)

# Label top 8 by PI with arrows (declutter)
top8_idx = metrics["PI_final"].rank(ascending=False, method="min") <= 8
for _, r in metrics[top8_idx].iterrows():
    ax.annotate(r["Horse"], xy=(r["Accel%"], r["Grind%"]), xytext=(5,5), textcoords="offset points", fontsize=9)

ax.axvline(np.nanmedian(x), linestyle="--", alpha=0.2)
ax.axhline(np.nanmedian(y), linestyle="--", alpha=0.2)
ax.set_xlabel("Kick (Accel%)")
ax.set_ylabel("Grind (last-200 vs avg)")
ax.set_title("Kick vs Grind — bubble size = PI")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig, use_container_width=True)

# ==========================
# VISUALS: Radar + Top-8 Bars
# ==========================
def _col(df, opts):
    for o in opts:
        if o in df.columns: return o
    return None

COL_HORSE   = _col(metrics, ["Horse","Runner","Name"])
COL_FINPOS  = _col(metrics, ["Finish_Pos","Finish","Placing"])
COL_PI      = _col(metrics, ["PI_final","PI_new","PI","PI_core+CB","PI_baseline"])
COL_F200    = _col(metrics, ["F200%","F200_pct","F200"])
COL_TSSPI   = _col(metrics, ["tsSPI%","tsSPI_pct","tsSPI"])
COL_ACCEL   = _col(metrics, ["Accel%","Acceleration%","Accel_pct","Accel"])
COL_GRIND   = _col(metrics, ["Grind%","Grind_pct","Grind"])

_req = [COL_HORSE, COL_PI, COL_F200, COL_TSSPI, COL_ACCEL, COL_GRIND]
if any(c is None for c in _req):
    st.warning("Radar/Contribution charts: required columns not found (need Horse, PI, F200%, tsSPI%, Accel%, Grind%).")
else:
    show_viz = st.checkbox("📈 Show advanced visuals (Radar per horse + Top-8 PI breakdown)", value=True)
    if show_viz:
        # weights by trip (same as PI)
        W = pi_weights(distance_m_input)

        def pct_rank_local(s: pd.Series):
            r = s.rank(pct=True)
            mx = r.max()
            if pd.isna(mx) or mx == 0: return r
            return r / mx

        viz = metrics[[COL_HORSE, COL_PI, COL_F200, COL_TSSPI, COL_ACCEL, COL_GRIND]].copy()
        viz.columns = ["Horse","PI","F200","tsSPI","Accel","Grind"]

        P = pd.DataFrame({
            "F200":  pct_rank_local(viz["F200"]),
            "tsSPI": pct_rank_local(viz["tsSPI"]),
            "Accel": pct_rank_local(viz["Accel"]),
            "Grind": pct_rank_local(viz["Grind"]),
        })
        spread = (P.max(axis=1) - P.min(axis=1))
        P["Eff"] = (1.0 - spread).clip(0, 1)

        # ---- Radar (solo) ----
        st.subheader("🎯 Radar — per horse sectional fingerprint")
        if COL_FINPOS:
            order_idx = metrics.sort_values(COL_FINPOS, na_position="last").index
        else:
            order_idx = viz.sort_values("PI", ascending=False).index
        horse_order = viz.loc[order_idx, "Horse"].tolist()
        selected = st.selectbox("Pick a horse", horse_order, index=0)

        idx = viz.index[viz["Horse"] == selected][0]
        labels = ["F200","tsSPI","Accel","Grind","Eff"]
        vals   = P.loc[idx, labels].values.astype(float)
        avg_vals = P[labels].mean().values.astype(float)

        def _radar(ax, values, ref, tickstep=0.25):
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
            values = np.r_[values, values[0]]
            ref    = np.r_[ref,    ref[0]]
            angles = np.r_[angles, angles[0]]

            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.01, tickstep))
            ax.plot(angles, ref, linewidth=2, linestyle="--", alpha=0.6)
            ax.fill(angles, ref, alpha=0.08)
            ax.plot(angles, values, linewidth=3)
            ax.fill(angles, values, alpha=0.22)

        fig_r, ax_r = plt.subplots(subplot_kw=dict(polar=True))
        _radar(ax_r, vals, avg_vals, tickstep=0.25)
        ax_r.set_title(f"{selected} — sectional fingerprint vs race average", pad=16)
        st.pyplot(fig_r, use_container_width=True)

        # ---- Top-8 Contribution Bars ----
        st.subheader("🧱 PI breakdown — Top 8 by PI")
        contrib = pd.DataFrame({
            "Horse": viz["Horse"],
            "F200":  pct_rank_local(viz["F200"])  * W["F200"],
            "tsSPI": pct_rank_local(viz["tsSPI"]) * W["tsSPI"],
            "Accel": pct_rank_local(viz["Accel"]) * W["Accel"],
            "Grind": pct_rank_local(viz["Grind"]) * W["Grind"],
            "PI":    viz["PI"]
        })
        top8 = contrib.sort_values("PI", ascending=False).head(8).reset_index(drop=True)

        x = np.arange(len(top8))
        fig_b, ax_b = plt.subplots()
        bottom = np.zeros(len(top8))
        for key in ["F200","tsSPI","Accel","Grind"]:
            ax_b.bar(x, top8[key].values, bottom=bottom, label=key)
            bottom += top8[key].values

        ax_b.set_xticks(x)
        ax_b.set_xticklabels(top8["Horse"], rotation=20, ha="right")
        ax_b.set_ylabel("PI contribution (weighted)")
        ax_b.set_title("Top-8 PI — sectional contribution breakdown")
        ax_b.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax_b.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        st.pyplot(fig_b, use_container_width=True)

        st.caption(
            "Reading tips: taller bars = higher PI. Colors show which phases contributed most. "
            "Big 'Grind' → strong late sustain; big 'Accel' → sharp kick; big 'tsSPI' → mid-race engine; "
            "F200 shows early gate speed. Radar above shows the same horse’s fingerprint (incl. consistency)."
        )

# ================================
# Runner-by-runner summaries
# ================================
st.subheader("Runner-by-runner notes")
med_tsspi = np.nanmedian(metrics["tsSPI%"])
med_accel = np.nanmedian(metrics["Accel%"])
med_grind = np.nanmedian(metrics["Grind%"])
for _, row in metrics.sort_values("Finish_Pos", na_position="last").iterrows():
    st.markdown(short_note(row, med_tsspi, med_accel, med_grind))
    st.markdown("---")
