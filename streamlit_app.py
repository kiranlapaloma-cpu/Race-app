# streamlit_app.py
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================
# App config (logo-free for simplicity)
# =========================================
st.set_page_config(page_title="Race Edge — Sectional Analysis", layout="wide")
st.title("🏇 Race Edge — Sectional Analysis")
st.caption("Pure sectionals: F200 • tsSPI • Acceleration • Grind • PI v2.4-B++ (Consistency Bonus + Class/Result Safeguard).")

# -----------------------------------------
# Sidebar
# -----------------------------------------
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input type", ["Upload CSV", "Manual entry"], index=0)
    distance_m_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=10)

    def round_up_200(x: float) -> int:
        x = int(round(float(x)))
        return int(np.ceil(x / 200.0) * 200)

    distance_m_grid = round_up_200(distance_m_input)
    st.caption(f"Manual grid rounds **up** to nearest 200m → **{distance_m_grid}m** (for grid labels).")

    st.divider()
    DEBUG = st.toggle("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# =========================================
# Helpers
# =========================================
# Accepts: '100', '100_Time', '100 Time', '1400', '1400_Time', '1400 Time', etc.
TIME_COL_RE = re.compile(r"^(\d+)(?:[_\s]?Time)?$", flags=re.IGNORECASE)

def parse_time(val):
    """Parse seconds or 'M:SS.sss' → seconds."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                m, sec, ms = parts
                return int(m) * 60 + int(sec) + int(ms) / 100.0
            if len(parts) == 2:
                m, sec = parts
                return int(m) * 60 + float(sec)
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def safe_div(num, den):
    """
    Safe division that works for scalars or Series in any combination.
    Returns Series when either input is a Series.
    """
    num_is_s = isinstance(num, pd.Series)
    den_is_s = isinstance(den, pd.Series)

    if num_is_s or den_is_s:
        if not num_is_s and den_is_s:
            num = pd.Series(float(num), index=den.index)
        if num_is_s and not den_is_s:
            den = pd.Series(float(den), index=num.index)
        num = num.astype(float)
        den = den.astype(float)
        out = num / den
        out[(den == 0) | den.isna()] = np.nan
        return out
    # both scalars
    if pd.isna(den) or den == 0:
        return np.nan
    return float(num) / float(den)

def _collect_time_cols(df):
    """
    Return {int -> colname} for 100m split columns with flexible headers.
    Also map 'Finish_Time'/'Finish' to key 0 (final 100 m) for countdown files.
    """
    mapping = {}
    for c in df.columns:
        c_norm = str(c).strip()
        lc = c_norm.lower().replace(" ", "_")
        if lc in ("finish_time", "finish"):
            mapping[0] = c
            continue
        m = TIME_COL_RE.match(c_norm)
        if m:
            try:
                d = int(m.group(1))
                mapping[d] = c
            except:
                pass
    return mapping

def _collect_200m_cols(df):
    """
    Return {int -> colname} for 200m splits (forward or countdown numeric headers),
    e.g., '200','400',... or '1600','1400',...,'0'
    """
    mapping = {}
    for c in df.columns:
        s = str(c).strip().lower().replace(" ", "").replace("m", "")
        if s.isdigit():
            d = int(s)
            if d % 200 == 0:
                mapping[d] = c
    return mapping

def build_metrics(df_in: pd.DataFrame, distance_m: int):
    """
    Compute RaceTime_s, Race_AvgSpeed, F200%, tsSPI%, Accel%, Grind% from either:
      • 100m splits (forward '100','200',... or countdown '1400','1300',...,'0'/Finish_Time)
      • 200m splits (forward '200','400',... or countdown 'distance-200',...,'0')
    """
    w = df_in.copy()
    w.columns = [str(c).strip() for c in w.columns]

    if DEBUG:
        st.write("🔧 build_metrics: distance_m =", distance_m)
        st.write("🔧 build_metrics: sample columns =", list(w.columns)[:20])

    # Horse id
    horse_col = None
    for c in ["Horse", "Runner", "Name"]:
        if c in w.columns:
            horse_col = c
            break
    if horse_col is None:
        w.insert(0, "Horse", [f"Runner {i+1}" for i in range(len(w))])
        horse_col = "Horse"

    if "Finish_Pos" not in w.columns:
        w["Finish_Pos"] = np.nan

    # Collect splits
    cols_100 = _collect_time_cols(w)
    cols_200 = _collect_200m_cols(w)

    def resolve_direction(keys, dm):
        if not keys:
            return "unknown"
        ks = sorted(keys)
        has_zero = (0 in keys)
        has_dm = (dm in keys)
        has_100 = (100 in keys)
        if has_zero or (abs(max(ks) - max(dm - 100, 0)) <= 20 and not has_dm):
            return "countdown"
        if has_100:
            return "forward"
        return "forward"

    dir100 = resolve_direction(set(cols_100.keys()), distance_m) if cols_100 else "unknown"
    dir200 = resolve_direction(set(cols_200.keys()), distance_m) if cols_200 else "unknown"

    def get100(d_forward):
        if not cols_100:
            return None
        return cols_100.get(d_forward) if dir100 == "forward" else cols_100.get(distance_m - d_forward)

    def get200(d_forward):
        if not cols_200:
            return None
        return cols_200.get(d_forward) if dir200 == "forward" else cols_200.get(distance_m - d_forward)

    # Race time (sum of splits if not present)
    if "Race Time" in w.columns:
        w["RaceTime_s"] = w["Race Time"].apply(parse_time)
    else:
        if cols_100:
            seg_cols = [cols_100[k] for k in sorted(cols_100.keys())]
            w["RaceTime_s"] = w[seg_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        elif cols_200:
            seg_cols = [cols_200[k] for k in sorted(cols_200.keys())]
            w["RaceTime_s"] = w[seg_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        else:
            w["RaceTime_s"] = np.nan

    # Infer finish pos by time if missing
    if w["Finish_Pos"].isna().all() and w["RaceTime_s"].notna().any():
        w["Finish_Pos"] = w["RaceTime_s"].rank(method="min").astype("Int64")

    # Average speed (row-wise safe division)
    w["Race_AvgSpeed"] = safe_div(distance_m, w["RaceTime_s"])

    # ---------- F200% (first 200 vs avg) ----------
    f100a, f100b = get100(100), get100(200)
    if f100a and f100b:
        f200_time = pd.to_numeric(w[f100a], errors="coerce") + pd.to_numeric(w[f100b], errors="coerce")
    else:
        f200_col = get200(200)
        f200_time = pd.to_numeric(w[f200_col], errors="coerce") if f200_col else pd.Series(np.nan, index=w.index)
    f200_speed = safe_div(200.0, f200_time)
    w["F200%"] = safe_div(f200_speed, w["Race_AvgSpeed"]) * 100.0

    # ---------- Grind% (last 100 preferred; else last 200) ----------
    last100_col = get100(distance_m)
    if not last100_col:
        finish_like = [c for c in w.columns if str(c).strip().lower().replace(" ", "_") in ("finish_time", "finish")]
        if finish_like:
            last100_col = finish_like[0]
    if last100_col:
        g100_speed = safe_div(100.0, pd.to_numeric(w[last100_col], errors="coerce"))
        w["Grind%"] = safe_div(g100_speed, w["Race_AvgSpeed"]) * 100.0
    else:
        last200_col = get200(distance_m)
        if last200_col:
            g200_speed = safe_div(200.0, pd.to_numeric(w[last200_col], errors="coerce"))
            w["Grind%"] = safe_div(g200_speed, w["Race_AvgSpeed"]) * 100.0
        else:
            w["Grind%"] = np.nan

    # ---------- tsSPI% (exclude first 200 & last 400) ----------
    if cols_100:
        mids = []
        for d in range(300, max(distance_m - 400, 0) + 1, 100):
            col = get100(d)
            if col:
                mids.append(col)
        if mids:
            mid_time = w[mids].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            mid_dist = 100.0 * len(mids)
            mid_speed = safe_div(mid_dist, mid_time)
            w["tsSPI%"] = safe_div(mid_speed, w["Race_AvgSpeed"]) * 100.0
        else:
            w["tsSPI%"] = np.nan
    elif cols_200:
        mids = []
        for d in range(400, max(distance_m - 600, 0) + 1, 200):
            col = get200(d)
            if col:
                mids.append(col)
        if mids:
            mid_time = w[mids].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            mid_dist = 200.0 * len(mids)
            mid_speed = safe_div(mid_dist, mid_time)
            w["tsSPI%"] = safe_div(mid_speed, w["Race_AvgSpeed"]) * 100.0
        else:
            w["tsSPI%"] = np.nan
    else:
        w["tsSPI%"] = np.nan

    # ---------- Accel% (200→100 vs mid; else last 200 vs mid) ----------
    prev100_col = get100(distance_m - 100)
    if prev100_col:
        a100_speed = safe_div(100.0, pd.to_numeric(w[prev100_col], errors="coerce"))
        mid_speed = safe_div(w["tsSPI%"], 100.0) * w["Race_AvgSpeed"]
        w["Accel%"] = safe_div(a100_speed, mid_speed) * 100.0
    else:
        last200_col = get200(distance_m)
        if last200_col:
            fin200_speed = safe_div(200.0, pd.to_numeric(w[last200_col], errors="coerce"))
            mid_speed = safe_div(w["tsSPI%"], 100.0) * w["Race_AvgSpeed"]
            w["Accel%"] = safe_div(fin200_speed, mid_speed) * 100.0
        else:
            w["Accel%"] = np.nan

    # If nothing usable, raise a clear error
    needed = ["F200%", "tsSPI%", "Accel%", "Grind%"]
    if all((col not in w.columns) or w[col].isna().all() for col in needed):
        raise ValueError(
            "No usable split columns detected. Expected 100m headers like '100_Time'/'100 Time'/100 "
            "or 200m headers like '200','400',... (forward) or countdown with 'Finish_Time'/0."
        )
    return w

# =========================================
# PI v2.4-B++ + Consistency Bonus + CRS
# =========================================
def pi_weights(dist_m: float):
    d = float(dist_m or 1400)
    if d <= 1200:   return dict(F200=0.08, tsSPI=0.30, Accel=0.20, Grind=0.42)
    if d <= 1600:   return dict(F200=0.10, tsSPI=0.32, Accel=0.24, Grind=0.34)
    if d <= 2000:   return dict(F200=0.08, tsSPI=0.34, Accel=0.24, Grind=0.34)
    return            dict(F200=0.06, tsSPI=0.34, Accel=0.22, Grind=0.38)

def pct_rank(s: pd.Series):
    r = s.rank(pct=True)
    mx = r.max()
    if pd.isna(mx) or mx == 0:
        return r
    return r / mx

def compute_pi_core(w: pd.DataFrame, distance_m: int):
    W = pi_weights(distance_m)
    sF = pct_rank(w["F200%"])
    sM = pct_rank(w["tsSPI%"])
    sA = pct_rank(w["Accel%"])
    sG = pct_rank(w["Grind%"])
    PI = (sF * W["F200"] + sM * W["tsSPI"] + sA * W["Accel"] + sG * W["Grind"])
    return PI, sF, sM, sA, sG

def consistency_bonus(sF, sM, sA, sG, spread_thresh=0.25, cap=0.010, field_size=None):
    S = pd.concat([sF, sM, sA, sG], axis=1)
    S.columns = ["F200", "tsSPI", "Accel", "Grind"]
    rng = (S.max(axis=1) - S.min(axis=1))
    S_min = S.min(axis=1)
    cb_raw = np.maximum(0.0, S_min - 0.55)
    cb = np.minimum(cap, cb_raw)
    cb[rng > spread_thresh] = 0.0
    if field_size is not None and field_size <= 6:
        cb *= 0.7
    return cb.fillna(0.0)

def pace_trust_flag(w: pd.DataFrame):
    T = 1.0
    N = len(w)
    if N <= 6:
        T *= 0.8
    med_accel = np.nanmedian(w["Accel%"])
    med_tsspi = np.nanmedian(w["tsSPI%"])
    if pd.notna(med_accel) and pd.notna(med_tsspi):
        if (med_accel >= 101.5) and (med_tsspi <= 99.0):
            T *= 0.7  # sprint-home suspicion
    rt = w["RaceTime_s"]
    if rt.notna().sum() >= 4:
        comp = np.nanstd(rt) / np.nanmean(rt)
        if comp >= 0.06:
            T *= 0.85
    return T

def apply_crs(w: pd.DataFrame, distance_m: int, PI_core_cb: pd.Series):
    if "Finish_Pos" in w.columns and w["Finish_Pos"].notna().any():
        idx_win = w["Finish_Pos"].idxmin()
    else:
        idx_win = w["RaceTime_s"].idxmin()
    t_win = w.loc[idx_win, "RaceTime_s"]

    sec_per_len = 0.17 if distance_m <= 1200 else (0.18 if distance_m <= 1600 else (0.20 if distance_m <= 2000 else 0.21))
    Lbehind = (w["RaceTime_s"] - t_win) / sec_per_len

    T = pace_trust_flag(w)

    if distance_m <= 1200:
        cap, scale = 0.010, 0.002
    elif distance_m <= 1800:
        cap, scale = 0.012, 0.0025
    else:
        cap, scale = 0.015, 0.003

    lb_win = np.nanmin(Lbehind[Lbehind > 0]) if np.any(Lbehind > 0) else 0.0
    ra_win = min(cap, max(0.0, scale * lb_win)) * T

    RA = pd.Series(0.0, index=w.index)
    RA.loc[idx_win] = ra_win

    if "Finish_Pos" in w.columns and w["Finish_Pos"].notna().any():
        order = w.sort_values("Finish_Pos").index.tolist()
    else:
        order = w.sort_values("RaceTime_s").index.tolist()
    for pos_i in order[1:3]:
        lbh = Lbehind.loc[pos_i]
        ra_p = max(0.0, 0.004 - 0.002 * lbh) * T if pd.notna(lbh) else 0.0
        RA.loc[pos_i] = min(0.006, ra_p)

    import math
    if distance_m <= 1200:
        PI_floor = max(0.60, 0.56 + 0.010 * math.log1p(lb_win))
    elif distance_m <= 1800:
        PI_floor = max(0.62, 0.57 + 0.012 * math.log1p(lb_win))
    else:
        PI_floor = max(0.64, 0.58 + 0.014 * math.log1p(lb_win))

    PI_final = (PI_core_cb + RA).copy()
    PI_final.loc[idx_win] = max(PI_final.loc[idx_win], min(PI_floor, PI_final.loc[idx_win] + 0.015))
    return PI_final, RA, T, lb_win

# -----------------------------------------
# Runner notes (compact)
# -----------------------------------------
def short_note(row, med_tsspi, med_accel, med_grind):
    name = str(row["Horse"])
    f200 = row.get("F200%", np.nan)
    tsm  = row.get("tsSPI%", np.nan)
    acc  = row.get("Accel%", np.nan)
    grd  = row.get("Grind%", np.nan)
    pi   = row.get("PI_final", np.nan)
    fin  = row.get("Finish_Pos", np.nan)

    tags = []
    # These percentiles are computed later on full metrics; guard if missing
    try:
        if pd.notna(acc) and acc > np.nanpercentile(metrics["Accel%"], 75): tags.append("Kick")
        if pd.notna(grd) and grd > np.nanpercentile(metrics["Grind%"], 75): tags.append("Grind")
        if pd.notna(tsm) and tsm > np.nanpercentile(metrics["tsSPI%"], 75): tags.append("Cruise")
        if pd.notna(f200) and f200 > np.nanpercentile(metrics["F200%"], 75): tags.append("Gate")
    except Exception:
        pass

    shape_bits = []
    if pd.notna(tsm) and tsm > med_tsspi + 1.0: shape_bits.append("strong mid-race engine")
    if pd.notna(acc) and acc > med_accel + 1.0: shape_bits.append("sharp late kick")
    if pd.notna(grd) and grd > med_grind + 1.0: shape_bits.append("kept grinding late")
    shape = "balanced profile" if not shape_bits else ", ".join(shape_bits)

    tag_str = f" **[{', '.join(tags)}]**" if tags else ""
    fpos = int(fin) if pd.notna(fin) else "—"
    return f"**{name}** — Finish **{fpos}**, PI **{pi:.3f}**{tag_str}. Sectional shape: {shape}."

# =========================================
# INPUTS
# =========================================
df_raw = None

if source == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload CSV (per-100m or 200m splits). Headers may be forward ('100','200',...) or countdown ('1400','1300',...,'0'/'Finish_Time').",
        type=["csv"]
    )
    if not uploaded:
        st.info("Upload a CSV, or switch to Manual entry.")
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        st.success("File loaded.")
    except Exception as e:
        st.error("Could not read CSV.")
        _dbg("CSV error", e)
        st.stop()
else:
    st.subheader("Manual Entry")
    st.caption("Enter horse names and **200m split times (seconds)**. Grid counts **down** from the rounded-up race distance.")
    n_horses = st.number_input("Number of horses", min_value=2, max_value=24, value=12, step=1)

    steps = list(range(distance_m_grid, 0, -200))  # countdown labels
    cols = ["Horse"] + [f"{d}" for d in steps]
    data = []
    for i in range(int(n_horses)):
        row = {"Horse": f"Runner {i+1}"}
        for c in cols[1:]:
            row[c] = np.nan
        data.append(row)
    df_edit = pd.DataFrame(data, columns=cols)

    st.write("**Enter 200m split times (seconds) for each runner):**")
    df_filled = st.data_editor(df_edit, use_container_width=True, num_rows="dynamic")

    if df_filled is not None and len(df_filled) > 0:
        work = df_filled.copy()
        for c in cols[1:]:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work["Race Time"] = work[cols[1:]].sum(axis=1, skipna=True)
        df_raw = work.copy()
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        st.info("Manual grid converted. Finish order will be inferred by total time.")
    else:
        st.stop()

# =========================================
# Show raw / converted
# =========================================
st.subheader("Raw / Converted Table")
st.dataframe(df_raw.head(12), use_container_width=True)
_dbg("Columns in df_raw", list(df_raw.columns))

# =========================================
# Compute metrics & PI
# =========================================
try:
    metrics = build_metrics(df_raw, distance_m=int(distance_m_input))
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG:
        st.exception(e)
    st.stop()

if ("Finish_Pos" not in metrics.columns) or metrics["Finish_Pos"].isna().all():
    if metrics["RaceTime_s"].notna().any():
        metrics["Finish_Pos"] = metrics["RaceTime_s"].rank(method="min").astype("Int64")

PI_core, sF, sM, sA, sG = compute_pi_core(metrics, int(distance_m_input))
CB = consistency_bonus(sF, sM, sA, sG, field_size=len(metrics))
PI_core_cb = PI_core + CB
PI_final, RA_vec, T_factor, win_margin_L = apply_crs(metrics, int(distance_m_input), PI_core_cb)

metrics["PI_core"]   = PI_core
metrics["CB"]        = CB
metrics["RA"]        = RA_vec
metrics["PI_final"]  = PI_final

# =========================================
# Sectional Metrics table (new-only)
# =========================================
st.subheader("Sectional Metrics")
disp = metrics[["Horse","Finish_Pos","RaceTime_s","F200%","tsSPI%","Accel%","Grind%","PI_final"]].copy()
for c in ["RaceTime_s","F200%","tsSPI%","Accel%","Grind%","PI_final"]:
    if c in disp.columns:
        disp[c] = pd.to_numeric(disp[c], errors="coerce").astype(float).round(3)
st.dataframe(disp.sort_values("PI_final", ascending=False), use_container_width=True)

# =========================================
# Visual 1: Sectional Shape Map (Kick vs Grind)
# =========================================
st.subheader("Sectional Shape Map — Kick vs Grind")
if metrics["Accel%"].notna().any() and metrics["Grind%"].notna().any():
    fig, ax = plt.subplots()
    x = metrics["Accel%"]
    y = metrics["Grind%"]
    span = float(metrics["PI_final"].max() - metrics["PI_final"].min()) if metrics["PI_final"].notna().any() else 0.0
    size = 60.0 if span == 0 else (200 * (metrics["PI_final"] - metrics["PI_final"].min()) / (span + 1e-9) + 60)
    ax.scatter(x, y, s=size, alpha=0.7, edgecolors="k", linewidths=0.5)

    # annotate top 8 by PI
    top8_idx = metrics["PI_final"].rank(ascending=False, method="min") <= 8
    for _, r in metrics[top8_idx].iterrows():
        if pd.notna(r["Accel%"]) and pd.notna(r["Grind%"]):
            ax.annotate(r["Horse"], xy=(r["Accel%"], r["Grind%"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.axvline(np.nanmedian(x), linestyle="--", alpha=0.2)
    ax.axhline(np.nanmedian(y), linestyle="--", alpha=0.2)
    ax.set_xlabel("Kick (Accel%)")
    ax.set_ylabel("Grind (last-200 vs avg)")
    ax.set_title("Kick vs Grind — bubble size = PI")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Not enough non-NaN values to plot the shape map.")

# =========================================
# Visual 2: Pace curve (avg + Top-8 finishers) over 200m
# =========================================
st.subheader("Pace Curve — average (black) + Top-8 finishers")

def extract_200m_speeds(df_src: pd.DataFrame, distance_m: int):
    local = df_src.copy()
    local.columns = [str(c).strip() for c in local.columns]
    cols_100 = _collect_time_cols(local)
    cols_200 = _collect_200m_cols(local)

    def dir_of(keys):
        if not keys:
            return "unknown"
        ks = sorted(keys)
        return "countdown" if (0 in keys or (abs(max(ks) - max(distance_m - 100, 0)) <= 20 and (distance_m not in keys))) else "forward"

    dir100 = dir_of(set(cols_100.keys())) if cols_100 else "unknown"
    dir200 = dir_of(set(cols_200.keys())) if cols_200 else "unknown"

    def get100(d_forward):
        if not cols_100: return None
        return cols_100.get(d_forward) if dir100 == "forward" else cols_100.get(distance_m - d_forward)

    def get200(d_forward):
        if not cols_200: return None
        return cols_200.get(d_forward) if dir200 == "forward" else cols_200.get(distance_m - d_forward)

    xs = list(range(200, distance_m + 1, 200))
    seg_speeds = pd.DataFrame(index=local.index)
    for d in xs:
        c200 = get200(d)
        if c200:
            t200 = pd.to_numeric(local[c200], errors="coerce")
        else:
            c100a, c100b = get100(d - 100), get100(d)
            if c100a and c100b:
                t200 = pd.to_numeric(local[c100a], errors="coerce") + pd.to_numeric(local[c100b], errors="coerce")
            else:
                t200 = pd.Series(np.nan, index=local.index)
        seg_speeds[d] = safe_div(200.0, t200)  # m/s
    return np.array(xs), seg_speeds

xs, seg_speed = extract_200m_speeds(df_raw, int(distance_m_input))
if seg_speed.notna().sum().sum() > 0:
    avg_speed = seg_speed.mean(axis=0).values
    fig_pc, ax_pc = plt.subplots()
    ax_pc.plot(xs, avg_speed, linewidth=3, color="black", marker="o", label="Average (Field)")

    # Top 8 by finish (fallback: by PI)
    if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
        top_idxs = metrics.sort_values("Finish_Pos").index[:8]
    else:
        top_idxs = metrics.sort_values("PI_final", ascending=False).index[:8]

    for idx in top_idxs:
        nm = str(metrics.loc[idx, "Horse"])
        yv = seg_speed.loc[idx, xs].values
        if np.isfinite(yv).any():
            ax_pc.plot(xs, yv, linewidth=2, marker="o", label=nm)

    ax_pc.set_xlabel("Distance (m)")
    ax_pc.set_ylabel("Speed (m/s)")
    ax_pc.set_title("Pace curve by 200m")
    ax_pc.grid(True, linestyle="--", alpha=0.3)
    ax_pc.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    st.pyplot(fig_pc, use_container_width=True)
else:
    st.info("Not enough split data to draw the pace curve.")

# =========================================
# Visual 3: Radar + Top-8 Contribution Bars
# =========================================
def _col(df, opts):
    for o in opts:
        if o in df.columns:
            return o
    return None

COL_HORSE   = _col(metrics, ["Horse", "Runner", "Name"])
COL_FINPOS  = _col(metrics, ["Finish_Pos", "Finish", "Placing"])
COL_PI      = _col(metrics, ["PI_final", "PI_new", "PI", "PI_core+CB", "PI_baseline"])
COL_F200    = _col(metrics, ["F200%", "F200_pct", "F200"])
COL_TSSPI   = _col(metrics, ["tsSPI%", "tsSPI_pct", "tsSPI"])
COL_ACCEL   = _col(metrics, ["Accel%", "Acceleration%", "Accel_pct", "Accel"])
COL_GRIND   = _col(metrics, ["Grind%", "Grind_pct", "Grind"])

_req = [COL_HORSE, COL_PI, COL_F200, COL_TSSPI, COL_ACCEL, COL_GRIND]
if any(c is None for c in _req):
    st.warning("Radar/Contribution charts: required columns not found (need Horse, PI, F200%, tsSPI%, Accel%, Grind%).")
else:
    show_viz = st.checkbox("📈 Show advanced visuals (Radar per horse + Top-8 PI breakdown)", value=True)
    if show_viz:
        W = pi_weights(distance_m_input)

        def pct_rank_local(s: pd.Series):
            r = s.rank(pct=True)
            mx = r.max()
            if pd.isna(mx) or mx == 0:
                return r
            return r / mx

        viz = metrics[[COL_HORSE, COL_PI, COL_F200, COL_TSSPI, COL_ACCEL, COL_GRIND]].copy()
        viz.columns = ["Horse", "PI", "F200", "tsSPI", "Accel", "Grind"]

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
        if COL_FINPOS and metrics[COL_FINPOS].notna().any():
            order_idx = metrics.sort_values(COL_FINPOS, na_position="last").index
        else:
            order_idx = viz.sort_values("PI", ascending=False).index
        horse_order = viz.loc[order_idx, "Horse"].tolist()
        selected = st.selectbox("Pick a horse", horse_order, index=0)

        idx = viz.index[viz["Horse"] == selected][0]
        labels = ["F200", "tsSPI", "Accel", "Grind", "Eff"]
        vals = P.loc[idx, labels].values.astype(float)
        avg_vals = P[labels].mean().values.astype(float)

        def _radar(ax, values, ref, tickstep=0.25):
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            values = np.r_[values, values[0]]
            ref = np.r_[ref, ref[0]]
            angles = np.r_[angles, angles[0]]
            ax.set_theta_offset(np.pi / 2)
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

        # ---- Top-8 Contribution Bars (sort toggle & winner highlight) ----
        st.subheader("🧱 PI breakdown — Top-8 (toggle sort & % labels)")
        sort_by = st.radio("Sort bars by:", ["PI", "Finish"], horizontal=True)
        show_pct = st.checkbox("Show % labels inside stacks", value=False)

        contrib = pd.DataFrame({
            "Horse": viz["Horse"],
            "F200":  pct_rank_local(viz["F200"])  * W["F200"],
            "tsSPI": pct_rank_local(viz["tsSPI"]) * W["tsSPI"],
            "Accel": pct_rank_local(viz["Accel"]) * W["Accel"],
            "Grind": pct_rank_local(viz["Grind"]) * W["Grind"],
            "PI":    viz["PI"]
        })

        if sort_by == "Finish" and COL_FINPOS and metrics[COL_FINPOS].notna().any():
            order_idx2 = metrics.sort_values(COL_FINPOS).index
        else:
            order_idx2 = contrib.sort_values("PI", ascending=False).index

        contrib = contrib.loc[order_idx2]
        top8 = contrib.head(8).reset_index(drop=True)

        winner_name = None
        if COL_FINPOS and metrics[COL_FINPOS].notna().any():
            idx_w = metrics[COL_FINPOS].idxmin()
            winner_name = str(metrics.loc[idx_w, COL_HORSE])

        x = np.arange(len(top8))
        fig_b, ax_b = plt.subplots()
        bottom = np.zeros(len(top8))
        segments = ["F200", "tsSPI", "Accel", "Grind"]

        for key in segments:
            b = ax_b.bar(x, top8[key].values, bottom=bottom, label=key)
            if show_pct:
                total = (top8[segments].sum(axis=1)).values
                pct_vals = np.divide(top8[key].values, total, out=np.zeros_like(top8[key].values), where=total > 0) * 100.0
                for rect, pct in zip(b, pct_vals):
                    if pct > 8.0:
                        ax_b.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                                  f"{pct:.0f}%", ha="center", va="center", fontsize=8)
            bottom += top8[key].values

        if winner_name is not None:
            try:
                win_pos = top8.index[top8["Horse"] == winner_name].tolist()
                if win_pos:
                    i = win_pos[0]
                    total_h = (top8[segments].sum(axis=1)).iloc[i]
                    ax_b.add_patch(plt.Rectangle((i - 0.5 + 0.05, 0), 0.90, total_h, fill=False, linewidth=2.0))
            except Exception:
                pass

        ax_b.set_xticks(x)
        ax_b.set_xticklabels(top8["Horse"], rotation=20, ha="right")
        ax_b.set_ylabel("PI contribution (weighted)")
        ttl_sort = "PI" if sort_by == "PI" else "Finish"
        ax_b.set_title(f"Top-8 — sectional contribution breakdown (sorted by {ttl_sort})")
        ax_b.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax_b.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        st.pyplot(fig_b, use_container_width=True)

        st.caption(
            "Reading tips: taller bars = higher PI. Colors show which phases contributed most. "
            "Winner has a thin outline. Toggle sorting to compare ability (PI) vs finishing order."
        )

# =========================================
# Runner-by-runner notes
# =========================================
st.subheader("Runner-by-runner notes")
med_tsspi = np.nanmedian(metrics["tsSPI%"])
med_accel = np.nanmedian(metrics["Accel%"])
med_grind = np.nanmedian(metrics["Grind%"])
for _, row in metrics.sort_values("Finish_Pos", na_position="last").iterrows():
    st.markdown(short_note(row, med_tsspi, med_accel, med_grind))
    st.markdown("---")
