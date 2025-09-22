# streamlit_app.py
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Race Edge — PI v2.4-B+ / GPI v0.95 + Hidden Horses", layout="wide")

# ------------------------
# Sidebar / Debug helpers
# ------------------------
with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

def relu(x):
    return np.maximum(0.0, x)

def winsorize(s: pd.Series, p_lo=5, p_hi=95) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    lo, hi = np.nanpercentile(s, [p_lo, p_hi])
    return s.clip(lo, hi)

def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(x)
    if sigma <= 0 or np.isnan(sigma):
        sigma = 1.0
    return (x - med) / sigma

def to_pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def round_up_200(x):
    x = int(x)
    return x if x % 200 == 0 else x + (200 - x % 200)

def sec_per_length_for_distance(distance_m: float) -> float:
    if distance_m <= 1600: return 0.14
    if distance_m <= 2000: return 0.16
    return 0.17

# ------------------------
# Robust time parsing
# ------------------------
def parse_race_time(val):
    """
    Accepts:
      • seconds:         73.30
      • M:SS.ms:         1:49.540
      • M:SS:ms|cs:      1:49:540
      • vendor numerics: 7330 (cs), 73300 (ms)
    Returns seconds (float).
    """
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            m, sec = parts
            try: return int(m) * 60 + float(sec)
            except: return np.nan
        if len(parts) == 3:
            m_str, ss_str, frac_str = parts
            try:
                m = int(m_str); ss = int(ss_str)
                if len(frac_str) >= 3: frac = int(frac_str) / 1000.0
                elif len(frac_str) == 2: frac = int(frac_str) / 100.0
                else: frac = int(frac_str) / 10.0
                return m * 60 + ss + frac
            except:
                return np.nan
    try:
        x = float(s)
    except:
        return np.nan
    if 5000 <= x < 20000:   # centiseconds
        return x / 100.0
    if 50000 <= x < 200000: # milliseconds
        return x / 1000.0
    return x

def normalize_time_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns: continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median()
        if 100 <= med < 20000:
            out[c] = out[c] / 100.0
        elif 10000 <= med < 200000:
            out[c] = out[c] / 1000.0
    return out

# ------------------------
# Manual input scaffolding
# ------------------------
def build_manual_template(distance_m: int, n_horses: int) -> pd.DataFrame:
    rows = list(range(distance_m, 0, -200))
    if rows[-1] != 0: rows.append(0)
    base = pd.DataFrame({"Marker_m": rows})
    for i in range(1, n_horses+1):
        base[f"H{i}_Pos"] = np.nan
        base[f"H{i}_Time"] = np.nan
    return base

def manual_compute_longform_to_wide(man_df: pd.DataFrame, horse_names: list, distance_m: int) -> pd.DataFrame:
    df = man_df.copy()
    out_rows = []
    for idx, name in enumerate(horse_names, start=1):
        hpos = pd.to_numeric(df.get(f"H{idx}_Pos"), errors="coerce")
        hseg = pd.to_numeric(df.get(f"H{idx}_Time"), errors="coerce")
        race_time = float(np.nansum([t for t in hseg if not pd.isna(t)])) if len(hseg) else np.nan

        def seg_time_for_span(start_m, end_m):
            try:
                start_idx = int(np.where(df["Marker_m"].values == start_m)[0][0])
            except Exception:
                return np.nan
            cnt = int((start_m - end_m) // 200)
            ts = []
            for k in range(cnt):
                if start_idx + k < len(hseg): ts.append(hseg[start_idx + k])
            if len(ts) != cnt or any(pd.isna(t) for t in ts): return np.nan
            return float(np.nansum(ts))

        mid400 = seg_time_for_span(800, 400)
        fin400 = seg_time_for_span(400, 0)

        def pos_at(marker):
            try: return float(df.loc[df["Marker_m"] == marker, f"H{idx}_Pos"].values[0])
            except Exception: return np.nan

        out_rows.append(dict(
            Horse=name,
            Finish_Pos=pos_at(0),
            RaceTime_s=race_time,
            **{"800-400": mid400 if pd.notna(mid400) else np.nan,
               "400-Finish": fin400 if pd.notna(fin400) else np.nan,
               "200_Pos": pos_at(200), "400_Pos": pos_at(400),
               "100_Time": np.nan, "200_Time": np.nan, "300_Time": np.nan, "Finish_Time": np.nan}
        ))
    return pd.DataFrame(out_rows)

# ------------------------
# Sectionals → metrics
# ------------------------
def pi_weights(distance_m: float):
    if distance_m <= 1200:
        return dict(F200=0.08, tsSPI=0.30, Accel=0.20, Grind=0.42)
    if distance_m <= 1600:
        return dict(F200=0.10, tsSPI=0.34, Accel=0.24, Grind=0.32)
    if distance_m <= 2000:
        return dict(F200=0.10, tsSPI=0.36, Accel=0.22, Grind=0.32)
    return dict(F200=0.08, tsSPI=0.38, Accel=0.18, Grind=0.36)

def build_sectionals(df: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    out = df.copy()
    if "Race Time" in out.columns:
        out["RaceTime_s"] = out["Race Time"].apply(parse_race_time)
    elif "RaceTime_s" in out.columns:
        out["RaceTime_s"] = pd.to_numeric(out["RaceTime_s"], errors="coerce")
    else:
        tcols = [c for c in out.columns if c.endswith("_Time") and c.split("_")[0].isdigit()]
        if tcols: out["RaceTime_s"] = out[tcols].sum(axis=1)

    med_rt = pd.to_numeric(out["RaceTime_s"], errors="coerce").median()
    if 5000 <= med_rt < 20000: out["RaceTime_s"] = out["RaceTime_s"] / 100.0
    elif 50000 <= med_rt < 200000: out["RaceTime_s"] = out["RaceTime_s"] / 1000.0

    out["Race_AvgSpeed"] = distance_m / out["RaceTime_s"]

    out = normalize_time_columns(out, ["800-400", "400-Finish", "100_Time", "200_Time", "300_Time", "Finish_Time"])
    out["Mid400_Speed"]   = np.where(out["800-400"].notna(), 400.0 / out["800-400"], np.nan)
    out["Final400_Speed"] = np.where(out["400-Finish"].notna(), 400.0 / out["400-Finish"], np.nan)
    out["tsSPI"]          = out["Mid400_Speed"] / out["Race_AvgSpeed"]

    # F200: earliest 200m from start (use explicit columns if present else two earliest _Time)
    cols = out.columns.tolist()
    def _first200_cols(distance_m: float, cols: list[str]):
        a = f"{int(distance_m)-100}_Time"; b = f"{int(distance_m)-200}_Time"
        if a in cols and b in cols: return a, b
        nums = sorted([int(c.split("_")[0]) for c in cols if c.endswith("_Time") and c.split("_")[0].isdigit()], reverse=True)
        if len(nums) >= 2: return f"{nums[0]}_Time", f"{nums[1]}_Time"
        return None, None
    c1, c2 = _first200_cols(distance_m, cols)
    out["F200"] = np.where(c1 and c2, (200.0 / (out[c1] + out[c2])) / out["Race_AvgSpeed"], np.nan)

    # Kick & Grind
    if {"200_Time", "300_Time"}.issubset(cols):
        sp_200_100 = 100.0 / out["200_Time"]
        sp_300_200 = 100.0 / out["300_Time"]
        out["Accel"] = sp_200_100 / sp_300_200
    else:
        out["Accel"] = np.nan

    if "Finish_Time" in cols:
        out["Grind"] = (100.0 / out["Finish_Time"]) / out["Race_AvgSpeed"]
    else:
        out["Grind"] = np.where(out["400-Finish"].notna(),
                                (200.0 / (out["400-Finish"] / 2.0)) / out["Race_AvgSpeed"], np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def compute_PI(df_sec: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    out = df_sec.copy()
    for col in ["F200", "tsSPI", "Accel", "Grind"]:
        if col in out.columns: out[col] = winsorize(out[col])

    w = pi_weights(distance_m)
    parts_list = []
    vals = []
    for _, r in out.iterrows():
        wparts = {}
        wsum = 0.0
        vsum = 0.0
        for k, ww in w.items():
            v = r.get(k, np.nan)
            if not pd.isna(v):
                wparts[k] = v * ww
                vsum += v * ww
                wsum += ww
        parts_list.append(wparts)
        vals.append(vsum / wsum if wsum else np.nan)

    out["PI_raw"] = vals
    z = robust_z(out["PI_raw"])
    k = 0.9
    out["PI"] = 10.0 * (1.0 / (1.0 + np.exp(-k * z)))

    # contribution shares
    C_F200, C_ts, C_Acc, C_Gr = [], [], [], []
    for wparts in parts_list:
        tot = sum(wparts.values())
        if tot:
            C_F200.append(wparts.get("F200", 0.0) / tot)
            C_ts.append(wparts.get("tsSPI", 0.0) / tot)
            C_Acc.append(wparts.get("Accel", 0.0) / tot)
            C_Gr.append(wparts.get("Grind", 0.0) / tot)
        else:
            C_F200.append(np.nan); C_ts.append(np.nan); C_Acc.append(np.nan); C_Gr.append(np.nan)
    out["C_F200"] = C_F200
    out["C_tsSPI"] = C_ts
    out["C_Accel"] = C_Acc
    out["C_Grind"] = C_Gr
    return out

def compute_GPI(df_sec_pi: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    out = df_sec_pi.copy()
    out["p_F200"]  = to_pct_rank(out["F200"])
    out["p_tsSPI"] = to_pct_rank(out["tsSPI"])
    out["p_Accel"] = to_pct_rank(out["Accel"])
    out["p_Grind"] = to_pct_rank(out["Grind"])
    out["p_PI"]    = to_pct_rank(out["PI"])

    pF, pM, pA, pG = out["p_F200"], out["p_tsSPI"], out["p_Accel"], out["p_Grind"]
    min4  = pd.concat([pF, pM, pA, pG], axis=1).min(axis=1)
    max4  = pd.concat([pF, pM, pA, pG], axis=1).max(axis=1)
    spread = max4 - min4

    BAL_star = relu(min4 - 0.50) / 0.50
    PEN      = 1.0 - relu(spread - 0.25) / 0.25
    BALp     = BAL_star * PEN

    S1 = 2.0 * relu(out["p_tsSPI"] - 0.50)
    S2 = 2.0 * relu(pd.concat([out["p_Accel"], out["p_Grind"]], axis=1).max(axis=1) - 0.50)
    SEPP = (S1 + S2) / 2.0

    DWB = np.where((out["p_Accel"] >= 0.75) & (out["p_Grind"] >= 0.75), 0.10,
          np.where((out["p_Accel"] >= 0.65) & (out["p_Grind"] >= 0.65), 0.05, 0.0))

    meds = {"pF": pF.median(), "pM": pM.median(), "pA": pA.median(), "pG": pG.median()}
    STRESS = np.where((pF >= meds["pF"]) & (pM >= meds["pM"]) & (pA >= meds["pA"]) & (pG >= meds["pG"]), 0.05, 0.0)

    # winner guard
    spl = sec_per_length_for_distance(distance_m)
    if "Finish_Pos" in out.columns and out["Finish_Pos"].notna().any():
        try: idx_w = out["Finish_Pos"].astype("Int64").idxmin(); winner_time = out.loc[idx_w, "RaceTime_s"]
        except Exception: winner_time = np.nan
    else:
        winner_time = np.nan
    L_behind = (out["RaceTime_s"] - winner_time) / spl if pd.notna(winner_time) else np.nan
    L_max = np.nanmax(L_behind.values) if isinstance(L_behind, pd.Series) else 0.0
    WINB = np.where(out["Finish_Pos"] == 1, np.minimum(0.10, 0.02 * np.log1p(max(L_max, 0.0))), 0.0)
    PLCB = np.where((out["Finish_Pos"] > 1) & (L_behind <= 1.0), np.maximum(0.0, 0.03 - 0.02 * L_behind), 0.0)

    # small-field guard
    N = out.shape[0]; guard = 0.8 if N <= 6 else 1.0
    BALp *= guard; SEPP *= guard; DWB *= guard; STRESS *= guard

    GPI01 = (0.15 * out["p_PI"] + 0.35 * BALp + 0.30 * SEPP + 0.10 * DWB + 0.10 * STRESS + WINB + PLCB)
    one_trick = (spread > 0.35).astype(float) * 0.03
    out["GPI"] = (np.clip(GPI01 - one_trick, 0, 1) * 10.0).round(3)
    return out

# ------------------------
# Hidden Horses: RSS + HAS + ASI (dual band)
# ------------------------
def compute_RSS_HAS_ASI(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- RSS (relative to field best per metric)
    metrics = ["F200", "tsSPI", "Accel", "Grind"]
    # ensure numeric
    for m in metrics:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce")
    field_best = {m: out[m].max() if m in out.columns else np.nan for m in metrics}
    def rss_row(r):
        vals = []
        for m in metrics:
            b = field_best.get(m, np.nan)
            v = r.get(m, np.nan)
            if pd.notna(v) and pd.notna(b) and b > 0:
                vals.append((v / b) * 100.0)
        return float(np.mean(vals)) if vals else np.nan
    out["RSS"] = out.apply(rss_row, axis=1)

    # --- pace shape (using medians)
    f200_med = out["F200"].median()
    tsspi_med = out["tsSPI"].median()
    if f200_med >= 105:
        shape = "fast"
    elif tsspi_med <= 95:
        shape = "slow"
    else:
        shape = "even"
    out["RaceShape"] = shape

    # --- HAS (>=2 checks)
    meds = {m: out[m].median() for m in metrics}
    def has_row(r):
        checks = 0
        if r["RSS"] >= 97: checks += 1
        if shape == "slow" and r["Accel"] > meds["Accel"]: checks += 1
        if shape == "fast" and r["Grind"] > meds["Grind"]: checks += 1
        if (r["Accel"] > meds["Accel"]) or (r["Grind"] > meds["Grind"]): checks += 1
        return bool(checks >= 2)
    out["HAS"] = out.apply(has_row, axis=1)

    # --- ASI (dual band)
    if shape == "fast":
        denom = out["Grind"].median()
        out["ASI"] = (out["Grind"] / denom) * 100.0
    elif shape == "slow":
        denom = out["Accel"].median()
        out["ASI"] = (out["Accel"] / denom) * 100.0
    else:
        denomK = out["Accel"].median()
        denomG = out["Grind"].median()
        out["ASI"] = 0.5 * (out["Accel"] / denomK + out["Grind"] / denomG) * 100.0

    def asi_band(v):
        if pd.isna(v): return ""
        if v >= 108: return "🔥 strong"
        if v >= 103: return "🟡 mild"
        return ""
    out["ASI_Band"] = out["ASI"].apply(asi_band)

    # concise reason for table
    def hh_reason(r):
        bits = []
        if r["HAS"]: bits.append("HAS")
        if isinstance(r["ASI_Band"], str) and r["ASI_Band"]:
            bits.append(f"ASI {r['ASI_Band']}")
        if r["RSS"] >= 97: bits.append("RSS≥97")
        return ", ".join(bits) if bits else "—"
    out["Hidden_Reason"] = out.apply(hh_reason, axis=1)

    return out

# ------------------------
# Full-race pace curve helpers
# ------------------------
RE_SEG = re.compile(r"^(\d+)_Time$")

def extract_segment_columns(df: pd.DataFrame):
    segs = []
    for c in df.columns:
        m = RE_SEG.match(str(c))
        if m:
            segs.append((int(m.group(1)), c))
    if "Finish_Time" in df.columns:
        segs.append((0, "Finish_Time"))
    if not segs: return []
    segs = sorted(segs, key=lambda x: x[0], reverse=True)
    markers = [m for m, _ in segs if m != 0]
    if len(markers) >= 2:
        diffs = np.diff(markers)
        step = int(abs(pd.Series(diffs).mode().iloc[0]))
    else:
        step = 100
    return segs, step

def build_pace_curves(df: pd.DataFrame, distance_m: float):
    seg_info = extract_segment_columns(df)
    if not seg_info:
        x = np.array([distance_m - 600, distance_m - 200])
        avg = np.array([df["Mid400_Speed"].mean(), df["Final400_Speed"].mean()])
        per = {}
        for _, r in df.iterrows():
            per[str(r.get("Horse", "Runner"))] = np.array([r.get("Mid400_Speed", np.nan), r.get("Final400_Speed", np.nan)])
        return x, avg, per

    (segs, step) = seg_info
    distances = []
    for (marker, col) in segs:
        distances.append(step if col != "Finish_Time" or len(segs) >= 2 else 100)
    x_edges = np.cumsum(distances)
    x_mid = x_edges - (np.array(distances) / 2.0)

    per = {}
    mat = []
    for _, r in df.iterrows():
        sp = []
        for (marker, col), dist in zip(segs, distances):
            t = r.get(col, np.nan)
            try: val = float(t)
            except Exception: val = np.nan
            sp.append(dist / val if pd.notna(val) and val > 0 else np.nan)
        arr = np.array(sp, dtype=float)
        per[str(r.get("Horse", "Runner"))] = arr
        mat.append(arr)
    avg = np.nanmean(np.vstack(mat), axis=0)
    return x_mid, avg, per

# ------------------------
# Narrative helpers (trip/pace/ride + ASI badges)
# ------------------------
def thresholds_for_distance(distance_m: float):
    if distance_m <= 1200:
        return dict(k_high=0.75, g_high=0.60, ts_high=0.70, low=0.35, mid=0.50)
    if distance_m <= 1600:
        return dict(k_high=0.70, g_high=0.70, ts_high=0.70, low=0.40, mid=0.50)
    if distance_m <= 2000:
        return dict(k_high=0.65, g_high=0.75, ts_high=0.70, low=0.40, mid=0.50)
    return dict(k_high=0.60, g_high=0.80, ts_high=0.70, low=0.40, mid=0.50)

def classify_trip_pace(row, thr):
    pK = float(row.get("p_Accel", np.nan))
    pG = float(row.get("p_Grind", np.nan))
    pM = float(row.get("p_tsSPI", np.nan))
    pF = float(row.get("p_F200", np.nan))

    wants = "Trip okay"
    if pG >= thr["g_high"] and (pK <= thr["mid"] or pF <= thr["mid"]):
        wants = "Wants further"
    elif pK >= thr["k_high"] and pG <= thr["low"]:
        wants = "Wants shorter"

    pace = "Pace-agnostic"
    if pM >= thr["ts_high"] and pK <= thr["mid"]:
        pace = "Wants faster early (genuine tempo)"
    elif pK >= thr["k_high"] and pM <= thr["mid"]:
        pace = "Prefers slower early (sit-and-sprint)"

    ride = "Flexible ride"
    if pF >= 0.65 and pM >= thr["mid"]:
        ride = "Forward/handy suits"
    elif pF <= 0.35 and pK >= thr["mid"]:
        ride = "Cold ride / cover then burst"

    return wants, pace, ride

def pi_gpi_note(row):
    pi = float(row.get("PI", np.nan))
    gpi = float(row.get("GPI", np.nan))
    if np.isnan(gpi): gband = "—"
    elif gpi >= 8.0: gband = "Group-class potential"
    elif gpi >= 6.0: gband = "Group-placed potential"
    elif gpi >= 4.0: gband = "Progressing / city-level"
    else: gband = "Needs to prove more"
    return f"PI {pi:.2f}; GPI {gpi:.2f} ({gband})."

def contribution_note(row):
    parts = []
    for k, col, label in [("F200","C_F200","F200"), ("tsSPI","C_tsSPI","tsSPI"), ("Accel","C_Accel","Kick"), ("Grind","C_Grind","Grind")]:
        v = row.get(col, np.nan)
        if not pd.isna(v):
            parts.append(f"{label} {v*100:.0f}%")
    return "PI mix: " + (", ".join(parts) if parts else "—") + "."

def metric_pct_line(row):
    def pct(x):
        try: return f"{x*100:.0f}%"
        except: return "—"
    return (
        f"Kick {pct(row.get('Accel'))}, Grind {pct(row.get('Grind'))}, "
        f"tsSPI {pct(row.get('tsSPI'))}, F200 {pct(row.get('F200'))}; "
        f"RSS {row.get('RSS', np.nan):.1f}; ASI {row.get('ASI', np.nan):.1f} {row.get('ASI_Band','')}"
    )

def runner_paragraph(row, distance_m: float):
    thr = thresholds_for_distance(distance_m)
    wants, pace, ride = classify_trip_pace(row, thr)
    asi_badge = f" {row.get('ASI_Band','')}" if isinstance(row.get('ASI_Band',''), str) and row.get('ASI_Band','') else ""
    notes = [
        f"**{str(row.get('Horse',''))}** — Finish: **{int(row['Finish_Pos']) if pd.notna(row.get('Finish_Pos')) else '—'}**{asi_badge}",
        pi_gpi_note(row),
        contribution_note(row),
        metric_pct_line(row),
        f"Trip: {wants}. Pace shape: {pace}. Ride: {ride}."
    ]
    return "  \n".join(notes)

# ------------------------
# UI ingest
# ------------------------
st.title("🏇 Race Edge — PI v2.4-B+ / GPI v0.95 + Hidden Horses (RSS • HAS • ASI)")
st.caption("Upload CSV/XLSX (100m splits preferred) or use Manual mode (200m grid). Adds RSS/HAS/ASI with dual-band ASI highlighting.")

with st.sidebar:
    mode = st.radio("Data Source", ["Upload file", "Manual input"], index=0)
    dist_in = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=10)
    rounded_dist = round_up_200(dist_in) if mode == "Manual input" else int(dist_in)

df_input = None
try:
    if mode == "Upload file":
        up = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
        if up is None:
            st.info("Upload a file or switch to Manual input.")
            st.stop()
        df_input = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        st.success("File loaded.")
    else:
        st.subheader("Manual Input (200m grid)")
        n_horses = st.sidebar.number_input("Number of horses", min_value=2, max_value=24, value=8, step=1)
        st.sidebar.caption(f"Grid uses distance rows from {rounded_dist} → 0 in 200m steps.")
        st.markdown("Enter horse names (one per line):")
        names_text = st.text_area("Horse names", value="\n".join([f"Horse {i}" for i in range(1, n_horses+1)]), height=150)
        horse_names = [n.strip() for n in names_text.splitlines() if n.strip()]
        if len(horse_names) != n_horses:
            st.warning("Please provide exactly the same number of horse names as 'Number of horses'.")
            st.stop()
        grid = build_manual_template(rounded_dist, n_horses)
        st.markdown("Fill **position** (1=leader) and **segment time (s)** for each 200m segment downward.")
        st.dataframe(grid, use_container_width=True)
        csv = st.text_area("Paste your filled grid here (CSV with same columns) to run analysis.", height=120)
        if not csv.strip():
            st.stop()
        try:
            man_df = pd.read_csv(io.StringIO(csv))
        except Exception:
            st.error("Could not parse the pasted CSV grid. Ensure headers and numeric values are valid.")
            st.stop()
        df_input = manual_compute_longform_to_wide(man_df, horse_names, rounded_dist)
        df_input["Race Time"] = df_input["RaceTime_s"]
except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw / Converted Table")
st.dataframe(df_input.head(12), use_container_width=True)
_dbg("Raw columns", list(df_input.columns))

# ------------------------
# Analysis pipeline
# ------------------------
distance_m = float(dist_in)
try:
    work = df_input.copy()
    for c in ["800-400", "400-Finish", "100_Time", "200_Time", "300_Time", "Finish_Time", "Finish_Pos"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    if ("Finish_Pos" not in work.columns) or work["Finish_Pos"].isna().all():
        if "Race Time" in work.columns:
            rt = work["Race Time"].apply(parse_race_time)
            work["Finish_Pos"] = rt.rank(method="min").astype("Int64")
        elif "RaceTime_s" in work.columns:
            work["Finish_Pos"] = work["RaceTime_s"].rank(method="min").astype("Int64")

    sec = build_sectionals(work, distance_m)
    pi_df = compute_PI(sec, distance_m)
    gpi_df = compute_GPI(pi_df, distance_m)
    hh_df  = compute_RSS_HAS_ASI(gpi_df)

except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# ------------------------
# Outputs: table
# ------------------------
st.subheader("Sectional Metrics (PI & GPI)")
disp_cols = ["Horse", "Finish_Pos", "RaceTime_s", "F200", "tsSPI", "Accel", "Grind", "PI", "GPI", "RSS", "ASI"]
present = [c for c in disp_cols if c in hh_df.columns]
disp = hh_df[present].copy()

for c in ["F200", "tsSPI", "Accel", "Grind"]:
    if c in disp.columns: disp[c] = (disp[c] * 100.0).round(2)
for c in ["PI", "GPI", "RSS", "ASI"]:
    if c in disp.columns: disp[c] = disp[c].round(3)

disp = disp.sort_values(["PI", "Finish_Pos"], ascending=[False, True])
st.dataframe(disp, use_container_width=True)

# ------------------------
# Visual 1: Sectional Shape Map — Kick vs Grind
# ------------------------
st.subheader("Sectional Shape Map — Kick (Accel) vs Grind")
fig, ax = plt.subplots()
x = hh_df["Accel"] * 100.0
y = hh_df["Grind"] * 100.0
ax.scatter(x, y, s=60, alpha=0.9)
if x.notna().any(): ax.axvline(x.median(), linestyle="--", alpha=0.4)
if y.notna().any(): ax.axhline(y.median(), linestyle="--", alpha=0.4)
for _, r in hh_df.iterrows():
    if pd.isna(r.get("Accel")) or pd.isna(r.get("Grind")): continue
    label = str(r.get("Horse", ""))
    ax.annotate(label,
                xy=(r["Accel"] * 100.0, r["Grind"] * 100.0),
                xytext=(5, 5), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5))
ax.set_xlabel("Kick — Acceleration (%)")
ax.set_ylabel("Grind — Late (%)")
ax.set_title("Kick vs Grind")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# ------------------------
# Visual 2: Full-Race Pace Curves — Field avg + Top 8 finishers
# ------------------------
st.subheader("Full-Race Pace Curves — Field Average (black) + Top 8 Finishers")
x_mid, avg_speed, per_horse = build_pace_curves(hh_df, distance_m)
fig2, ax2 = plt.subplots()
ax2.plot(x_mid, avg_speed, marker="o", linewidth=3, color="black", label="Average (Field)")
top8_names = hh_df.sort_values("Finish_Pos").head(8)["Horse"].astype(str).tolist()
for name in top8_names:
    if name in per_horse:
        ax2.plot(x_mid, per_horse[name], marker="o", linewidth=2, label=name)
ax2.set_xlabel("Distance from Start (m)")
ax2.set_ylabel("Segment Speed (m/s)")
ax2.set_title("Full-Race Pace Curves")
ax2.grid(True, linestyle="--", alpha=0.3)
fig2.legend(loc="lower center", ncol=min(4, len(top8_names)+1), bbox_to_anchor=(0.5, -0.18))
st.pyplot(fig2)

# ------------------------
# Visual 3: Top-8 PI stacked by contributions
# ------------------------
st.subheader("Top 8 by PI — Contribution Breakdown")
top8 = hh_df.sort_values("PI", ascending=False).head(8).copy()
for k, col in [("F200","C_F200"), ("tsSPI","C_tsSPI"), ("Kick","C_Accel"), ("Grind","C_Grind")]:
    if col in top8.columns:
        top8[f"{k}_part"] = top8[col] * top8["PI"]
    else:
        top8[f"{k}_part"] = np.nan

fig3, ax3 = plt.subplots()
labels = top8["Horse"].astype(str).tolist()[::-1]
b1 = top8["F200_part"].iloc[::-1]
b2 = top8["tsSPI_part"].iloc[::-1]
b3 = top8["Kick_part"].iloc[::-1]
b4 = top8["Grind_part"].iloc[::-1]
ax3.barh(labels, b1, label="F200")
ax3.barh(labels, b2, left=b1, label="tsSPI")
ax3.barh(labels, b3, left=(b1+b2), label="Kick")
ax3.barh(labels, b4, left=(b1+b2+b3), label="Grind")
ax3.set_xlabel("PI (0–10)")
ax3.set_title("Top 8 — PI Contribution Breakdown")
ax3.legend(loc="lower right")
st.pyplot(fig3)

# ------------------------
# NEW: Hidden Horses (RSS + HAS) + ASI band
# ------------------------
st.subheader("Hidden Horses — RSS & HAS (with ASI bands)")
hidden = hh_df.loc[hh_df["HAS"] == True].copy()
if hidden.empty:
    st.write("No hidden horses flagged today under RSS/HAS rules.")
else:
    tbl = hidden[["Horse","Finish_Pos","RSS","ASI","ASI_Band","Hidden_Reason"]].copy()
    tbl["RSS"] = tbl["RSS"].round(1)
    tbl["ASI"] = tbl["ASI"].round(1)
    tbl = tbl.sort_values(["RSS","ASI"], ascending=[False, False])
    st.dataframe(tbl, use_container_width=True)

# ------------------------
# Runner-by-Runner Analysis (with ASI badge)
# ------------------------
st.subheader("Runner-by-Runner Analysis")
ordered = hh_df.sort_values(["PI", "Finish_Pos"], ascending=[False, True]).reset_index(drop=True)
for _, row in ordered.iterrows():
    try:
        st.markdown(runner_paragraph(row, distance_m))
        st.markdown("---")
    except Exception as e:
        if DEBUG:
            st.write(f"Note generation failed for {row.get('Horse','?')}: {e}")

st.caption(
    "PI v2.4-B+: distance-aware blend of F200 / tsSPI / Kick / Grind → robust-z → logistic (0–10). "
    "GPI v0.95 estimates group potential from balance & separation with winner/place guards. "
    "Hidden Horses use RSS (vs field best) + HAS checks; ASI shows against-shape strength with dual bands: "
    "🟡 mild ≥103, 🔥 strong ≥108."
)
