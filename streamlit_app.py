# streamlit_app.py
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Race Edge — PI v2.4-B+ & GPI v0.95", layout="wide")

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

# ------------------------
# Generic utils
# ------------------------
def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(x)
    if sigma <= 0 or np.isnan(sigma):
        sigma = 1.0
    return (x - med) / sigma

def winsorize(s: pd.Series, p_lo=5, p_hi=95) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    lo, hi = np.nanpercentile(s, [p_lo, p_hi])
    return s.clip(lo, hi)

def to_pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def relu(x):
    return np.maximum(0.0, x)

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
    Robust parser for race times. Accepts:
      • seconds:         73.30
      • M:SS.ms:         1:49.540
      • M:SS:ms|cs:      1:49:540   (milliseconds/centiseconds/tenths)
      • vendor numerics: 7330 (centiseconds), 73300 (milliseconds)
    Returns seconds (float).
    """
    if pd.isna(val): return np.nan
    s = str(val).strip()

    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            m, sec = parts
            try:
                return int(m) * 60 + float(sec)
            except Exception:
                return np.nan
        if len(parts) == 3:
            m_str, ss_str, frac_str = parts
            try:
                m = int(m_str); ss = int(ss_str)
                if len(frac_str) >= 3: frac = int(frac_str) / 1000.0
                elif len(frac_str) == 2: frac = int(frac_str) / 100.0
                else: frac = int(frac_str) / 10.0
                return m * 60 + ss + frac
            except Exception:
                return np.nan

    try:
        x = float(s)
    except Exception:
        return np.nan

    if 5000 <= x < 20000:   # centiseconds
        return x / 100.0
    if 50000 <= x < 200000: # milliseconds
        return x / 1000.0
    return x

def normalize_time_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Downscale vendor centiseconds/milliseconds to seconds where needed."""
    out = df.copy()
    for c in cols:
        if c not in out.columns: continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median()
        if 100 <= med < 20000:        # likely centiseconds
            out[c] = out[c] / 100.0
        elif 10000 <= med < 200000:   # likely milliseconds
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
        base[f"H{i}_Time"] = np.nan  # per-200m segment time
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

    # Race time from "Race Time" when present
    if "Race Time" in out.columns:
        out["RaceTime_s"] = out["Race Time"].apply(parse_race_time)
    elif "RaceTime_s" in out.columns:
        out["RaceTime_s"] = pd.to_numeric(out["RaceTime_s"], errors="coerce")
    else:
        tcols = [c for c in out.columns if c.endswith("_Time") and c.split("_")[0].isdigit()]
        if tcols: out["RaceTime_s"] = out[tcols].sum(axis=1)

    # Downscale vendor numerics if needed
    med_rt = pd.to_numeric(out["RaceTime_s"], errors="coerce").median()
    if 5000 <= med_rt < 20000: out["RaceTime_s"] = out["RaceTime_s"] / 100.0
    elif 50000 <= med_rt < 200000: out["RaceTime_s"] = out["RaceTime_s"] / 1000.0

    out["Race_AvgSpeed"] = distance_m / out["RaceTime_s"]

    # Normalize common window columns
    out = normalize_time_columns(out, ["800-400", "400-Finish", "100_Time", "200_Time", "300_Time", "Finish_Time"])

    # Derived speeds
    out["Mid400_Speed"]   = np.where(out["800-400"].notna(), 400.0 / out["800-400"], np.nan)
    out["Final400_Speed"] = np.where(out["400-Finish"].notna(), 400.0 / out["400-Finish"], np.nan)
    out["tsSPI"]          = out["Mid400_Speed"] / out["Race_AvgSpeed"]

    # F200 from the race start (find two earliest segment columns)
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
    """
    PI v2.4-B+; also returns contribution shares (C_F200, C_tsSPI, C_Accel, C_Grind) for stacked bar.
    """
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

    # contribution shares (sum to 1 when possible)
    C_F200, C_ts, C_Acc, C_Gr = [], [], [], []
    for wparts in parts_list:
        tot = sum(wparts.values())
        if tot and tot != 0:
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
    """GPI v0.95 — only added as a column (no visuals)."""
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
# Full-race pace curve helpers
# ------------------------
RE_SEG = re.compile(r"^(\d+)_Time$")

def extract_segment_columns(df: pd.DataFrame):
    """
    Find per-segment time columns like '1400_Time', '1300_Time', ..., '100_Time', 'Finish_Time'.
    Returns list of (start_marker_m, colname) sorted from race start → finish.
    """
    segs = []
    for c in df.columns:
        m = RE_SEG.match(str(c))
        if m:
            segs.append((int(m.group(1)), c))
    # include 'Finish_Time' as 0→finish segment if present
    if "Finish_Time" in df.columns:
        segs.append((0, "Finish_Time"))

    if not segs: return []

    # sort from largest marker (near the start) to 0
    segs = sorted(segs, key=lambda x: x[0], reverse=True)

    # infer segment length (100 or 200)
    markers = [m for m, _ in segs if m != 0]
    if len(markers) >= 2:
        diffs = np.diff(markers)
        step = int(abs(pd.Series(diffs).mode().iloc[0]))
    else:
        step = 100  # default

    # Build x positions from 0→distance in meters covered
    # Convert markers (distance-from-finish style) to distance-from-start edges
    # We want speeds plotted at segment midpoints along 0..distance
    return segs, step

def build_pace_curves(df: pd.DataFrame, distance_m: float):
    """
    Returns:
      x_m (array of segment midpoints from 0..distance),
      avg_speed (field average),
      per_horse (dict name -> speeds array)
    Works with 100m or 200m splits when present; falls back to 2-point curve otherwise.
    """
    seg_info = extract_segment_columns(df)
    if not seg_info:
        # fallback to mid vs final 2-point curve
        x = np.array([distance_m - 600, distance_m - 200])  # just to draw something monotonic
        avg = np.array([df["Mid400_Speed"].mean(), df["Final400_Speed"].mean()])
        per = {}
        for _, r in df.iterrows():
            per[str(r.get("Horse", "Runner"))] = np.array([r.get("Mid400_Speed", np.nan), r.get("Final400_Speed", np.nan)])
        return x, avg, per

    (segs, step) = seg_info
    # Build speeds per segment (distance / time)
    distances = []
    for i, (marker, col) in enumerate(segs):
        if col == "Finish_Time":
            dist = step if len(segs) >= 2 else 100
        else:
            # segment covers [marker, marker-step]
            dist = step
        distances.append(dist)

    # x positions = cumulative from start, at segment midpoints
    x_edges = np.cumsum(distances)
    x_mid = x_edges - (np.array(distances) / 2.0)

    # speeds per runner
    per = {}
    mat = []
    for _, r in df.iterrows():
        sp = []
        for (marker, col), dist in zip(segs, distances):
            t = r.get(col, np.nan)
            try:
                val = float(t)
            except Exception:
                val = np.nan
            sp.append(dist / val if pd.notna(val) and val > 0 else np.nan)
        per[str(r.get("Horse", "Runner"))] = np.array(sp, dtype=float)
        mat.append(per[str(r.get("Horse", "Runner"))])

    avg = np.nanmean(np.vstack(mat), axis=0)
    return x_mid, avg, per

# ------------------------
# UI ingest
# ------------------------
st.title("🏇 Race Edge — PI v2.4-B+ & GPI v0.95")
st.caption("Upload CSV/XLSX (100m splits preferred) or use Manual mode (200m grid). Calculates sectionals, PI (0–10), and GPI (0–10).")

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

except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# ------------------------
# Outputs: table
# ------------------------
st.subheader("Sectional Metrics (PI & GPI)")
disp_cols = ["Horse", "Finish_Pos", "RaceTime_s", "F200", "tsSPI", "Accel", "Grind", "PI", "GPI"]
present = [c for c in disp_cols if c in gpi_df.columns]
disp = gpi_df[present].copy()

for c in ["F200", "tsSPI", "Accel", "Grind"]:
    if c in disp.columns: disp[c] = (disp[c] * 100.0).round(2)
if "PI" in disp.columns:  disp["PI"]  = disp["PI"].round(3)
if "GPI" in disp.columns: disp["GPI"] = disp["GPI"].round(3)

disp = disp.sort_values(["PI", "Finish_Pos"], ascending=[False, True])
st.dataframe(disp, use_container_width=True)

# ------------------------
# Visual 1: Sectional Shape Map — Kick vs Grind
# ------------------------
st.subheader("Sectional Shape Map — Kick (Accel) vs Grind")
fig, ax = plt.subplots()
x = gpi_df["Accel"] * 100.0
y = gpi_df["Grind"] * 100.0
ax.scatter(x, y, s=60, alpha=0.9)
if x.notna().any(): ax.axvline(x.median(), linestyle="--", alpha=0.4)
if y.notna().any(): ax.axhline(y.median(), linestyle="--", alpha=0.4)
for _, r in gpi_df.iterrows():
    if pd.isna(r.get("Accel")) or pd.isna(r.get("Grind")): continue
    ax.annotate(str(r.get("Horse", "")),
                xy=(r["Accel"] * 100.0, r["Grind"] * 100.0),
                xytext=(5, 5), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5))
ax.set_xlabel("Kick — Acceleration (%)")
ax.set_ylabel("Grind — Late (%)")
ax.set_title("Kick vs Grind")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# ------------------------
# Visual 2: Full-Race Pace Curves
# ------------------------
st.subheader("Full-Race Pace Curves — Field Average (black) + Top 8 Finishers")
x_mid, avg_speed, per_horse = build_pace_curves(gpi_df, distance_m)
fig2, ax2 = plt.subplots()
ax2.plot(x_mid, avg_speed, marker="o", linewidth=3, color="black", label="Average (Field)")
top8_names = gpi_df.sort_values("Finish_Pos").head(8)["Horse"].astype(str).tolist()
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
top8 = gpi_df.sort_values("PI", ascending=False).head(8).copy()
# shares * PI to show stacked height equals PI
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

st.caption(
    "PI v2.4-B+: distance-aware blend of F200 / tsSPI / Kick / Grind → robust-z → logistic (0–10). "
    "Stacked bars show each metric’s share of the horse’s PI. "
    "Full-race pace curves use every available split (100 m or 200 m). "
    "GPI v0.95 estimates group potential (0–10) from balance/separation with winner/place guards."
)
