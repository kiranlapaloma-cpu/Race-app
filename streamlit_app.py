import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Page config (no logo)
# ----------------------------
st.set_page_config(page_title="RaceEdge — vPI 2.3G", layout="wide")

# ----------------------------
# Debug toggle
# ----------------------------
with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# =====================================================
# Utilities
# =====================================================
def parse_race_time(val):
    """Accept seconds or 'MM:SS.ms' (or 'M:SS.ms'). Return float seconds."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    try:
        # bare float/seconds
        return float(s)
    except Exception:
        pass
    try:
        parts = s.split(":")
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        if len(parts) == 3:
            m, sec, ms = parts
            return int(m) * 60 + int(sec) + int(ms) / 1000.0
    except Exception:
        return np.nan
    return np.nan

def percent_rank(series):
    return pd.to_numeric(series, errors="coerce").rank(pct=True)

def norm_to_band(x, lo, hi):
    if pd.isna(x):
        return np.nan
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def clamp(x, lo=0.0, hi=1.0):
    return float(min(hi, max(lo, x)))

def to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

# =====================================================
# Distance-aware PI v2.3G profiles (weights + bonuses)
# =====================================================
def trip_profile(distance_m: int):
    """
    Return dict with weights and bonus/penalty magnitudes based on trip.
    Buckets:
      ≤1200 (SPRINT), 1300–1500 (SEVEN), 1550–1700 (MILE-ish),
      1750–2050 (MIDDLE), ≥2100 (CLASSIC)
    """
    d = int(distance_m)
    if d <= 1200:
        return dict(bucket="SPRINT",
                    wF200=0.08, wSPI=0.30, wACC=0.20, wGR=0.42,
                    end_hi=0.015, end_lo=0.010,
                    dom=0.020, win_hi=0.025, win_lo=0.015,
                    cap_lo=-0.05, cap_hi=0.06)
    if d <= 1500:
        return dict(bucket="SEVEN",
                    wF200=0.09, wSPI=0.28, wACC=0.27, wGR=0.36,
                    end_hi=0.015, end_lo=0.010,
                    dom=0.020, win_hi=0.025, win_lo=0.015,
                    cap_lo=-0.05, cap_hi=0.06)
    if d <= 1700:
        return dict(bucket="MILE",
                    wF200=0.08, wSPI=0.27, wACC=0.25, wGR=0.40,
                    end_hi=0.015, end_lo=0.010,
                    dom=0.020, win_hi=0.025, win_lo=0.015,
                    cap_lo=-0.05, cap_hi=0.06)
    if d <= 2050:
        return dict(bucket="MIDDLE",
                    wF200=0.06, wSPI=0.25, wACC=0.22, wGR=0.47,
                    end_hi=0.015, end_lo=0.010,
                    dom=0.025, win_hi=0.030, win_lo=0.020,
                    cap_lo=-0.06, cap_hi=0.07)
    return dict(bucket="CLASSIC",
                wF200=0.05, wSPI=0.23, wACC=0.20, wGR=0.52,
                end_hi=0.018, end_lo=0.012,
                dom=0.030, win_hi=0.035, win_lo=0.025,
                cap_lo=-0.07, cap_hi=0.08)

# =====================================================
# Metric computation from 100m or manual 200m inputs
# =====================================================
def detect_100m_columns(df, distance_m: int):
    """
    For a race distance D, 100m countdown columns are like:
      D-100, D-200, ..., 100, Finish
    Return list of present col names like '1200_Time','1100_Time',...,'100_Time','Finish_Time'
    """
    cols = []
    for m in range(distance_m - 100, 0, -100):
        name = f"{m}_Time"
        if name in df.columns:
            cols.append(name)
    if "Finish_Time" in df.columns:
        cols.append("Finish_Time")
    return cols

def build_metrics(df_in: pd.DataFrame, distance_m: int, manual_mode: bool) -> pd.DataFrame:
    """
    Compute F200%, tsSPI%, Accel%, Grind%, PI v2.3G and helper columns.
    - CSV mode expects 100m splits.
    - Manual mode uses 200m splits + Finish (100m).
    """
    work = df_in.copy()

    # Ensure RaceTime_s & Race_AvgSpeed
    work["RaceTime_s"] = work["Race Time"].apply(parse_race_time) if "Race Time" in work.columns else np.nan
    if work["RaceTime_s"].isna().all() and "Race Time" not in work.columns:
        # If user supplied 'RaceTime' or other variants
        for alt in ["Race_Time", "RaceTime", "Time"]:
            if alt in work.columns:
                work["RaceTime_s"] = work[alt].apply(parse_race_time)
                break
    # Fallback: sum of all sectionals (rough)
    if work["RaceTime_s"].isna().any():
        sec_cols = [c for c in work.columns if c.endswith("_Time")]
        if sec_cols:
            rough = work[sec_cols].sum(axis=1, skipna=True)
            work["RaceTime_s"] = work["RaceTime_s"].fillna(rough)

    work["Race_AvgSpeed"] = distance_m / work["RaceTime_s"]

    # Cast numeric sectionals
    for c in work.columns:
        if c.endswith("_Time"):
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # ------ Phase times ------
    if not manual_mode:
        # 100m sheets
        hundred_cols = detect_100m_columns(work, distance_m)
        # F200 = first two 100m splits (true first 200m)
        first_200 = []
        for m in [distance_m - 100, distance_m - 200]:
            name = f"{m}_Time"
            if name in work.columns:
                first_200.append(name)
        work["F200_time"] = work[first_200].sum(axis=1, skipna=False) if first_200 else np.nan

        # Accel = 200->100 + 100->Finish (two 100m pieces)
        a_cols = []
        if "200_Time" in work.columns: a_cols.append("200_Time")
        if "100_Time" in work.columns: a_cols.append("100_Time")
        work["Accel_time"] = work[a_cols].sum(axis=1, skipna=False) if a_cols else np.nan

        # Grind = final 100m
        work["Grind_time"] = work["Finish_Time"] if "Finish_Time" in work.columns else np.nan

        # tsSPI: exclude first 200m and last 400m
        # last 400m = 400→300 + 300→Finish(=200→100 + 100→Finish)
        mid_cols = []
        for m in range(distance_m - 300, 300, -100):  # from D-300 down to 400 (exclusive), step 100
            name = f"{m}_Time"
            if name in work.columns:
                mid_cols.append(name)
        work["Mid_time"] = work[mid_cols].sum(axis=1, skipna=True)
        work["Mid_dist"] = 100.0 * work[mid_cols].notna().sum(axis=1)

    else:
        # Manual 200m splits + Finish (100m)
        # Expected columns like: 'Horse', '<D-200>_Time', '<D-400>_Time', ..., '200_Time', 'Finish_Time'
        # F200 = first 200m cell
        first200_col = f"{distance_m-200}_Time"
        work["F200_time"] = work[first200_col] if first200_col in work.columns else np.nan
        # Accel = 200->100 + 100->Finish; in manual, we have Finish (100m); ask also '100_Time'? Usually absent.
        # Approximate Accel using last 200m split if 100_Time not available:
        if "200_Time" in work.columns and "100_Time" in work.columns:
            work["Accel_time"] = work["200_Time"] + work["100_Time"]
        elif "200_Time" in work.columns and "Finish_Time" in work.columns:
            # assume last 100m ~= Finish_Time; 200->100 approximated as half of 200_Time
            work["Accel_time"] = (work["200_Time"] / 2.0) + work["Finish_Time"]
        else:
            work["Accel_time"] = np.nan
        # Grind = Finish 100m
        work["Grind_time"] = work["Finish_Time"] if "Finish_Time" in work.columns else np.nan
        # tsSPI: exclude first 200 and last 400 (two 200m cells)
        mid_cols = []
        for m in range(distance_m - 400, 400, -200):  # from D-400 down to 600 (exclusive), step 200
            name = f"{m}_Time"
            if name in work.columns:
                mid_cols.append(name)
        work["Mid_time"] = work[mid_cols].sum(axis=1, skipna=True)
        work["Mid_dist"] = 200.0 * work[mid_cols].notna().sum(axis=1)

    # Speeds / ratios
    work["F200_speed"] = 200.0 / work["F200_time"]
    work["Mid_speed"] = np.where(work["Mid_time"] > 0, work["Mid_dist"] / work["Mid_time"], np.nan)
    work["Accel_speed"] = 200.0 / work["Accel_time"]
    work["Grind_speed"] = 100.0 / work["Grind_time"]

    work["F200%"]  = (work["F200_speed"]  / work["Race_AvgSpeed"]) * 100.0
    work["tsSPI%"] = (work["Mid_speed"]   / work["Race_AvgSpeed"]) * 100.0
    work["Accel%"] = (work["Accel_speed"] / work["Mid_speed"])     * 100.0
    work["Grind%"] = (work["Grind_speed"] / work["Race_AvgSpeed"]) * 100.0

    # If Finish_Pos missing, infer from time
    if ("Finish_Pos" not in work.columns) or work["Finish_Pos"].isna().all():
        work["Finish_Pos"] = work["RaceTime_s"].rank(method="min").astype("Int64")

    # Margin (lengths @ 0.20s/length)
    winner_time = work.loc[work["Finish_Pos"] == work["Finish_Pos"].min(), "RaceTime_s"].min()
    work["Margin_L"] = (work["RaceTime_s"] - winner_time) / 0.20

    return work

# =====================================================
# PI v2.3G computation
# =====================================================
def compute_pi(df: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    prof = trip_profile(distance_m)

    # Field medians for context
    spi_med = float(np.nanmedian(df["tsSPI%"]))
    acc_med = float(np.nanmedian(df["Accel%"]))
    gr_med  = float(np.nanmedian(df["Grind%"]))

    # Scaled scores (percentile + band)
    bands = {"F200%": (86, 96),
             "tsSPI%": (100, 109),
             "Accel%": (97, 105),
             "Grind%": (95, 104)}
    for k, (lo, hi) in bands.items():
        df[f"p_{k}"] = percent_rank(df[k])
        df[f"a_{k}"] = df[k].apply(lambda x: norm_to_band(x, lo, hi))
        df[f"s_{k}"] = 0.7 * df[f"p_{k}"] + 0.3 * df[f"a_{k}"]

    # Base PI (distance-aware weights)
    wF, wS, wA, wG = prof["wF200"], prof["wSPI"], prof["wACC"], prof["wGR"]
    df["PI_base"] = (
        df["s_F200%"] * wF +
        df["s_tsSPI%"] * wS +
        df["s_Accel%"] * wA +
        df["s_Grind%"] * wG
    )

    # Bonuses / penalties
    def endurance_bonus(r):
        if r["Grind%"] >= gr_med + 3.5 and r["tsSPI%"] >= spi_med:
            return prof["end_hi"]
        if r["Grind%"] >= gr_med + 1.5 and r["tsSPI%"] >= spi_med - 0.5:
            return prof["end_lo"]
        return 0.0

    def integrity_penalty(r):
        if r["Accel%"] >= acc_med + 4 and r["Grind%"] <= gr_med - 2:
            return -0.020  # kept constant across trips
        return 0.0

    def dominance_bonus(r, winner_time):
        if r["tsSPI%"] >= spi_med + 3 and (r["RaceTime_s"] - winner_time) / 0.20 >= 3:
            return prof["dom"]
        return 0.0

    def winner_bonus(r, winner_time):
        margin_L = (r["RaceTime_s"] - winner_time) / 0.20
        if r["Finish_Pos"] == 1 and margin_L >= 5:
            return prof["win_hi"]
        if r["Finish_Pos"] == 1 and margin_L >= 3:
            return prof["win_lo"]
        return 0.0

    winner_time = df.loc[df["Finish_Pos"] == df["Finish_Pos"].min(), "RaceTime_s"].min()
    df["EndBonus"] = df.apply(endurance_bonus, axis=1)
    df["IntPenalty"] = df.apply(integrity_penalty, axis=1)
    df["DomBonus"] = df.apply(lambda r: dominance_bonus(r, winner_time), axis=1)
    df["WinBonus"] = df.apply(lambda r: winner_bonus(r, winner_time), axis=1)

    extras = (df["EndBonus"] + df["IntPenalty"] + df["DomBonus"] + df["WinBonus"]).clip(prof["cap_lo"], prof["cap_hi"])
    df["PI_v2_3G"] = (df["PI_base"] + extras).clip(0, 1)

    return df

# =====================================================
# Manual input builder (200 m segments + Finish)
# =====================================================
def make_manual_template(horses: int, distance_m: int) -> pd.DataFrame:
    # Round distance up to nearest 200 for grid shape (your earlier preference)
    if distance_m % 200 != 0:
        distance_m = int(np.ceil(distance_m / 200.0) * 200)

    cols = ["Horse"]
    # countdown 200 m columns
    for m in range(distance_m - 200, 0, -200):
        cols.append(f"{m}_Time")
    cols.append("Finish_Time")  # final 100 m
    # optional Finish_Pos (user can leave blank)
    cols.append("Finish_Pos")
    df = pd.DataFrame({c: [np.nan] * horses for c in cols})
    df["Horse"] = [f"Runner {i+1}" for i in range(horses)]
    return df, distance_m

# =====================================================
# Charts
# =====================================================
def chart_pi_bar(df):
    ranked = df.sort_values("PI_v2_3G", ascending=True)
    names = ranked["Horse"].astype(str).tolist()
    pis = ranked["PI_v2_3G"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    ax.barh(names, pis)
    ax.set_xlabel("PI v2.3G (0–1)")
    ax.set_title("PI Ranking")
    for i, v in enumerate(pis):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    # glyphs on left
    glyphs = []
    for _, r in ranked.iterrows():
        g = ""
        if r["EndBonus"] > 0: g += "⛽"
        if r["DomBonus"] > 0: g += "★"
        if r["WinBonus"] > 0: g += "🛡"
        if r["IntPenalty"] < 0: g += "⚠️"
        glyphs.append(g or " ")
    for i, g in enumerate(glyphs):
        ax.text(0.01, i, g, va="center", fontsize=10)
    st.pyplot(fig)

def chart_shape_map(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    x = df["Accel%"].astype(float)
    y = df["Grind%"].astype(float)
    pi = df["PI_v2_3G"].astype(float)
    size = (df["tsSPI%"].astype(float) - df["tsSPI%"].min() + 1.0) * 18.0

    x_med, y_med = np.nanmedian(x), np.nanmedian(y)
    # quadrant shading + median lines
    ax.axvspan(x_med, max(np.nanmax(x), x_med), ymin=0, ymax=0.5, alpha=0.08)
    ax.axvspan(min(np.nanmin(x), x_med), x_med, ymin=0.5, ymax=1, alpha=0.08)
    ax.axhline(y_med, linestyle="--", linewidth=1.2)
    ax.axvline(x_med, linestyle="--", linewidth=1.2)

    sc = ax.scatter(x, y, s=size, c=pi, cmap="viridis", alpha=0.85, edgecolors="white", linewidths=0.8)
    # label top 8 by PI
    top_idx = df["PI_v2_3G"].nlargest(min(8, len(df))).index
    for i, r in df.iterrows():
        if i in top_idx:
            ax.annotate(str(r["Horse"]), (r["Accel%"], r["Grind%"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold")

    cb = plt.colorbar(sc, ax=ax, shrink=0.9)
    cb.set_label("PI v2.3G")
    ax.set_xlabel("Accel% (200→100 vs Mid)")
    ax.set_ylabel("Grind% (Final 100 vs Avg)")
    ax.set_title("Sectional Shape Map — Accel vs Grind (bubble = tsSPI, color = PI)")
    st.pyplot(fig)

# =====================================================
# UI
# =====================================================
st.title("🏇 RaceEdge — Performance Index v2.3G")

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose input type", ["Upload CSV", "Manual input"], index=0)
    distance_m = st.number_input("Race distance (m)", min_value=800, max_value=4000, value=1400, step=50)
    if mode == "Manual input":
        n_horses = st.number_input("Number of horses (manual)", min_value=2, max_value=24, value=8, step=1)

df_raw = None
manual_mode = (mode == "Manual input")

try:
    if not manual_mode:
        uploaded = st.file_uploader("Upload CSV with 100 m sectionals", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV or switch to Manual input.")
            st.stop()
        df_raw = pd.read_csv(uploaded)
        st.success("File loaded.")
    else:
        st.markdown("Manual mode: enter **200 m split times in countdown order** and the **Finish** (last 100 m) time.")
        tmpl, rounded_dist = make_manual_template(int(n_horses), int(distance_m))
        if rounded_dist != int(distance_m):
            st.info(f"Distance rounded up to **{rounded_dist} m** for the grid.")
            distance_m = rounded_dist
        df_raw = st.data_editor(tmpl, num_rows="fixed", use_container_width=True, key="manual_grid")
        st.success("Manual grid ready. Fill times in seconds (e.g., 11.05).")
except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw / Manual table preview")
st.dataframe(df_raw.head(12), use_container_width=True)

# ---------- Compute metrics ----------
try:
    work = build_metrics(df_raw, int(distance_m), manual_mode)
    work = compute_pi(work, int(distance_m))
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# Display core metrics only
show_cols = ["Horse", "Finish_Pos", "RaceTime_s", "Margin_L", "F200%", "tsSPI%", "Accel%", "Grind%", "PI_base",
             "EndBonus", "IntPenalty", "DomBonus", "WinBonus", "PI_v2_3G"]
disp = work[show_cols].copy()

for c in ["RaceTime_s", "Margin_L", "F200%", "tsSPI%", "Accel%", "Grind%", "PI_base", "PI_v2_3G"]:
    if c in disp.columns:
        disp[c] = pd.to_numeric(disp[c], errors="coerce")

st.subheader("Sectional Metrics (new system)")
st.dataframe(disp.sort_values(["PI_v2_3G"], ascending=False), use_container_width=True)

# ---------- Charts ----------
st.subheader("PI Ranking")
chart_pi_bar(work)

st.subheader("Sectional Shape Map")
chart_shape_map(work)

st.caption(
    "Definitions: F200% = first 200 m vs race avg; tsSPI% = sustained mid-race pace (excludes first 200 m & last 400 m); "
    "Accel% = 200→100 vs mid; Grind% = final 100 vs race avg. PI v2.3G applies distance-aware weighting with endurance/dominance/winner protection bonuses."
)
