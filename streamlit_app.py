import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# Page config (no logo)
# =========================
st.set_page_config(page_title="RaceEdge — PI v2.3G", layout="wide")

# =========================
# Debug toggle
# =========================
with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# =========================
# Generic helpers
# =========================
def parse_race_time(val):
    """Accept seconds or 'MM:SS.ms' (or 'M:SS.ms'). Return float seconds."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # bare float?
    try:
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

# =========================
# PI v2.3G trip profile
# =========================
def trip_profile(distance_m: int):
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

# =========================
# Metric computation
# =========================
def _list_100m_cols(df, distance_m: int):
    cols = []
    for m in range(distance_m - 100, 0, -100):
        c = f"{m}_Time"
        if c in df.columns:
            cols.append(c)
    if "Finish_Time" in df.columns:
        cols.append("Finish_Time")
    return cols

def detect_100m_columns(df, distance_m: int):
    cols = []
    for m in range(distance_m - 100, 0, -100):
        name = f"{m}_Time"
        if name in df.columns:
            cols.append(name)
    if "Finish_Time" in df.columns:
        cols.append("Finish_Time")
    return cols

def build_metrics(df_in: pd.DataFrame, distance_m: int, manual_mode: bool) -> pd.DataFrame:
    work = df_in.copy()

    # normalize RaceTime_s
    if "Race Time" in work.columns:
        work["RaceTime_s"] = work["Race Time"].apply(parse_race_time)
    else:
        rt_col = next((c for c in ["Race_Time", "RaceTime", "Time"] if c in work.columns), None)
        if rt_col:
            work["RaceTime_s"] = work[rt_col].apply(parse_race_time)
        else:
            work["RaceTime_s"] = np.nan

    # fallback: sum of sectionals
    if work["RaceTime_s"].isna().any():
        sec_cols = [c for c in work.columns if c.endswith("_Time")]
        if sec_cols:
            rough = work[sec_cols].sum(axis=1, skipna=True)
            work["RaceTime_s"] = work["RaceTime_s"].fillna(rough)

    work["Race_AvgSpeed"] = distance_m / work["RaceTime_s"]

    # cast numeric
    for c in work.columns:
        if c.endswith("_Time"):
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # ----- Phase times -----
    if not manual_mode:
        # 100m sheets
        first_200 = []
        for m in [distance_m - 100, distance_m - 200]:
            name = f"{m}_Time"
            if name in work.columns:
                first_200.append(name)
        work["F200_time"] = work[first_200].sum(axis=1, skipna=False) if first_200 else np.nan

        a_cols = []
        if "200_Time" in work.columns: a_cols.append("200_Time")
        if "100_Time" in work.columns: a_cols.append("100_Time")
        work["Accel_time"] = work[a_cols].sum(axis=1, skipna=False) if a_cols else np.nan

        work["Grind_time"] = work["Finish_Time"] if "Finish_Time" in work.columns else np.nan

        # tsSPI: exclude first 200 & last 400
        mid_cols = []
        for m in range(distance_m - 300, 300, -100):  # D-300 down to 400, step 100
            name = f"{m}_Time"
            if name in work.columns:
                mid_cols.append(name)
        work["Mid_time"] = work[mid_cols].sum(axis=1, skipna=True)
        work["Mid_dist"] = 100.0 * work[mid_cols].notna().count(axis=1)

    else:
        # Manual (200m grid + Finish 100m)
        # round up distance for grid shape already done earlier
        first200_col = f"{distance_m - 200}_Time"
        work["F200_time"] = work[first200_col] if first200_col in work.columns else np.nan

        if "200_Time" in work.columns and "100_Time" in work.columns:
            work["Accel_time"] = work["200_Time"] + work["100_Time"]
        elif "200_Time" in work.columns and "Finish_Time" in work.columns:
            work["Accel_time"] = (work["200_Time"] / 2.0) + work["Finish_Time"]
        else:
            work["Accel_time"] = np.nan

        work["Grind_time"] = work["Finish_Time"] if "Finish_Time" in work.columns else np.nan

        mid_cols = []
        for m in range(distance_m - 400, 400, -200):  # D-400 down to 600, step 200
            name = f"{m}_Time"
            if name in work.columns:
                mid_cols.append(name)
        work["Mid_time"] = work[mid_cols].sum(axis=1, skipna=True)
        work["Mid_dist"] = 200.0 * work[mid_cols].notna().count(axis=1)

    # speeds & ratios
    work["F200_speed"]  = 200.0 / work["F200_time"]
    work["Mid_speed"]   = np.where(work["Mid_time"] > 0, work["Mid_dist"] / work["Mid_time"], np.nan)
    work["Accel_speed"] = 200.0 / work["Accel_time"]
    work["Grind_speed"] = 100.0 / work["Grind_time"]

    work["F200%"]  = (work["F200_speed"]  / work["Race_AvgSpeed"]) * 100.0
    work["tsSPI%"] = (work["Mid_speed"]   / work["Race_AvgSpeed"]) * 100.0
    work["Accel%"] = (work["Accel_speed"] / work["Mid_speed"])     * 100.0
    work["Grind%"] = (work["Grind_speed"] / work["Race_AvgSpeed"]) * 100.0

    # Finish_Pos fallback
    if ("Finish_Pos" not in work.columns) or work["Finish_Pos"].isna().all():
        work["Finish_Pos"] = work["RaceTime_s"].rank(method="min").astype("Int64")

    # Margin (lengths @ 0.20s)
    winner_time = work.loc[work["Finish_Pos"] == work["Finish_Pos"].min(), "RaceTime_s"].min()
    work["Margin_L"] = (work["RaceTime_s"] - winner_time) / 0.20

    return work

# =========================
# PI computation
# =========================
def compute_pi(df: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    prof = trip_profile(distance_m)

    spi_med = float(np.nanmedian(df["tsSPI%"]))
    acc_med = float(np.nanmedian(df["Accel%"]))
    gr_med  = float(np.nanmedian(df["Grind%"]))

    bands = {"F200%": (86, 96),
             "tsSPI%": (100, 109),
             "Accel%": (97, 105),
             "Grind%": (95, 104)}
    for k, (lo, hi) in bands.items():
        df[f"p_{k}"] = percent_rank(df[k])
        df[f"a_{k}"] = df[k].apply(lambda x: norm_to_band(x, lo, hi))
        df[f"s_{k}"] = 0.7 * df[f"p_{k}"] + 0.3 * df[f"a_{k}"]

    wF, wS, wA, wG = prof["wF200"], prof["wSPI"], prof["wACC"], prof["wGR"]
    df["PI_base"] = (
        df["s_F200%"]  * wF +
        df["s_tsSPI%"] * wS +
        df["s_Accel%"] * wA +
        df["s_Grind%"] * wG
    )

    # Winner time (for margins)
    winner_time = df.loc[df["Finish_Pos"] == df["Finish_Pos"].min(), "RaceTime_s"].min()

    def endurance_bonus(r):
        if r["Grind%"] >= gr_med + 3.5 and r["tsSPI%"] >= spi_med:
            return prof["end_hi"]
        if r["Grind%"] >= gr_med + 1.5 and r["tsSPI%"] >= spi_med - 0.5:
            return prof["end_lo"]
        return 0.0

    def integrity_penalty(r):
        if r["Accel%"] >= acc_med + 4 and r["Grind%"] <= gr_med - 2:
            return -0.020
        return 0.0

    def dominance_bonus(r):
        margin_L = (r["RaceTime_s"] - winner_time) / 0.20
        if r["tsSPI%"] >= spi_med + 3 and margin_L >= 3:
            return prof["dom"]
        return 0.0

    def winner_bonus(r):
        margin_L = (r["RaceTime_s"] - winner_time) / 0.20
        if r["Finish_Pos"] == 1 and margin_L >= 5:
            return prof["win_hi"]
        if r["Finish_Pos"] == 1 and margin_L >= 3:
            return prof["win_lo"]
        return 0.0

    df["EndBonus"]   = df.apply(endurance_bonus, axis=1)
    df["IntPenalty"] = df.apply(integrity_penalty, axis=1)
    df["DomBonus"]   = df.apply(dominance_bonus, axis=1)
    df["WinBonus"]   = df.apply(winner_bonus, axis=1)

    extras = (df["EndBonus"] + df["IntPenalty"] + df["DomBonus"] + df["WinBonus"]).clip(prof["cap_lo"], prof["cap_hi"])
    df["PI_v2_3G"] = (df["PI_base"] + extras).clip(0, 1)

    return df

# =========================
# Manual input template
# =========================
def make_manual_template(horses: int, distance_m: int) -> pd.DataFrame:
    # Round up to nearest 200 for grid
    if distance_m % 200 != 0:
        distance_m = int(np.ceil(distance_m / 200.0) * 200)

    cols = ["Horse"]
    for m in range(distance_m - 200, 0, -200):
        cols.append(f"{m}_Time")
    cols.append("Finish_Time")  # final 100 m
    cols.append("Finish_Pos")
    df = pd.DataFrame({c: [np.nan] * horses for c in cols})
    df["Horse"] = [f"Runner {i+1}" for i in range(horses)]
    return df, distance_m

# =========================
# Pace curve (200m bins)
# =========================
def compute_field_pace_200m(df: pd.DataFrame, distance_m: int, manual_mode: bool):
    labels = []
    seg_times = []

    if not manual_mode:
        hcols = _list_100m_cols(df, distance_m)  # e.g. [D-100, D-200, ..., 100, Finish]
        pairs = []
        for i in range(0, len(hcols) - 1, 2):
            pairs.append((hcols[i], hcols[i+1]))

        # labels from start to finish
        for k in range(distance_m - 200, -1, -200):
            labels.append("0" if k == 0 else f"{k}_to_{k+200}")

        for (a, b) in pairs:
            t = pd.to_numeric(df[a], errors="coerce") + pd.to_numeric(df[b], errors="coerce")
            seg_times.append(t.values)
    else:
        mcols = [c for c in df.columns if c.endswith("_Time") and c != "Finish_Time"]
        def _dist(c): return int(c.split("_")[0])
        mcols = sorted(mcols, key=_dist, reverse=True)  # D-200, D-400, ...
        for c in mcols:
            k = int(c.split("_")[0])
            labels.append("0" if k == 200 else f"{k-200}_to_{k}")
            seg_times.append(pd.to_numeric(df[c], errors="coerce").values)

    avg_speeds = []
    for t in seg_times:
        s = pd.Series(t).dropna()
        if len(s) == 0 or s.mean() <= 0:
            avg_speeds.append(np.nan)
        else:
            avg_speeds.append(200.0 / s.mean())

    n = min(len(labels), len(avg_speeds))
    return labels[:n], avg_speeds[:n]

def ema(series, alpha=0.3):
    y = []
    prev = None
    for v in series:
        if np.isnan(v):
            y.append(np.nan)
            continue
        if prev is None or np.isnan(prev):
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        y.append(prev)
    return y

# =========================
# Charts
# =========================
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

def chart_shape_map(df, label_mode="Top 8"):
    fig, ax = plt.subplots(figsize=(9, 7))
    x = df["Accel%"].astype(float)
    y = df["Grind%"].astype(float)
    pi = df["PI_v2_3G"].astype(float)
    size = (df["tsSPI%"].astype(float) - df["tsSPI%"].min() + 1.0) * 18.0

    x_med, y_med = np.nanmedian(x), np.nanmedian(y)
    # quadrant shading
    ax.axvspan(x_med, max(np.nanmax(x), x_med), ymin=0, ymax=0.5, alpha=0.08)
    ax.axvspan(min(np.nanmin(x), x_med), x_med, ymin=0.5, ymax=1, alpha=0.08)
    ax.axhline(y_med, linestyle="--", linewidth=1.2)
    ax.axvline(x_med, linestyle="--", linewidth=1.2)

    sc = ax.scatter(x, y, s=size, c=pi, cmap="viridis", alpha=0.85, edgecolors="white", linewidths=0.8)

    # Decide which to label
    label_idx = []
    if label_mode == "None":
        label_idx = []
    elif label_mode == "All":
        label_idx = df.index.tolist()
    elif label_mode == "Top 5":
        label_idx = df["PI_v2_3G"].nlargest(min(5, len(df))).index.tolist()
    else:  # Top 8
        label_idx = df["PI_v2_3G"].nlargest(min(8, len(df))).index.tolist()

    # Arrowed labels (offset + arrowprops)
    for i, r in df.iterrows():
        if i in label_idx:
            ax.annotate(
                str(r["Horse"]),
                (r["Accel%"], r["Grind%"]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
                arrowprops=dict(arrowstyle="-", color="black", alpha=0.6, lw=0.75),
            )

    cb = plt.colorbar(sc, ax=ax, shrink=0.9)
    cb.set_label("PI v2.3G")
    ax.set_xlabel("Accel% (200→100 vs Mid)")
    ax.set_ylabel("Grind% (Final 100 vs Avg)")
    ax.set_title("Sectional Shape Map — Accel vs Grind (bubble = tsSPI, color = PI)")
    st.pyplot(fig)

def chart_pace_curve_200m(df: pd.DataFrame, distance_m: int, manual_mode: bool,
                          overlays="Top 5 finishers", smooth=False, normalize=False):
    labels, speeds = compute_field_pace_200m(df, distance_m, manual_mode)
    # X left->right from start to finish
    x = list(range(len(labels)))[::-1]
    y = list(reversed(speeds))
    xlabels = list(reversed(labels))

    y_plot = ema(y) if smooth else y

    fig, ax = plt.subplots(figsize=(10, 4.8))
    # Field average in black
    ax.plot(x, y_plot, marker="o", linewidth=2.5, label="Field avg (200m)", zorder=5, color="black")

    # Prepare per-horse overlays (top N finishers)
    # We'll reconstruct each horse's 200m speeds with same bins
    # Normalize flag: if True, plot (runner - field_avg) per bin
    # Determine N
    overlay_n = {"None": 0, "Top 3 finishers": 3, "Top 5 finishers": 5, "Top 8 finishers": 8}.get(overlays, 5)
    if overlay_n > 0:
        # Get top N by Finish_Pos ascending
        top_finish = df.sort_values("Finish_Pos").head(overlay_n)
        palette = [
            # colorblind-friendly-ish distinct set
            "#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB","#000000"
        ]
        # Build each horse's 200m vector
        # Helper to get per-horse bin times
        def horse_bin_times(row):
            if not manual_mode:
                hcols = _list_100m_cols(df, distance_m)
                pairs = []
                for i in range(0, len(hcols) - 1, 2):
                    pairs.append((hcols[i], hcols[i+1]))
                times = []
                for (a, b) in pairs:
                    t = pd.to_numeric(pd.Series([row.get(a), row.get(b)])).sum()
                    times.append(t if pd.notna(t) and t > 0 else np.nan)
                return times
            else:
                mcols = [c for c in df.columns if c.endswith("_Time") and c != "Finish_Time"]
                def _dist(c): return int(c.split("_")[0])
                mcols = sorted(mcols, key=_dist, reverse=True)
                times = [row.get(c) for c in mcols]
                return times

        for idx, (_, r) in enumerate(top_finish.iterrows()):
            tvec = horse_bin_times(r)
            # convert to speeds
            svec = [200.0 / t if (pd.notna(t) and t > 0) else np.nan for t in tvec]
            # align & reverse to left->right
            svec = list(reversed(svec))[:len(y)]
            if normalize:
                svec = [sv - fv if (pd.notna(sv) and pd.notna(fv)) else np.nan for sv, fv in zip(svec, y)]
            line = ax.plot(x, svec, marker="o", linewidth=1.8, label=f"{int(r['Finish_Pos'])}° {r['Horse']}",
                           zorder=3, color=palette[idx % len(palette)])
            # make overlay slightly transparent to let black stand out
            for l in line:
                l.set_alpha(0.9)

    ax.set_xticks(x)
    pretty = []
    for lab in xlabels:
        if lab == "0":
            pretty.append("Finish")
        else:
            a, _, b = lab.partition("_to_")
            a, b = int(a), int(b)
            pretty.append(f"{a}–{b}")
    ax.set_xticklabels(pretty, rotation=30, ha="right")
    ax.set_ylabel("Speed (m/s)" if not normalize else "Δ Speed vs Field (m/s)")
    ax.set_title("Race Pace Curve — 200 m bins")
    ax.grid(True, linestyle="--", alpha=0.3)
    if overlay_n > 0:
        ax.legend(loc="best", fontsize=8, ncol=2)
    st.pyplot(fig)

# =========================
# UI
# =========================
st.title("🏇 RaceEdge — Performance Index v2.3G")

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose input type", ["Upload CSV", "Manual input"], index=0)
    distance_m = st.number_input("Race distance (m)", min_value=800, max_value=4000, value=1400, step=50)
    if mode == "Manual input":
        n_horses = st.number_input("Number of horses (manual)", min_value=2, max_value=24, value=8, step=1)

    st.header("Charts")
    label_mode = st.selectbox("Shape map labels", ["Top 8", "Top 5", "All", "None"], index=0)
    overlays = st.selectbox("Pace curve overlays", ["Top 5 finishers", "Top 3 finishers", "Top 8 finishers", "None"], index=0)
    smooth = st.checkbox("Smooth field avg (EMA)", value=False)
    normalize = st.checkbox("Normalize overlays to field (Δ m/s)", value=False)

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
_dbg("Columns", list(df_raw.columns))

# ---------- Compute metrics ----------
try:
    work = build_metrics(df_raw, int(distance_m), manual_mode)
    work = compute_pi(work, int(distance_m))
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# ---------- Display core metrics ----------
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
chart_shape_map(work, label_mode=label_mode)

st.subheader("Race Pace Curve (200 m bins)")
chart_pace_curve_200m(work, int(distance_m), manual_mode,
                      overlays=overlays, smooth=smooth, normalize=normalize)

st.caption(
    "Definitions: F200% = first 200 m vs race avg; tsSPI% = sustained mid-race pace (excludes first 200 m & last 400 m); "
    "Accel% = 200→100 vs mid; Grind% = final 100 vs race avg. PI v2.3G applies distance-aware weighting with endurance/dominance/winner protection bonuses."
)
