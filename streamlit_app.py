import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# Matplotlib / PIL safety defaults
# =============================================================================
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.dpi"] = 110
mpl.rcParams["figure.max_open_warning"] = 0
# Belt & braces: increase max pixels (we still clamp figures below)
Image.MAX_IMAGE_PIXELS = 500_000_000

# =============================================================================
# Streamlit page config (logo removed per request)
# =============================================================================
st.set_page_config(page_title="RaceEdge — PI v2.3G", layout="wide")

# =============================================================================
# Debug toggle
# =============================================================================
with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# =============================================================================
# Safe pyplot wrapper (prevents huge images and font glyph issues)
# =============================================================================
def safe_pyplot(fig, max_w_px=1000, max_h_px=800, dpi=96):
    """
    Render fig as PNG in-memory at a safe DPI/size and display via st.image.
    This bypasses streamlit.pyplot's internal marshalling that can trigger
    PIL decompression-bomb checks when the canvas is too large.
    """
    # Clamp DPI/size to keep pixel budget sane
    fig.set_dpi(dpi)
    w_in, h_in = fig.get_size_inches()
    w_px, h_px = max(w_in * dpi, 1), max(h_in * dpi, 1)
    scale = min(max_w_px / w_px, max_h_px / h_px, 1.0)
    if scale < 1.0:
        fig.set_size_inches(w_in * scale, h_in * scale, forward=True)

    buf = io.BytesIO()
    # bbox_inches=None avoids autosizing that can inflate canvas
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, caption=None, width=max_w_px)

# =============================================================================
# Generic helpers
# =============================================================================
def parse_race_time(val):
    """Accept seconds or 'MM:SS.ms' (or 'M:SS.ms'). Return float seconds."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
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
    try:
        return float(min(hi, max(lo, x)))
    except Exception:
        return np.nan

def _safe_median(s, default=np.nan):
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanmedian(s)) if s.notna().any() else default

# =============================================================================
# Trip profile (distance-aware weights & bonuses) — PI v2.3G
# =============================================================================
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

# =============================================================================
# Column utilities (100 m sheets)
# =============================================================================
def _list_100m_cols(df, distance_m: int):
    cols = []
    for m in range(distance_m - 100, 0, -100):
        c = f"{m}_Time"
        if c in df.columns:
            cols.append(c)
    if "Finish_Time" in df.columns:
        cols.append("Finish_Time")
    return cols

# =============================================================================
# Manual template (200 m grid + Finish 100 m)
# =============================================================================
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

# =============================================================================
# Build sectional metrics (supports CSV 100 m sheets and manual 200 m grid)
# =============================================================================
def build_metrics(df_in: pd.DataFrame, distance_m: int, manual_mode: bool) -> pd.DataFrame:
    work = df_in.copy()

    # --- Race time normalization
    if "Race Time" in work.columns:
        work["RaceTime_s"] = work["Race Time"].apply(parse_race_time)
    else:
        rt_col = next((c for c in ["Race_Time", "RaceTime", "Time"] if c in work.columns), None)
        if rt_col:
            work["RaceTime_s"] = work[rt_col].apply(parse_race_time)
        else:
            work["RaceTime_s"] = np.nan

    # Fallback: sum of sectionals if RaceTime_s missing
    if work["RaceTime_s"].isna().any():
        sec_cols = [c for c in work.columns if c.endswith("_Time")]
        if sec_cols:
            rough = work[sec_cols].sum(axis=1, skipna=True)
            work["RaceTime_s"] = work["RaceTime_s"].fillna(rough)

    work["Race_AvgSpeed"] = distance_m / work["RaceTime_s"]

    # Cast sectionals to numeric
    for c in work.columns:
        if c.endswith("_Time"):
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # --- Phase definitions
    if not manual_mode:
        # First 200 (D-100 + D-200)
        first_200 = []
        for m in [distance_m - 100, distance_m - 200]:
            name = f"{m}_Time"
            if name in work.columns:
                first_200.append(name)
        work["F200_time"] = work[first_200].sum(axis=1, skipna=False) if first_200 else np.nan

        # Accel (200->100): prefer 200_Time + 100_Time
        a_cols = []
        if "200_Time" in work.columns: a_cols.append("200_Time")
        if "100_Time" in work.columns: a_cols.append("100_Time")
        work["Accel_time"] = work[a_cols].sum(axis=1, skipna=False) if a_cols else np.nan

        # Grind: Finish 100
        work["Grind_time"] = work["Finish_Time"] if "Finish_Time" in work.columns else np.nan

        # tsSPI mid: exclude first 200 and last 400
        mid_cols = []
        for m in range(distance_m - 300, 300, -100):  # D-300 down to 400
            name = f"{m}_Time"
            if name in work.columns:
                mid_cols.append(name)
        work["Mid_time"] = work[mid_cols].sum(axis=1, skipna=True)
        work["Mid_dist"] = (work[mid_cols].notna().sum(axis=1).astype(float)) * 100.0

    else:
        # Manual (200m bins + Finish 100)
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
        work["Mid_dist"] = (work[mid_cols].notna().sum(axis=1).astype(float)) * 200.0

    # --- speeds & guarded ratios
    work["F200_speed"]  = 200.0 / work["F200_time"]
    work["Mid_speed"]   = np.where(work["Mid_time"] > 0, work["Mid_dist"] / work["Mid_time"], np.nan)
    work["Accel_speed"] = 200.0 / work["Accel_time"]
    work["Grind_speed"] = 100.0 / work["Grind_time"]

    work["F200%"]  = np.where(work["Race_AvgSpeed"] > 0, (work["F200_speed"]  / work["Race_AvgSpeed"]) * 100.0, np.nan)
    work["tsSPI%"] = np.where(work["Race_AvgSpeed"] > 0, (work["Mid_speed"]   / work["Race_AvgSpeed"]) * 100.0, np.nan)
    work["Accel%"] = np.where(work["Mid_speed"]     > 0, (work["Accel_speed"] / work["Mid_speed"])     * 100.0, np.nan)
    work["Grind%"] = np.where(work["Race_AvgSpeed"] > 0, (work["Grind_speed"] / work["Race_AvgSpeed"]) * 100.0, np.nan)

    # Finish_Pos fallback
    if ("Finish_Pos" not in work.columns) or work["Finish_Pos"].isna().all():
        work["Finish_Pos"] = work["RaceTime_s"].rank(method="min").astype("Int64")

    # Margin in lengths (0.20s per length)
    winner_time = work.loc[work["Finish_Pos"] == work["Finish_Pos"].min(), "RaceTime_s"].min()
    work["Margin_L"] = (work["RaceTime_s"] - winner_time) / 0.20

    return work

# =============================================================================
# PI v2.3G computation
# =============================================================================
def compute_pi(df: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    prof = trip_profile(distance_m)

    spi_med = _safe_median(df["tsSPI%"], default=100.0)
    acc_med = _safe_median(df["Accel%"], default=100.0)
    gr_med  = _safe_median(df["Grind%"], default=100.0)

    # Soft bands for assistance (lo, hi)
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
        if pd.notna(r["Grind%"]) and pd.notna(r["tsSPI%"]):
            if r["Grind%"] >= gr_med + 3.5 and r["tsSPI%"] >= spi_med:
                return prof["end_hi"]
            if r["Grind%"] >= gr_med + 1.5 and r["tsSPI%"] >= spi_med - 0.5:
                return prof["end_lo"]
        return 0.0

    def integrity_penalty(r):
        if pd.notna(r["Accel%"]) and pd.notna(r["Grind%"]):
            if r["Accel%"] >= acc_med + 4 and r["Grind%"] <= gr_med - 2:
                return -0.020
        return 0.0

    def dominance_bonus(r):
        if pd.isna(r["RaceTime_s"]):
            return 0.0
        margin_L = (r["RaceTime_s"] - winner_time) / 0.20
        if pd.notna(r["tsSPI%"]) and r["tsSPI%"] >= spi_med + 3 and margin_L >= 3:
            return prof["dom"]
        return 0.0

    def winner_bonus(r):
        if pd.isna(r["RaceTime_s"]):
            return 0.0
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

# =============================================================================
# Pace curve (200 m bins)
# =============================================================================
def compute_field_pace_200m(df: pd.DataFrame, distance_m: int, manual_mode: bool):
    labels = []
    seg_times = []

    if not manual_mode:
        hcols = _list_100m_cols(df, distance_m)  # [D-100, D-200, ..., 100, Finish]
        pairs = []
        for i in range(0, len(hcols) - 1, 2):
            pairs.append((hcols[i], hcols[i+1]))

        # labels: start->finish
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
    out = []
    prev = None
    for v in series:
        if np.isnan(v):
            out.append(np.nan)
            continue
        if prev is None or np.isnan(prev):
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out

# =============================================================================
# Charts (use safe_pyplot)
# =============================================================================
def chart_pi_bar(df):
    ranked = df.sort_values("PI_v2_3G", ascending=True)
    names = ranked["Horse"].astype(str).tolist()
    pis = ranked["PI_v2_3G"].astype(float).tolist()

    max_h_inches = 10.0
    fig_h = min(max(4, 0.35 * len(names)), max_h_inches)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(names, pis)
    ax.set_xlabel("PI v2.3G (0–1)")
    ax.set_title("PI Ranking")
    for i, v in enumerate(pis):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    # ASCII badges (no emoji font issues)
    badges = []
    for _, r in ranked.iterrows():
        g = ""
        if r["EndBonus"] > 0: g += "[END]"
        if r["DomBonus"] > 0: g += " [DOM]"
        if r["WinBonus"] > 0: g += " [WIN]"
        if r["IntPenalty"] < 0: g += " [INT]"
        badges.append(g.strip() or " ")
    for i, g in enumerate(badges):
        ax.text(0.01, i, g, va="center", fontsize=9)

    ax.tick_params(labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    safe_pyplot(fig)

def chart_shape_map(df, label_mode="Top 8"):
    fig, ax = plt.subplots(figsize=(9, 7))
    x = pd.to_numeric(df["Accel%"], errors="coerce")
    y = pd.to_numeric(df["Grind%"], errors="coerce")
    pi = pd.to_numeric(df["PI_v2_3G"], errors="coerce")
    size = (pd.to_numeric(df["tsSPI%"], errors="coerce") - pd.to_numeric(df["tsSPI%"], errors="coerce").min() + 1.0) * 18.0

    x_med, y_med = _safe_median(x), _safe_median(y)
    # quadrant shading + medians
    ax.axhline(y_med, linestyle="--", linewidth=1.2, color="grey", alpha=0.8)
    ax.axvline(x_med, linestyle="--", linewidth=1.2, color="grey", alpha=0.8)
    ax.axvspan(x_med, max(np.nanmax(x), x_med), ymin=0, ymax=0.5, alpha=0.08, color="C0")
    ax.axvspan(min(np.nanmin(x), x_med), x_med, ymin=0.5, ymax=1, alpha=0.08, color="C2")

    sc = ax.scatter(x, y, s=size, c=pi, cmap="viridis", alpha=0.85, edgecolors="white", linewidths=0.8)

    # Label subset
    if label_mode == "None":
        label_idx = []
    elif label_mode == "All":
        label_idx = df.index.tolist()
    elif label_mode == "Top 5":
        label_idx = df["PI_v2_3G"].nlargest(min(5, len(df))).index.tolist()
    else:  # Top 8
        label_idx = df["PI_v2_3G"].nlargest(min(8, len(df))).index.tolist()

    # Arrow labels pointing to bubbles
    for i, r in df.iterrows():
        if i in label_idx and pd.notna(r["Accel%"]) and pd.notna(r["Grind%"]):
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
    safe_pyplot(fig)

def chart_pace_curve_200m(df: pd.DataFrame, distance_m: int, manual_mode: bool,
                          overlays="Top 5 finishers", smooth=False, normalize=False):
    labels, speeds = compute_field_pace_200m(df, distance_m, manual_mode)
    # x left->right from start to finish
    x = list(range(len(labels)))[::-1]
    y = list(reversed(speeds))
    xlabels = list(reversed(labels))
    y_plot = ema(y) if smooth else y

    fig, ax = plt.subplots(figsize=(10, 4.8))
    # Field average in black
    ax.plot(x, y_plot, marker="o", linewidth=2.5, label="Field avg (200m)", zorder=5, color="black")

    # Overlays: top N finishers
    overlay_n = {"None": 0, "Top 3 finishers": 3, "Top 5 finishers": 5, "Top 8 finishers": 8}.get(overlays, 5)
    if overlay_n > 0:
        top_finish = df.sort_values("Finish_Pos").head(overlay_n)
        palette = ["#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB","#000000"]

        def horse_bin_times(row):
            if not manual_mode:
                hcols = _list_100m_cols(df, distance_m)
                pairs = []
                for i in range(0, len(hcols) - 1, 2):
                    pairs.append((hcols[i], hcols[i+1]))
                times = []
                for (a, b) in pairs:
                    a_val = pd.to_numeric(pd.Series([row.get(a)]), errors="coerce").iloc[0]
                    b_val = pd.to_numeric(pd.Series([row.get(b)]), errors="coerce").iloc[0]
                    t = (a_val if pd.notna(a_val) else np.nan) + (b_val if pd.notna(b_val) else np.nan)
                    times.append(t if pd.notna(t) and t > 0 else np.nan)
                return times
            else:
                mcols = [c for c in df.columns if c.endswith("_Time") and c != "Finish_Time"]
                def _dist(c): return int(c.split("_")[0])
                mcols = sorted(mcols, key=_dist, reverse=True)
                return [row.get(c) for c in mcols]

        for idx, (_, r) in enumerate(top_finish.iterrows()):
            tvec = horse_bin_times(r)
            svec = [200.0 / t if (pd.notna(t) and t > 0) else np.nan for t in tvec]
            svec = list(reversed(svec))[:len(y)]
            if normalize:
                svec = [sv - fv if (pd.notna(sv) and pd.notna(fv)) else np.nan for sv, fv in zip(svec, y)]
            line = ax.plot(x, svec, marker="o", linewidth=1.8, label=f"{int(r['Finish_Pos'])}° {r['Horse']}",
                           zorder=3, color=palette[idx % len(palette)])
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
    safe_pyplot(fig)

# =============================================================================
# UI
# =============================================================================
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

# =============================================================================
# Data ingestion
# =============================================================================
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
        df_raw = st.data_editor(tmpl, num_rows="fixed", width="stretch", key="manual_grid")
        st.success("Manual grid ready. Fill times in seconds (e.g., 11.05).")
except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw / Manual table preview")
st.dataframe(df_raw.head(12), width="stretch")
_dbg("Columns", list(df_raw.columns))

# =============================================================================
# Compute metrics & PI
# =============================================================================
try:
    work = build_metrics(df_raw, int(distance_m), manual_mode)
    work = compute_pi(work, int(distance_m))
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# =============================================================================
# Display core metrics (new system)
# =============================================================================
show_cols = [
    "Horse", "Finish_Pos", "RaceTime_s", "Margin_L",
    "F200%", "tsSPI%", "Accel%", "Grind%",
    "PI_base", "EndBonus", "IntPenalty", "DomBonus", "WinBonus", "PI_v2_3G"
]
disp = work.copy()
for c in ["RaceTime_s", "Margin_L", "F200%", "tsSPI%", "Accel%", "Grind%", "PI_base", "PI_v2_3G",
          "EndBonus", "IntPenalty", "DomBonus", "WinBonus"]:
    if c in disp.columns:
        disp[c] = pd.to_numeric(disp[c], errors="coerce")

st.subheader("Sectional Metrics (new system)")
st.dataframe(disp[show_cols].sort_values(["PI_v2_3G"], ascending=False), width="stretch")

# =============================================================================
# Charts
# =============================================================================
st.subheader("PI Ranking")
chart_pi_bar(work)

st.subheader("Sectional Shape Map")
chart_shape_map(work, label_mode=label_mode)

st.subheader("Race Pace Curve (200 m bins)")
chart_pace_curve_200m(work, int(distance_m), manual_mode,
                      overlays=overlays, smooth=smooth, normalize=normalize)

st.caption(
    "Definitions: F200% = first 200 m vs race avg; tsSPI% = sustained mid-race pace (excludes first 200 m & last 400 m); "
    "Accel% = 200→100 vs mid; Grind% = final 100 vs race avg. "
    "PI v2.3G applies distance-aware weighting with endurance/dominance/winner protection bonuses. "
    "Manual input rounds distance up to the nearest 200 m for the grid."
)
