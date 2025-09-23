# streamlit_app.py
import io
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Page config (logo optional)
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
CANDIDATE_LOGO_PATHS = [
    APP_DIR / "assets" / "logos.png",
    APP_DIR / "assets" / "logo.png",
    APP_DIR / "logos.png",
    APP_DIR / "logo.png",
]
LOGO_PATH = next((p for p in CANDIDATE_LOGO_PATHS if p.exists()), None)

st.set_page_config(page_title="Race Edge — PI v3.1 + Hidden Horses v2", layout="wide", page_icon=str(LOGO_PATH) if LOGO_PATH else None)
if LOGO_PATH:
    st.image(str(LOGO_PATH), width=260)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.write("### Data source")
    mode = st.radio(" ", ["Upload file", "Manual input"], index=0, label_visibility="collapsed")
    distance_m_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=50)
    st.caption("Rounded distance used: **{} m** (manual grid counts down from here).".format(int(math.ceil(distance_m_input / 200.0) * 200)))
    DEBUG = st.toggle("Debug info", value=False)

def _dbg(*args):
    if DEBUG:
        st.write(*args)

# -----------------------------
# Parsing helpers
# -----------------------------
def parse_time_cell(x):
    """Accept seconds or MM:SS.ms → seconds (float). Empty/invalid → NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    # already numeric?
    try:
        return float(s)
    except Exception:
        pass
    # mm:ss.ms
    m = re.match(r"^(\d+):(\d{2}(?:\.\d+)?)$", s)
    if m:
        return 60 * int(m.group(1)) + float(m.group(2))
    # h:mm:ss?
    m2 = re.match(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$", s)
    if m2:
        return 3600 * int(m2.group(1)) + 60 * int(m2.group(2)) + float(m2.group(3))
    # plain float with stray chars
    m3 = re.findall(r"\d+(?:\.\d+)?", s)
    try:
        return float(m3[-1]) if m3 else np.nan
    except Exception:
        return np.nan

def round_distance_for_grid(distance_m):
    """Manual grid counts down from rounded-up 200 m."""
    return int(math.ceil(distance_m / 200.0) * 200)

# -------------------------------------
# Manual grid constructor (200 m steps)
# -------------------------------------
def manual_grid(distance_m, n_horses):
    rows = []
    rounded = round_distance_for_grid(distance_m)
    markers = list(range(rounded, 0, -200)) + ["Finish"]
    cols = ["Horse"] + [f"{m}_Time" for m in markers] + [f"{m}_Pos" for m in markers]
    for i in range(n_horses):
        row = {c: "" for c in cols}
        row["Horse"] = f"Runner {i+1}"
        rows.append(row)
    return pd.DataFrame(rows), markers

# -------------------------------------
# Build core metrics from raw dataframe
# -------------------------------------
def build_metrics(df_raw, distance_m: int):
    df = df_raw.copy()

    # normalise time columns: anything like "<integer>_Time" and "Finish_Time"
    time_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_Time")]
    # keep only numeric prefixes or "Finish"
    def _prefix_ok(c):
        p = c.split("_")[0]
        return p.isdigit() or p.lower() == "finish"
    time_cols = [c for c in time_cols if _prefix_ok(c)]
    if not time_cols:
        raise ValueError("No *_Time columns detected.")

    # Order segments from START→FINISH (left to right)
    def _key(c):
        p = c.split("_")[0]
        return (10**6 if p.lower()=="finish" else int(p))
    time_cols_sorted = sorted(time_cols, key=_key, reverse=True)  # highest marker first → start of race

    # Parse times to seconds
    for c in time_cols_sorted:
        df[c] = pd.to_numeric(df[c].apply(parse_time_cell), errors="coerce")

    # Race time (if not supplied)
    if "Race Time" in df.columns:
        df["RaceTime_s"] = pd.to_numeric(df["Race Time"].apply(parse_time_cell), errors="coerce")
    else:
        # sum 100 m segments OR 200 m if provided
        seg_len = infer_seg_len(time_cols_sorted)
        total_m = seg_len * len(time_cols_sorted)
        scale = distance_m / total_m if total_m else 1.0
        df["RaceTime_s"] = df[time_cols_sorted].sum(axis=1) * (1.0/scale)  # rescale if needed

    # Build per-100m speeds by splitting 200 m segments if necessary
    seg_len = infer_seg_len(time_cols_sorted)
    # Construct a list of per-100m speeds from available columns:
    # - if seg_len==100 → direct 100 m speeds
    # - if seg_len==200 → duplicate each 200 m time into two equal 100 m (assumption for visual pacing)
    per100_speeds = []
    if seg_len == 100:
        per100_speeds = [100.0 / df[c] for c in time_cols_sorted]
    else:
        # split each 200 m time equally into two 100 m chunks
        for c in time_cols_sorted:
            t200 = df[c]
            s100 = 100.0 / (t200 / 2.0)
            per100_speeds.extend([s100, s100])

    # make a 2D array (n, k) speeds from start→finish
    S = np.column_stack([s.values for s in per100_speeds]) if per100_speeds else np.empty((df.shape[0], 0))

    # Race avg speed (per-runner)
    with np.errstate(divide="ignore", invalid="ignore"):
        race_avg = distance_m / df["RaceTime_s"].values

    # Pointers into S for phases
    n_cols = S.shape[1]
    # ensure we have at least 10x100m for robust splits; if not, degrade gracefully
    first2 = slice(0, min(2, n_cols))
    last2 = slice(max(0, n_cols-2), n_cols)
    last6 = slice(max(0, n_cols-6), n_cols-2)
    mid = slice(2, max(2, n_cols-6))  # exclude first 200 & last 600

    # Component indices (each around ~100)
    def _comp_idx(speed_slice):
        if speed_slice.stop <= speed_slice.start:
            return np.full(df.shape[0], np.nan)
        v = np.nanmean(S[:, speed_slice], axis=1)
        return 100.0 * v / race_avg

    f200_idx = _comp_idx(first2)
    accel_idx = _comp_idx(last6)     # 600→200 (mean of last6)
    grind_idx = _comp_idx(last2)     # 200→Finish (mean of last2)
    tsspi_idx = _comp_idx(mid)       # cruising excluding first 200 & last 600

    metrics = pd.DataFrame({
        "Horse": df.get("Horse", pd.Series([f"Runner {i+1}" for i in range(df.shape[0])])) ,
        "RaceTime_s": df["RaceTime_s"].round(2),
        "F200_idx": f200_idx,
        "tsSPI": tsspi_idx,
        "Accel": accel_idx,
        "Grind": grind_idx
    })

    # Replace impossible or inf
    for c in ["F200_idx", "tsSPI", "Accel", "Grind"]:
        metrics[c] = metrics[c].replace([np.inf, -np.inf], np.nan)

    # Dist. buckets (for nudge)
    def bucket(dm):
        if dm <= 1200: return "SPRINT"
        if dm <= 1600: return "MILE"
        if dm <= 2000: return "MIDDLE"
        return "STAY"

    bkt = bucket(distance_m)

    # Field medians for "points vs field"
    med = metrics[["F200_idx", "tsSPI", "Accel", "Grind"]].median(numeric_only=True)
    dev_cols = {}
    for c in ["F200_idx", "tsSPI", "Accel", "Grind"]:
        dev_cols[c + "_pts"] = metrics[c] - med[c]
    dev = pd.DataFrame(dev_cols)

    # PI v3 weights + shape nudge
    wF, wS, wA, wG = 0.08, 0.37, 0.30, 0.25

    # base PI is weighted sum of *deviations vs field* (in index points)
    base_PI = (wF*dev["F200_idx_pts"] + wS*dev["tsSPI_pts"] + wA*dev["Accel_pts"] + wG*dev["Grind_pts"])

    # shape nudge: reward balance; penalise one-legged profiles
    # scale differs a touch by trip
    if bkt == "SPRINT":
        pos_gate = (dev["Accel_pts"] > 0) & (dev["Grind_pts"] > 0)
        neg_gate = (dev["Accel_pts"] < 0) ^ (dev["Grind_pts"] < 0)
        nudge = np.where(pos_gate, 0.35, 0.0) + np.where(neg_gate, -0.20, 0.0)
    elif bkt == "MILE":
        nudge = np.where((dev["Accel_pts"] > 0) & (dev["Grind_pts"] > 0), 0.25, 0.0) + \
                np.where((dev["Accel_pts"] < 0) ^ (dev["Grind_pts"] < 0), -0.15, 0.0)
    elif bkt == "MIDDLE":
        nudge = np.where((dev["Accel_pts"] > 0) & (dev["Grind_pts"] > 0), 0.20, 0.0) + \
                np.where((dev["Accel_pts"] < 0) ^ (dev["Grind_pts"] < 0), -0.10, 0.0)
    else:
        nudge = np.where((dev["Accel_pts"] > 0) & (dev["Grind_pts"] > 0), 0.15, 0.0) + \
                np.where((dev["Accel_pts"] < 0) ^ (dev["Grind_pts"] < 0), -0.10, 0.0)

    PI = base_PI + nudge

    # Lightweight contextual GCI: late quality + on-pace merit
    # (kept simple here so the table always shows something stable)
    late_quality = np.maximum(0.0, metrics["Grind"] - 100.0) * 0.6 + np.maximum(0.0, metrics["Accel"] - 100.0) * 0.4
    cruise_bonus = np.maximum(0.0, metrics["tsSPI"] - 100.0) * 0.25
    GCI = (late_quality * 0.02) + (cruise_bonus * 0.02)  # compress into ~0–5 band

    out = metrics.copy()
    out["PI"] = PI.round(3)
    out["GPI"] = GCI.round(3)  # named GPI in table as requested earlier
    # Attach finish pos if present
    fin_col = None
    for c in ["Finish_Pos", "finish_pos", "Placing", "Result"]:
        if c in df.columns:
            fin_col = c; break
    if fin_col:
        out["Finish_Pos"] = pd.to_numeric(df[fin_col], errors="coerce")
    # draw if present
    if "Draw" in df.columns:
        out["Draw"] = df["Draw"]

    # Hidden Horses v2 (hybrid): against-shape + under-ranked PI
    #    - tsSPI <= field median and PI >= median OR
    #    - Strong Accel & Grind but outside the first 3 home
    field_med_PI = np.nanmedian(out["PI"])
    field_med_ts = np.nanmedian(out["tsSPI"])
    hidden_flags = (
        ((out["tsSPI"] <= field_med_ts) & (out["PI"] >= field_med_PI)) |
        ((out["Accel"] >= med["Accel"]) & (out["Grind"] >= med["Grind"]) & (out.get("Finish_Pos", 99) > 3))
    )
    out["Hidden"] = hidden_flags

    # keep devs for visuals + stacked bars
    for c in dev.columns:
        out[c] = dev[c]

    return out, time_cols_sorted, seg_len

def infer_seg_len(time_cols_sorted):
    """Guess segment length from column cadence (100 m vs 200 m)."""
    # if we see both 1100_Time and 1000_Time between, probably 100 m
    numeric_markers = []
    for c in time_cols_sorted:
        p = c.split("_")[0]
        if p.isdigit():
            numeric_markers.append(int(p))
    if len(numeric_markers) >= 2:
        diffs = [abs(a - b) for a, b in zip(numeric_markers[:-1], numeric_markers[1:])]
        step = pd.Series(diffs).median()
        if step <= 150:
            return 100
        return 200
    # fallback
    return 200

# -------------------------------------
# Visuals
# -------------------------------------
def shape_map(df_metrics):
    """Accel (x) vs Grind (y), colour = tsSPI deviation (vs field), label with arrows."""
    tdf = df_metrics.copy()
    # deviations vs field (points)
    for c in ["Accel", "Grind", "tsSPI"]:
        tdf[c + "_dev"] = tdf[c] - np.nanmedian(tdf[c])
    x = tdf["Accel_dev"]; y = tdf["Grind_dev"]; c = tdf["tsSPI_dev"]
    names = tdf["Horse"].fillna("Runner")

    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    sc = ax.scatter(x, y, c=c, cmap="coolwarm", s=80, edgecolor="white", linewidth=0.6, alpha=0.95)
    # zero axes
    ax.axhline(0, color="#666", ls="--", lw=1, alpha=0.7)
    ax.axvline(0, color="#666", ls="--", lw=1, alpha=0.7)
    # light quadrant tint
    ax.set_facecolor("#fff")
    ax.grid(True, ls=":", alpha=0.35)
    ax.set_xlabel("Acceleration vs field (points →, 600→200)")
    ax.set_ylabel("Grind vs field (points ↑, 200→Finish)")
    ax.set_title("Sectional Shape Map — Accel vs Grind\nColour = tsSPI (points vs field)")

    # annotate all points with unobtrusive arrows to text
    for xi, yi, name in zip(x, y, names):
        if pd.isna(xi) or pd.isna(yi): 
            continue
        # offset text slightly away from point
        tx, ty = xi + 0.15, yi + 0.25
        ax.annotate(str(name), xy=(xi, yi), xytext=(tx, ty),
                    textcoords="data", fontsize=8, color="#222",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.7, alpha=0.9))

    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("tsSPI – field (points)")
    st.pyplot(fig)
    st.caption("Each bubble is a runner.  Right = stronger late acceleration; Up = stronger final 200.  Warmer colour = faster cruise (tsSPI) vs field.")

def pace_curve(df_raw, df_metrics, time_cols_sorted, seg_len, top_n=8):
    """Pace over 200 m segments. Field avg in black + top-N finishers in thin distinct colours."""
    # Build per-200m speeds from raw (if we have 100m, aggregate pairs).
    df = df_raw.copy()
    for c in time_cols_sorted:
        df[c] = pd.to_numeric(df[c].apply(parse_time_cell), errors="coerce")

    # Get horses and optional finish ordering
    names = df.get("Horse", pd.Series([f"Runner {i+1}" for i in range(df.shape[0])]))
    finish = pd.to_numeric(df.get("Finish_Pos", pd.Series(np.arange(1, df.shape[0]+1))), errors="coerce")

    # Build per-100 speeds as earlier
    if seg_len == 100:
        per100 = [100.0 / df[c] for c in time_cols_sorted]  # list of series
    else:
        per100 = []
        for c in time_cols_sorted:
            s100 = 100.0 / (df[c]/2.0)
            per100 += [s100, s100]

    S100 = np.column_stack([s.values for s in per100]) if per100 else np.empty((df.shape[0], 0))

    # Aggregate into 200 m segments: pair consecutive 100s
    if S100.shape[1] % 2 == 1:
        S100 = S100[:, :-1]  # drop last odd if any
    S200 = (S100[:, 0::2] + S100[:, 1::2]) / 2.0
    nseg = S200.shape[1]

    # Labels from left (early) to right (home straight)
    # Derive from highest marker; e.g., 1400→1200→...→0–200
    # Best-effort textual ticks:
    if len(time_cols_sorted) > 0 and time_cols_sorted[0].split("_")[0].isdigit():
        start_m = int(time_cols_sorted[0].split("_")[0])
    else:
        start_m = round_distance_for_grid(int(st.session_state.get("_last_distance", 1400)))
    # 200 m ticks descending:
    xticks = [f"{start_m - 200*i}m" if start_m - 200*i > 0 else "0–200m" for i in range(nseg)]
    xticks = xticks[:nseg]

    # Field average
    field_avg = np.nanmean(S200, axis=0)

    # Top-N finishers
    order = np.argsort(finish.values)
    keep_idx = order[:min(top_n, len(order))]

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.plot(range(nseg), field_avg, color="black", lw=2.8, marker="o", ms=4, label="Field average")
    for i in keep_idx:
        ax.plot(range(nseg), S200[i], lw=1.3, marker="o", ms=3.2, alpha=0.85, label=str(names.iloc[i]))
    ax.set_xticks(range(nseg))
    ax.set_xticklabels(xticks, rotation=35, ha="right")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Pace over 200 m segments (left = early, right = home straight)")
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    st.pyplot(fig)
    st.caption("Black = field average. Coloured = top 8 by finish. Thinner lines & small markers keep overlapping traces readable.")

def stacked_pi_bars(df_metrics, top_n=8):
    """Stacked bar showing PI contributions by metric for the top-N PI runners."""
    t = df_metrics.sort_values("PI", ascending=False).head(top_n).copy()
    # contributions are weights × (points vs field)
    w = dict(F200_idx=0.08, tsSPI=0.37, Accel=0.30, Grind=0.25)
    contrib = pd.DataFrame({
        "F200": w["F200_idx"] * t["F200_idx_pts"],
        "tsSPI": w["tsSPI"] * t["tsSPI_pts"],
        "Accel": w["Accel"] * t["Accel_pts"],
        "Grind": w["Grind"] * t["Grind_pts"],
    }, index=t["Horse"])
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bottom = np.zeros(len(contrib))
    colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]
    for i, col in enumerate(contrib.columns):
        ax.bar(contrib.index, contrib[col], bottom=bottom, label=col, color=colors[i], alpha=0.9)
        bottom += contrib[col].values
    ax.set_ylabel("PI contribution (points)")
    ax.set_title("Top-8 PI — stacked contributions")
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    st.pyplot(fig)
    st.caption("Each bar shows where the PI came from (F200 / tsSPI / Accel / Grind).")

def runner_notes(df_metrics, distance_m):
    """Simple runner-by-runner readout."""
    t = df_metrics.copy()
    med = t[["F200_idx", "tsSPI", "Accel", "Grind"]].median(numeric_only=True)
    lines = []
    for _, r in t.sort_values("Finish_Pos", na_position="last").iterrows():
        name = str(r["Horse"])
        pi = r["PI"]; gpi = r["GPI"]
        desc = []
        # shape
        ax, gy = r["Accel"] - med["Accel"], r["Grind"] - med["Grind"]
        if not (pd.isna(ax) or pd.isna(gy)):
            if ax > 1 and gy > 1: desc.append("powerful close (accel+grind)")
            elif ax > 1: desc.append("late acceleration evident")
            elif gy > 1: desc.append("kept finding late")
            elif ax < -1 and gy < -1: desc.append("flattened late")
        # distance hint
        tss = r["tsSPI"] - 100
        if gy > 1.5 and tss < 0:
            desc.append("likely to appreciate further")
        elif ax < -1 and tss > 1.5:
            desc.append("may prefer a shade shorter or stronger early pace")
        # hidden flag
        if bool(r.get("Hidden", False)):
            desc.append("hidden merit flagged")
        # assemble
        lines.append(f"**{name}** — PI {pi:.3f}; GPI {gpi:.3f}. " + ("; ".join(desc) if desc else "clean run."))
    st.markdown("\n\n".join(lines))

# -------------------------------------
# Input
# -------------------------------------
df_raw = None
rounded_for_manual = round_distance_for_grid(distance_m_input)

try:
    if mode == "Upload file":
        up = st.file_uploader("Upload CSV/XLSX with *_Time columns (100 m or 200 m), optional *_Pos, Finish_Pos", type=["csv", "xlsx"])
        if up is None:
            st.stop()
        if up.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(up)
        else:
            df_raw = pd.read_excel(up)
        st.success("File loaded.")
    else:
        n_h = st.number_input("Number of horses (manual)", 2, 20, 8)
        grid, markers = manual_grid(distance_m_input, n_h)
        st.write("Enter **segment times** (seconds) counting down from the rounded distance, and optional positions.")
        df_entry = st.data_editor(grid, use_container_width=True, num_rows="fixed")
        # Build df_raw from editor (keep as-is; we’ll parse times later)
        df_raw = df_entry.copy()
        st.info("Manual mode: enter segment times only. We’ll compute all indices from this table.")
except Exception as e:
    st.error("Input parsing failed.")
    _dbg(e)
    st.stop()

st.markdown("### Raw / Converted Table")
st.dataframe(df_raw.head(20), use_container_width=True)
st.session_state["_last_distance"] = distance_m_input

# -------------------------------------
# Compute metrics
# -------------------------------------
try:
    metrics, time_cols_sorted, seg_len = build_metrics(df_raw, int(distance_m_input))
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG:
        st.exception(e)
    st.stop()

# Display metrics table
show_cols = ["Horse"]
if "Finish_Pos" in metrics.columns:
    show_cols += ["Finish_Pos"]
show_cols += ["RaceTime_s", "F200_idx", "tsSPI", "Accel", "Grind", "PI", "GPI"]
st.markdown("## Sectional Metrics (PI & GPI)")
st.dataframe(metrics[show_cols].round(3), use_container_width=True)

# -------------------------------------
# Visual 1: Sectional Shape Map
# -------------------------------------
shape_map(metrics)

# -------------------------------------
# Visual 2: Pace Curve
# -------------------------------------
pace_curve(df_raw, metrics, time_cols_sorted, seg_len, top_n=8)

# -------------------------------------
# Visual 3: Top-8 PI stacked bars
# -------------------------------------
stacked_pi_bars(metrics, top_n=8)

# -------------------------------------
# Hidden Horses table
# -------------------------------------
st.markdown("## Hidden Horses")
hh = metrics.loc[metrics["Hidden"]].copy()
if hh.empty:
    st.write("No hidden horses flagged today under the current hybrid rules.")
else:
    st.dataframe(hh[["Horse", "Finish_Pos", "PI", "tsSPI", "Accel", "Grind"]].round(3).sort_values("PI", ascending=False), use_container_width=True)

# -------------------------------------
# Runner-by-runner notes
# -------------------------------------
st.markdown("## Runner by runner")
runner_notes(metrics, int(distance_m_input))

# Footer
st.caption("PI v3.1 (F200 0.08, tsSPI 0.37, Accel 0.30, Grind 0.25) with shape nudge; tsSPI excludes first 200 m and last 600 m.  "
           "GPI is a lightweight group potential read (late quality + cruise).  Pace curve aggregates 100 m into 200 m segments.")
