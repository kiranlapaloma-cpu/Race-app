# streamlit_app.py
# RaceEdge — Sectional Analysis (PI v3.1+ / Hidden Horses v2 / GPI v0.95)
# ----------------------------------------------------------------------
# What this file does:
# • Robustly parses CSV/XLSX race split tables (100m or 200m segments).
# • Orders *_Time columns numerically (e.g., 1400_Time, ..., 100_Time, Finish_Time).
# • Treats each *_Time as a segment time (NOT cumulative) in seconds.
# • Computes segment speeds correctly (handles 100m and 200m).
# • Implements F200 / tsSPI / Accel (600→200) / Grind (200→Finish) metrics.
# • PI = weighted blend of the four metrics (+ small shape nudge), field-size guarded.
# • GPI (group potential) kept alongside PI in the table (no extra visuals, per your ask).
# • Visuals: Sectional Shape Map (fully labeled), Pace Curve (field avg + top-8),
#             Top-8 PI stacked bars, Hidden Horses table.
# • Clear legends/keys; thin lines + small markers for readability.

from __future__ import annotations
import math
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------- Utilities ----------------------------------------------------------

def to_seconds(x):
    """Convert time-like values to seconds. Accepts float/int, 'SS.sss',
    or 'MM:SS(.sss)'. Invalid/empty -> np.nan."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        v = float(x)
        return np.nan if v <= 0 else v
    s = str(x).strip()
    if not s:
        return np.nan
    # mm:ss(.sss) ?
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(mm) * 60.0 + float(ss)
        except Exception:
            return np.nan
    # plain seconds
    try:
        v = float(s)
        return np.nan if v <= 0 else v
    except Exception:
        return np.nan


def extract_marker(col: str) -> int | None:
    """Return numeric distance marker from '<NNNN>_Time' columns."""
    m = re.match(r"^(\d+)_Time$", col.strip())
    if not m:
        return None
    return int(m.group(1))


def infer_segment_len(markers: List[int]) -> int:
    """Infer segment length from consecutive markers (100 or 200)."""
    if len(markers) < 2:
        return 100  # default
    diffs = sorted({abs(a - b) for a, b in zip(markers[:-1], markers[1:])})
    # keep the smallest positive difference
    for d in diffs:
        if d > 0:
            return d
    return 100


def order_time_columns(df: pd.DataFrame) -> Tuple[List[str], List[int], int]:
    """Return ordered *_Time columns (far -> near), their numeric markers, and seg_len."""
    time_cols = [c for c in df.columns if c.endswith("_Time")]
    markers = [extract_marker(c) for c in time_cols]
    pairs = [(m, c) for m, c in zip(markers, time_cols) if m is not None]
    # order farthest first (e.g., 1400, 1300, ..., 100), Finish handled separately
    pairs_sorted = sorted(pairs, key=lambda z: z[0], reverse=True)
    ordered_cols = [c for _, c in pairs_sorted]
    ordered_markers = [m for m, _ in pairs_sorted]
    seg_len = infer_segment_len(ordered_markers) if ordered_markers else 100
    return ordered_cols, ordered_markers, seg_len


def safe_mean(a: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    a = a.replace([np.inf, -np.inf], np.nan)
    return float(np.nanmean(a.values)) if a.size else np.nan


def safe_std(a: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    a = a.replace([np.inf, -np.inf], np.nan)
    return float(np.nanstd(a.values, ddof=0)) if a.size else np.nan


def idx_vs_field(series_vals: pd.Series) -> float:
    """Return points vs field average = 100 * (runner/field - 1)."""
    series_vals = pd.to_numeric(series_vals, errors="coerce")
    fld = safe_mean(series_vals)
    me = series_vals.iloc[0] if series_vals.size else np.nan
    if np.isnan(me) or np.isnan(fld) or fld <= 0:
        return np.nan
    return 100.0 * (float(me) / float(fld) - 1.0)


def standardize(values: pd.Series) -> pd.Series:
    """z-score with NaN safety (population std)."""
    m = safe_mean(values)
    s = safe_std(values)
    if np.isnan(m) or np.isnan(s) or s == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - m) / s


# ---------- Metric engine ------------------------------------------------------

def compute_segment_speeds(row: pd.Series, time_cols: List[str], seg_len: int) -> Dict[int, float]:
    """Return dict {marker : speed(m/s)} for each time column.
    Each *_Time is the segment time for that slice (NOT cumulative)."""
    out = {}
    for c in time_cols:
        marker = extract_marker(c)
        if marker is None:
            continue
        t = to_seconds(row.get(c, np.nan))
        if np.isnan(t) or t <= 0:
            out[marker] = np.nan
        else:
            out[marker] = float(seg_len) / float(t)  # m/s
    return out


def compute_all_speeds(df: pd.DataFrame, time_cols: List[str], seg_len: int) -> Dict[str, Dict[int, float]]:
    """For each horse -> {marker: speed}."""
    speeds: Dict[str, Dict[int, float]] = {}
    for _, r in df.iterrows():
        name = str(r.get("Horse", f"Runner {r.name}"))
        speeds[name] = compute_segment_speeds(r, time_cols, seg_len)
    return speeds


def window_markers(markers: List[int], start_from: int, end_to: int) -> List[int]:
    """Markers are distances-to-finish (e.g., 1400,1300,...,100).
       Return those in [end_to, start_from], inclusive, descending."""
    return [m for m in markers if end_to <= m <= start_from]


def compute_metrics(df: pd.DataFrame, distance_m: int) -> Tuple[pd.DataFrame, List[int], int, Dict[str, Dict[int, float]]]:
    # 1) Identify & order time columns
    time_cols, markers, seg_len = order_time_columns(df)
    if not time_cols:
        raise ValueError("No '*_Time' columns found.")
    # Ensure we only use columns within this race distance
    markers = [m for m in markers if m <= distance_m]
    time_cols = [f"{m}_Time" for m in markers if f"{m}_Time" in time_cols]

    # 2) Build segment speed map
    speed_map = compute_all_speeds(df, time_cols, seg_len)

    # 3) Assemble speed matrix (rows=runners, cols=markers desc)
    names = [str(x) for x in df["Horse"].astype(str)]
    S = pd.DataFrame(
        {name: [speed_map[name].get(m, np.nan) for m in markers] for name in names},
        index=markers  # index are markers (desc)
    ).T  # shape: (n_horses, n_markers)

    # 4) Field means by marker (for indices vs field)
    field_by_marker = S.apply(safe_mean, axis=0)  # per column

    # 5) Metric slices
    # F200: first 200 m of race (the two farthest markers from finish)
    f200_window = markers[: max(1, int(200 / seg_len))]
    # Accel: late surge 600→200 m (exclude last 200)
    accel_window = window_markers(markers, start_from=600, end_to=300 if seg_len == 100 else 400)
    # Grind: last 200 m (average of last 2 x 100m, or last 1 x 200m)
    grind_window = window_markers(markers, start_from=200, end_to=seg_len)

    # tsSPI: trip-scaled mid-race cruise — exclude first 200 m and last 600 m
    ts_start = distance_m - 200
    ts_end = 700 if seg_len == 100 else 600  # exclude to 600
    ts_window = [m for m in markers if (m <= ts_start) and (m >= ts_end)]
    if len(ts_window) < max(2, 400 // seg_len):  # fallback if too short
        # Use middle third as backup
        k = len(markers)
        ts_window = markers[max(1, k // 3): max(2, 2 * k // 3)]

    def avg_speed_over(cols: List[int], row: pd.Series) -> float:
        return safe_mean(row[cols]) if cols else np.nan

    # Runner metrics
    f200 = S.apply(lambda r: avg_speed_over(f200_window, r), axis=1)
    accel = S.apply(lambda r: avg_speed_over(accel_window, r), axis=1)
    grind = S.apply(lambda r: avg_speed_over(grind_window, r), axis=1)
    tsspi_raw = S.apply(lambda r: avg_speed_over(ts_window, r), axis=1)

    # Convert each to "points vs field" (relative to field means over the same windows)
    def points_vs_field(series_vals: pd.Series, cols: List[int]) -> pd.Series:
        if not cols:
            return pd.Series([np.nan] * len(series_vals), index=series_vals.index)
        fld = safe_mean(S[cols].apply(safe_mean, axis=0))  # average field speed across those cols
        if np.isnan(fld) or fld <= 0:
            return pd.Series([np.nan] * len(series_vals), index=series_vals.index)
        return 100.0 * (series_vals / fld - 1.0)

    f200_pts = points_vs_field(f200, f200_window)
    accel_pts = points_vs_field(accel, accel_window)
    grind_pts = points_vs_field(grind, grind_window)
    tsspi_pts = points_vs_field(tsspi_raw, ts_window)

    # 6) PI — weights + shape nudge (guard small fields)
    # Defaults you set:
    w_f200, w_ts, w_acc, w_gr = 0.08, 0.37, 0.30, 0.25
    # Shape nudge: reward balanced profiles slightly (within 1 std of field on all)
    # (Compute z-scores on points)
    z_f200 = standardize(f200_pts)
    z_ts = standardize(tsspi_pts)
    z_acc = standardize(accel_pts)
    z_gr = standardize(grind_pts)
    balanced_mask = (z_acc.abs() < 1.0) & (z_gr.abs() < 1.0) & (z_ts.abs() < 1.0)
    shape_nudge = balanced_mask.astype(float) * 0.15  # small, not dominant

    # Field-size guard: shrink volatility when n small
    n = len(S)
    guard = 1.0 if n >= 10 else (0.85 + 0.015 * n)  # from ~1.0 down to 0.985 for small fields

    PI = guard * (w_f200 * f200_pts + w_ts * tsspi_pts + w_acc * accel_pts + w_gr * grind_pts) + shape_nudge

    # 7) GPI — keep simple: standardized blend of (tsSPI + Accel + Grind), scaled to 0..~5
    gpi_z = (standardize(tsspi_pts) * 0.5 + standardize(accel_pts) * 0.25 + standardize(grind_pts) * 0.25)
    GPI = (gpi_z - gpi_z.min()) / (gpi_z.max() - gpi_z.min() + 1e-9) * 5.0

    # Build metrics table
    out = pd.DataFrame({
        "Horse": names,
        "RaceTime_s": pd.to_numeric(df.get("Race Time", df.get("RaceTime_s", np.nan)).apply(to_seconds), errors="coerce"),
        "F200": f200_pts.round(3),
        "tsSPI": tsspi_pts.round(3),
        "Accel": accel_pts.round(3),
        "Grind": grind_pts.round(3),
        "PI": PI.round(3),
        "GPI": GPI.round(3),
    })
    # Attach Finish_Pos if present
    if "Finish_Pos" in df.columns:
        out.insert(1, "Finish_Pos", pd.to_numeric(df["Finish_Pos"], errors="coerce"))
    elif "Finish_Pos" not in out.columns and "Finish Position" in df.columns:
        out.insert(1, "Finish_Pos", pd.to_numeric(df["Finish Position"], errors="coerce"))

    # Sort by PI desc
    out = out.sort_values("PI", ascending=False).reset_index(drop=True)
    return out, markers, seg_len, speed_map


# ---------- Hidden horses (current version you approved) ----------------------

def hidden_horses_table(metrics: pd.DataFrame) -> pd.DataFrame:
    # Hidden if: PI below winner by <= 1.0 “point” but tsSPI ≥ +2, or
    # Accel/Grind jointly ≥ +4 with tsSPI ≥ 0.
    if metrics.empty:
        return metrics
    pi_max = float(metrics["PI"].max())
    candidates = metrics[
        ((pi_max - metrics["PI"]) <= 1.0) & (metrics["tsSPI"] >= 2.0)
        |
        (((metrics["Accel"] + metrics["Grind"]) >= 4.0) & (metrics["tsSPI"] >= 0.0))
    ].copy()
    return candidates.sort_values(["PI", "tsSPI"], ascending=False).reset_index(drop=True)


# ---------- Visuals -----------------------------------------------------------

def plot_shape_map(metrics: pd.DataFrame):
    # Scatter: X = Accel pts vs field, Y = Grind pts vs field, color = tsSPI deviation
    x = metrics["Accel"].astype(float)
    y = metrics["Grind"].astype(float)
    c = metrics["tsSPI"].astype(float)  # already points vs field
    names = metrics["Horse"].astype(str)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=150)
    sc = ax.scatter(x, y, c=c, cmap="coolwarm", s=90, edgecolor="white", linewidth=0.6, alpha=0.95, zorder=3)

    # Quadrants and zero-lines
    ax.axhline(0, color="#999999", lw=1.0, ls="--", zorder=1)
    ax.axvline(0, color="#999999", lw=1.0, ls="--", zorder=1)
    ax.set_xlabel("Acceleration vs field (points) →")
    ax.set_ylabel("Grind vs field (points) ↑")
    ax.set_title("Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)\nColour = tsSPI (points vs field)")

    # Label each bubble with an offset + hairline to improve readability
    for xi, yi, nm in zip(x, y, names):
        if pd.isna(xi) or pd.isna(yi):
            continue
        dx = 0.12
        dy = 0.12
        ax.annotate(nm, xy=(xi, yi), xytext=(xi + dx, yi + dy),
                    textcoords="data",
                    fontsize=8, color="#222222",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6, shrinkA=0, shrinkB=0))

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("tsSPI – field (points)")
    ax.grid(True, linestyle=":", alpha=0.4)
    st.pyplot(fig)


def plot_pace_curve(df: pd.DataFrame, markers: List[int], seg_len: int, speed_map: Dict[str, Dict[int, float]]):
    # Field average
    cols = [f"{m}_Time" for m in markers if f"{m}_Time" in df.columns]
    names = [str(x) for x in df["Horse"].astype(str)]
    # Speeds table
    S = pd.DataFrame({name: [speed_map[name].get(m, np.nan) for m in markers] for name in names}, index=markers).T
    field = S.apply(safe_mean, axis=0)

    # Order x-axis from earliest (left) to finish (right): i.e., far→near decreasing markers reversed for plotting
    x_labels = [f"{m}-{m - seg_len if (m - seg_len) > 0 else 0}m" for m in markers]
    x = list(range(len(markers)))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot field average in black, thick
    ax.plot(x, field.values, label="Field average", lw=2.2, color="black", marker="o", markersize=2.5)

    # Top-8 by PI (if PI already computed; else use median of last segment)
    # We'll accept any 'PI' order present in df_metrics later — for now plot all with thin lines
    palette = plt.cm.tab10(np.linspace(0, 1, min(10, len(names))))
    for i, name in enumerate(names[:8]):
        y = [speed_map[name].get(m, np.nan) for m in markers]
        ax.plot(x, y, lw=1.2, marker="o", markersize=2.5, label=name, color=palette[i % len(palette)], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Pace over 200 m segments (left = early, right = home straight)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left")
    st.pyplot(fig)

    st.caption(
        "Key: each dot is a segment. Leftmost = first run segment (farthest from finish); "
        "rightmost = home straight. Lines are thin with small markers for readability. "
        "Black line is field average."
    )


def plot_top8_bars(metrics: pd.DataFrame):
    # Stacked bars for PI contributions (Top-8 by PI)
    top = metrics.nlargest(8, "PI").copy()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

    x = np.arange(len(top))
    wF, wT, wA, wG = 0.08, 0.37, 0.30, 0.25
    # Convert “points vs field” into weighted contributions
    F = wF * top["F200"]
    T = wT * top["tsSPI"]
    A = wA * top["Accel"]
    G = wG * top["Grind"]

    ax.bar(x, F, label="F200", width=0.6)
    ax.bar(x, T, bottom=F, label="tsSPI", width=0.6)
    ax.bar(x, A, bottom=(F + T), label="Accel", width=0.6)
    ax.bar(x, G, bottom=(F + T + A), label="Grind", width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(top["Horse"], rotation=30, ha="right")
    ax.set_ylabel("PI contribution (points weighted)")
    ax.set_title("Top-8 PI — stacked contributions")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(ncol=4, loc="upper left", fontsize=8, frameon=False)
    st.pyplot(fig)


# ---------- App UI ------------------------------------------------------------

st.set_page_config(page_title="Race Edge — PI v3.1+", layout="wide")

st.title("The Sharpest Edge")
st.caption("Upload CSV/XLSX split table or use manual entry (200 m grid). Calculates PI & GPI; shows Shape Map, Pace Curve, and Hidden Horses.")

# Sidebar
with st.sidebar:
    st.subheader("Data source")
    mode = st.radio("", ["Upload file", "Manual input"], index=0, horizontal=False)
    distance_m = st.number_input("Race Distance (m)", min_value=800, max_value=3600, step=100, value=1200)
    st.caption("Using rounded distance: 200 m rounding for manual grid (counts down from here).")
    debug = st.toggle("Debug info", value=False)

# Data intake
if mode == "Upload file":
    f = st.file_uploader("CSV or Excel with split columns like '1400_Time' ... '100_Time', 'Finish_Time'", type=["csv", "xlsx", "xls"])
    if not f:
        st.info("Upload a file to begin.")
        st.stop()
    if f.name.lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(f)
    else:
        df_raw = pd.read_csv(f)
else:
    # Minimal manual grid: you can extend columns as needed; we keep it simple here
    # Create a blank grid based on rounded distance (countdown 200m)
    rounded = int(math.ceil(distance_m / 200.0) * 200)
    segs = list(range(rounded, 0, -200))
    base_cols = ["Horse"]
    for m in segs:
        base_cols.append(f"{m}_Time")
    base_cols.append("Finish_Time")
    st.caption(f"Manual grid (countdown every 200m) from {rounded}m.")
    df_raw = st.data_editor(pd.DataFrame(columns=base_cols), num_rows="dynamic", use_container_width=True)
    if df_raw.empty:
        st.info("Enter at least one row of manual times.")
        st.stop()

# Safety: standardize col names
df_raw.columns = [str(c).strip() for c in df_raw.columns]

# Compute metrics
try:
    metrics, markers, seg_len, speed_map = compute_metrics(df_raw, distance_m)
except Exception as e:
    st.error(f"Metric computation failed.\n\n{e}")
    if debug:
        st.code(repr(e))
    st.stop()

# ---------- Output: Metrics table ----------
st.markdown("## Sectional Metrics (PI & GPI)")
st.dataframe(metrics[["Horse"] + ([c for c in ("Finish_Pos",) if c in metrics.columns]) +
                     ["RaceTime_s", "F200", "tsSPI", "Accel", "Grind", "PI", "GPI"]],
             use_container_width=True)

# ---------- Visual 1: Shape map ----------
plot_shape_map(metrics)

# ---------- Visual 2: Pace curve ----------
plot_pace_curve(df_raw, markers, seg_len, speed_map)

# ---------- Visual 3: Top-8 PI bars ----------
plot_top8_bars(metrics)

# ---------- Hidden Horses ----------
st.markdown("## Hidden Horses")
hh = hidden_horses_table(metrics)
if hh.empty:
    st.info("None flagged by the current hidden-horse rules.")
else:
    st.dataframe(hh[["Horse", "PI", "tsSPI", "Accel", "Grind"]], use_container_width=True)

# ---------- Footnote / Explainers ----------
with st.expander("What the metrics mean"):
    st.markdown(
        "- **F200**: first 200 m speed relative to field (points = 100×(you/field − 1)).\n"
        "- **tsSPI**: mid-race cruise (excludes first 200 m and last 600 m).\n"
        "- **Accel (600→200)**: late surge before the straight.\n"
        "- **Grind (200→Finish)**: ability to sustain through the line.\n"
        "- **PI**: weighted blend (F200 0.08, tsSPI 0.37, Accel 0.30, Grind 0.25) + small balance nudge.\n"
        "- **GPI**: scaled standardized blend of tsSPI, Accel, Grind (0..5)."
    )

if debug:
    st.markdown("### 🔧 Debug")
    st.write({"markers(desc)": markers, "segment_length": seg_len})
    st.write("Sample rows:", df_raw.head())
