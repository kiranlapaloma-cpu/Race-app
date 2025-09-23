# streamlit_app.py
# RaceEdge — PI v3 (F200/tsSPI/Accel/Grind) + Shape Map + Pace Curve + Top-8 bars
# Includes robust finish-time parsing and fallback from split times.

from __future__ import annotations
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Page / theme
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Race Edge — PI v3", layout="wide")

# ----------------------------
# Sidebar: controls
# ----------------------------
with st.sidebar:
    st.header("Data source")
    source = st.radio("",
                      ["Upload file", "Manual input"],
                      index=0,
                      label_visibility="collapsed")
    distance_m_input = st.number_input("Race Distance (m)",
                                       min_value=800, max_value=4000,
                                       value=1200, step=50)
    st.caption("Using rounded distance: **{} m** (manual grid counts down from here).".format(
        int(np.ceil(distance_m_input/100.0)*100)
    ))
    DEBUG = st.toggle("Debug info", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# ----------------------------
# Robust time parsing
# ----------------------------
def parse_time_cell(x) -> float:
    """Return seconds as float for strings like:
       - '73.25'
       - '01:11.95'
       - '01:11:950'  (mm:ss:ms)
       - '0:59:12'    (m:s:cs)
       - '1:12'       (m:ss)
       Fallback: NaN
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan

    # numeric
    try:
        return float(s)
    except Exception:
        pass

    # mm:ss.ms
    m = re.match(r"^(\d+):(\d{2})(?:\.(\d+))?$", s)
    if m:
        mm = int(m.group(1)); ss = int(m.group(2))
        frac = m.group(3)
        f = float("0."+frac) if frac else 0.0
        return 60*mm + ss + f

    # mm:ss:ms (3 digits) or :cs (2 digits)
    m = re.match(r"^(\d+):(\d{2}):(\d{2,3})$", s)
    if m:
        mm = int(m.group(1)); ss = int(m.group(2)); tail = m.group(3)
        denom = 1000.0 if len(tail) == 3 else 100.0
        return 60*mm + ss + (int(tail)/denom)

    # h:mm:ss(.ms)
    m = re.match(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$", s)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2)); ss = int(m.group(3))
        frac = m.group(4)
        f = float("0."+frac) if frac else 0.0
        return 3600*hh + 60*mm + ss + f

    return np.nan

# ----------------------------
# Schema helpers
# ----------------------------
def find_countdown_time_cols(df: pd.DataFrame) -> list[str]:
    """Return countdown segment time cols like '1400_Time', ..., '100_Time', 'Finish_Time'."""
    cols = []
    for c in df.columns:
        if not str(c).endswith("_Time"):
            continue
        head = str(c).split("_")[0]
        if head.isdigit() or head.lower() == "finish":
            cols.append(str(c))
    # Sort numerics desc, keep Finish_Time at the end
    def keyer(c):
        h = c.split("_")[0]
        return (10**6 if h.lower()=="finish" else -int(h))
    cols_sorted = sorted(cols, key=keyer)
    return cols_sorted

def markers_from_cols(time_cols: list[str]) -> list[int]:
    """Return numeric markers (e.g., [1200,1100,...,100]) for all non-finish time cols."""
    out = []
    for c in time_cols:
        head = c.split("_")[0]
        if head.isdigit():
            out.append(int(head))
    return sorted(out, reverse=True)

def infer_seg_len(time_cols: list[str]) -> int:
    """Infer 100 or 200 from gaps between numeric markers."""
    nums = markers_from_cols(time_cols)
    if len(nums) < 2:
        return 100
    diffs = [nums[i]-nums[i+1] for i in range(len(nums)-1)]
    # median diff
    if not diffs:
        return 100
    d = int(pd.Series(diffs).median())
    return 200 if abs(d-200) < 50 else 100

def safe_div(num, den):
    if isinstance(den, (pd.Series, np.ndarray)):
        return np.where((den==0)|pd.isna(den), np.nan, num/den)
    return np.nan if (den==0 or pd.isna(den)) else num/den

# ----------------------------
# Metric builders (100-based vs field)
# ----------------------------
def speed_from_time(meters: float, secs):
    return safe_div(meters, secs)

def idx_vs_field(series_speed: pd.Series) -> pd.Series:
    fld = series_speed.mean(skipna=True)
    return 100.0 * (series_speed / fld)

def sum_cols(df, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([np.nan]*len(df), index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce").applymap(parse_time_cell).sum(axis=1)

def build_metrics(df_raw: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    w = df_raw.copy()

    # strip strings for time-like cols (helps bad CSVs)
    for c in w.columns:
        if str(c).endswith("_Time") or str(c) in ("Race Time", "Finish_Time"):
            w[c] = w[c].astype(str).str.strip()

    time_cols = find_countdown_time_cols(w)
    _dbg("time_cols", time_cols)
    seg_len = infer_seg_len(time_cols)
    _dbg("segment length", seg_len)

    # parse all segment times into seconds
    for c in time_cols:
        w[c] = pd.to_numeric(w[c].apply(parse_time_cell), errors="coerce")

    # --- RaceTime_s: trust Race Time if sane; else recompute from segments
    if "Race Time" in w.columns:
        rt = pd.to_numeric(w["Race Time"].apply(parse_time_cell), errors="coerce")
    else:
        rt = pd.Series([np.nan]*len(w), index=w.index)

    total_m_splits = seg_len * (len([c for c in time_cols if c.lower()!="finish_time"]))
    scale = (distance_m / total_m_splits) if total_m_splits else 1.0
    # Sum only numeric markers (+ Finish if 100m splits to cover to the line)
    seg_sum_cols = [c for c in time_cols if c.lower()!="race time"]
    rt_from = w[seg_sum_cols].sum(axis=1) * (1.0/scale)

    # sanity band by trip
    dm = distance_m
    if dm <= 1400: lo, hi = 35, 110
    elif dm <= 2000: lo, hi = 60, 160
    else: lo, hi = 100, 260
    rt_ok = rt.between(lo, hi)

    w["RaceTime_s"] = np.where(rt_ok, rt, rt_from)

    # speeds
    w["Race_AvgSpeed"] = speed_from_time(distance_m, w["RaceTime_s"])

    # ---- F200 (last 200 = 100_Time + Finish_Time for 100m splits;
    #             for 200m splits: just '200_Time')
    if seg_len == 100:
        f200_time = sum_cols(w, ["100_Time", "Finish_Time"])
    else:
        f200_time = sum_cols(w, ["200_Time"])
    f200_speed = speed_from_time(200.0, f200_time)
    F200_idx = idx_vs_field(f200_speed)

    # ---- Accel (600→200) four 100m segments; if 200m splits: use 600+400?
    if seg_len == 100:
        accel_time = sum_cols(w, ["600_Time", "500_Time", "400_Time", "300_Time"])
        accel_m = 400.0
    else:
        # in 200m mode, 600→200 = 600+400 (400→200 isn’t present), approximate with 2x200m
        accel_time = sum_cols(w, ["600_Time", "400_Time"])
        accel_m = 400.0
    accel_speed = speed_from_time(accel_m, accel_time)
    Accel_idx = idx_vs_field(accel_speed)

    # ---- Grind (last 200)
    Grind_idx = F200_idx.copy()

    # ---- tsSPI: exclude first 200 and last 600
    # keep markers strictly between (top two) and > 600
    num_markers = markers_from_cols(time_cols)
    if num_markers:
        top = sorted(num_markers, reverse=True)
        excl_first = set(top[:2])  # first 200
        # include those > 600 (i.e., exclude <= 600)
        mids = [m for m in num_markers if (m not in excl_first and m > 600)]
        mids_cols = [f"{m}_Time" for m in mids if f"{m}_Time" in w.columns]
    else:
        mids_cols = []

    ts_m = seg_len * len(mids_cols)
    if ts_m <= 0:
        ts_speed = pd.Series([np.nan]*len(w), index=w.index)
    else:
        mid_sum = w[mids_cols].sum(axis=1) if mids_cols else pd.Series([np.nan]*len(w), index=w.index)
        ts_speed = speed_from_time(ts_m, mid_sum)

    tsSPI_idx = idx_vs_field(ts_speed)

    # ---- PI (index points above/below 100, weighted)
    # Weights: F200 0.08, tsSPI 0.37, Accel 0.30, Grind 0.25
    comp = pd.DataFrame({
        "F200": F200_idx,
        "tsSPI": tsSPI_idx,
        "Accel": Accel_idx,
        "Grind": Grind_idx
    }, index=w.index)

    # shape nudge (very small): +0.15*(min(Accel-100,0)) to avoid over-rewarding backmarkers when no kick
    shape_nudge = 0.15 * np.minimum(comp["Accel"] - 100.0, 0.0)

    PI = (0.08*(comp["F200"]-100.0)
          + 0.37*(comp["tsSPI"]-100.0)
          + 0.30*(comp["Accel"]-100.0)
          + 0.25*(comp["Grind"]-100.0)
          + shape_nudge)

    # ---- Simple GPI (0–~5): positive cruise + positive late work
    GPI = (np.maximum(0.0, comp["tsSPI"]-100.0)*0.05
           + np.maximum(0.0, comp["Accel"]-100.0)*0.03
           + np.maximum(0.0, comp["Grind"]-100.0)*0.02)

    out = pd.DataFrame({
        "Horse": w.get("Horse", pd.Series([f"Runner {i+1}" for i in range(len(w))])),
        "Finish_Pos": pd.to_numeric(w.get("Finish_Pos", pd.Series([np.nan]*len(w))), errors="coerce"),
        "RaceTime_s": w["RaceTime_s"].round(3),
        "F200_idx": comp["F200"].round(3),
        "tsSPI": comp["tsSPI"].round(3),
        "Accel": comp["Accel"].round(3),
        "Grind": comp["Grind"].round(3),
        "PI": PI.round(3),
        "GPI": GPI.round(3),
    })

    return out, time_cols, seg_len

# ----------------------------
# Upload / Manual ingestion
# ----------------------------
df_raw = None
if source == "Upload file":
    up = st.file_uploader("Upload CSV/XLSX (countdown split columns)", type=["csv","xlsx"])
    if up is None:
        st.stop()
    if up.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(up)
    else:
        df_raw = pd.read_excel(up)
else:
    # Minimal manual sheet: Horse + countdown times (e.g., 1200_Time ... 100_Time, Finish_Time)
    st.info("Manual entry: create a small grid with **Horse** and countdown split columns "
            "(e.g., 1200_Time … 100_Time, Finish_Time).")
    example = pd.DataFrame({
        "Horse": [f"Runner {i+1}" for i in range(5)],
        f"{int(np.ceil(distance_m_input/100.0)*100)}_Time": ["" for _ in range(5)],
        f"{int(np.ceil(distance_m_input/100.0)*100) - 100}_Time": ["" for _ in range(5)],
        "100_Time": ["" for _ in range(5)],
        "Finish_Time": ["" for _ in range(5)],
    })
    df_raw = st.data_editor(example, num_rows="dynamic", use_container_width=True)

st.subheader("Raw / Converted Table")
st.dataframe(df_raw.head(20), use_container_width=True)

# ----------------------------
# Build metrics
# ----------------------------
try:
    distance_m = int(distance_m_input)
    metrics, time_cols_sorted, seg_len = build_metrics(df_raw, distance_m)
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG:
        st.exception(e)
    st.stop()

# ----------------------------
# Metrics table
# ----------------------------
st.markdown("## Sectional Metrics (PI & GPI)")
st.dataframe(metrics.sort_values(["Finish_Pos","PI"], na_position="last"),
             use_container_width=True)

# ----------------------------
# Visual 1: Sectional Shape Map (Accel vs Grind, color=tsSPI-100, size ~ PI)
# ----------------------------
st.markdown("## Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)")
def shape_map(df: pd.DataFrame):
    x = df["Accel"] - 100.0
    y = df["Grind"] - 100.0
    c = df["tsSPI"] - 100.0
    s = np.clip((df["PI"] + 10.0)*10.0, 20.0, 400.0)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    sc = ax.scatter(x, y, c=c, s=s, cmap="coolwarm", alpha=0.85, edgecolor="k", linewidths=0.4)
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.axhline(0, color="gray", lw=1, ls="--")

    # annotate all runners (tight labels with short arrows)
    for xi, yi, name in zip(x, y, df["Horse"]):
        ax.annotate(str(name),
                    xy=(xi, yi),
                    xytext=(xi + 0.15, yi + 0.15),
                    textcoords="data",
                    fontsize=8,
                    va="bottom", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", shrinkA=0, shrinkB=0))

    ax.set_xlabel("Acceleration (points vs field, 600→200) →")
    ax.set_ylabel("Grind (points vs field, 200→Finish) ↑")
    ax.set_title("Quadrants: +X = late acceleration; +Y = strong last 200. Colour = tsSPI deviation")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("tsSPI – 100")
    ax.grid(True, linestyle=":", alpha=0.4)
    st.pyplot(fig)

shape_map(metrics)

st.caption("Each bubble is a runner. Size ≈ PI (bigger = stronger overall). "
           "X: late acceleration (600→200) vs field; Y: last-200 grind vs field. "
           "Colour shows cruise strength (tsSPI vs field): red = faster mid-race, blue = slower.")

# ----------------------------
# Visual 2: Pace curve (field avg in black + Top-8 finishers)
# ----------------------------
st.markdown("## Pace over 200 m segments (left = early, right = home straight)")

def build_pace_curve(df_raw: pd.DataFrame, time_cols: list[str], seg_len: int):
    # segment order: earliest -> latest (left to right). Our columns are countdown, so reverse.
    # keep only numeric markers; handle Finish_Time as last point (100m -> finish).
    num_cols = [c for c in time_cols if c.split("_")[0].isdigit()]
    finish_col = [c for c in time_cols if c.split("_")[0].lower()=="finish"]
    ordered = list(reversed(num_cols)) + finish_col  # earliest to latest

    # labels like '1200m','1100m',...,'0–100m'
    labels = []
    for c in ordered:
        h = c.split("_")[0]
        if h.isdigit():
            labels.append(f"{h}-{int(h)-seg_len}m")
        else:
            labels.append("0–100m" if seg_len==100 else "0–200m")

    T = df_raw[ordered].applymap(parse_time_cell)
    S = seg_len / T  # m/s
    field_avg = S.mean(axis=0)

    # pick top-8 by Finish_Pos if present
    if "Finish_Pos" in df_raw.columns:
        pos = pd.to_numeric(df_raw["Finish_Pos"], errors="coerce")
        order_idx = np.argsort(pos.fillna(9999).values)[:8]
        top8 = S.iloc[order_idx]
        names = df_raw["Horse"].iloc[order_idx] if "Horse" in df_raw.columns else pd.Series([f"Runner {i+1}" for i in order_idx])
    else:
        top8 = S.head(8)
        names = df_raw["Horse"].head(8) if "Horse" in df_raw.columns else pd.Series([f"Runner {i+1}" for i in range(len(top8))])

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.plot(range(len(field_avg)), field_avg.values, lw=3, marker="o", label="Field average", color="black")

    # thinner lines, smaller markers for runners
    for i, (idx, row) in enumerate(top8.iterrows()):
        ax.plot(range(len(row)), row.values, lw=1.4, marker="o", ms=3, label=str(names.iloc[i]))

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    st.pyplot(fig)

try:
    build_pace_curve(df_raw, time_cols_sorted, seg_len)
except Exception as e:
    st.error("Could not render pace curve.")
    if DEBUG:
        st.exception(e)

# ----------------------------
# Visual 3: Top-8 PI — stacked bars of contributions
# ----------------------------
st.markdown("## Top-8 PI — stacked contributions")

def top8_stacked(df_metrics: pd.DataFrame):
    df = df_metrics.copy()
    # top-8 = best PI (or by finish if all NaN)
    df = df.sort_values("PI", ascending=False).head(8)

    parts = pd.DataFrame({
        "F200": (df["F200_idx"]-100.0)*0.08,
        "tsSPI": (df["tsSPI"]-100.0)*0.37,
        "Accel": (df["Accel"]-100.0)*0.30,
        "Grind": (df["Grind"]-100.0)*0.25,
        "Nudge": 0.15 * np.minimum(df["Accel"]-100.0, 0.0),
    }, index=df["Horse"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(parts))
    colors = ["#92c5de", "#4393c3", "#2166ac", "#053061", "#999999"]
    for i, col in enumerate(parts.columns):
        vals = parts[col].values
        ax.bar(range(len(parts)), vals, bottom=bottom, label=col, color=colors[i], alpha=0.9)
        bottom += vals

    ax.set_xticks(range(len(parts))); ax.set_xticklabels(parts.index, rotation=30, ha="right")
    ax.set_ylabel("PI points (stacked)")
    ax.set_title("How each horse built its PI")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper left")
    st.pyplot(fig)

top8_stacked(metrics)

# ----------------------------
# Notes / keys
# ----------------------------
st.markdown("""
**Keys**
- **F200**: last 200 m index vs field (100 = field average).
- **tsSPI**: mid-race cruising index, excluding the **first 200 m** and the **last 600 m**.
- **Accel**: 600→200 section index vs field.
- **Grind**: last 200 m index vs field (same basis as F200).
- **PI** = 0.08·(F200−100) + 0.37·(tsSPI−100) + 0.30·(Accel−100) + 0.25·(Grind−100) + small shape nudge.
- **GPI**: light group-potential hint from positive cruise + late work (0–≈5).
""")
