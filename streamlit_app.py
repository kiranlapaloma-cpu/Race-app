import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="Race Edge — PI v3", layout="wide")
st.title("🏇 Race Edge — PI v3 (sectional-first)")
st.caption(
    "Metrics: F200 (0–200), tsSPI (200 → D−600), Accel (600→200), Grind (200→Finish). "
    "PI fixed weights + shape nudge, robust scaling, fairness guards, rank-blend to 0–10."
)

# ----------------------------
# Small helpers
# ----------------------------
def parse_race_time(val):
    """Parse 'Race Time' into seconds. Accepts:
       - float seconds ('72.4')
       - 'mm:ss.ms'
       - 'mm:ss:ms' (rare)
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # try pure float
    try:
        return float(s)
    except Exception:
        pass
    parts = s.split(":")
    try:
        if len(parts) == 3:  # mm:ss:ms
            m, ss, ms = parts
            return int(m) * 60 + int(ss) + float(ms) / 1000.0
        if len(parts) == 2:  # mm:ss.ms
            m, ss = parts
            return int(m) * 60 + float(ss)
        if len(parts) == 1:
            return float(parts[0])
    except Exception:
        return np.nan
    return np.nan


def iqr_above_median_01(series: pd.Series) -> pd.Series:
    """Robust mapper to [0,1]: (x - Q2) / max(Q3-Q2, Q2-Q1, eps), clipped to [0,1]."""
    s = pd.to_numeric(series, errors="coerce")
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    upper = q3 - q2
    lower = q2 - q1
    denom = max(upper, lower, 1e-9)
    return ((s - q2) / denom).clip(0, 1)


def nearest_200m_up(x: int) -> int:
    """Round distance up to nearest 200m (e.g., 1160 -> 1200)."""
    return int(math.ceil(x / 200.0) * 200)


def make_countdown_markers(distance_m: int, step: int = 200):
    """Return list of markers counting down from distance to 0 by 'step'."""
    return list(range(distance_m, -step, -step))


def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    if not base:
        return ["#1f77b4"] * n
    res = []
    i = 0
    while len(res) < n:
        res.append(base[i % len(base)])
        i += 1
    return res[:n]


# ----------------------------
# Sidebar: data source + distance
# ----------------------------
with st.sidebar:
    source = st.radio("Data source", ["Upload file", "Manual input"], index=0)
    user_distance = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=10)
    distance_m = nearest_200m_up(int(user_distance))
    st.caption(f"Using rounded distance: **{distance_m} m** (manual grid counts down from here).")
    show_debug = st.checkbox("Debug info", value=False)

# ----------------------------
# Data ingestion
# ----------------------------
df_raw = None

def detect_available_time_cols(cols):
    """Return dictionary mapping like {'1700_Time': True, ...} for any *Time columns."""
    return {c: True for c in cols if str(c).endswith("_Time")}

def manual_grid(distance_m: int, n_horses: int) -> pd.DataFrame:
    """Build a manual-entry grid: Horse + per-200m split times counting down from distance."""
    markers = make_countdown_markers(distance_m, 200)  # e.g., [1400,1200,...,0]
    # We'll need segment splits: e.g., '1400_Time' meaning 1400->1200 segment, etc., down to '200_Time' for 200->Finish
    time_cols = [f"{m}_Time" for m in markers if m >= 200]  # no 0_Time
    cols = ["Horse", "Finish_Pos", "Race Time"] + time_cols
    data = []
    for i in range(n_horses):
        row = {c: "" for c in cols}
        row["Horse"] = f"Runner {i+1}"
        data.append(row)
    df = pd.DataFrame(data, columns=cols)
    return df

try:
    if source == "Upload file":
        uploaded = st.file_uploader("Upload CSV or XLSX (sectional splits per 100m or 200m)", type=["csv", "xlsx"])
        if uploaded is None:
            st.info("Upload a file, or switch to Manual input.")
            st.stop()
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
        st.success("File loaded.")
    else:
        # Manual mode
        with st.sidebar:
            n_horses = st.number_input("Number of horses", min_value=2, max_value=24, value=12, step=1)
        st.subheader("Manual input grid (enter split times in seconds)")
        grid = manual_grid(distance_m, int(n_horses))
        edited = st.data_editor(grid, use_container_width=True, num_rows="dynamic")
        # convert types carefully
        df_raw = edited.copy()
        st.success("Manual table captured.")

except Exception as e:
    st.error("Failed to load data.")
    if show_debug:
        st.exception(e)
    st.stop()

st.subheader("Raw preview")
st.dataframe(df_raw.head(12), use_container_width=True)

# ----------------------------
# Normalize columns, ensure numeric
# ----------------------------
df = df_raw.copy()

# Identify split-time columns like '1700_Time', '1600_Time', ..., '200_Time'
time_cols = [c for c in df.columns if str(c).endswith("_Time")]
# Some feeds include helper columns like '800-400' and '400-Finish'
has_400_segments = ("800-400" in df.columns) and ("400-Finish" in df.columns)
if has_400_segments:
    df["800-400"] = pd.to_numeric(df["800-400"], errors="coerce")
    df["400-Finish"] = pd.to_numeric(df["400-Finish"], errors="coerce")

# Coerce all *_Time columns to numeric where possible
for c in time_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Ensure 'Race Time' present and parse into seconds
if "Race Time" not in df.columns and "Finish_Time" in df.columns:
    # if feed has Finish_Time per horse and maybe 'Race Time' absent, use 'Finish_Time' for winner but keep field-specific
    df["Race Time"] = df["Finish_Time"]

if "Race Time" not in df.columns:
    df["Race Time"] = np.nan

df["RaceTime_s"] = df["Race Time"].apply(parse_race_time)

# If Finish_Pos missing, infer via RaceTime_s
if "Finish_Pos" not in df.columns or df["Finish_Pos"].isna().all():
    if df["RaceTime_s"].notna().any():
        df["Finish_Pos"] = df["RaceTime_s"].rank(method="min").astype("Int64")

# ----------------------------
# Sectional windows & metrics
# ----------------------------
# 1) F200 (first 200m): pick the top-most distance marker split (e.g., '1400_Time' for 1400->1200)
first_seg_col = None
if time_cols:
    # Choose the largest leading marker (e.g., 1700_Time over 1500_Time)
    try:
        lead_vals = [(int(str(c).split("_")[0]), c) for c in time_cols]
        lead_vals.sort(reverse=True)  # highest distance first
        first_seg_col = lead_vals[0][1]  # e.g., '1400_Time'
    except Exception:
        first_seg_col = None

# 2) Grind: must be last 200m (200->Finish) split (i.e., '200_Time')
grind_col = "200_Time" if "200_Time" in df.columns else None

# 3) Accel: 600->200 split sum
# Accept either four 100m splits, or two 200m splits, or precomputed '800-400'/'400-Finish' (we'll compute from splits if available).
def get_accel_time_cols(all_cols, distance_m: int):
    # Try 100m splits: '600_Time', '500_Time', '400_Time', '300_Time'
    cols_100 = [f"{m}_Time" for m in [600, 500, 400, 300]]
    if all(c in all_cols for c in cols_100):
        return cols_100
    # Try 200m splits: '600_Time', '400_Time' (meaning 600->400) AND '400_Time' (400->200) — but many feeds label both as same.
    # So safest generic: sum known 600 and 400 splits if they represent those segments.
    cols_200 = [c for c in all_cols if c in ["600_Time", "400_Time"]]
    if len(cols_200) >= 2:
        return cols_200
    # Fallback to 400-segment helpers if present: Accel ~ Final400 vs Mid400 needs real splits,
    # but if we only have '800-400' & '400-Finish', we can approximate Speed_600_200 = 400 / t where t = (400-Finish) + (some 200m portion before 400)
    return None

accel_cols = get_accel_time_cols(time_cols, distance_m)

# 4) tsSPI window: 200m after start to Distance-600
# Build expected markers present in df and sum their times
def markers_from_df(all_cols):
    """Return available numeric markers (as ints) for *_Time columns."""
    out = []
    for c in all_cols:
        if str(c).endswith("_Time"):
            try:
                m = int(str(c).split("_")[0])
                out.append(m)
            except Exception:
                pass
    return sorted(list(set(out)), reverse=True)  # e.g. [1700,1600,...,200]

markers = markers_from_df(time_cols)

def compute_metric_columns(df: pd.DataFrame, distance_m: int):
    """Compute F200_Speed, tsSPI, Accel, Grind for each runner."""
    out = df.copy()
    # Race average speed
    out["Race_AvgSpeed"] = distance_m / out["RaceTime_s"]

    # F200
    if first_seg_col and first_seg_col in out.columns:
        out["F200_Speed"] = 200.0 / out[first_seg_col]
    else:
        out["F200_Speed"] = np.nan

    # Grind (last 200m)
    if grind_col and grind_col in out.columns:
        out["Grind_Speed"] = 200.0 / out[grind_col]
    else:
        out["Grind_Speed"] = np.nan

    # Accel (600->200)
    if accel_cols:
        total = 0.0
        for c in accel_cols:
            total = total + pd.to_numeric(out[c], errors="coerce")
        out["Accel_Time"] = total
        out["Accel_Speed"] = 400.0 / out["Accel_Time"]
    elif has_400_segments:
        # Approximation is NOT used here; we require real 600→200 splits for accuracy.
        out["Accel_Time"] = np.nan
        out["Accel_Speed"] = np.nan
    else:
        out["Accel_Time"] = np.nan
        out["Accel_Speed"] = np.nan

    # tsSPI window (200 -> distance-600)
    # Identify segments present in df that lie strictly between 200 and distance-600.
    start_excl = 200
    end_excl = distance_m - 600
    if end_excl <= start_excl:
        # For very short trips (1000m), there may be no interior mid window; handle gracefully
        out["Mid_Time"] = np.nan
        out["Mid_Dist"] = 0.0
        out["Mid_Speed"] = np.nan
    else:
        # We assume *_Time columns are 200m segments counting down, e.g., 1400_Time means 1400→1200.
        # So include all c where marker in (start_excl, distance] and next marker >= end_excl.
        # Simpler: include segment c where segment END >= end_excl and START <= start_excl? (But we have per-segment labels as start marker.)
        # Because labels are "marker_Time" for segment marker→marker-200, we include those where:
        # marker <= distance_m and (marker-200) >= end_excl, also marker <= (distance_m - 200) ... Too complicated; take direct set:
        candidates = []
        for c in time_cols:
            try:
                m = int(str(c).split("_")[0])
            except Exception:
                continue
            seg_start = m
            seg_end = m - 200
            if seg_end < 0:
                continue
            # We want all segments fully inside (200, distance-600)
            if (seg_start <= distance_m) and (seg_end >= end_excl) and (seg_start <= distance_m - 200) and (seg_end >= start_excl):
                # This condition may include overlapping edges incorrectly; safer approach is range inclusion:
                if (seg_start <= distance_m - 600) and (seg_end >= 200):
                    candidates.append(c)
        # If above is too strict and yields none, relax: include all segments whose centers lie inside (200, distance-600)
        if not candidates:
            for c in time_cols:
                try:
                    m = int(str(c).split("_")[0])
                except Exception:
                    continue
                seg_center = m - 100
                if (seg_center > 200) and (seg_center < (distance_m - 600)):
                    candidates.append(c)

        if candidates:
            mid_time = pd.DataFrame({c: pd.to_numeric(out[c], errors="coerce") for c in candidates}).sum(axis=1)
            out["Mid_Time"] = mid_time
            out["Mid_Dist"] = 200.0 * len(candidates)
            out["Mid_Speed"] = out["Mid_Dist"] / out["Mid_Time"]
        else:
            out["Mid_Time"] = np.nan
            out["Mid_Dist"] = 0.0
            out["Mid_Speed"] = np.nan

    # Convert to indices (field-relative where specified)
    # F200 index relative to field (speed vs median speed)
    if out["F200_Speed"].notna().any():
        med_f200 = out["F200_Speed"].median()
        out["F200_idx"] = (out["F200_Speed"] / med_f200) * 100.0
    else:
        out["F200_idx"] = np.nan

    # Accel index (speed vs median Accel speed)
    if out["Accel_Speed"].notna().any():
        med_acc = out["Accel_Speed"].median()
        out["Accel"] = (out["Accel_Speed"] / med_acc) * 100.0
    else:
        out["Accel"] = np.nan

    # Grind index (speed vs median last-200 speed)
    if out["Grind_Speed"].notna().any():
        med_gr = out["Grind_Speed"].median()
        out["Grind"] = (out["Grind_Speed"] / med_gr) * 100.0
    else:
        out["Grind"] = np.nan

    # tsSPI: (Mid_Speed / Race_AvgSpeed)*100 (note: mid is absolute, relative to race avg)
    out["tsSPI"] = (out["Mid_Speed"] / out["Race_AvgSpeed"]) * 100.0

    return out

try:
    work = compute_metric_columns(df, distance_m)
except Exception as e:
    st.error("Metric computation failed (splits may be missing for the required windows).")
    if show_debug:
        st.exception(e)
    st.stop()

# ----------------------------
# PI v3 compositor
# ----------------------------
# Robust 0..1
F200_01  = iqr_above_median_01(work["F200_idx"])
tsSPI_01 = iqr_above_median_01(work["tsSPI"])
Accel_01 = iqr_above_median_01(work["Accel"])
Grind_01 = iqr_above_median_01(work["Grind"])

# Fixed weights
wF, wS, wA, wG = 0.08, 0.37, 0.30, 0.25

# Shape nudge (±0.02 shift between Accel and Grind)
spi_med = work["tsSPI"].median()
shape = "even"
if pd.notna(spi_med):
    if spi_med <= 97.0:
        shape = "sprint-home"
        wA += 0.02; wG -= 0.02
    elif spi_med >= 103.0:
        shape = "fast-early"
        wG += 0.02; wA -= 0.02
# re-normalize safely
total_w = wF + wS + wA + wG
wF, wS, wA, wG = (wF/total_w, wS/total_w, wA/total_w, wG/total_w)

PI_base = (wF*F200_01) + (wS*tsSPI_01) + (wA*Accel_01) + (wG*Grind_01)

# Consistency bonus (≥3 of 4 raw metrics >= medians)
m_meds = {
    "F200_idx": work["F200_idx"].median(),
    "tsSPI":    work["tsSPI"].median(),
    "Accel":    work["Accel"].median(),
    "Grind":    work["Grind"].median()
}
def consistency(row):
    cnt = int(row["F200_idx"] >= m_meds["F200_idx"]) + int(row["tsSPI"] >= m_meds["tsSPI"]) \
          + int(row["Accel"] >= m_meds["Accel"]) + int(row["Grind"] >= m_meds["Grind"])
    return 0.03 if cnt >= 3 else 0.0

# One-trick penalty (spiky profile) unless aligned with shape
q1 = {
    "F200_idx": work["F200_idx"].quantile(0.25),
    "tsSPI":    work["tsSPI"].quantile(0.25),
    "Accel":    work["Accel"].quantile(0.25),
    "Grind":    work["Grind"].quantile(0.25),
}
q3 = {
    "F200_idx": work["F200_idx"].quantile(0.75),
    "tsSPI":    work["tsSPI"].quantile(0.75),
    "Accel":    work["Accel"].quantile(0.75),
    "Grind":    work["Grind"].quantile(0.75),
}

def one_trick(row, shape: str):
    low_any = (row["F200_idx"] <= q1["F200_idx"]) or (row["tsSPI"] <= q1["tsSPI"]) \
              or (row["Accel"] <= q1["Accel"]) or (row["Grind"] <= q1["Grind"])
    high_any = (row["F200_idx"] >= q3["F200_idx"]) or (row["tsSPI"] >= q3["tsSPI"]) \
               or (row["Accel"] >= q3["Accel"]) or (row["Grind"] >= q3["Grind"])
    if not (low_any and high_any):
        return 0.0
    # waive penalty if spike aligns with shape
    if shape == "sprint-home" and (row["Accel"] >= q3["Accel"]):
        return 0.0
    if shape == "fast-early" and (row["Grind"] >= q3["Grind"]):
        return 0.0
    return -0.02

# Small-field guard & DQ
N = work.shape[0]
sf_guard = 0.92 if N <= 6 else 1.00

def dq_factor(row):
    bad = 0
    if "200_Time" in row and (pd.isna(row["200_Time"]) or row["200_Time"] <= 0): bad += 1
    if "600_Time" in row and (pd.isna(row.get("600_Time", np.nan)) or (pd.notna(row.get("600_Time")) and row.get("600_Time") <= 0)): bad += 1
    if pd.isna(row["RaceTime_s"]) or row["RaceTime_s"] <= 0: bad += 1
    if bad == 0: return 1.00
    if bad == 1: return 0.95
    return 0.90

cons = work.apply(consistency, axis=1)
spike = work.apply(lambda r: one_trick(r, shape), axis=1)
dq = work.apply(dq_factor, axis=1)

PI_guarded01 = (PI_base + cons + spike) * sf_guard * dq

# Winner protection
if "Finish_Pos" in work.columns:
    q3_field = PI_guarded01.quantile(0.75)
    is_winner = work["Finish_Pos"] == work["Finish_Pos"].min()
    PI_guarded01 = np.where((is_winner) & (PI_guarded01 < q3_field),
                            np.maximum(PI_guarded01, q3_field - 0.02),
                            PI_guarded01)

# Final scaling: rank blend to 0–10
rank_pct = 1.0 - (pd.Series(PI_guarded01).rank(method="min", ascending=True) - 1) / (N - 1 if N > 1 else 1)
PI_final = 10.0 * (0.70 * np.clip(PI_guarded01, 0, 1) + 0.30 * rank_pct)
work["PI"] = np.round(PI_final, 3)

# For stacked bar (component contributions BEFORE guards; just to “see the build”)
work["PI_F200"]  = 10.0 * (wF * F200_01)
work["PI_tsSPI"] = 10.0 * (wS * tsSPI_01)
work["PI_Accel"] = 10.0 * (wA * Accel_01)
work["PI_Grind"] = 10.0 * (wG * Grind_01)

# ----------------------------
# Display: metrics table
# ----------------------------
st.subheader("Sectional Metrics & PI")
disp_cols = ["Horse","Finish_Pos","RaceTime_s","F200_idx","tsSPI","Accel","Grind","PI"]
# Try to keep Horse present
if "Horse" not in work.columns:
    work["Horse"] = [f"Runner {i+1}" for i in range(len(work))]
table = work[disp_cols].copy()
st.dataframe(table.sort_values("PI", ascending=False), use_container_width=True)

# ----------------------------
# Visual 1: Sectional Shape Map (Accel vs Grind)
# ----------------------------
st.subheader("Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)")
fig1, ax1 = plt.subplots()
x = work["Accel"]
y = work["Grind"]
sizes = 40 + 60 * (work["PI"] / (work["PI"].max() if work["PI"].max() > 0 else 1))
sc = ax1.scatter(x, y, s=sizes, c=work["PI"], cmap="viridis", alpha=0.85, edgecolors="k", linewidths=0.4)
ax1.set_xlabel("Acceleration (index vs field, 600→200)")
ax1.set_ylabel("Grind (index vs field, 200→Finish)")
ax1.grid(True, linestyle="--", alpha=0.3)
cb = fig1.colorbar(sc, ax=ax1, shrink=0.85)
cb.set_label("PI (0–10)")

# Name annotations as arrows (top 8 by PI to reduce clutter)
top8 = work.sort_values("PI", ascending=False).head(8)
for _, r in top8.iterrows():
    ax1.annotate(
        str(r["Horse"]),
        xy=(r["Accel"], r["Grind"]),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.6, color="grey")
    )
st.pyplot(fig1)

# ----------------------------
# Visual 2: Pace curve — full race average + Top 8 finishers
# ----------------------------
st.subheader("Pace Curve — field average (black) + Top 8 finishers")
# Build per-200m speed vectors if possible
seg_markers = sorted([int(c.split("_")[0]) for c in time_cols if c.endswith("_Time")], reverse=True)
seg_markers = [m for m in seg_markers if m >= 200]  # ensure have 200->Finish

if seg_markers:
    # For each horse, speeds along race segments
    speed_cols = []
    for m in seg_markers:
        col = f"{m}_Time"
        if col in work.columns:
            speed_cols.append(col)
    # Construct speeds df (m/s)
    spds = pd.DataFrame({c: 200.0 / pd.to_numeric(work[c], errors="coerce") for c in speed_cols})
    field_avg = spds.mean(axis=0).values.tolist()

    # x-axis labels as descending markers
    x_labels = [f"{m-200}–{m}m" for m in seg_markers]  # counting forward visually
    x_idx = list(range(len(speed_cols)))

    fig2, ax2 = plt.subplots()
    ax2.plot(x_idx, field_avg, linewidth=3, color="black", label="Field average")
    # top 8 by Finish_Pos (if present); otherwise by PI
    if "Finish_Pos" in work.columns and work["Finish_Pos"].notna().any():
        top8_finish = work.sort_values("Finish_Pos").head(8)
    else:
        top8_finish = work.sort_values("PI", ascending=False).head(8)

    cols = color_cycle(8)
    for (i, (_, r)) in enumerate(top8_finish.iterrows()):
        y_vals = []
        for c in speed_cols:
            y_vals.append(200.0 / r[c] if pd.notna(r[c]) and r[c] > 0 else np.nan)
        ax2.plot(x_idx, y_vals, linewidth=2, marker="o", label=str(r["Horse"]), color=cols[i])

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Pace over race segments (200m resolution)")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", ncol=2, fontsize=9)
    st.pyplot(fig2)
else:
    st.info("Not enough *_Time columns to draw the full pace curve at 200m resolution.")

# ----------------------------
# Visual 3: Top-8 PI stacked bar (component contributions)
# ----------------------------
st.subheader("Top 8 — PI stacked bar (component build)")
top8_pi = work.sort_values("PI", ascending=False).head(8).copy()
bar_labels = top8_pi["Horse"].tolist()
cF = top8_pi["PI_F200"].tolist()
cS = top8_pi["PI_tsSPI"].tolist()
cA = top8_pi["PI_Accel"].tolist()
cG = top8_pi["PI_Grind"].tolist()

x = np.arange(len(bar_labels))
fig3, ax3 = plt.subplots(figsize=(max(8, len(bar_labels)*0.9), 4.8))
p1 = ax3.bar(x, cF, label="F200", width=0.6)
p2 = ax3.bar(x, cS, bottom=cF, label="tsSPI", width=0.6)
p3 = ax3.bar(x, cA, bottom=np.array(cF)+np.array(cS), label="Accel", width=0.6)
p4 = ax3.bar(x, cG, bottom=np.array(cF)+np.array(cS)+np.array(cA), label="Grind", width=0.6)

ax3.set_xticks(x)
ax3.set_xticklabels(bar_labels, rotation=30, ha="right")
ax3.set_ylabel("Contribution to PI (0–10 scale, pre-guards)")
ax3.set_ylim(0, max(10.0, (np.array(cF)+np.array(cS)+np.array(cA)+np.array(cG)).max()*1.1))
ax3.grid(True, axis="y", linestyle="--", alpha=0.3)
ax3.legend()
st.pyplot(fig3)

# ----------------------------
# Runner-by-runner quick notes
# ----------------------------
st.subheader("Runner-by-runner quick notes")
def note_row(r):
    parts = []
    # high / mid / low tags per metric vs field median
    def tag(val, med, name):
        if pd.isna(val): return None
        if val >= med + 3: return f"{name}↑"
        if val <= med - 3: return f"{name}↓"
        return None

    tF = tag(r["F200_idx"], m_meds["F200_idx"], "Gate")
    tS = tag(r["tsSPI"],    m_meds["tsSPI"],    "Cruise")
    tA = tag(r["Accel"],    m_meds["Accel"],    "Accel")
    tG = tag(r["Grind"],    m_meds["Grind"],    "Grind")
    tags = [t for t in [tF,tS,tA,tG] if t]
    shape_hint = "even"
    if shape == "sprint-home": shape_hint = "sprint-home (late pop rewarded)"
    elif shape == "fast-early": shape_hint = "fast-early (stamina rewarded)"
    parts.append(f"**{r['Horse']}** — PI {r['PI']:.2f} | Finish: {int(r['Finish_Pos']) if pd.notna(r['Finish_Pos']) else '—'} | Race shape: {shape_hint}")
    if tags:
        parts.append(" • " + ", ".join(tags))
    # Distance hint
    if pd.notna(r["Accel"]) and pd.notna(r["Grind"]):
        if r["Accel"] > r["Grind"] + 2:
            parts.append(" • Distance hint: might thrive a touch shorter or with a faster early setup.")
        elif r["Grind"] > r["Accel"] + 2:
            parts.append(" • Distance hint: likely better further or under a genuine tempo.")
    return "  \n".join(parts)

for _, row in work.sort_values("PI", ascending=False).iterrows():
    st.markdown(note_row(row))
    st.markdown("---")

# ----------------------------
# Debug (optional)
# ----------------------------
if show_debug:
    st.subheader("Debug — internals")
    st.write("Weights:", dict(F200=wF, tsSPI=wS, Accel=wA, Grind=wG))
    st.write("Shape:", shape, " | tsSPI median:", spi_med)
    st.write("IQR scaled head:")
    dbg = pd.DataFrame({
        "Horse": work["Horse"],
        "F200_01": iqr_above_median_01(work["F200_idx"]),
        "tsSPI_01": iqr_above_median_01(work["tsSPI"]),
        "Accel_01": iqr_above_median_01(work["Accel"]),
        "Grind_01": iqr_above_median_01(work["Grind"]),
        "PI_base01": PI_base
    })
    st.dataframe(dbg.head(12), use_container_width=True)
