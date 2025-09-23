import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Race Edge — PI v3", layout="wide")

# ----------------------------
# Small helpers
# ----------------------------
def _dbg_show(label, obj=None):
    if st.session_state.get("_DEBUG", False):
        st.write(f"🔧 {label}")
        if obj is not None:
            st.write(obj)

def as_num(s):
    return pd.to_numeric(s, errors="coerce")

def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### Data source")
    mode = st.radio("",
                    options=["Upload file", "Manual input"],
                    index=0,
                    label_visibility="collapsed")

    race_distance_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, step=50, value=1200)
    st.caption("Using rounded distance: **{} m** (manual grid counts down from here).".format(int(np.ceil(race_distance_input/200.0)*200)))

    if mode == "Manual input":
        n_horses = st.number_input("Number of horses", min_value=2, max_value=20, value=8, step=1)

    st.markdown("---")
    _DEBUG = st.toggle("Debug info", value=False, key="_DEBUG")

st.title("Sectional Analysis & PI")

# ----------------------------
# Input handling
# ----------------------------
work = None
rounded_distance = int(np.ceil(race_distance_input / 200.0) * 200)

try:
    if mode == "Upload file":
        up = st.file_uploader("Upload CSV or Excel with 200 m split columns like `1200_Time`, `1000_Time`, … and optional `*_Pos`.", type=["csv", "xlsx", "xls"])
        if not up:
            st.stop()
        if up.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(up)
        else:
            df_raw = pd.read_excel(up)
        st.success("File loaded.")
        work = df_raw.copy()

    else:
        # Build a manual grid for 200 m segments
        segs = list(range(rounded_distance, 0, -200))  # 1200,1000,...,200
        cols = ["Horse", "Finish_Pos"]
        for m in segs:
            cols += [f"{m}_Time", f"{m}_Pos"]
        manual_template = pd.DataFrame([[None, None] + [None]*(2*len(segs)) for _ in range(n_horses)], columns=cols)
        st.info("Enter **segment times (s)** and (optionally) **positions**. Times are for each 200 m segment.")
        work = st.data_editor(manual_template, num_rows="dynamic", use_container_width=True, key="manual_grid").copy()
        st.success("Manual grid captured.")

except Exception as e:
    st.error("Input parsing failed.")
    if _DEBUG:
        st.exception(e)
    st.stop()

st.markdown("### Raw / Converted Table")
st.dataframe(work.head(12), use_container_width=True)
_dbg_show("Columns", list(work.columns))

# ----------------------------
# Metric builder (F200_idx, tsSPI, Accel, Grind, PI)
# ----------------------------
def build_metrics(df_in: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    w = df_in.copy()

    # Normalize Finish_Pos numeric if present
    if "Finish_Pos" in w.columns:
        w["Finish_Pos"] = as_num(w["Finish_Pos"])

    # Identify 200m time columns
    time_cols = [c for c in w.columns if str(c).endswith("_Time")]
    seg_markers = []
    for c in time_cols:
        try:
            m = int(str(c).split("_")[0])
            seg_markers.append(m)
        except Exception:
            pass
    seg_markers = sorted(set(seg_markers), reverse=True)

    # Convert times to speeds per segment
    for m in seg_markers:
        col = f"{m}_Time"
        w[f"spd_{m}"] = 200.0 / as_num(w[col])

    # Race time as sum of available segment times
    if seg_markers:
        sum_cols = [f"{m}_Time" for m in seg_markers if f"{m}_Time" in w.columns]
        w["RaceTime_s"] = w[sum_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    else:
        # fallback if provided
        w["RaceTime_s"] = as_num(w.get("Race Time", np.nan))

    # ---------- F200_idx (first 200 m segment relative to field) ----------
    first_mark = max(seg_markers) if seg_markers else None
    if first_mark and f"spd_{first_mark}" in w.columns:
        field_f200 = w[f"spd_{first_mark}"].median(skipna=True)
        w["F200_idx"] = 100.0 * (w[f"spd_{first_mark}"] / field_f200)
    else:
        w["F200_idx"] = np.nan

    # ---------- tsSPI: exclude first 200 and last 600 ----------
    def _tsspi_row(row):
        if not seg_markers:
            return np.nan
        mids = seg_markers[1:-3] if len(seg_markers) >= 5 else []
        if not mids:
            return np.nan
        vals = [row.get(f"spd_{m}") for m in mids]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))

    w["_mid_run_spd"] = w.apply(_tsspi_row, axis=1)
    field_mid = w["_mid_run_spd"].median(skipna=True)
    w["tsSPI"] = 100.0 * (w["_mid_run_spd"] / field_mid)

    # ---------- Accel (600→200) ----------
    last_mark = min(seg_markers) if seg_markers else None
    pre_marks = [m for m in seg_markers if last_mark and m > last_mark]  # segments before last 200
    accel_window = pre_marks[-3:] if len(pre_marks) >= 3 else pre_marks  # up to 600 m window
    def _avg(row, marks):
        vals = [row.get(f"spd_{m}") for m in marks if f"spd_{m}" in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))
    w["_accel_spd"] = w.apply(lambda r: _avg(r, accel_window), axis=1)
    field_accel = w["_accel_spd"].median(skipna=True)
    w["Accel"] = 100.0 * (w["_accel_spd"] / field_accel)

    # ---------- Grind (last 200) ----------
    if last_mark and f"spd_{last_mark}" in w.columns:
        w["_grind_spd"] = w[f"spd_{last_mark}"]
        field_grind = w["_grind_spd"].median(skipna=True)
        w["Grind"] = 100.0 * (w["_grind_spd"] / field_grind)
    else:
        w["_grind_spd"] = np.nan
        w["Grind"] = np.nan

    # ---------- PI v3 (weights + reweighting; baseline 100 = neutral) ----------
    PI_WEIGHTS = {"F200_idx": 0.08, "tsSPI": 0.37, "Accel": 0.30, "Grind": 0.25}

    def _pi_row(row):
        parts = []
        weights = []
        for k, wgt in PI_WEIGHTS.items():
            v = row.get(k)
            if pd.notna(v):
                parts.append(wgt * (v - 100.0))
                weights.append(wgt)
        if not weights:
            return np.nan
        scaled = sum(parts) / sum(weights)    # reweight if any missing
        # Convert to a 0–10-ish scale; +5 over baseline ~ 1 PI point
        return max(0.0, round(scaled / 5.0, 3))

    w["PI"] = w.apply(_pi_row, axis=1)

    # housekeeping: round presentation columns
    for c in ["F200_idx", "tsSPI", "Accel", "Grind", "PI"]:
        if c in w.columns:
            w[c] = w[c].round(3)

    return w, seg_markers

try:
    metrics, seg_markers = build_metrics(work, rounded_distance)
except Exception as e:
    st.error("Metric computation failed.")
    if _DEBUG:
        st.exception(e)
    st.stop()

# ----------------------------
# Section: Metrics table
# ----------------------------
st.markdown("## Sectional Metrics & PI")
cols_show = ["Horse", "Finish_Pos", "RaceTime_s", "F200_idx", "tsSPI", "Accel", "Grind", "PI"]
for c in cols_show:
    if c not in metrics.columns:
        metrics[c] = np.nan
disp = metrics[cols_show].copy()
st.dataframe(disp.sort_values(["PI","Finish_Pos"], ascending=[False, True]), use_container_width=True)

# ----------------------------
# Visual 1: Sectional Shape Map (Accel vs Grind)
# ----------------------------
st.markdown("## Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)")
fig, ax = plt.subplots()
data = metrics.copy()
x = data["Accel"] - 100.0
y = data["Grind"] - 100.0
cval = data["tsSPI"] - 100.0
mask = x.notna() & y.notna()
xv = x[mask].values
yv = y[mask].values
cv = cval[mask].values
ax.scatter(xv, yv, c=cv, cmap="coolwarm", s=60, alpha=0.9, edgecolor="k", linewidth=0.4)
for name, xx, yy in zip(data.loc[mask, "Horse"], xv, yv):
    ax.annotate(str(name), (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.axvline(0, color="gray", lw=1, ls="--")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("Acceleration (index vs field, 600→200)")
ax.set_ylabel("Grind (index vs field, 200→Finish)")
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax)
# set colorbar ticks around data range
if cv.size:
    vmin, vmax = float(np.nanmin(cv)), float(np.nanmax(cv))
    cb.set_ticks(np.linspace(vmin, vmax, 5))
cb.set_label("tsSPI deviation from field")
st.pyplot(fig)

# ----------------------------
# Visual 2: Pace curve (field average + Top 8 finishers)
# ----------------------------
st.markdown("## Pace Curve — field average (black) + Top 8 finishers")

# robust parse of *_Time markers
valid_markers = []
for c in [c for c in work.columns if str(c).endswith("_Time")]:
    try:
        valid_markers.append(int(str(c).split("_")[0]))
    except Exception:
        pass
valid_markers = sorted(set(valid_markers), reverse=True)
valid_markers = [m for m in valid_markers if m >= 200]

if valid_markers:
    speed_cols = [f"{m}_Time" for m in valid_markers if f"{m}_Time" in work.columns]
    spd_df = pd.DataFrame({c: 200.0 / as_num(work[c]) for c in speed_cols})
    field_avg = spd_df.mean(axis=0).values.tolist()

    x_labels = [f"{m-200}–{m}m" for m in valid_markers]
    x_idx = list(range(len(speed_cols)))

    fig2, ax2 = plt.subplots()
    ax2.plot(x_idx, field_avg, linewidth=3, color="black", label="Field average")

    # choose Top 8 (Finish_Pos preferred)
    if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
        top8 = metrics.sort_values("Finish_Pos").head(8)
    else:
        top8 = metrics.sort_values("PI", ascending=False).head(8)

    cols = color_cycle(8)
    for i, (_, r) in enumerate(top8.iterrows()):
        y_vals = []
        for c in speed_cols:
            val = r.get(c, np.nan)
            if pd.isna(val):
                # Try to draw from original work row by matching Horse
                if "Horse" in r and "Horse" in work.columns:
                    row0 = work[work["Horse"] == r["Horse"]]
                    if not row0.empty:
                        val = row0.iloc[0].get(c, np.nan)
            y_vals.append(200.0 / val if pd.notna(val) and val > 0 else np.nan)
        ax2.plot(x_idx, y_vals, linewidth=2, marker="o", label=str(r.get("Horse", f"H{i+1}")), color=cols[i])

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Pace over race segments (200 m resolution)")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", ncol=2, fontsize=9)
    st.pyplot(fig2)
else:
    st.info("Not enough 200 m *_Time columns to draw the pace curve.")

# ----------------------------
# Visual 3: Top-8 PI stacked bars (contribution by metric)
# ----------------------------
st.markdown("## Top-8 PI — contribution by metric")
PI_WEIGHTS = {"F200_idx": 0.08, "tsSPI": 0.37, "Accel": 0.30, "Grind": 0.25}

top8_pi = metrics.sort_values("PI", ascending=False).head(8).copy()
if not top8_pi.empty:
    # compute contributions using same reweight logic
    contribs = {"F200": [], "tsSPI": [], "Accel": [], "Grind": []}
    horses = []
    for _, row in top8_pi.iterrows():
        horses.append(str(row.get("Horse", "")))
        parts = []
        weights = []
        for k, wgt in PI_WEIGHTS.items():
            v = row.get(k)
            if pd.notna(v):
                parts.append(wgt * (v - 100.0))
                weights.append(wgt)
        scale = sum(weights) if weights else 1.0
        # bar heights in PI units (same division by 5.0)
        def to_pi(units): return (units / scale) / 5.0 if scale > 0 else 0.0
        # push each contribution aligned with labels
        contribs["F200"].append(to_pi(PI_WEIGHTS["F200_idx"] * (row.get("F200_idx", np.nan) - 100.0)) if pd.notna(row.get("F200_idx")) else 0.0)
        contribs["tsSPI"].append(to_pi(PI_WEIGHTS["tsSPI"] * (row.get("tsSPI", np.nan) - 100.0)) if pd.notna(row.get("tsSPI")) else 0.0)
        contribs["Accel"].append(to_pi(PI_WEIGHTS["Accel"] * (row.get("Accel", np.nan) - 100.0)) if pd.notna(row.get("Accel")) else 0.0)
        contribs["Grind"].append(to_pi(PI_WEIGHTS["Grind"] * (row.get("Grind", np.nan) - 100.0)) if pd.notna(row.get("Grind")) else 0.0)

    fig3, ax3 = plt.subplots(figsize=(max(6, 0.9*len(horses)), 4))
    idx = np.arange(len(horses))
    bottoms = np.zeros(len(horses))
    colors = {"F200": "#6baed6", "tsSPI": "#9e9ac8", "Accel": "#74c476", "Grind": "#fd8d3c"}

    for key in ["F200", "tsSPI", "Accel", "Grind"]:
        vals = np.array(contribs[key])
        ax3.bar(idx, vals, bottom=bottoms, label=key, color=colors[key], edgecolor="black", linewidth=0.4)
        bottoms += vals

    ax3.set_xticks(idx)
    ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylabel("PI (stacked contributions)")
    ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend()
    st.pyplot(fig3)
else:
    st.info("Need at least some PI values to draw the stacked contribution chart.")

# ----------------------------
# Footer / Notes
# ----------------------------
st.caption(
    "Notes: **F200_idx** compares the first 200 m to the field (100 = par). "
    "**tsSPI** measures sustained mid-race pace excluding the first 200 m and last 600 m. "
    "**Accel** is the 600→200 window; **Grind** is the last 200 m. "
    "**PI** blends these (weights: 0.08/0.37/0.30/0.25) with auto-reweighting if a component is missing."
)
