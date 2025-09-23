import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =============== Page config ===============
st.set_page_config(page_title="Race Edge — PI v3 + Hidden Horses", layout="wide")

# =============== Utilities ===============
def as_num(x): return pd.to_numeric(x, errors="coerce")

def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

def clamp01(x): return max(0.0, min(1.0, float(x)))

# =============== Sidebar ===============
with st.sidebar:
    st.markdown("### Data source")
    mode = st.radio("", ["Upload file", "Manual input"], index=0, label_visibility="collapsed")

    race_distance_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, step=50, value=1200)
    rounded_distance = int(np.ceil(race_distance_input/200.0)*200)
    st.caption(f"Using rounded distance: **{rounded_distance} m** (manual grid counts down from here).")

    if mode == "Manual input":
        n_horses = st.number_input("Number of horses", 2, 20, 8, 1)

    st.markdown("---")
    DEBUG = st.toggle("Debug info", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔧 {label}")
        if obj is not None: st.write(obj)

# =============== Input ===============
work = None
try:
    if mode == "Upload file":
        up = st.file_uploader("Upload CSV/XLSX with 200 m segments (`1200_Time`, `1000_Time`, …) and optional `*_Pos`.", type=["csv","xlsx","xls"])
        if not up: st.stop()
        work = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        st.success("File loaded.")
    else:
        segs = list(range(rounded_distance, 0, -200))  # e.g., 1200,1000,...,200
        cols = ["Horse","Finish_Pos"]
        for m in segs:
            cols += [f"{m}_Time", f"{m}_Pos"]
        template = pd.DataFrame([[None, None] + [None]*(2*len(segs)) for _ in range(8)], columns=cols)
        st.info("Enter **segment times (s)** per 200 m and (optionally) positions.")
        work = st.data_editor(template if 'manual_grid' not in st.session_state else st.session_state['manual_grid'],
                              num_rows="dynamic", use_container_width=True, key="manual_grid").copy()
        st.success("Manual grid captured.")
except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.markdown("### Raw / Converted Table")
st.dataframe(work.head(12), use_container_width=True)
_dbg("Columns", list(work.columns))

# =============== Metrics Builder (PI v3) ===============
def build_metrics(df_in: pd.DataFrame, rounded_m: int):
    w = df_in.copy()
    # numeric Finish_Pos if present
    if "Finish_Pos" in w.columns: w["Finish_Pos"] = as_num(w["Finish_Pos"])

    # identify 200 m *_Time columns -> markers
    time_cols = [c for c in w.columns if str(c).endswith("_Time")]
    seg_markers = []
    for c in time_cols:
        try: seg_markers.append(int(str(c).split("_")[0]))
        except Exception: pass
    seg_markers = sorted(set(seg_markers), reverse=True)

    # speeds per segment
    for m in seg_markers:
        col = f"{m}_Time"
        w[f"spd_{m}"] = 200.0 / as_num(w[col])

    # Race time as sum of segment times (fallback to provided column)
    if seg_markers:
        sum_cols = [f"{m}_Time" for m in seg_markers if f"{m}_Time" in w.columns]
        w["RaceTime_s"] = w[sum_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    else:
        w["RaceTime_s"] = as_num(w.get("Race Time", np.nan))

    # F200 index (first available segment)
    first_mark = max(seg_markers) if seg_markers else None
    if first_mark and f"spd_{first_mark}" in w.columns:
        f_med = w[f"spd_{first_mark}"].median(skipna=True)
        w["F200_idx"] = 100.0 * (w[f"spd_{first_mark}"] / f_med)
    else: w["F200_idx"] = np.nan

    # tsSPI (exclude first 200, last 600)
    def _tsspi_row(row):
        if len(seg_markers) < 5: return np.nan
        mids = seg_markers[1:-3]
        vals = [row.get(f"spd_{m}") for m in mids]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))
    w["_mid_spd"] = w.apply(_tsspi_row, axis=1)
    mid_med = w["_mid_spd"].median(skipna=True)
    w["tsSPI"] = 100.0 * (w["_mid_spd"] / mid_med)

    # Accel (600→200)
    last_mark = min(seg_markers) if seg_markers else None
    pre_marks = [m for m in seg_markers if last_mark and m > last_mark]
    accel_window = pre_marks[-3:] if len(pre_marks) >= 3 else pre_marks
    def _mean(row, marks):
        vals = [row.get(f"spd_{m}") for m in marks if f"spd_{m}" in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))
    w["_accel_spd"] = w.apply(lambda r: _mean(r, accel_window), axis=1)
    accel_med = w["_accel_spd"].median(skipna=True)
    w["Accel"] = 100.0 * (w["_accel_spd"] / accel_med)

    # Grind (last 200)
    if last_mark and f"spd_{last_mark}" in w.columns:
        w["_grind_spd"] = w[f"spd_{last_mark}"]
        grind_med = w["_grind_spd"].median(skipna=True)
        w["Grind"] = 100.0 * (w["_grind_spd"] / grind_med)
    else:
        w["_grind_spd"] = np.nan
        w["Grind"] = np.nan

    # PI v3 (weights with reweight on missing parts; 100 baseline -> 0)
    PI_W = {"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25}
    def _pi_row(row):
        parts, weights = [], []
        for k,wgt in PI_W.items():
            v = row.get(k)
            if pd.notna(v):
                parts.append(wgt*(v-100.0))
                weights.append(wgt)
        if not weights: return np.nan
        scaled = sum(parts)/sum(weights)
        return max(0.0, round(scaled/5.0, 3))  # 1 PI per +5 over field
    w["PI"] = w.apply(_pi_row, axis=1)

    # ---- GCI (0–10) distance-aware using time deficit + late quality + sustained pace + efficiency ----
    def bucket(dm):
        if dm <= 1400: return "SPRINT"
        if dm < 1800:  return "MILE"
        if dm < 2200:  return "MIDDLE"
        return "STAY"

    prof = {
        "SPRINT": dict(wT=0.20, wPACE=0.45, wSS=0.25, wEFF=0.10),
        "MILE":   dict(wT=0.24, wPACE=0.40, wSS=0.26, wEFF=0.10),
        "MIDDLE": dict(wT=0.26, wPACE=0.38, wSS=0.26, wEFF=0.10),
        "STAY":   dict(wT=0.28, wPACE=0.35, wSS=0.27, wEFF=0.10),
    }[bucket(rounded_m)]

    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try: winner_time = w["RaceTime_s"].min()
        except Exception: winner_time = None

    def map_pct(x, lo=98.0, hi=104.0):
        if pd.isna(x): return 0.0
        return clamp01((float(x)-lo)/(hi-lo))  # 98 -> 0, 104 -> 1

    G = []
    for _, r in w.iterrows():
        # T (deficit to winner)
        T = 0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d = float(r["RaceTime_s"] - winner_time)
            if d <= 0.30: T = 1.0
            elif d <= 0.60: T = 0.7
            elif d <= 1.00: T = 0.4
            else: T = 0.2

        # PACE from late quality (Accel+Grind, emphasised) and sustained speed
        LQ = 0.6*map_pct(r.get("Accel")) + 0.4*map_pct(r.get("Grind"))
        SS = map_pct(r.get("tsSPI"))

        # EFF: how close Accel & Grind are to 100 (balanced finish)
        acc = r.get("Accel"); grd = r.get("Grind")
        if pd.isna(acc) or pd.isna(grd):
            EFF = 0.0
        else:
            dev = (abs(acc-100.0) + abs(grd-100.0))/2.0
            EFF = clamp01(1.0 - dev/8.0)  # within ±8 ≈ good

        score01 = prof["wT"]*T + prof["wPACE"]*LQ + prof["wSS"]*SS + prof["wEFF"]*EFF
        G.append(round(10.0*score01, 3))
    w["GCI"] = G

    # presentation rounding
    for c in ["F200_idx","tsSPI","Accel","Grind","PI","GCI"]: 
        if c in w.columns: w[c] = w[c].round(3)

    return w, seg_markers

try:
    metrics, seg_markers = build_metrics(work, rounded_distance)
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# =============== Metrics table (now with GCI) ===============
st.markdown("## Sectional Metrics (PI & GCI)")
show_cols = ["Horse","Finish_Pos","RaceTime_s","F200_idx","tsSPI","Accel","Grind","PI","GCI"]
for c in show_cols:
    if c not in metrics.columns: metrics[c] = np.nan
tbl = metrics[show_cols].copy()
st.dataframe(tbl.sort_values(["PI","Finish_Pos"], ascending=[False, True]), use_container_width=True)

# =============== Visual 1: Shape Map ===============
st.markdown("## Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)")
fig, ax = plt.subplots()
x = metrics["Accel"] - 100.0
y = metrics["Grind"] - 100.0
cval = metrics["tsSPI"] - 100.0
mask = x.notna() & y.notna()
xv = x[mask].values; yv = y[mask].values; cv = cval[mask].values
sc = ax.scatter(xv, yv, c=cv, cmap="coolwarm", s=60, alpha=0.95, edgecolor="k", linewidth=0.4)
for name, xx, yy in zip(metrics.loc[mask,"Horse"], xv, yv):
    ax.annotate(str(name), (xx, yy), xytext=(4,4), textcoords="offset points", fontsize=8)
ax.axvline(0, color="gray", ls="--", lw=1); ax.axhline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("Acceleration (index vs field, 600→200)")
ax.set_ylabel("Grind (index vs field, 200→Finish)")
ax.set_title("Quadrants: +X = late acceleration; +Y = strong last 200 (‘grind’). Colour = tsSPI vs field.")
# legend below chart
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("tsSPI deviation")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
st.pyplot(fig)
st.caption("Tip: top-right = true closers; bottom-right = sharp kick but faded late; top-left = on-pace grinds; bottom-left = off-pace and faded.")

# =============== Visual 2: Pace curve ===============
st.markdown("## Pace Curve — field average (black) + Top 8 finishers")
valid_markers = []
for c in [c for c in work.columns if str(c).endswith("_Time")]:
    try: valid_markers.append(int(str(c).split("_")[0]))
    except Exception: pass
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

    # choose Top 8
    if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
        top8 = metrics.sort_values("Finish_Pos").head(8)
    else:
        top8 = metrics.sort_values("PI", ascending=False).head(8)

    cols = color_cycle(8)
    for i, (_, r) in enumerate(top8.iterrows()):
        y_vals = []
        for c in speed_cols:
            val = r.get(c, np.nan)
            if pd.isna(val) and "Horse" in metrics.columns and "Horse" in work.columns:
                row0 = work[work["Horse"]==r.get("Horse")]
                if not row0.empty: val = row0.iloc[0].get(c, np.nan)
            y_vals.append(200.0/val if pd.notna(val) and val>0 else np.nan)
        ax2.plot(x_idx, y_vals, linewidth=2, marker="o", label=str(r.get("Horse","")), color=cols[i])

    ax2.set_xticks(x_idx); ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_ylabel("Speed (m/s)"); ax2.set_title("Pace over 200 m segments")
    ax2.grid(True, linestyle="--", alpha=0.3)
    # legend below plot
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9)
    st.pyplot(fig2)
    st.caption("Black line is field average. Coloured lines = Top 8 finishers. Left = early race, right = home straight.")
else:
    st.info("Not enough 200 m *_Time columns to draw the pace curve.")

# =============== Visual 3: Top-8 PI stacked bars ===============
st.markdown("## Top-8 PI — contributions by metric")
PI_W = {"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25}
top8_pi = metrics.sort_values("PI", ascending=False).head(8).copy()
if not top8_pi.empty:
    horses = []
    contrib = {"F200":[], "tsSPI":[], "Accel":[], "Grind":[]}
    for _, row in top8_pi.iterrows():
        horses.append(str(row.get("Horse","")))
        parts, weights = [], []
        for k,wgt in PI_W.items():
            v = row.get(k)
            if pd.notna(v):
                parts.append(wgt*(v-100.0)); weights.append(wgt)
        scale = sum(weights) if weights else 1.0
        conv = lambda units: (units/scale)/5.0 if scale>0 else 0.0
        contrib["F200"].append(conv(PI_W["F200_idx"]*(row.get("F200_idx",np.nan)-100.0)) if pd.notna(row.get("F200_idx")) else 0.0)
        contrib["tsSPI"].append(conv(PI_W["tsSPI"]*(row.get("tsSPI",np.nan)-100.0)) if pd.notna(row.get("tsSPI")) else 0.0)
        contrib["Accel"].append(conv(PI_W["Accel"]*(row.get("Accel",np.nan)-100.0)) if pd.notna(row.get("Accel")) else 0.0)
        contrib["Grind"].append(conv(PI_W["Grind"]*(row.get("Grind",np.nan)-100.0)) if pd.notna(row.get("Grind")) else 0.0)

    fig3, ax3 = plt.subplots(figsize=(max(6,0.9*len(horses)),4))
    idx = np.arange(len(horses)); bottoms = np.zeros(len(horses))
    palette = {"F200":"#6baed6","tsSPI":"#9e9ac8","Accel":"#74c476","Grind":"#fd8d3c"}
    for key in ["F200","tsSPI","Accel","Grind"]:
        vals = np.array(contrib[key])
        ax3.bar(idx, vals, bottom=bottoms, label=key, color=palette[key], edgecolor="black", linewidth=0.4)
        bottoms += vals
    ax3.set_xticks(idx); ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylabel("PI (stacked contributions)"); ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    # legend below
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    st.pyplot(fig3)
    st.caption("Each bar splits the horse’s PI into contributions from F200, tsSPI, Accel and Grind (after reweighting).")

# =============== Hidden Horses (RSS + HAS + ASI) ===============
st.markdown("## Hidden Horses")

hh = metrics.copy()

# RSS: Relative Sectional Superiority (z across tsSPI, Accel, Grind)
def z(series):
    s = series.astype(float)
    mu, sd = np.nanmean(s), np.nanstd(s)
    if not np.isfinite(sd) or sd == 0: return pd.Series([0.0]*len(s), index=s.index)
    return (s - mu)/sd

hh["z_ts"] = z(hh["tsSPI"]); hh["z_ac"] = z(hh["Accel"]); hh["z_gr"] = z(hh["Grind"])
hh["RSS"] = 0.40*hh["z_ts"] + 0.35*hh["z_ac"] + 0.25*hh["z_gr"]

# HAS: Hidden Against Shape — if race shape clearly favoured Accel or Grind, reward opposite specialist
shape_bias = (hh["Accel"].median(skipna=True) - 100.0) - (hh["Grind"].median(skipna=True) - 100.0)
if shape_bias >= 1.0:
    # kick-favouring; reward grinders
    hh["HAS"] = (hh["Grind"] - hh["Accel"]).clip(lower=0.0)/10.0
elif shape_bias <= -1.0:
    # grind-favouring; reward late accelerators
    hh["HAS"] = (hh["Accel"] - hh["Grind"]).clip(lower=0.0)/10.0
else:
    hh["HAS"] = 0.0

# ASI: Against-Shape dual-band flag (1 if strongly opposite-style)
hh["ASI"] = 0
hh.loc[((hh["Accel"]>=105)&(hh["Grind"]<=97)) | ((hh["Grind"]>=105)&(hh["Accel"]<=97)), "ASI"] = 1

# Composite "Hidden" score (scaled to ~0–3)
hh["HiddenScore"] = (hh["RSS"].fillna(0)/2.5) + hh["HAS"].fillna(0) + hh["ASI"]*0.2

# Short note
def hh_note(r):
    bits = []
    if r["ASI"]==1:
        if r["Accel"]>=105: bits.append("closed strongly against a grind-leaning shape")
        else: bits.append("ground late against a kick-leaning shape")
    if r["HAS"]>0:
        bits.append("ran opposite to race bias")
    if r["RSS"]>0.6:
        bits.append("sectionals superior to field")
    if not bits:
        return "No clear hidden angle."
    return "; ".join(bits)+"."

hh["Note"] = hh.apply(hh_note, axis=1)

cols_hh = ["Horse","Finish_Pos","PI","GCI","tsSPI","Accel","Grind","RSS","HAS","ASI","HiddenScore","Note"]
for c in cols_hh:
    if c not in hh.columns: hh[c] = np.nan

hidden_table = hh.sort_values(["HiddenScore","PI"], ascending=[False, False]).head(8)[cols_hh]
st.dataframe(hidden_table, use_container_width=True)
st.caption("Hidden Horses surfacing runners whose sectionals outperformed the shape: RSS = sectional z-score blend, HAS = against-shape bonus, ASI = strong opposite-style flag.")

# =============== Footer notes ===============
st.caption(
    "Definitions — F200_idx: first 200 m vs field (100 = par). "
    "tsSPI: sustained mid-race pace excluding first 200 and last 600. "
    "Accel: 600→200 window; Grind: last 200. "
    "PI: blended sectional performance (reweighted if parts missing). "
    "GCI: distance-aware class index (0–10) mixing time deficit, late quality, sustained pace and efficiency."
)
