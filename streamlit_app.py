import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ======================= Page config =======================
st.set_page_config(page_title="Race Edge — PI v3.1 + Hidden Horses v2", layout="wide")

# ======================= Small helpers =====================
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def mad_std(x):
    # robust sigma from MAD
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def winsorize(s, p_lo=0.10, p_hi=0.90):
    lo = s.quantile(p_lo)
    hi = s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)

# ======================= Sidebar ===========================
with st.sidebar:
    st.markdown("### Data source")
    mode = st.radio("", ["Upload file", "Manual input"], index=0, label_visibility="collapsed")

    race_distance_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, step=50, value=1200)
    rounded_distance = int(np.ceil(race_distance_input / 200.0) * 200)
    st.caption(f"Rounded distance used: **{rounded_distance} m** (manual grid counts **down** from here).")

    if mode == "Manual input":
        n_horses = st.number_input("Number of horses", min_value=2, max_value=20, value=8, step=1)

    st.markdown("---")
    DEBUG = st.toggle("Debug info", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔧 {label}")
        if obj is not None:
            st.write(obj)

# ======================= Input handling ====================
work = None
try:
    if mode == "Upload file":
        up = st.file_uploader("Upload CSV/XLSX with 200 m segments like `1200_Time`, `1000_Time`, … and optional `*_Pos`.", type=["csv","xlsx","xls"])
        if not up:
            st.stop()
        work = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        st.success("File loaded.")
    else:
        # Build manual grid: 1200, 1000, ..., 200 (counts down)
        segs = list(range(rounded_distance, 0, -200))
        cols = ["Horse", "Finish_Pos"]
        for m in segs:
            cols += [f"{m}_Time", f"{m}_Pos"]
        template = pd.DataFrame([[None, None] + [None] * (2 * len(segs)) for _ in range(n_horses)], columns=cols)
        st.info("Enter **segment times (seconds)** for each 200 m leg; positions optional.")
        work = st.data_editor(template, num_rows="dynamic", use_container_width=True, key="manual_grid").copy()
        st.success("Manual grid captured.")
except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG:
        st.exception(e)
    st.stop()

st.markdown("### Raw / Converted Table")
st.dataframe(work.head(12), use_container_width=True)
_dbg("Columns", list(work.columns))

# ======================= Core metric build =================
def build_metrics(df_in: pd.DataFrame, distance_m: int):
    w = df_in.copy()

    # numeric finish pos (if present)
    if "Finish_Pos" in w.columns:
        w["Finish_Pos"] = as_num(w["Finish_Pos"])

    # find 200m *_Time columns -> integer markers (e.g., 1200, 1000, ..., 200)
    time_cols = [c for c in w.columns if str(c).endswith("_Time")]
    seg_markers = []
    for c in time_cols:
        try:
            seg_markers.append(int(str(c).split("_")[0]))
        except Exception:
            pass
    seg_markers = sorted(set(seg_markers), reverse=True)

    # per-segment speeds
    for m in seg_markers:
        w[f"spd_{m}"] = 200.0 / as_num(w[f"{m}_Time"])

    # race time = sum of segment times (fallback to provided)
    if seg_markers:
        sum_cols = [f"{m}_Time" for m in seg_markers if f"{m}_Time" in w.columns]
        w["RaceTime_s"] = w[sum_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    else:
        w["RaceTime_s"] = as_num(w.get("Race Time", np.nan))

    # ---------- small-field tools ----------
    def shrink_center(idx_series):
        """Return blended center ~ median with small-N shrink towards 100."""
        x = idx_series.dropna().values
        N_eff = len(x)
        if N_eff == 0:
            return 100.0, 0
        med_race = float(np.median(x))
        alpha = N_eff / (N_eff + 6.0)  # kappa = 6 (gentle)
        center = alpha * med_race + (1 - alpha) * 100.0
        return center, N_eff

    def dispersion_equalizer(delta_series, N_eff, N_ref=10, beta=0.20, cap=1.20):
        """Scale (idx-100) by a mild factor when field is small."""
        gamma = 1.0 + beta * max(0, N_ref - N_eff) / N_ref
        gamma = min(gamma, cap)
        return delta_series * gamma

    def variance_floor(idx_series, floor=1.5, cap=1.25):
        """If robust spread of (idx-100) too low, gently stretch."""
        deltas = idx_series - 100.0
        sigma = mad_std(deltas)
        if not np.isfinite(sigma) or sigma <= 0:
            return idx_series
        if sigma < floor:
            factor = min(cap, floor / sigma)
            return 100.0 + deltas * factor
        return idx_series

    # ---------- F200 index ----------
    first_mark = max(seg_markers) if seg_markers else None
    if first_mark and f"spd_{first_mark}" in w.columns:
        base = w[f"spd_{first_mark}"]
        # preliminary index to derive center & guards
        prelim = 100.0 * (base / base.median(skipna=True))
        center, n_eff = shrink_center(prelim)
        f200 = 100.0 * (base / (center / 100.0 * base.median(skipna=True)))
        # apply dispersion equalizer and variance floor
        f200 = 100.0 + dispersion_equalizer(f200 - 100.0, n_eff)
        f200 = variance_floor(f200)
        w["F200_idx"] = f200
    else:
        w["F200_idx"] = np.nan

    # ---------- tsSPI: exclude first 200 and last 600; adaptive fallback ----------
    last_mark = min(seg_markers) if seg_markers else None
    def tsspi_avg(row):
        if len(seg_markers) == 0:
            return np.nan
        mids = seg_markers[1:-3]  # drop first and last 600
        if len(mids) < 2:
            mids = seg_markers[1:-2]  # drop first and last 400
        if len(mids) < 1:
            mids = seg_markers[1:2]   # at least one mid segment if exists
        speeds = [row.get(f"spd_{m}") for m in mids if f"spd_{m}" in row.index]
        speeds = [s for s in speeds if pd.notna(s)]
        return np.nan if not speeds else float(np.mean(speeds))

    w["_mid_spd"] = w.apply(tsspi_avg, axis=1)
    # preliminary tsSPI index vs median to compute center & guards
    mid_med_raw = w["_mid_spd"].median(skipna=True)
    w["tsSPI_raw"] = 100.0 * (w["_mid_spd"] / mid_med_raw)
    center_ts, n_ts = shrink_center(w["tsSPI_raw"])
    # re-center
    tsSPI = 100.0 * (w["_mid_spd"] / (center_ts / 100.0 * mid_med_raw))
    # equalizer + floor
    tsSPI = 100.0 + dispersion_equalizer(tsSPI - 100.0, n_ts)
    tsSPI = variance_floor(tsSPI)
    w["tsSPI"] = tsSPI

    # ---------- Accel: 600→200 (up to 3 segments; adaptive) ----------
    pre_marks = [m for m in seg_markers if last_mark and m > last_mark]
    accel_win = pre_marks[-3:] if len(pre_marks) >= 3 else (pre_marks[-2:] if len(pre_marks) >= 2 else pre_marks[-1:])
    def mean_marks(row, marks):
        vals = [row.get(f"spd_{m}") for m in marks if f"spd_{m}" in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))
    w["_accel_spd"] = w.apply(lambda r: mean_marks(r, accel_win), axis=1)
    a_med_raw = w["_accel_spd"].median(skipna=True)
    w["Accel_raw"] = 100.0 * (w["_accel_spd"] / a_med_raw)
    center_a, n_a = shrink_center(w["Accel_raw"])
    Accel = 100.0 * (w["_accel_spd"] / (center_a / 100.0 * a_med_raw))
    Accel = 100.0 + dispersion_equalizer(Accel - 100.0, n_a)
    Accel = variance_floor(Accel)
    w["Accel"] = Accel

    # ---------- Grind: last 200 ----------
    if last_mark and f"spd_{last_mark}" in w.columns:
        w["_grind_spd"] = w[f"spd_{last_mark}"]
        g_med_raw = w["_grind_spd"].median(skipna=True)
        w["Grind_raw"] = 100.0 * (w["_grind_spd"] / g_med_raw)
        center_g, n_g = shrink_center(w["Grind_raw"])
        Grind = 100.0 * (w["_grind_spd"] / (center_g / 100.0 * g_med_raw))
        Grind = 100.0 + dispersion_equalizer(Grind - 100.0, n_g)
        Grind = variance_floor(Grind)
        w["Grind"] = Grind
    else:
        w["_grind_spd"] = np.nan
        w["Grind"] = np.nan

    # ---------- PI v3.1 (reweighting; baseline 100 -> 0) ----------
    PI_W = {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
    def pi_row(row):
        parts, weights = [], []
        for k, wgt in PI_W.items():
            v = row.get(k, np.nan)
            if pd.notna(v):
                parts.append(wgt * (v - 100.0))
                weights.append(wgt)
        if not weights:
            return np.nan
        scaled = sum(parts) / sum(weights)
        return max(0.0, round(scaled / 5.0, 3))  # ~0–10 range
    w["PI"] = w.apply(pi_row, axis=1)

    # ---------- GCI (0–10) ----------
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
    }[bucket(distance_m)]

    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try:
            winner_time = w["RaceTime_s"].min()
        except Exception:
            winner_time = None

    def map_pct(x, lo=98.0, hi=104.0):
        if pd.isna(x): return 0.0
        return clamp((float(x) - lo) / (hi - lo), 0.0, 1.0)

    gci_vals = []
    for _, r in w.iterrows():
        # T
        T = 0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d = float(r["RaceTime_s"] - winner_time)
            if d <= 0.30: T = 1.0
            elif d <= 0.60: T = 0.7
            elif d <= 1.00: T = 0.4
            else: T = 0.2
        # PACE via late quality + sustained
        LQ = 0.6 * map_pct(r.get("Accel")) + 0.4 * map_pct(r.get("Grind"))
        SS = map_pct(r.get("tsSPI"))
        # EFF (balance)
        acc, grd = r.get("Accel"), r.get("Grind")
        if pd.isna(acc) or pd.isna(grd):
            EFF = 0.0
        else:
            dev = (abs(acc - 100.0) + abs(grd - 100.0)) / 2.0
            EFF = clamp(1.0 - dev / 8.0, 0.0, 1.0)
        score01 = prof["wT"] * T + prof["wPACE"] * LQ + prof["wSS"] * SS + prof["wEFF"] * EFF
        gci_vals.append(round(10.0 * score01, 3))
    w["GCI"] = gci_vals

    # tidy
    for c in ["F200_idx", "tsSPI", "Accel", "Grind", "PI", "GCI", "RaceTime_s"]:
        if c in w.columns:
            w[c] = w[c].round(3)

    return w, seg_markers

try:
    metrics, seg_markers = build_metrics(work, rounded_distance)
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG:
        st.exception(e)
    st.stop()

# ======================= Metrics table =====================
st.markdown("## Sectional Metrics (PI v3.1 & GCI)")
show_cols = ["Horse", "Finish_Pos", "RaceTime_s", "F200_idx", "tsSPI", "Accel", "Grind", "PI", "GCI"]
for c in show_cols:
    if c not in metrics.columns:
        metrics[c] = np.nan
st.dataframe(metrics[show_cols].sort_values(["PI","Finish_Pos"], ascending=[False, True]),
             use_container_width=True)

# ======================= Visual: Shape Map (improved) =======================
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

st.markdown("## Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)")

# knobs you can tweak
TOP_LABELS = 8                 # how many runners get arrowed labels
LABEL_BY   = "both"            # "swing" (|Accel|+|Grind|), "tsspi", or "both"
DOT_MIN, DOT_MAX = 40, 140     # point-size range for PI bubbles
TINT = 0.06                    # quadrant tint alpha (0–1)

# data
x = (metrics["Accel"] - 100.0).rename("AccelΔ")     # + = late acceleration
y = (metrics["Grind"] - 100.0).rename("GrindΔ")     # + = strong last 200
c = (metrics["tsSPI"] - 100.0).rename("tsSPIΔ")     # colour
pi = metrics["PI"].fillna(0)
names = metrics["Horse"].astype(str)

mask = x.notna() & y.notna() & c.notna()
if mask.sum() == 0:
    st.info("Not enough data to draw the shape map.")
else:
    xv, yv, cv = x[mask].to_numpy(), y[mask].to_numpy(), c[mask].to_numpy()
    piv = pi[mask].to_numpy()
    namev = names[mask].tolist()

    # bubble size from PI (safe scaling)
    if np.isfinite(piv).any():
        piv_norm = (piv - np.nanmin(piv)) / (np.nanmax(piv) - np.nanmin(piv) + 1e-9)
        sizes = DOT_MIN + piv_norm * (DOT_MAX - DOT_MIN)
    else:
        sizes = np.full_like(xv, DOT_MIN)

    # choose whom to label
    swing = np.abs(xv) + np.abs(yv)
    rank_swing = np.argsort(-swing)
    rank_tsspi = np.argsort(-np.abs(cv))
    if LABEL_BY == "swing":
        label_idx = list(rank_swing[:TOP_LABELS])
    elif LABEL_BY == "tsspi":
        label_idx = list(rank_tsspi[:TOP_LABELS])
    else:  # both
        label_idx = list(dict.fromkeys(list(rank_swing[:TOP_LABELS//2]) + list(rank_tsspi[:TOP_LABELS - TOP_LABELS//2])))

    # figure
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # quadrant tinting (very light)
    lim = max(4.5, np.ceil(max(np.max(np.abs(xv)), np.max(np.abs(yv))) / 1.5) * 1.5)
    ax.add_patch(Rectangle((0, 0),  lim,  lim, facecolor="#4daf4a", alpha=TINT, edgecolor="none"))  # +accel +grind
    ax.add_patch(Rectangle((-lim, 0), lim,  lim, facecolor="#377eb8", alpha=TINT, edgecolor="none")) # -accel +grind
    ax.add_patch(Rectangle((0, -lim), lim, lim, facecolor="#ff7f00", alpha=TINT, edgecolor="none"))  # +accel -grind
    ax.add_patch(Rectangle((-lim, -lim), lim, lim, facecolor="#984ea3", alpha=TINT, edgecolor="none"))# -accel -grind

    # zero lines
    ax.axvline(0, color="gray", lw=1.3, ls=(0,(3,3)))
    ax.axhline(0, color="gray", lw=1.3, ls=(0,(3,3)))

    # colour centered at 0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(-1.0, float(np.min(cv))), vmax=max(1.0, float(np.max(cv))))
    sc = ax.scatter(xv, yv, s=sizes, c=cv, cmap="coolwarm", norm=norm,
                    edgecolor="black", linewidth=0.5, alpha=0.95)

    # arrowed labels for top runners (simple collision-lite offsets)
    angles = np.linspace(20, 340, num=max(1, len(label_idx)), endpoint=False)
    for k, i in enumerate(label_idx):
        dx = 0.85 * np.cos(np.deg2rad(angles[k]))
        dy = 0.85 * np.sin(np.deg2rad(angles[k]))
        ax.annotate(
            namev[i],
            xy=(xv[i], yv[i]),
            xytext=(xv[i] + dx, yv[i] + dy),
            fontsize=8,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.7, shrinkA=0, shrinkB=3, color="black", alpha=0.9),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
        )

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Acceleration vs field (points)  →")
    ax.set_ylabel("Grind vs field (points)  ↑")
    ax.set_title("Quadrants: +X = late acceleration; +Y = strong last 200. Colour = tsSPI deviation")

    # size legend (PI)
    from matplotlib.lines import Line2D
    s_ex = [DOT_MIN, 0.5*(DOT_MIN+DOT_MAX), DOT_MAX]
    h_ex = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                   markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
    ax.legend(h_ex, ["PI: low", "PI: mid", "PI: high"],
              loc="upper left", frameon=False, fontsize=8)

    # colorbar, below plot
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("tsSPI − 100")

    ax.grid(True, linestyle=":", alpha=0.25)
    st.pyplot(fig)
    st.caption(
        "Bubble size = PI (bigger = stronger overall). Red = faster cruise (tsSPI>field), blue = slower. "
        "Top labels mark the most shape-extreme or cruise-extreme runners."
    )

# ======================= Visual 2: Pace Curve =============
st.markdown("## Pace Curve — field average (black) + Top 8 finishers")

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

    # choose top 8 by finish (else by PI)
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
                row0 = work[work["Horse"] == r.get("Horse")]
                if not row0.empty:
                    val = row0.iloc[0].get(c, np.nan)
            y_vals.append(200.0 / val if pd.notna(val) and val > 0 else np.nan)
        ax2.plot(x_idx, y_vals, linewidth=2, marker="o", label=str(r.get("Horse", "")), color=cols[i])

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Pace over 200 m segments (left = early, right = home straight)")
    ax2.grid(True, linestyle="--", alpha=0.3)
    # legend below plot
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9)
    st.pyplot(fig2)
    st.caption("Black = field average. Coloured lines = Top 8 finishers over the same segments.")
else:
    st.info("Not enough 200 m *_Time columns to draw the pace curve.")

# ======================= Visual 3: Top-8 PI bars ==========
st.markdown("## Top-8 PI — stacked contributions")
PI_W = {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
top8_pi = metrics.sort_values("PI", ascending=False).head(8).copy()
if not top8_pi.empty:
    horses = []
    contrib = {"F200": [], "tsSPI": [], "Accel": [], "Grind": []}
    for _, row in top8_pi.iterrows():
        horses.append(str(row.get("Horse", "")))
        parts, weights = [], []
        for k, wgt in PI_W.items():
            v = row.get(k)
            if pd.notna(v):
                parts.append(wgt * (v - 100.0))
                weights.append(wgt)
        scale = sum(weights) if weights else 1.0
        conv = lambda units: (units / scale) / 5.0 if scale > 0 else 0.0
        contrib["F200"].append(conv(PI_W["F200_idx"] * (row.get("F200_idx", np.nan) - 100.0)) if pd.notna(row.get("F200_idx")) else 0.0)
        contrib["tsSPI"].append(conv(PI_W["tsSPI"] * (row.get("tsSPI", np.nan) - 100.0)) if pd.notna(row.get("tsSPI")) else 0.0)
        contrib["Accel"].append(conv(PI_W["Accel"] * (row.get("Accel", np.nan) - 100.0)) if pd.notna(row.get("Accel")) else 0.0)
        contrib["Grind"].append(conv(PI_W["Grind"] * (row.get("Grind", np.nan) - 100.0)) if pd.notna(row.get("Grind")) else 0.0)

    fig3, ax3 = plt.subplots(figsize=(max(6, 0.9 * len(horses)), 4))
    idx = np.arange(len(horses))
    bottoms = np.zeros(len(horses))
    palette = {"F200": "#6baed6", "tsSPI": "#9e9ac8", "Accel": "#74c476", "Grind": "#fd8d3c"}
    for key in ["F200", "tsSPI", "Accel", "Grind"]:
        vals = np.array(contrib[key])
        ax3.bar(idx, vals, bottom=bottoms, label=key, color=palette[key], edgecolor="black", linewidth=0.4)
        bottoms += vals
    ax3.set_xticks(idx)
    ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylabel("PI (stacked contributions)")
    ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    # legend below
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    st.pyplot(fig3)
    st.caption("Bars split each horse’s PI into F200 / tsSPI / Accel / Grind contributions (after reweighting).")

# ======================= Hidden Horses v2 =================
st.markdown("## Hidden Horses (v2)")

hh = metrics.copy()

# --- SOS (winsorized + robust z) ---
if {"tsSPI","Accel","Grind"}.issubset(hh.columns):
    ts_w = winsorize(hh["tsSPI"])
    ac_w = winsorize(hh["Accel"])
    gr_w = winsorize(hh["Grind"])

    def rz(s):
        mu = np.nanmedian(s)
        sd = mad_std(s)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    z_ts = rz(ts_w).clip(-2.5, 3.5)
    z_ac = rz(ac_w).clip(-2.5, 3.5)
    z_gr = rz(gr_w).clip(-2.5, 3.5)

    hh["SOS_raw"] = 0.45 * z_ts + 0.35 * z_ac + 0.20 * z_gr

    # normalize SOS → ~[0, 2] using robust min/max in-race
    q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom = (q95 - q5) if (pd.notna(q95) and pd.notna(q5) and q95 > q5) else 1.0
    hh["SOS"] = 2.0 * (hh["SOS_raw"] - q5) / denom
    hh["SOS"] = hh["SOS"].clip(lower=0.0, upper=2.0)
else:
    hh["SOS"] = 0.0

# --- ASI² (against-shape magnitude with bias scaling) ---
acc_med = hh["Accel"].median(skipna=True)
grd_med = hh["Grind"].median(skipna=True)
bias = (acc_med - 100.0) - (grd_med - 100.0)  # >0 = kick-leaning shape
B = min(1.0, abs(bias) / 4.0)  # 4 pts = strong bias
S = hh["Accel"] - hh["Grind"]
if bias >= 0:
    hh["ASI2"] = B * ((-S).clip(lower=0.0)) / 5.0  # reward grinders vs kick bias
else:
    hh["ASI2"] = B * ((S).clip(lower=0.0)) / 5.0   # reward kick vs grind bias

# --- TFS (trip friction) ---
# last 3 segments (600 m) std normalized by horse mid speed
last3 = []
if len(seg_markers) >= 3:
    last3 = sorted(seg_markers)[0:3]  # e.g., [200,400,600] in ascending by distance covered
    last3 = sorted(last3, reverse=True)  # [600,400,200] markers (still refer to "m to go" style)

def tfs_row(row):
    spds = [row.get(f"spd_{m}") for m in last3 if f"spd_{m}" in row.index]
    spds = [s for s in spds if pd.notna(s)]
    if len(spds) < 2:
        return np.nan
    sigma = float(np.std(spds, ddof=0))
    mid = float(row.get("_mid_spd", np.nan))
    if not np.isfinite(mid) or mid <= 0:
        return np.nan
    return 100.0 * (sigma / mid)

hh["TFS"] = hh.apply(tfs_row, axis=1)

# distance-aware TFS gate
if rounded_distance <= 1200:
    gate = 4.0
elif rounded_distance < 1800:
    gate = 3.5
else:
    gate = 3.0
def tfs_plus(x):
    if pd.isna(x) or x < gate:
        return 0.0
    return min(0.6, (x - gate) / 3.0)
hh["TFS_plus"] = hh["TFS"].apply(tfs_plus)

# --- UEI (underused engine) ---
def uei_row(r):
    ts, ac, gr = r.get("tsSPI"), r.get("Accel"), r.get("Grind")
    if pd.isna(ts) or pd.isna(ac) or pd.isna(gr):
        return 0.0
    val = 0.0
    # sprint-home: strong cruise but little late show
    if ts >= 102 and ac <= 98 and gr <= 98:
        gap = min((ts - 102) / 3.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    # fast-early: strong stay/grind but shape suppressed kick
    if ts >= 102 and gr >= 102 and ac <= 100:
        gap = min(((ts - 102) + (gr - 102)) / 6.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    return round(val, 3)
hh["UEI"] = hh.apply(uei_row, axis=1)

# --- HiddenScore v2 (0..3) with guards ---
hidden = 0.55 * hh["SOS"].fillna(0) + 0.30 * hh["ASI2"].fillna(0) + 0.10 * hh["TFS_plus"].fillna(0) + 0.05 * hh["UEI"].fillna(0)
# field-size guard
N = int(hh.shape[0])
if N <= 6:
    hidden *= 0.9
hh["HiddenScore"] = hidden.clip(lower=0.0, upper=3.0)

# tiers
def hh_tier(s):
    if pd.isna(s):
        return ""
    if s >= 1.8:
        return "🔥 Top Hidden"
    if s >= 1.2:
        return "🟡 Notable Hidden"
    return ""
hh["Tier"] = hh["HiddenScore"].apply(hh_tier)

# notes
def hh_note(r):
    bits = []
    if r.get("Tier","") != "":
        if r["SOS"] >= 1.2:
            bits.append("sectionals superior")
        if r["ASI2"] >= 0.8:
            bits.append("ran against strong bias")
        elif r["ASI2"] >= 0.4:
            bits.append("ran against bias")
        if r["TFS_plus"] > 0:
            bits.append("trip friction late")
        if r["UEI"] >= 0.5:
            bits.append("latent potential if shape flips")
    return "; ".join(bits).capitalize() + ("." if bits else "")
hh["Note"] = hh.apply(hh_note, axis=1)

cols_hh = ["Horse", "Finish_Pos", "PI", "GCI", "tsSPI", "Accel", "Grind", "SOS", "ASI2", "TFS", "UEI", "HiddenScore", "Tier", "Note"]
for c in cols_hh:
    if c not in hh.columns:
        hh[c] = np.nan

st.dataframe(hh.sort_values(["Tier","HiddenScore","PI"], ascending=[True, False, False]).loc[:, cols_hh], use_container_width=True)
st.caption("Hidden Horses v2: SOS = sectional outlier score (robust), ASI² = against-shape magnitude, TFS = trip friction, UEI = underused engine. Tiering: 🔥 ≥1.8, 🟡 ≥1.2.")

# ======================= Footer ===========================
st.caption(
    "Definitions — F200_idx: first 200 m vs field (100 = par). "
    "tsSPI: sustained mid-race pace (excl. first 200 & last 600; adaptive). "
    "Accel: 600→200 window (adaptive); Grind: last 200. "
    "PI v3.1: blended sectional performance with small-field stabilization. "
    "GCI: distance-aware class index (0–10)."
)
