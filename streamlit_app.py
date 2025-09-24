import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import io
import math

# ======================= Page config =======================
st.set_page_config(page_title="Race Edge — PI v3.1 (distance+context) + Hidden Horses v2", layout="wide")

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

def _lerp(a, b, t):
    return a + (b - a) * float(t)

def _interpolate_weights(dm, a_dm, a_w, b_dm, b_w):
    # linear interpolate between two anchors a_dm -> b_dm
    span = float(b_dm - a_dm)
    t = 0.0 if span <= 0 else (float(dm) - a_dm) / span
    return {
        "F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
        "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
        "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
        "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t),
    }

# ======================= Sidebar ===========================
with st.sidebar:
    st.markdown("### Data source")
    mode = st.radio("", ["Upload file", "Manual input"], index=0, label_visibility="collapsed")

    # Use the *actual* distance (not rounded) for weighting logic
    race_distance_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, step=50, value=1200)
    # Rounded distance used only for manual grid row count
    rounded_distance = int(np.ceil(race_distance_input / 200.0) * 200)
    st.caption(f"Rounded distance used for manual grid: **{rounded_distance} m** (grid counts **down**).")

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
        up = st.file_uploader(
            "Upload CSV/XLSX with 200 m segments like `1200_Time`, `1000_Time`, … and optional `*_Pos`.",
            type=["csv","xlsx","xls"]
        )
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

# ======================= Distance + Context PI weights =======================
def pi_weights_distance_and_context(distance_m: float,
                                    acc_median: float | None,
                                    grd_median: float | None) -> dict:
    """
    Distance logic:
      - 1000m anchor: F200=0.12, tsSPI=0.35, Accel=0.36, Grind=0.17
      - 1100m anchor: F200=0.10, tsSPI=0.36, Accel=0.34, Grind=0.20
      - 1200m anchor: F200=0.08, tsSPI=0.37, Accel=0.30, Grind=0.25
      - >1200m: shift 0.01 per +100m from tsSPI → Grind, cap Grind at 0.40
                (F200 stays 0.08, Accel stays 0.30; tsSPI = 1 - F200 - Accel - Grind)

    Context nudge (tiny, ±0.02 max total):
      - If race is kick-biased (median Accel > median Grind), move up to 0.02 from Accel→Grind
      - If grind-biased (median Grind > median Accel), move up to 0.02 from Grind→Accel
      - Smoothly scaled by tanh(|bias| / 6), so a ~6pt bias ~ 0.02 shift
    """
    dm = float(distance_m or 1200)

    # ---- distance base weights ----
    if dm <= 1000:
        base = {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17}
    elif dm < 1100:
        base = _interpolate_weights(
            dm,
            1000, {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17},
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20}
        )
    elif dm < 1200:
        base = _interpolate_weights(
            dm,
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20},
            1200, {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
        )
    elif dm == 1200:
        base = {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
    else:
        # shift from tsSPI to Grind, +0.01 per +100m, cap Grind at 0.40
        shift_units = max(0.0, (dm - 1200.0) / 100.0) * 0.01
        grind = min(0.25 + shift_units, 0.40)
        F200  = 0.08
        ACC   = 0.30
        ts    = max(0.0, 1.0 - F200 - ACC - grind)
        base  = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":grind}

    # ---- context nudge (very small) ----
    acc_med = float(acc_median) if acc_median is not None else None
    grd_med = float(grd_median) if grd_median is not None else None

    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        # bias in "index points"
        bias = acc_med - grd_med  # +ve = kick-leaning, −ve = grind-leaning
        # scale 0..~1 using tanh, 6 pts ≈ strong bias → ~1
        scale = math.tanh(abs(bias) / 6.0)
        max_shift = 0.02 * scale  # don’t exceed 0.02 total on large bias

        F200 = base["F200_idx"]; ts = base["tsSPI"]; ACC = base["Accel"]; GR = base["Grind"]

        if bias > 0:
            # move Accel → Grind (kick-biased race)
            delta = min(max_shift, ACC - 0.26)  # keep a sensible floor on Accel
            ACC -= delta; GR += delta
        elif bias < 0:
            # move Grind → Accel (grind-biased race)
            delta = min(max_shift, GR - 0.18)   # keep a sensible floor on Grind
            GR  -= delta; ACC += delta

        # enforce grind cap and renormalize tsSPI to keep sum = 1.0
        GR = min(GR, 0.40)
        ts = max(0.0, 1.0 - F200 - ACC - GR)
        base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":GR}

    # final tiny numeric clean
    s = sum(base.values())
    if abs(s - 1.0) > 1e-6:
        base = {k: v / s for k, v in base.items()}

    return base

# ======================= Core metric build =================
def build_metrics(df_in: pd.DataFrame, distance_m: int, actual_distance_m: float):
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

    # per-segment speeds (m/s)
    for m in seg_markers:
        w[f"spd_{m}"] = 200.0 / as_num(w[f"{m}_Time"])

    # race time = sum of segment times (fallback to provided)
    if seg_markers:
        sum_cols = [f"{m}_Time" for m in seg_markers if f"{m}_Time" in w.columns]
        w["RaceTime_s"] = w[sum_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    else:
        w["RaceTime_s"] = as_num(w.get("Race Time", np.nan))

    # ---------- small-field stabilizers ----------
    def shrink_center(idx_series):
        x = idx_series.dropna().values
        N_eff = len(x)
        if N_eff == 0:
            return 100.0, 0
        med_race = float(np.median(x))
        alpha = N_eff / (N_eff + 6.0)    # gentle pull to 100 when N is small
        return alpha * med_race + (1 - alpha) * 100.0, N_eff

    def dispersion_equalizer(delta_series, N_eff, N_ref=10, beta=0.20, cap=1.20):
        gamma = 1.0 + beta * max(0, N_ref - N_eff) / N_ref
        return delta_series * min(gamma, cap)

    def variance_floor(idx_series, floor=1.5, cap=1.25):
        deltas = idx_series - 100.0
        sigma = mad_std(deltas)
        if not np.isfinite(sigma) or sigma <= 0:
            return idx_series
        if sigma < floor:
            factor = min(cap, floor / sigma)
            return 100.0 + deltas * factor
        return idx_series

    # ---------- F200 ----------
    first_mark = max(seg_markers) if seg_markers else None
    if first_mark and f"spd_{first_mark}" in w.columns:
        base = w[f"spd_{first_mark}"]
        prelim = 100.0 * (base / base.median(skipna=True))
        center, n_eff = shrink_center(prelim)
        f200 = 100.0 * (base / (center / 100.0 * base.median(skipna=True)))
        f200 = 100.0 + dispersion_equalizer(f200 - 100.0, n_eff)
        f200 = variance_floor(f200)
        w["F200_idx"] = f200
    else:
        w["F200_idx"] = np.nan

    # ---------- tsSPI (exclude first 200 & last 600; adaptive fallback) ----------
    last_mark = min(seg_markers) if seg_markers else None

    def tsspi_avg(row):
        if len(seg_markers) == 0:
            return np.nan
        mids = seg_markers[1:-3]          # drop first & last 600
        if len(mids) < 2: mids = seg_markers[1:-2]
        if len(mids) < 1: mids = seg_markers[1:2]
        vals = [row.get(f"spd_{m}") for m in mids if f"spd_{m}" in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))

    w["_mid_spd"] = w.apply(tsspi_avg, axis=1)
    mid_med = w["_mid_spd"].median(skipna=True)
    w["tsSPI_raw"] = 100.0 * (w["_mid_spd"] / mid_med)
    center_ts, n_ts = shrink_center(w["tsSPI_raw"])
    tsSPI = 100.0 * (w["_mid_spd"] / (center_ts / 100.0 * mid_med))
    tsSPI = 100.0 + dispersion_equalizer(tsSPI - 100.0, n_ts)
    tsSPI = variance_floor(tsSPI)
    w["tsSPI"] = tsSPI

    # ---------- Accel: 600→200 (adaptive up to 3 segments) ----------
    pre_marks = [m for m in seg_markers if last_mark and m > last_mark]
    accel_win = pre_marks[-3:] if len(pre_marks) >= 3 else (pre_marks[-2:] if len(pre_marks) >= 2 else pre_marks[-1:])

    def mean_marks(row, marks):
        vals = [row.get(f"spd_{m}") for m in marks if f"spd_{m}" in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))

    w["_accel_spd"] = w.apply(lambda r: mean_marks(r, accel_win), axis=1)
    a_med = w["_accel_spd"].median(skipna=True)
    w["Accel_raw"] = 100.0 * (w["_accel_spd"] / a_med)
    center_a, n_a = shrink_center(w["Accel_raw"])
    Accel = 100.0 * (w["_accel_spd"] / (center_a / 100.0 * a_med))
    Accel = 100.0 + dispersion_equalizer(Accel - 100.0, n_a)
    Accel = variance_floor(Accel)
    w["Accel"] = Accel

    # ---------- Grind: last 200 ----------
    if last_mark and f"spd_{last_mark}" in w.columns:
        g_base = w[f"spd_{last_mark}"]
        g_med = g_base.median(skipna=True)
        w["Grind_raw"] = 100.0 * (g_base / g_med)
        center_g, n_g = shrink_center(w["Grind_raw"])
        Grind = 100.0 * (g_base / (center_g / 100.0 * g_med))
        Grind = 100.0 + dispersion_equalizer(Grind - 100.0, n_g)
        Grind = variance_floor(Grind)
        w["Grind"] = Grind
    else:
        w["Grind"] = np.nan

    # ---------- PI v3.1 (distance- & context-aware weights, robust 0–10 scaling) ----------
    acc_med = w["Accel"].median(skipna=True)
    grd_med = w["Grind"].median(skipna=True)
    PI_W = pi_weights_distance_and_context(float(actual_distance_m), acc_med, grd_med)

    def pi_pts_row(row):
        parts, weights = [], []
        for k, wgt in PI_W.items():
            v = row.get(k, np.nan)
            if pd.notna(v):
                parts.append(wgt * (v - 100.0))  # index points above/below par
                weights.append(wgt)
        if not weights:
            return np.nan
        return sum(parts) / sum(weights)       # weighted delta in "points"
    w["PI_pts"] = w.apply(pi_pts_row, axis=1)

    # robust race-by-race scaling: median -> 5, ±~2σ -> [~0..10]
    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered = pts - med

    # robust sigma via MAD; add a small floor so ultra-flat races don't collapse
    sigma = mad_std(centered)
    if not np.isfinite(sigma) or sigma < 0.75:
        sigma = 0.75

    # map to 0..10 (tune 2.2 if you want a slightly wider/narrower spread)
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)

# ---------- GCI (0–10) — distance & context aligned ----------
def _gci_profile(dm: float) -> dict:
    # Distance-aware class weighting (Time/ Pace / Sustained / Efficiency)
    if dm <= 1400:   # SPRINT
        return dict(wT=0.20, wPACE=0.45, wSS=0.25, wEFF=0.10)
    elif dm < 1800:  # MILE
        return dict(wT=0.24, wPACE=0.40, wSS=0.26, wEFF=0.10)
    elif dm < 2200:  # MIDDLE
        return dict(wT=0.26, wPACE=0.38, wSS=0.26, wEFF=0.10)
    else:            # STAY
        return dict(wT=0.28, wPACE=0.35, wSS=0.27, wEFF=0.10)

def _map_pct(x, lo=98.0, hi=104.0):
    if pd.isna(x): return 0.0
    return clamp((float(x) - lo) / (hi - lo), 0.0, 1.0)

def compute_gci(df: pd.DataFrame, distance_m: float) -> pd.Series:
    dm = float(distance_m or 1200)
    prof = _gci_profile(dm)

    # Winner time (in seconds) from summed splits
    winner_time = None
    if "RaceTime_s" in df.columns and df["RaceTime_s"].notna().any():
        try:
            winner_time = float(df["RaceTime_s"].min())
        except Exception:
            winner_time = None

    # Context bias (same idea as PI): +ve → kick-leaning; −ve → grind-leaning
    acc_med = pd.to_numeric(df.get("Accel"), errors="coerce").median(skipna=True)
    grd_med = pd.to_numeric(df.get("Grind"), errors="coerce").median(skipna=True)
    bias = (acc_med - grd_med) if np.isfinite(acc_med) and np.isfinite(grd_med) else 0.0
    # Tiny nudge of the late-quality blend using tanh scaling (≤ ±0.10 total shift)
    shift = 0.10 * math.tanh(abs(bias) / 6.0)
    wA, wG = 0.60, 0.40  # baseline inside LQ
    if bias > 0:   # kick-biased → emphasise Grind inside LQ
        wA, wG = max(0.40, wA - shift), min(0.60, wG + shift)
    elif bias < 0: # grind-biased → emphasise Accel inside LQ
        wA, wG = min(0.70, wA + shift), max(0.30, wG - shift)

    # Distance-scaled time tolerance (bigger distances → bigger margins reasonable)
    # Base thresholds: 0.30 / 0.60 / 1.00s @1200m, scale ~ sqrt(dm/1200)
    tol_scale = math.sqrt(max(0.5, dm / 1200.0))
    t1, t2, t3 = 0.30*tol_scale, 0.60*tol_scale, 1.00*tol_scale

    out = []
    for _, r in df.iterrows():
        # T: closeness to winner on raw race time
        T = 0.0
        if (winner_time is not None) and pd.notna(r.get("RaceTime_s")):
            d = float(r["RaceTime_s"]) - winner_time
            if d <= t1:   T = 1.0
            elif d <= t2: T = 0.7
            elif d <= t3: T = 0.4
            else:         T = 0.2

        # LQ: bias-aware blend of Accel & Grind as class signal
        A = _map_pct(r.get("Accel"))
        G = _map_pct(r.get("Grind"))
        LQ = wA*A + wG*G

        # SS: sustained mid-race class (already distance-consistent via tsSPI definition)
        SS = _map_pct(r.get("tsSPI"))

        # EFF: balance of late profile (penalise lopsidedness)
        acc = r.get("Accel"); grd = r.get("Grind")
        if pd.isna(acc) or pd.isna(grd):
            EFF = 0.0
        else:
            dev = (abs(acc - 100.0) + abs(grd - 100.0)) / 2.0
            EFF = clamp(1.0 - dev / 8.0, 0.0, 1.0)

        score01 = (prof["wT"]*T + prof["wPACE"]*LQ + prof["wSS"]*SS + prof["wEFF"]*EFF)
        out.append(round(10.0 * score01, 3))
    return pd.Series(out, index=df.index)

# ---- apply (replace your old GCI block with this call) ----
metrics["GCI"] = compute_gci(metrics, float(race_distance_input))
# ======================= Metrics table =====================
st.markdown("## Sectional Metrics (PI v3.1 & GCI)")
show_cols = ["Horse", "Finish_Pos", "RaceTime_s", "F200_idx", "tsSPI", "Accel", "Grind", "PI", "GCI"]
for c in show_cols:
    if c not in metrics.columns:
        metrics[c] = np.nan
st.dataframe(metrics[show_cols].sort_values(["PI","Finish_Pos"], ascending=[False, True]),
             use_container_width=True)

# ===================== Sectional Shape Map — Accel (600→200) vs Grind (200→Finish) =====================
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import io

# --- neat labeling helpers (uses adjustText if available; otherwise a built-in repeller) ---
def _repel_labels_builtin(ax, x, y, labels, *,
                          init_shift=0.18, k_attract=0.006, k_repel=0.012,
                          max_iter=250):
    import numpy as np
    from matplotlib.lines import Line2D
    trans = ax.transData
    renderer = ax.figure.canvas.get_renderer()
    xy = np.column_stack([x, y]).astype(float)
    offs = np.zeros_like(xy)
    for i, (xi, yi) in enumerate(xy):
        offs[i] = [init_shift if xi >= 0 else -init_shift,
                   init_shift if yi >= 0 else -init_shift]

    texts, lines = [], []
    for (xi, yi), (dx, dy), lab in zip(xy, offs, labels):
        t = ax.text(xi+dx, yi+dy, lab, fontsize=8.6,
                    va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))
        texts.append(t)
        ln = Line2D([xi, xi+dx], [yi, yi+dy], lw=0.75, color="black", alpha=0.9)
        ax.add_line(ln); lines.append(ln)

    inv = ax.transData.inverted()
    for _ in range(max_iter):
        moved = False
        bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.15) for t in texts]
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if not bbs[i].overlaps(bbs[j]): 
                    continue
                # push apart in display coords
                ci = ((bbs[i].x0+bbs[i].x1)/2, (bbs[i].y0+bbs[i].y1)/2)
                cj = ((bbs[j].x0+bbs[j].x1)/2, (bbs[j].y0+bbs[j].y1)/2)
                vx, vy = ci[0]-cj[0], ci[1]-cj[1]
                if vx == 0 and vy == 0: vx = 1.0
                n = (vx**2 + vy**2)**0.5
                dx, dy = (vx/n)*k_repel*72, (vy/n)*k_repel*72  # ~points
                for t, s in ((texts[i], +1), (texts[j], -1)):
                    tx, ty = t.get_position()
                    px = trans.transform((tx, ty)) + s*np.array([dx, dy])
                    t.set_position(inv.transform(px)); moved = True
        # spring toward ~25 px leader length
        for t, (xi, yi) in zip(texts, xy):
            tx, ty = t.get_position()
            pt = trans.transform((tx, ty)); pp = trans.transform((xi, yi))
            d = ((pt[0]-pp[0])**2 + (pt[1]-pp[1])**2)**0.5
            tgt = 25.0
            if abs(d - tgt) > 1.0:
                v = (pt - pp) / (d + 1e-9)
                pt2 = pt + v * (0.6 * (tgt - d))
                t.set_position(inv.transform(pt2)); moved = True
        if not moved:
            break
    # update leader lines
    for t, ln, (xi, yi) in zip(texts, lines, xy):
        tx, ty = t.get_position()
        ln.set_data([xi, tx], [yi, ty])

def label_points_neatly(ax, x, y, names):
    """Prefer adjustText if present; fall back to built-in repeller."""
    try:
        from adjustText import adjust_text
        texts = []
        for xi, yi, nm in zip(x, y, names):
            texts.append(ax.text(xi, yi, nm, fontsize=8.6,
                                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70)))
        adjust_text(
            texts, x=x, y=y, ax=ax,
            only_move={'points': 'y', 'text': 'xy'},
            force_points=0.6, force_text=0.7,
            expand_text=(1.05, 1.15), expand_points=(1.05, 1.15),
            arrowprops=dict(arrowstyle="->", lw=0.75, color="black", alpha=0.9, shrinkA=0, shrinkB=3)
        )
    except Exception:
        _repel_labels_builtin(ax, x, y, names)

st.markdown("## Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)")

needed_cols = {"Horse", "Accel", "Grind", "tsSPI", "PI"}
if not needed_cols.issubset(metrics.columns):
    st.warning("Shape Map: required columns missing: " + ", ".join(sorted(needed_cols - set(metrics.columns))))
else:
    # slice & clean
    dfm = metrics.loc[:, ["Horse", "Accel", "Grind", "tsSPI", "PI"]].copy()
    for c in ["Accel", "Grind", "tsSPI", "PI"]:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
    dfm = dfm.dropna(subset=["Accel", "Grind", "tsSPI"])

    if dfm.empty:
        st.info("Not enough data to draw the shape map.")
    else:
        # deltas vs field (index points above/below 100)
        dfm["AccelΔ"] = dfm["Accel"] - 100.0
        dfm["GrindΔ"] = dfm["Grind"] - 100.0
        dfm["tsSPIΔ"] = dfm["tsSPI"] - 100.0

        names = dfm["Horse"].astype(str).to_list()
        xv = dfm["AccelΔ"].to_numpy()
        yv = dfm["GrindΔ"].to_numpy()
        cv = dfm["tsSPIΔ"].to_numpy()
        piv = dfm["PI"].fillna(0).to_numpy()

        # guards
        if not np.isfinite(xv).any() or not np.isfinite(yv).any():
            st.info("No valid sectional differentials available for this race.")
        else:
            # axis span (safe default if tiny)
            try:
                span = float(np.nanmax([np.nanmax(np.abs(xv)), np.nanmax(np.abs(yv))]))
            except Exception:
                span = 1.0
            if not np.isfinite(span) or span <= 0:
                span = 1.0
            lim = max(4.5, float(np.ceil(span / 1.5) * 1.5))

            # bubble sizes from PI
            DOT_MIN, DOT_MAX = 40.0, 140.0
            pmin, pmax = float(np.nanmin(piv)), float(np.nanmax(piv))
            if not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax - pmin) < 1e-9:
                sizes = np.full_like(xv, DOT_MIN)
            else:
                sizes = DOT_MIN + (piv - pmin) / (pmax - pmin) * (DOT_MAX - DOT_MIN)

            # figure
            fig, ax = plt.subplots(figsize=(7.6, 6.4))

            # quadrant tint
            TINT = 0.06
            ax.add_patch(Rectangle((0, 0),  lim,  lim, facecolor="#4daf4a", alpha=TINT, edgecolor="none"))   # +accel +grind
            ax.add_patch(Rectangle((-lim,0), lim,  lim, facecolor="#377eb8", alpha=TINT, edgecolor="none"))   # -accel +grind
            ax.add_patch(Rectangle((0,-lim), lim, lim, facecolor="#ff7f00", alpha=TINT, edgecolor="none"))    # +accel -grind
            ax.add_patch(Rectangle((-lim,-lim),lim, lim, facecolor="#984ea3", alpha=TINT, edgecolor="none"))  # -accel -grind

            # zero axes
            ax.axvline(0, color="gray", lw=1.3, ls=(0, (3, 3)))
            ax.axhline(0, color="gray", lw=1.3, ls=(0, (3, 3)))

            # colour map centered at 0 (tsSPI deviation)
            vmin = float(np.nanmin(cv)) if np.isfinite(cv).any() else -1.0
            vmax = float(np.nanmax(cv)) if np.isfinite(cv).any() else  1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = -1.0, 1.0
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

            sc = ax.scatter(
                xv, yv, s=sizes, c=cv, cmap="coolwarm", norm=norm,
                edgecolor="black", linewidth=0.6, alpha=0.95
            )

            # neat labels (adjustText if installed; else fallback)
            label_points_neatly(ax, xv, yv, names)

            # limits & labels
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("Acceleration vs field (points)  →")
            ax.set_ylabel("Grind vs field (points)  ↑")
            ax.set_title("Quadrants: +X = late acceleration; +Y = strong last 200. Colour = tsSPI deviation")

            # PI size legend
            s_ex = [DOT_MIN, 0.5*(DOT_MIN+DOT_MAX), DOT_MAX]
            h_ex = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                           markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
            ax.legend(h_ex, ["PI: low", "PI: mid", "PI: high"],
                      loc="upper left", frameon=False, fontsize=8)

            # colorbar
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("tsSPI − 100")

            ax.grid(True, linestyle=":", alpha=0.25)
            st.pyplot(fig)

            # download
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.download_button("Download shape map (PNG)", buf.getvalue(),
                               file_name="shape_map.png", mime="image/png")

            st.caption(
                "Each bubble is a runner. Size = PI (bigger = stronger overall). "
                "X: late acceleration (600→200) vs field; Y: last-200 grind vs field. "
                "Colour shows cruise strength (tsSPI vs field): red = faster mid-race, blue = slower."
            )
# =======================================================================================================

# ======================= Visual 2: Pace Curve — accurate segments & thin lines =================
st.markdown("## Pace Curve — field average (black) + Top 8 finishers")

# Collect all *_Time markers present
all_time_cols = [c for c in work.columns if str(c).endswith("_Time")]
markers = []
for c in all_time_cols:
    try:
        markers.append(int(str(c).split("_")[0]))
    except Exception:
        pass
markers = sorted(set(markers), reverse=True)  # e.g. [1600, 1400, 1200, ..., 200]

if len(markers) == 0:
    st.info("Not enough *_Time columns to draw the pace curve.")
else:
    # Build ordered segments as (start_m, end_m, length_m); last goes to 0
    segs = []
    for i, start in enumerate(markers):
        end = markers[i + 1] if (i + 1) < len(markers) else 0
        seg_len = float(start - end)
        if seg_len <= 0:
            continue
        segs.append((start, end, seg_len))  # left = early, right = late

    if len(segs) == 0:
        st.info("Could not infer segment lengths.")
    else:
        # speeds for field (m/s = length / time)
        seg_cols = [f"{start}_Time" for (start, end, seg_len) in segs]
        times_df = work[seg_cols].apply(pd.to_numeric, errors="coerce")
        speed_df = pd.DataFrame(index=work.index)
        for (start, end, seg_len) in segs:
            c = f"{start}_Time"
            speed_df[c] = seg_len / times_df[c]

        field_avg = speed_df.mean(axis=0).to_numpy()

        # choose top 8: finish pos if present, else PI
        if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
            top8 = metrics.sort_values("Finish_Pos").head(8)
        else:
            top8 = metrics.sort_values("PI", ascending=False).head(8)

        # X axis
        x_idx = list(range(len(segs)))
        x_labels = [f"{int(s)}–{int(e)}m" for (s, e, _) in segs]

        fig2, ax2 = plt.subplots(figsize=(8.6, 5.2))
        # Field average — thicker black
        ax2.plot(x_idx, field_avg, linewidth=2.2, color="black", label="Field average", marker=None)

        # Overlay top 8 — thin lines & small markers
        palette = color_cycle(len(top8))
        for i, (_, r) in enumerate(top8.iterrows()):
            # find raw times by name in original table if available
            if "Horse" in work.columns and "Horse" in metrics.columns:
                row0 = work[work["Horse"] == r.get("Horse")]
                row_times = row0.iloc[0] if not row0.empty else r
            else:
                row_times = r

            y_vals = []
            for (start, end, seg_len) in segs:
                c = f"{start}_Time"
                t = pd.to_numeric(row_times.get(c, np.nan), errors="coerce")
                y_vals.append(seg_len / t if pd.notna(t) and t > 0 else np.nan)

            ax2.plot(x_idx, y_vals, linewidth=1.1, marker="o", markersize=2.5,
                     label=str(r.get("Horse", "")), color=palette[i])

        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(x_labels, rotation=45, ha="right")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Pace over 200 m segments (left = early, right = home straight)")
        ax2.grid(True, linestyle="--", alpha=0.30)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
        st.pyplot(fig2)
        st.caption("Black = field average. Coloured lines = top 8 finishers using true segment lengths (start→end labels).")

# ======================= Visual 3: Top-8 PI — stacked contributions (sum matches PI) =========
st.markdown("## Top-8 PI — stacked contributions")

# NOTE: for the bar-slice calculation we need the *distance-aware* weights used in PI
# Recompute the race-level weights (using medians already calculated)
acc_med_for_bars = metrics["Accel"].median(skipna=True)
grd_med_for_bars = metrics["Grind"].median(skipna=True)
PI_W_BARS = pi_weights_distance_and_context(float(race_distance_input), acc_med_for_bars, grd_med_for_bars)

def parts_scaled_to_total(row, total_pi, weights, zero_floor=True):
    """Return component contributions rescaled so their sum equals total_pi."""
    raw = {
        "F200_idx": weights["F200_idx"] * (float(row.get("F200_idx", 100.0)) - 100.0),
        "tsSPI":    weights["tsSPI"]    * (float(row.get("tsSPI",    100.0)) - 100.0),
        "Accel":    weights["Accel"]    * (float(row.get("Accel",    100.0)) - 100.0),
        "Grind":    weights["Grind"]    * (float(row.get("Grind",    100.0)) - 100.0),
    }
    if zero_floor:
        raw = {k: max(0.0, v) for k, v in raw.items()}
    s = sum(raw.values())
    if not np.isfinite(total_pi) or total_pi <= 0 or not np.isfinite(s) or s <= 0:
        return {"F200_idx": 0.0, "tsSPI": 0.0, "Accel": 0.0, "Grind": 0.0}
    scale = float(total_pi) / float(s)
    return {k: v * scale for k, v in raw.items()}

top8_pi = metrics.sort_values(["PI","Finish_Pos"], ascending=[False, True]).head(8).copy()
if not top8_pi.empty:
    horses = []
    stacks = {"F200_idx": [], "tsSPI": [], "Accel": [], "Grind": []}
    totals = []
    is_winner = []

    for _, r in top8_pi.iterrows():
        total_pi = float(r.get("PI", 0.0))
        parts = parts_scaled_to_total(r, total_pi, PI_W_BARS, zero_floor=True)
        for k in stacks:
            stacks[k].append(parts[k])
        totals.append(total_pi)
        horses.append(str(r.get("Horse", "")))
        is_winner.append(int(r.get("Finish_Pos", 0)) == 1)

    fig3, ax3 = plt.subplots(figsize=(max(7.5, 0.95*len(horses)), 4.8))
    x = np.arange(len(horses))
    palette = {"F200_idx": "#6baed6", "tsSPI": "#9e9ac8", "Accel": "#74c476", "Grind": "#fd8d3c"}

    bottoms = np.zeros(len(horses))
    for key, label in [("F200_idx","F200"), ("tsSPI","tsSPI"), ("Accel","Accel"), ("Grind","Grind")]:
        vals = np.array(stacks[key], dtype=float)
        ax3.bar(x, vals, bottom=bottoms, label=label, color=palette[key], edgecolor="black", linewidth=0.4)
        bottoms += vals

    ymax = max(0.1, max(totals)*1.20)
    for i, tot in enumerate(totals):
        if is_winner[i]:
            ax3.add_patch(plt.Rectangle((i-0.5, 0), 1.0, max(tot, bottoms[i]), fill=False, lw=2.0, ec="#d4af37"))
            horses[i] = f"★ {horses[i]}"
        ax3.text(i, tot + ymax*0.03, f"{tot:.2f}", ha="center", va="bottom", fontsize=9)

    ax3.set_xticks(x); ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylim(0, ymax)
    ax3.set_ylabel("PI (stacked contributions)")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    st.pyplot(fig3)
    st.caption("Slices are rescaled to sum exactly to each horse’s PI. ★ = race winner.")
else:
    st.info("No PI values available to plot the stacked contributions.")

# ======================= Hidden Horses (v2) =================
st.markdown("## Hidden Horses (v2)")

# Work on a copy so we don't mutate `metrics`
hh = metrics.copy()

# ---------- 1) SOS = sectional outlier score (robust) ----------
# Winsorise each component to mute single outliers,
# then convert to robust z-scores using MAD.
need_cols = {"tsSPI", "Accel", "Grind"}
if need_cols.issubset(hh.columns) and len(hh) > 0:
    ts_w = winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
    ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
    gr_w = winsorize(pd.to_numeric(hh["Grind"], errors="coerce"))

    def rz(s: pd.Series) -> pd.Series:
        mu = np.nanmedian(s)
        sd = mad_std(s)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    z_ts = rz(ts_w).clip(-2.5, 3.5)
    z_ac = rz(ac_w).clip(-2.5, 3.5)
    z_gr = rz(gr_w).clip(-2.5, 3.5)

    hh["SOS_raw"] = 0.45 * z_ts + 0.35 * z_ac + 0.20 * z_gr

    # Normalize SOS into ~[0, 2] range using robust in-race 5–95% window
    q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom = (q95 - q5) if (pd.notna(q95) and pd.notna(q5) and (q95 > q5)) else 1.0
    hh["SOS"] = (2.0 * (hh["SOS_raw"] - q5) / denom).clip(lower=0.0, upper=2.0)
else:
    hh["SOS"] = 0.0

# ---------- 2) ASI² = against-shape magnitude (bias-aware) ----------
acc_med = pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
grd_med = pd.to_numeric(hh.get("Grind"), errors="coerce").median(skipna=True)
bias = (acc_med - 100.0) - (grd_med - 100.0)  # >0 → kick-leaning; <0 → grind-leaning
B = min(1.0, abs(bias) / 4.0)                  # scale by strength of bias (4 pts ≈ strong)
S = pd.to_numeric(hh.get("Accel"), errors="coerce") - pd.to_numeric(hh.get("Grind"), errors="coerce")
if bias >= 0:
    # race favoured kick → reward grinders
    hh["ASI2"] = (B * (-S).clip(lower=0.0) / 5.0).fillna(0.0)
else:
    # race favoured grind → reward kickers
    hh["ASI2"] = (B * (S).clip(lower=0.0) / 5.0).fillna(0.0)

# ---------- 3) TFS = trip friction (late variability vs mid pace) ----------
# Use last 3 segments available (e.g., 600, 400, 200) relative to finish.
last3 = []
if len(seg_markers) >= 3:
    last3 = sorted(seg_markers)[0:3]          # smallest markers (closest to finish)
    last3 = sorted(last3, reverse=True)       # ensure [600, 400, 200] style ordering

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

# Distance-aware TFS gate (more forgiving as distance increases)
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

# ---------- 4) UEI = underused engine indicators ----------
def uei_row(r):
    ts = pd.to_numeric(r.get("tsSPI"), errors="coerce")
    ac = pd.to_numeric(r.get("Accel"), errors="coerce")
    gr = pd.to_numeric(r.get("Grind"), errors="coerce")
    if pd.isna(ts) or pd.isna(ac) or pd.isna(gr):
        return 0.0
    val = 0.0
    # Sprint-home: strong cruise but little late show
    if ts >= 102 and ac <= 98 and gr <= 98:
        gap = min((ts - 102) / 3.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    # Fast-early: strong stay/grind but shape suppressed kick
    if ts >= 102 and gr >= 102 and ac <= 100:
        gap = min(((ts - 102) + (gr - 102)) / 6.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    return round(val, 3)

hh["UEI"] = hh.apply(uei_row, axis=1)

# ---------- 5) HiddenScore v2 (0..3) ----------
# Base weighted blend
hidden = (
    0.55 * pd.to_numeric(hh["SOS"], errors="coerce").fillna(0.0) +
    0.30 * pd.to_numeric(hh["ASI2"], errors="coerce").fillna(0.0) +
    0.10 * pd.to_numeric(hh["TFS_plus"], errors="coerce").fillna(0.0) +
    0.05 * pd.to_numeric(hh["UEI"], errors="coerce").fillna(0.0)
)

# Field-size guard (very small races → slightly softer)
if int(hh.shape[0]) <= 6:
    hidden = hidden * 0.90

# Robust race-by-race normalization to keep tiers consistent
h_med = float(np.nanmedian(hidden))
h_mad = float(np.nanmedian(np.abs(hidden - h_med)))
h_sigma = max(1e-6, 1.4826 * h_mad)                # robust sigma

# Vectorised scaling + clamping (fixes the previous clamp(Series) error)
hh["HiddenScore"] = (1.2 + (hidden - h_med) / (2.5 * h_sigma)).clip(lower=0.0, upper=3.0)

# ---------- 6) Tiering & Notes ----------
def hh_tier(s):
    if pd.isna(s): return ""
    if s >= 1.8:   return "🔥 Top Hidden"
    if s >= 1.2:   return "🟡 Notable Hidden"
    return ""

hh["Tier"] = hh["HiddenScore"].apply(hh_tier)

def hh_note(r):
    bits = []
    if r.get("Tier", "") != "":
        if pd.to_numeric(r.get("SOS"), errors="coerce") >= 1.2:
            bits.append("sectionals superior")
        asi2 = pd.to_numeric(r.get("ASI2"), errors="coerce")
        if asi2 >= 0.8:
            bits.append("ran against strong bias")
        elif asi2 >= 0.4:
            bits.append("ran against bias")
        if pd.to_numeric(r.get("TFS_plus"), errors="coerce") > 0:
            bits.append("trip friction late")
        if pd.to_numeric(r.get("UEI"), errors="coerce") >= 0.5:
            bits.append("latent potential if shape flips")
    return ("; ".join(bits).capitalize() + ".") if bits else ""

hh["Note"] = hh.apply(hh_note, axis=1)

# ---------- 7) Display ----------
cols_hh = [
    "Horse", "Finish_Pos", "PI", "GCI",
    "tsSPI", "Accel", "Grind",
    "SOS", "ASI2", "TFS", "UEI",
    "HiddenScore", "Tier", "Note"
]
for c in cols_hh:
    if c not in hh.columns:
        hh[c] = np.nan

st.dataframe(
    hh.sort_values(["Tier", "HiddenScore", "PI"], ascending=[True, False, False])[cols_hh],
    use_container_width=True
)

st.caption(
    "Hidden Horses v2: SOS = sectional outlier (robust), ASI² = against-shape magnitude (bias-aware), "
    "TFS = trip friction, UEI = underused engine. Tiering: 🔥 ≥ 1.8, 🟡 ≥ 1.2."
)
# ======================= Footer ===========================
st.caption(
    "Definitions — F200_idx: first 200 m vs field (100 = par). "
    "tsSPI: sustained mid-race pace (excl. first 200 & last 600; adaptive). "
    "Accel: 600→200 window (adaptive); Grind: last 200. "
    "PI v3.1: blended sectional performance with distance- & context-aware weights; small-field stabilized. "
    "GCI: distance-aware class index (0–10)."
)
