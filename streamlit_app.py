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
Image.MAX_IMAGE_PIXELS = 500_000_000  # belt-and-braces

# =============================================================================
# Streamlit page (no logo/download per your baseline)
# =============================================================================
st.set_page_config(page_title="RaceEdge — PI v2.4-B++", layout="wide")

with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# =============================================================================
# Safe pyplot (prevents huge canvas & decompression warnings)
# =============================================================================
def safe_pyplot(fig, max_w_px=1000, max_h_px=800, dpi=96):
    fig.set_dpi(dpi)
    w_in, h_in = fig.get_size_inches()
    w_px, h_px = max(w_in * dpi, 1), max(h_in * dpi, 1)
    scale = min(max_w_px / w_px, max_h_px / h_px, 1.0)
    if scale < 1.0:
        fig.set_size_inches(w_in * scale, h_in * scale, forward=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, caption=None, width=max_w_px)

# =============================================================================
# Helpers
# =============================================================================
def parse_race_time(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    try:
        return float(s)
    except Exception:
        pass
    parts = s.split(":")
    try:
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
    if pd.isna(x): return np.nan
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def clamp(x, lo=0.0, hi=1.0):
    try:
        return float(min(hi, max(lo, x)))
    except Exception:
        return np.nan

def _safe_median(s, default=np.nan):
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanmedian(s)) if s.notna().any() else default

def _mad(arr):
    x = pd.to_numeric(arr, errors="coerce").dropna().values
    if x.size == 0: return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return mad

# =============================================================================
# Distance profiles (baseline weights from your v2.3G)
# =============================================================================
def trip_profile(distance_m: int):
    d = int(distance_m)
    if d <= 1200:
        return dict(bucket="SPRINT",
                    wF200=0.08, wSPI=0.30, wACC=0.20, wGR=0.42,
                    band={"F200%":(86,96), "tsSPI%":(100,109), "Accel%":(97,105), "Grind%":(95,104)})
    if d <= 1500:
        return dict(bucket="SEVEN",
                    wF200=0.09, wSPI=0.28, wACC=0.27, wGR=0.36,
                    band={"F200%":(86,96), "tsSPI%":(99.5,107.5), "Accel%":(97,104), "Grind%":(96,104)})
    if d <= 1700:
        return dict(bucket="MILE",
                    wF200=0.08, wSPI=0.27, wACC=0.25, wGR=0.40,
                    band={"F200%":(86,95), "tsSPI%":(99.5,106.5), "Accel%":(99,105), "Grind%":(97,104)})
    if d <= 2050:
        return dict(bucket="MIDDLE",
                    wF200=0.06, wSPI=0.25, wACC=0.22, wGR=0.47,
                    band={"F200%":(85,94), "tsSPI%":(99,105.5), "Accel%":(99,104), "Grind%":(97,103.5)})
    return dict(bucket="CLASSIC",
                wF200=0.05, wSPI=0.23, wACC=0.20, wGR=0.52,
                band={"F200%":(84,94), "tsSPI%":(98.5,104.5), "Accel%":(98,103), "Grind%":(97,103)})

# =============================================================================
# Manual template (200 m grid + Finish 100 m)
# =============================================================================
def make_manual_template(horses: int, distance_m: int) -> tuple[pd.DataFrame, int]:
    if distance_m % 200 != 0:
        distance_m = int(np.ceil(distance_m / 200.0) * 200)
    cols = ["Horse"]
    for m in range(distance_m - 200, 0, -200):
        cols.append(f"{m}_Time")
    cols.append("Finish_Time")  # final 100 m
    cols.append("Finish_Pos")
    df = pd.DataFrame({c: [np.nan]*horses for c in cols})
    df["Horse"] = [f"Runner {i+1}" for i in range(horses)]
    return df, distance_m

# =============================================================================
# 100 m column lister
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
# Build new-era metrics (F200%, tsSPI%, Accel%, Grind%)
# =============================================================================
def build_metrics(df_in: pd.DataFrame, distance_m: int, manual_mode: bool) -> pd.DataFrame:
    work = df_in.copy()

    # Race time normalization
    if "Race Time" in work.columns:
        work["RaceTime_s"] = work["Race Time"].apply(parse_race_time)
    else:
        rt_col = next((c for c in ["Race_Time", "RaceTime", "Time"] if c in work.columns), None)
        work["RaceTime_s"] = work[rt_col].apply(parse_race_time) if rt_col else np.nan

    # Fallback from sectionals
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

    # Phase definitions
    if not manual_mode:
        # F200: (D-100 + D-200)
        f200_parts = []
        for m in [distance_m - 100, distance_m - 200]:
            nm = f"{m}_Time"
            if nm in work.columns:
                f200_parts.append(nm)
        work["F200_time"] = work[f200_parts].sum(axis=1, skipna=False) if f200_parts else np.nan

        # Accel: 200 + 100
        a_cols = []
        if "200_Time" in work.columns: a_cols.append("200_Time")
        if "100_Time" in work.columns: a_cols.append("100_Time")
        work["Accel_time"] = work[a_cols].sum(axis=1, skipna=False) if a_cols else np.nan

        # Grind: Finish 100
        work["Grind_time"] = work["Finish_Time"] if "Finish_Time" in work.columns else np.nan

        # tsSPI mid: exclude first 200 and last 400 (so 300..(D-300))
        mid_cols = []
        for m in range(distance_m - 300, 300, -100):
            nm = f"{m}_Time"
            if nm in work.columns:
                mid_cols.append(nm)
        work["Mid_time"] = work[mid_cols].sum(axis=1, skipna=True)
        work["Mid_dist"] = (work[mid_cols].notna().sum(axis=1).astype(float)) * 100.0
    else:
        # Manual (200 m bins + Finish 100)
        f200_col = f"{distance_m - 200}_Time"
        work["F200_time"] = work[f200_col] if f200_col in work.columns else np.nan

        if "200_Time" in work.columns and "100_Time" in work.columns:
            work["Accel_time"] = work["200_Time"] + work["100_Time"]
        elif "200_Time" in work.columns and "Finish_Time" in work.columns:
            work["Accel_time"] = (work["200_Time"] / 2.0) + work["Finish_Time"]
        else:
            work["Accel_time"] = np.nan

        work["Grind_time"] = work["Finish_Time"] if "Finish_Time" in work.columns else np.nan

        mid_cols = []
        for m in range(distance_m - 400, 400, -200):  # D-400..600 step 200
            nm = f"{m}_Time"
            if nm in work.columns:
                mid_cols.append(nm)
        work["Mid_time"] = work[mid_cols].sum(axis=1, skipna=True)
        work["Mid_dist"] = (work[mid_cols].notna().sum(axis=1).astype(float)) * 200.0

    # Speeds
    work["F200_speed"]  = 200.0 / work["F200_time"]
    work["Mid_speed"]   = np.where(work["Mid_time"] > 0, work["Mid_dist"] / work["Mid_time"], np.nan)
    work["Accel_speed"] = 200.0 / work["Accel_time"]
    work["Grind_speed"] = 100.0 / work["Grind_time"]

    # Guarded ratios
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
# Context indices for PI v2.4-B++
# =============================================================================
def context_indices(df):
    # field medians
    f200_med = _safe_median(df["F200%"], default=90.0)
    grind_med = _safe_median(df["Grind%"], default=98.0)

    # Early-Heat (positive = fast early). Use (median(F200) - 92)/4 clipped.
    EHI = (f200_med - 92.0) / 4.0
    EHI = float(np.clip(EHI, -2.0, 2.0))

    # Sprint-Home index (higher = more late slowdown / sprint-home)
    SHI = (100.0 - grind_med) / 2.5
    SHI = float(np.clip(SHI, 0.0, 2.0))

    return EHI, SHI

def renorm_weights(w):
    s = sum(w.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in w.items()}

# =============================================================================
# PI v2.4-B++ (Balanced++): context weighting + graded TBB + penalties
# =============================================================================
def compute_pi_v24_bpp(df: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    prof = trip_profile(distance_m)

    # Build scoring blends (same approach as v2.3G)
    # s_metric = 0.7 * percentile + 0.3 * band-fit
    s_cols = {}
    for k, (lo, hi) in prof["band"].items():
        pr = percent_rank(df[k])
        af = df[k].apply(lambda x: norm_to_band(x, lo, hi))
        s = 0.7 * pr + 0.3 * af
        s_cols[f"s_{k}"] = s
    for c, s in s_cols.items():
        df[c] = s

    # Context-aware reweighting
    w = dict(wF200=prof["wF200"], wSPI=prof["wSPI"], wACC=prof["wACC"], wGR=prof["wGR"])
    EHI, SHI = context_indices(df)

    w["wACC"] = w["wACC"] * (1 + 0.10 * max(0.0, -EHI) + 0.05 * SHI)
    w["wGR"]  = w["wGR"]  * (1 + 0.10 * max(0.0,  EHI) + 0.05 * (2 - SHI))
    w["wSPI"] = w["wSPI"] * (1 + 0.05 * abs(EHI))
    # wF200 unchanged (optionally: w["wF200"] *= (1 + 0.05*EHI) for burn-ups)

    w = renorm_weights(w)
    df["_wF200"] = w["wF200"]; df["_wSPI"] = w["wSPI"]; df["_wACC"] = w["wACC"]; df["_wGR"] = w["wGR"]

    # Base with context weights
    df["PI_base_ctx"] = (
        df["s_F200%"]  * w["wF200"] +
        df["s_tsSPI%"] * w["wSPI"]  +
        df["s_Accel%"] * w["wACC"]  +
        df["s_Grind%"] * w["wGR"]
    )

    # Robust medians & MAD
    metrics = ["F200%", "tsSPI%", "Accel%", "Grind%"]
    med = {m: _safe_median(df[m]) for m in metrics}
    mad = {m: _mad(df[m]) for m in metrics}
    # Guards for zero MAD
    eff_mad = {m: (mad[m] if (not pd.isna(mad[m]) and mad[m] > 1e-6) else np.nan) for m in metrics}

    # Below-median penalty (BMP)
    # Sprint bucket lambdas per your spec; adjust by distance
    bucket = prof["band"]
    if distance_m <= 1200:
        lam = dict(F200=0.010, tsSPI=0.010, Accel=0.020, Grind=0.015)
    elif distance_m <= 1700:
        lam = dict(F200=0.008, tsSPI=0.012, Accel=0.015, Grind=0.018)
    else:
        lam = dict(F200=0.006, tsSPI=0.012, Accel=0.010, Grind=0.022)

    def _bmp_row(r):
        k = 1.5
        parts = []
        for M, L in [("F200%", "F200"), ("tsSPI%", "tsSPI"), ("Accel%", "Accel"), ("Grind%", "Grind")]:
            if pd.isna(eff_mad[M]) or pd.isna(r[M]): 
                parts.append(0.0); continue
            z = max(0.0, (med[M] - r[M]) / (1.4826 * eff_mad[M]))
            p = lam[L] * (z / (z + k))
            parts.append(p)
        val = sum(parts)
        return min(val, 0.05)

    df["BMP"] = df.apply(_bmp_row, axis=1)

    # Imbalance surcharge (IS) when both Accel & Grind well below
    if distance_m <= 1200: mu, k2 = 0.015, 2.0
    elif distance_m <= 1700: mu, k2 = 0.012, 2.0
    else: mu, k2 = 0.010, 2.0

    def _robust_z(val, M):
        if pd.isna(eff_mad[M]) or pd.isna(val): return 0.0
        return max(0.0, (med[M] - val) / (1.4826 * eff_mad[M]))

    def _is_row(r):
        zA = _robust_z(r["Accel%"], "Accel%")
        zG = _robust_z(r["Grind%"], "Grind%")
        if (zA >= 0.8) and (zG >= 0.8):
            t = zA + zG
            return mu * (t / (t + k2))
        return 0.0

    df["IS"] = df.apply(_is_row, axis=1)

    # Spread penalty (max-min across Accel, Grind, tsSPI with tolerance)
    if distance_m <= 1200: gamma = 0.015
    elif distance_m <= 1700: gamma = 0.012
    else: gamma = 0.010
    tau, k_sp = 6.0, 4.0

    a_med, g_med = med["Accel%"], med["Grind%"]

    def _spread_row(r):
        trio = [r["Accel%"], r["Grind%"], r["tsSPI%"]]
        if any(pd.isna(x) for x in trio):
            return 0.0
        spread = max(trio) - min(trio)
        raw = 0.0
        if spread > tau:
            raw = gamma * ((spread - tau) / ((spread - tau) + k_sp))
        # Archetype half-penalty rule
        half = False
        if (pd.notna(r["Accel%"]) and pd.notna(r["Grind%"])):
            if (r["Accel%"] >= a_med + 2.0 and r["Grind%"] >= g_med - 1.0) or \
               (r["Grind%"] >= g_med + 2.0 and r["Accel%"] >= a_med - 1.0):
                half = True
        return raw * (0.5 if half else 1.0)

    df["Spread"] = df.apply(_spread_row, axis=1)

    # Triple Balance Bonus (graded): clear all three medians (Accel, Grind, tsSPI)
    def _tbb_row(r):
        vals = {"Accel%": r["Accel%"], "Grind%": r["Grind%"], "tsSPI%": r["tsSPI%"]}
        if any(pd.isna(vals[k]) for k in vals): return 0.0
        if (vals["Accel%"] >= a_med) and (vals["Grind%"] >= g_med) and (vals["tsSPI%"] >= med["tsSPI%"]):
            worst_surplus = min(vals["Accel%"] - a_med, vals["Grind%"] - g_med, vals["tsSPI%"] - med["tsSPI%"])
            extra = max(0.0, min(0.010, (worst_surplus / 2.0) * 0.010))  # up to +0.010 when worst ≥ +2pp
            return 0.010 + extra
        return 0.0

    df["TBB"] = df.apply(_tbb_row, axis=1)

    # Winner protection with field compression
    winner_time = df.loc[df["Finish_Pos"] == df["Finish_Pos"].min(), "RaceTime_s"].min()
    times = pd.to_numeric(df["RaceTime_s"], errors="coerce")
    comp = float(times.std(skipna=True) / times.mean(skipna=True)) if times.notna().any() else 0.05
    comp_clip = float(np.clip(comp, 0.02, 0.08))
    win_cap = 0.03 * (0.8 + 0.2 / comp_clip)  # ≈ 0.03 in tight fields, slightly less if strung out

    # Small-field damping on penalties
    N = len(df)
    pen_scale = min(1.0, max(0.0, (N - 5) / 7.0))  # 0 at N<=5 → 1 by N>=12

    df["BMP_s"] = df["BMP"] * pen_scale
    df["IS_s"]  = df["IS"]  * pen_scale
    df["Spread_s"] = df["Spread"] * pen_scale

    # Compose final PI
    df["Adj_ctx0"] = df["PI_base_ctx"]  # for delta
    # Existing context bonuses from v2.3G you liked (keep as light hooks)
    # We'll keep EndBonus/DomBonus/WinBonus shape as zeros (or plug if you have)
    df["EndBonus"] = 0.0
    df["DomBonus"] = 0.0
    df["WinBonus"] = 0.0

    # Winner soft-landing: cap total negative correction for decisive winners
    def _winner_soft_cap(r, net_corr):
        if pd.isna(r["RaceTime_s"]): return net_corr
        margin_L = (r["RaceTime_s"] - winner_time) / 0.20
        if r["Finish_Pos"] == 1 and margin_L >= 3.0:
            return max(net_corr, -win_cap)
        return net_corr

    # Net corrections (clamped ±0.06)
    df["Corr_raw"] = -df["BMP_s"] - df["IS_s"] - df["Spread_s"] + df["TBB"]
    df["Corr_capped"] = df["Corr_raw"].apply(lambda x: float(np.clip(x, -0.06, 0.06)))
    df["Corr_final"] = df.apply(lambda r: _winner_soft_cap(r, r["Corr_capped"]), axis=1)

    df["PI_v2_4Bpp"] = (df["PI_base_ctx"] + df["EndBonus"] + df["DomBonus"] + df["WinBonus"] + df["Corr_final"]).clip(0, 1)

    # Context move tag
    df["PI_base_plain"] = (
        df["s_F200%"] * prof["wF200"] +
        df["s_tsSPI%"] * prof["wSPI"] +
        df["s_Accel%"] * prof["wACC"] +
        df["s_Grind%"] * prof["wGR"]
    )
    df["CTX_delta"] = df["PI_base_ctx"] - df["PI_base_plain"]

    # Tags
    def _tags(r):
        t = []
        if r["CTX_delta"] >= 0.007: t.append("[CTX+]")
        if r["CTX_delta"] <= -0.007: t.append("[CTX-]")
        if r["TBB"] >= 0.012: t.append("[BAL+]")
        if r["Spread_s"] >= 0.008: t.append("[LOPS]")
        # Winner protection fired if Corr_final > Corr_capped (i.e., soft cap raised result)
        if r["Corr_final"] > r["Corr_capped"] + 1e-9: t.append("[SAFE]")
        return " ".join(t) if t else ""

    df["Notes_Tag"] = df.apply(_tags, axis=1)

    # Narrative line (short)
    def _note_text(r):
        shape = "fast-early" if EHI > 0.5 else ("slow-early / sprint-home" if EHI < -0.5 else "even tempo")
        bits = [f"Race shape: {shape}; weights favoured " + ("Accel" if (EHI < -0.5) else ("Grind" if (EHI > 0.5) else "balance")) + "."]
        prof_str = []
        if pd.notna(r["Accel%"]) and pd.notna(med["Accel%"]):
            if r["Accel%"] >= med["Accel%"] + 2: prof_str.append("good kick")
            elif r["Accel%"] <= med["Accel%"] - 2: prof_str.append("limited kick")
        if pd.notna(r["Grind%"]) and pd.notna(med["Grind%"]):
            if r["Grind%"] >= med["Grind%"] + 2: prof_str.append("strong late")
            elif r["Grind%"] <= med["Grind%"] - 2: prof_str.append("weak late")
        if pd.notna(r["tsSPI%"]) and pd.notna(med["tsSPI%"]):
            if r["tsSPI%"] >= med["tsSPI%"] + 2: prof_str.append("solid mid engine")
        if r["Notes_Tag"]:
            prof_str.append(r["Notes_Tag"].replace("[","").replace("]",""))
        if prof_str:
            bits.append("Profile: " + ", ".join(prof_str) + ".")
        return " ".join(bits)

    df["Notes_Text"] = df.apply(_note_text, axis=1)

    return df

# =============================================================================
# Pace curve (200 m) + helpers
# =============================================================================
def _list_100m_pairs(df, distance_m):
    cols = _list_100m_cols(df, distance_m)  # D-100, D-200, ..., 100, Finish
    pairs = []
    for i in range(0, len(cols) - 1, 2):
        pairs.append((cols[i], cols[i+1]))
    return pairs

def compute_field_pace_200m(df: pd.DataFrame, distance_m: int, manual_mode: bool):
    labels, seg_times = [], []

    if not manual_mode:
        pairs = _list_100m_pairs(df, distance_m)
        for k in range(distance_m - 200, -1, -200):
            labels.append("0" if k == 0 else f"{k}_to_{k+200}")
        for (a, b) in pairs:
            t = pd.to_numeric(df[a], errors="coerce") + pd.to_numeric(df[b], errors="coerce")
            seg_times.append(t.values)
    else:
        mcols = [c for c in df.columns if c.endswith("_Time") and c != "Finish_Time"]
        def _dist(c): return int(c.split("_")[0])
        mcols = sorted(mcols, key=_dist, reverse=True)
        for c in mcols:
            k = int(c.split("_")[0])
            labels.append("0" if k == 200 else f"{k-200}_to_{k}")
            seg_times.append(pd.to_numeric(df[c], errors="coerce").values)

    avg_speeds = []
    for t in seg_times:
        s = pd.Series(t).dropna()
        avg_speeds.append(200.0 / s.mean() if len(s) and s.mean() > 0 else np.nan)

    n = min(len(labels), len(avg_speeds))
    return labels[:n], avg_speeds[:n]

def ema(series, alpha=0.3):
    out, prev = [], None
    for v in series:
        if np.isnan(v):
            out.append(np.nan); continue
        prev = v if (prev is None or np.isnan(prev)) else (alpha * v + (1 - alpha) * prev)
        out.append(prev)
    return out

# =============================================================================
# Charts
# =============================================================================
def chart_pi_bar(df):
    ranked = df.sort_values("PI_v2_4Bpp", ascending=True)
    names = ranked["Horse"].astype(str).tolist()
    vals  = ranked["PI_v2_4Bpp"].astype(float).tolist()
    tags  = ranked["Notes_Tag"].astype(str).tolist()

    max_h_inches = 10.0
    fig_h = min(max(4, 0.35 * len(names)), max_h_inches)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(names, vals)
    ax.set_xlabel("PI v2.4-B++ (0–1)")
    ax.set_title("PI Ranking")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    for i, g in enumerate(tags):
        if g:
            ax.text(0.01, i, g, va="center", fontsize=8)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    safe_pyplot(fig)

def chart_shape_map(df, label_mode="Top 8"):
    fig, ax = plt.subplots(figsize=(9, 7))
    x = pd.to_numeric(df["Accel%"], errors="coerce")
    y = pd.to_numeric(df["Grind%"], errors="coerce")
    pi = pd.to_numeric(df["PI_v2_4Bpp"], errors="coerce")
    size = (pd.to_numeric(df["tsSPI%"], errors="coerce") - pd.to_numeric(df["tsSPI%"], errors="coerce").min() + 1.0) * 18.0

    x_med, y_med = _safe_median(x), _safe_median(y)
    ax.axhline(y_med, linestyle="--", linewidth=1.2, color="grey", alpha=0.8)
    ax.axvline(x_med, linestyle="--", linewidth=1.2, color="grey", alpha=0.8)
    ax.axvspan(x_med, max(np.nanmax(x), x_med), ymin=0, ymax=0.5, alpha=0.08, color="C0")
    ax.axvspan(min(np.nanmin(x), x_med), x_med, ymin=0.5, ymax=1, alpha=0.08, color="C2")

    sc = ax.scatter(x, y, s=size, c=pi, cmap="viridis", alpha=0.85, edgecolors="white", linewidths=0.8)

    # label subset
    if label_mode == "None":
        label_idx = []
    elif label_mode == "All":
        label_idx = df.index.tolist()
    elif label_mode == "Top 5":
        label_idx = df["PI_v2_4Bpp"].nlargest(min(5, len(df))).index.tolist()
    else:
        label_idx = df["PI_v2_4Bpp"].nlargest(min(8, len(df))).index.tolist()

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
    cb.set_label("PI v2.4-B++")
    ax.set_xlabel("Accel% (200→100 vs Mid)")
    ax.set_ylabel("Grind% (Final 100 vs Avg)")
    ax.set_title("Sectional Shape Map — Accel vs Grind (bubble = tsSPI, color = PI)")
    safe_pyplot(fig)

def chart_pace_curve_200m(df: pd.DataFrame, distance_m: int, manual_mode: bool,
                          overlays="Top 5 finishers", smooth=False, normalize=False):
    labels, speeds = compute_field_pace_200m(df, distance_m, manual_mode)
    x = list(range(len(labels)))[::-1]
    y = list(reversed(speeds))
    xlabels = list(reversed(labels))
    y_plot = ema(y) if smooth else y

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y_plot, marker="o", linewidth=2.5, label="Field avg (200m)", zorder=5, color="black")

    overlay_n = {"None": 0, "Top 3 finishers": 3, "Top 5 finishers": 5, "Top 8 finishers": 8}.get(overlays, 5)
    if overlay_n > 0:
        top_finish = df.sort_values("Finish_Pos").head(overlay_n)
        palette = ["#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB","#000000"]

        def horse_bin_times(row):
            if not manual_mode:
                pairs = _list_100m_pairs(df, distance_m)
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
st.title("🏇 RaceEdge — Performance Index v2.4-B++")

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

# Data ingestion
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

# Compute metrics + PI
try:
    work = build_metrics(df_raw, int(distance_m), manual_mode)
    work = compute_pi_v24_bpp(work, int(distance_m))
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# Display metrics
show_cols = [
    "Horse","Finish_Pos","RaceTime_s","Margin_L",
    "F200%","tsSPI%","Accel%","Grind%",
    "PI_base_ctx","BMP_s","IS_s","Spread_s","TBB","CTX_delta","PI_v2_4Bpp","Notes_Tag","Notes_Text"
]
disp = work.copy()
for c in ["RaceTime_s","Margin_L","F200%","tsSPI%","Accel%","Grind%",
          "PI_base_ctx","BMP_s","IS_s","Spread_s","TBB","CTX_delta","PI_v2_4Bpp"]:
    if c in disp.columns:
        disp[c] = pd.to_numeric(disp[c], errors="coerce")

st.subheader("Sectional Metrics (new system + PI v2.4-B++)")
st.dataframe(disp[show_cols].sort_values(["PI_v2_4Bpp"], ascending=False), width="stretch")

# Charts
st.subheader("PI Ranking")
chart_pi_bar(work)

st.subheader("Sectional Shape Map")
chart_shape_map(work, label_mode=label_mode)

st.subheader("Race Pace Curve (200 m bins)")
chart_pace_curve_200m(work, int(distance_m), manual_mode,
                      overlays=overlays, smooth=smooth, normalize=normalize)

st.caption(
    "Definitions: F200% = first 200 m vs race avg; tsSPI% = sustained mid-race pace (excl. first 200 m & last 400 m); "
    "Accel% = 200→100 vs mid; Grind% = final 100 vs race avg. "
    "PI v2.4-B++ adds context-aware weights (EHI/SHI), graded triple-balance bonus, and balanced penalties (below-median, imbalance, spread) "
    "with winner protection, small-field damping, and zero-MAD guards."
)
