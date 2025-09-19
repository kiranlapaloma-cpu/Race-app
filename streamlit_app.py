# streamlit_app.py
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================================
# Branding / Page
# ================================
APP_DIR = Path(__file__).resolve().parent
CANDIDATE_LOGO_PATHS = [
    APP_DIR / "assets" / "logos.png",
    APP_DIR / "assets" / "logo.png",
    APP_DIR / "logos.png",
    APP_DIR / "logo.png",
]
LOGO_PATH = next((p for p in CANDIDATE_LOGO_PATHS if p.exists()), None)

st.set_page_config(
    page_title="Race Edge — v3.3 (KickQ Sleepers)",
    page_icon=str(LOGO_PATH) if LOGO_PATH else None,
    layout="wide",
)

if LOGO_PATH:
    st.image(str(LOGO_PATH), width=220)

st.title("🏇 Race Edge — v3.3")
st.caption(
    "GCI v3.3 (EPI-free, sectional). Sleepers use KickQ (quantile-aware) with balanced gates/tiers, "
    "heat bonus, field-size guards, and a safety-net."
)

# -------------------
# Debug toggle
# -------------------
with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# ================================
# Helpers
# ================================
def parse_race_time(val):
    """Parse Race Time seconds or 'MM:SS.ms'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # already seconds?
    try:
        return float(s)
    except Exception:
        pass
    parts = s.split(":")
    try:
        if len(parts) == 3:
            m, sec, ms = parts
            return int(m) * 60 + int(sec) + int(ms) / 1000.0
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except Exception:
        return np.nan
    return np.nan

def _norm_header(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(",", "").replace(" ", "")
    return s

def _has_200m_headers(df: pd.DataFrame) -> bool:
    norm_cols = {_norm_header(c) for c in df.columns}
    keys = {"1000m", "800m", "600m", "400m", "200m"}
    return len(keys & norm_cols) > 0

# ====== 200m→app converter (handles fused text) ======
TIME_RE = re.compile(r"(\d+:\d{2}\.\d{1,3}|\d+\.\d{1,3})")

def _last_float(cell):
    if pd.isna(cell): return np.nan
    m = re.findall(r"\d+\.\d{1,3}", str(cell))
    return float(m[-1]) if m else np.nan

def _first_text(cell):
    if pd.isna(cell): return ""
    return str(cell).splitlines()[0].strip()

def _first_int_and_time(cell):
    if pd.isna(cell):
        return np.nan, np.nan
    s = str(cell).strip()
    if s.upper() == "NR":
        return np.nan, np.nan
    m = re.match(r"^(\d{1,2})(\d+\.\d{1,3})$", s)
    if m:
        return int(m.group(1)), float(m.group(2))
    m_time = TIME_RE.findall(s)
    m_pos  = re.findall(r"\b\d+\b", s)
    pos = int(m_pos[0]) if m_pos else np.nan
    if m_time:
        t = m_time[0]
        if ":" in t:
            mm, rest = t.split(":")
            return pos, int(mm) * 60 + float(rest)
        return pos, float(t)
    return pos, np.nan

def convert_200m_table_to_app(df):
    cols = {str(c).lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            n = n.lower()
            if n in cols:
                return cols[n]
        return None
    col_horse  = pick("horse", "runner", "name")
    col_result = pick("result", "finish", "time")
    col_800    = pick("800m")
    col_600    = pick("600m")
    col_400    = pick("400m")
    col_200    = pick("200m")

    out = pd.DataFrame()
    out["Horse"] = (df[col_horse].apply(_first_text) if col_horse else df.iloc[:, 0].apply(_first_text))

    if col_result:
        ft = df[col_result].apply(_first_int_and_time)
        out["Finish_Pos"] = ft.apply(lambda x: x[0])
        out["Race Time"]  = ft.apply(lambda x: x[1])
    else:
        out["Finish_Pos"] = np.nan
        out["Race Time"]  = np.nan

    seg_800 = df[col_800].apply(_last_float) if col_800 else np.nan
    seg_600 = df[col_600].apply(_last_float) if col_600 else np.nan
    seg_400 = df[col_400].apply(_last_float) if col_400 else np.nan
    seg_200 = df[col_200].apply(_last_float) if col_200 else np.nan

    out["800-400"]    = seg_800 + seg_600
    out["400-Finish"] = seg_400 + seg_200

    for c in ["Finish_Pos", "Race Time", "800-400", "400-Finish"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ====== Distance buckets / profiles ======
def _distance_bucket(distance_m: float) -> str:
    d = float(distance_m)
    if d <= 1400: return "SPRINT"
    if d < 1800:  return "MILE"
    if d < 2200:  return "MIDDLE"
    return "STAY"

def distance_profile(distance_m: float) -> dict:
    b = _distance_bucket(distance_m)
    if b == "SPRINT":
        return dict(bucket=b, wT=0.22, wPACE=0.38, wSS=0.20, wEFF=0.20,
                    ss_lo=99.0, ss_hi=105.0, lq_floor=0.35)
    if b == "STAY":
        return dict(bucket=b, wT=0.30, wPACE=0.28, wSS=0.30, wEFF=0.12,
                    ss_lo=97.5, ss_hi=104.5, lq_floor=0.25)
    if b == "MIDDLE":
        return dict(bucket=b, wT=0.27, wPACE=0.32, wSS=0.28, wEFF=0.13,
                    ss_lo=98.0, ss_hi=105.0, lq_floor=0.30)
    return dict(bucket="MILE", wT=0.25, wPACE=0.35, wSS=0.25, wEFF=0.15,
                ss_lo=98.0, ss_hi=105.0, lq_floor=0.30)

def clip01(x):
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def map01(x, lo, hi):
    if pd.isna(x) or hi == lo:
        return 0.0
    return clip01((float(x) - lo) / (hi - lo))

# ====== Core metrics ======
def compute_metrics(df, distance_m=1400.0):
    out = df.copy()
    out["Race_AvgSpeed"]  = distance_m / out["RaceTime_s"]
    out["Mid400_Speed"]   = 400.0 / out["800-400"]
    out["Final400_Speed"] = 400.0 / out["400-Finish"]

    out["Basic_FSP_%"]   = (out["Final400_Speed"] / out["Race_AvgSpeed"]) * 100.0
    out["Refined_FSP_%"] = (out["Final400_Speed"] / out["Mid400_Speed"]) * 100.0
    out["SPI_%"]         = (out["Mid400_Speed"]   / out["Race_AvgSpeed"]) * 100.0
    return out

def compute_pressure_context(metrics_df):
    spi_med = metrics_df["SPI_%"].median()
    early_heat = 0.0 if pd.isna(spi_med) else max(0.0, min(1.0, (spi_med - 100.0) / 3.0))  # ~103 → 1.0
    mid_pct = metrics_df["Mid400_Speed"].rank(pct=True, method="average")
    pressure_ratio_prime = float((mid_pct >= 0.65).mean()) if len(mid_pct) else 0.0
    return dict(spi_median=spi_med, early_heat=early_heat, pressure_ratio_prime=pressure_ratio_prime,
                field_size=int(metrics_df.shape[0]))

# ====== Anchored Kick01 (for GCI only) & KickQ (for Sleepers/Display) ======
def kick01_anchored(refined_fsp, basic_fsp):
    # 98→0, 104→1
    return 0.70 * map01(refined_fsp, 98.0, 104.0) + 0.30 * map01(basic_fsp, 98.0, 104.0)

def kickQ_quantile(df_fsp):
    """Return KickQ 0..100 using quantile-aware highs per race (Refined/Basic)."""
    ref = df_fsp["Refined_FSP_%"].astype(float)
    bas = df_fsp["Basic_FSP_%"].astype(float)
    base_hi_ref, base_hi_bas = 106.0, 105.5
    q75_ref = float(np.nanpercentile(ref.dropna(), 75)) if ref.notna().any() else base_hi_ref
    q75_bas = float(np.nanpercentile(bas.dropna(), 75)) if bas.notna().any() else base_hi_bas
    hi_ref = max(base_hi_ref, q75_ref + 1.0)
    hi_bas = max(base_hi_bas, q75_bas + 0.8)
    lo_ref, lo_bas = 98.0, 98.0
    k01 = 0.70 * ref.apply(lambda x: map01(x, lo_ref, hi_ref)) + 0.30 * bas.apply(lambda x: map01(x, lo_bas, hi_bas))
    return (k01 * 100.0).astype(float), dict(hi_ref=hi_ref, hi_bas=hi_bas, q75_ref=q75_ref, q75_bas=q75_bas)

# ====== GCI v3.3 (EPI-free, sectional-driven) ======
def distance_profile_for(distance_m):  # alias
    return distance_profile(distance_m)

def compute_gci_v33(metrics_df, distance_m):
    prof = distance_profile_for(distance_m)
    spi_median = metrics_df["SPI_%"].median()
    early_heat = 0.0 if pd.isna(spi_median) else clip01((spi_median - 100.0)/3.0)

    mid_vals = metrics_df["Mid400_Speed"]
    mid_pct = mid_vals.rank(pct=True, method="average")  # 0..1
    pressure_ratio_prime = float((mid_pct >= 0.65).mean())
    field_size = int(len(metrics_df))

    # T vs winner
    wtime = None
    if "Finish_Pos" in metrics_df.columns and "RaceTime_s" in metrics_df.columns and metrics_df["RaceTime_s"].notna().any():
        try:
            wtime = float(metrics_df.loc[metrics_df["RaceTime_s"].idxmin(), "RaceTime_s"])
        except Exception:
            wtime = None

    def to01(val, lo, hi):
        if pd.isna(val) or hi == lo:
            return 0.0
        return clip01((val - lo) / (hi - lo))

    def gci_row(row):
        refined = row["Refined_FSP_%"]; basic = row["Basic_FSP_%"]; spi = row["SPI_%"]
        rtime   = row["RaceTime_s"]; mid_rk = mid_pct.loc[row.name]
        k01     = kick01_anchored(refined, basic)  # anchored for comparability
        sprint_home = (spi_median <= 97)

        # 1) T
        T = 0.0
        if (wtime is not None) and (not pd.isna(rtime)):
            deficit = rtime - wtime
            if deficit <= 0.30: T = 1.0
            elif deficit <= 0.60: T = 0.7
            elif deficit <= 1.00: T = 0.4
            else: T = 0.2
            if sprint_home:
                T *= 0.85  # crawl deflate

        # 2) LQ via Kick
        LQ = k01

        # 3) OP' by mid-race speed + basic
        OPp = 0.0
        if (mid_rk >= 0.55) and (basic >= 99.0): OPp = 0.5
        if (mid_rk >= 0.70) and (basic >= 100.0): OPp = max(OPp, 0.7)
        if (early_heat >= 0.6) and (pressure_ratio_prime >= 0.35) and (mid_rk >= 0.70) and (basic >= 100.0):
            OPp = max(OPp, 1.0)
        if field_size <= 7 and OPp >= 0.7 and basic < 100.5:
            OPp = 0.5

        # 4) CM'
        CMp = 0.0
        if (early_heat >= 0.6) and (k01 >= 0.65):
            CMp = 0.60
            if k01 >= 0.80: CMp += 0.10
            if mid_rk <= 0.40: CMp += 0.05
            CMp = min(1.0, CMp)

        # 5) LT'
        LT_raw = 0.6*(1.0 - early_heat) + 0.5*(1.0 - pressure_ratio_prime)
        LT_raw = max(0.0, min(0.65, LT_raw))
        if (mid_rk >= 0.75) and (k01 < 0.60):
            LT_raw = min(0.65, LT_raw + 0.10)
        if sprint_home:
            LT_raw = min(0.65, LT_raw + 0.05)
        if early_heat >= 0.7:
            LT_raw = max(0.0, LT_raw - 0.15)
        LTp = min(LT_raw, 0.30 if early_heat >= 0.7 else 0.60)

        # 6) SS & EFF
        SS = 0.0 if (pd.isna(spi) or pd.isna(basic)) else to01((spi + basic)/2.0, prof["ss_lo"], prof["ss_hi"])
        EFF = 0.0 if pd.isna(refined) else max(0.0, 1.0 - abs(refined - 100.0)/8.0)

        PACE = max(LQ, OPp, CMp) * (1.0 - LTp)
        if LQ < prof["lq_floor"]:
            PACE = min(PACE, 0.60)

        score01 = (prof["wT"]*T) + (prof["wPACE"]*PACE) + (prof["wSS"]*SS) + (prof["wEFF"]*EFF)
        return round(10.0 * min(1.0, score01), 2), T, LQ, OPp, CMp, LTp, SS, EFF, k01, mid_rk

    parts = metrics_df.apply(gci_row, axis=1, result_type="expand")
    parts.columns = ["GCI","T","LQ","OPp","CMp","LTp","SS","EFF","Kick01","mid_pct"]
    return pd.concat([metrics_df.reset_index(drop=True), parts.reset_index(drop=True)], axis=1), dict(
        spi_median=spi_median, early_heat=early_heat, pressure_ratio_prime=pressure_ratio_prime, field_size=field_size
    )

# ====== Sleepers v3.3 (KickQ + Balanced + dual gate + tiers + heat bonus + guards + safety-net) ======
KICK_THR = {"SPRINT":68.0, "MILE":67.0, "MIDDLE":65.0, "STAY":63.0}
DEFICIT  = {"SPRINT":(0.25,1.05), "MILE":(0.33,1.05), "MIDDLE":(0.42,1.15), "STAY":(0.52,1.35)}
S_GATE   = {"SPRINT":0.58, "MILE":0.59, "MIDDLE":0.60, "STAY":0.60}
TIER_EDGES = [0.58, 0.68, 0.78]  # Bronze/Silver/Gold

def _tier_label(x):
    if x < TIER_EDGES[0]: return "—"
    if x < TIER_EDGES[1]: return "Bronze"
    if x < TIER_EDGES[2]: return "Silver"
    return "Gold"

def flag_sleepers_v33_balanced(df_in, distance_m, ctx):
    df = df_in.copy()
    bucket = _distance_bucket(distance_m)
    kick_thr = KICK_THR[bucket]
    lo_def, hi_def = DEFICIT[bucket]
    gate = S_GATE[bucket]

    # field-size guards
    fs = ctx.get("field_size", int(df.shape[0]))
    if fs <= 8:
        kick_thr += 1.0
    elif fs >= 13:
        gate = max(0.0, gate - 0.02)

    # KickQ for sleepers/display
    kickQ, qinfo = kickQ_quantile(df[["Refined_FSP_%","Basic_FSP_%"]])
    df["KickQ"] = kickQ
    # Absolute anchored Kick (for dual gate)
    df["Kick_abs"] = 100.0 * (0.70*df["Refined_FSP_%"].apply(lambda x: map01(x, 98.0, 104.0)) +
                              0.30*df["Basic_FSP_%"].apply(lambda x: map01(x, 98.0, 104.0)))

    # mid_pct already exists in GCI block; recompute here if missing
    if "mid_pct" not in df.columns:
        df["mid_pct"] = df["Mid400_Speed"].rank(pct=True, method="average").fillna(0.5)

    # winner time
    wtime = float(df.loc[df["RaceTime_s"].idxmin(), "RaceTime_s"])

    # Components (using KickQ for 'kick_comp')
    df["kick_comp"]  = df["KickQ"].apply(lambda k: clip01((k - (kick_thr - 10.0)) / 20.0))
    df["chase_comp"] = ctx["early_heat"] * (1.0 - df["mid_pct"]) * (0.5 + 0.5*df["Refined_FSP_%"].apply(lambda r: map01(r, 100.0, 104.0)))
    df.loc[df["Basic_FSP_%"] >= 100.0, "chase_comp"] += 0.10
    df["chase_comp"] = df["chase_comp"].apply(clip01)
    df["deficit_s"]  = df["RaceTime_s"] - wtime
    df["time_comp"]  = 1.0 - ((df["deficit_s"] - lo_def) / (hi_def - lo_def)).apply(lambda x: clip01(x))

    df["sleeper01"]  = 0.60*df["kick_comp"] + 0.25*df["chase_comp"] + 0.15*df["time_comp"]
    # Heat bonus
    if ctx["early_heat"] >= 0.60:
        df["sleeper01"] = (df["sleeper01"] + 0.03).clip(upper=1.0)

    # Dual gate: outside top-3 AND
    #    (score ≥ gate) OR (abs Kick ≥ threshold AND chase/time reasonable) OR (KickQ pct ≥ 75th)
    kickq_pct = df["KickQ"].rank(pct=True, method="average")
    df["Sleeper"] = (df["Finish_Pos"] > 3) & (
        (df["sleeper01"] >= gate) |
        ((df["Kick_abs"] >= kick_thr) & ((df["chase_comp"] >= 0.35) | (df["time_comp"] >= 0.50))) |
        (kickq_pct >= 0.75)
    )

    # Safety net: if none flagged, widen time hi +0.05 then lower Kick line −1
    flagged = df["Sleeper"].sum()
    if flagged == 0:
        df["time_comp"]  = 1.0 - ((df["deficit_s"] - lo_def) / (hi_def + 0.05 - lo_def)).apply(lambda x: clip01(x))
        df["sleeper01"]  = 0.60*df["kick_comp"] + 0.25*df["chase_comp"] + 0.15*df["time_comp"]
        if ctx["early_heat"] >= 0.60:
            df["sleeper01"] = (df["sleeper01"] + 0.03).clip(upper=1.0)
        df["Sleeper"] = (df["Finish_Pos"] > 3) & (
            (df["sleeper01"] >= gate) |
            ((df["Kick_abs"] >= (kick_thr - 1.0)) & ((df["chase_comp"] >= 0.35) | (df["time_comp"] >= 0.50))) |
            (kickq_pct >= 0.75)
        )

    df["Sleeper_Tier"] = df["sleeper01"].apply(_tier_label)
    return df, qinfo

# ====== Display helpers ======
def round_display(df):
    for c in ["Basic_FSP_%", "Refined_FSP_%", "SPI_%", "Race_AvgSpeed", "Mid400_Speed", "Final400_Speed", "KickQ", "Kick_abs"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)
    for c in ["T","LQ","OPp","CMp","LTp","SS","EFF","Kick01","mid_pct","sleeper01","kick_comp","chase_comp","time_comp"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(3)
    if "RaceTime_s" in df.columns:
        df["RaceTime_s"] = df["RaceTime_s"].round(3)
    return df

def runner_summary(row, spi_med):
    name = str(row.get("Horse", ""))
    fin  = row.get("Finish_Pos", np.nan)
    rt   = row.get("RaceTime_s", np.nan)
    ref  = row.get("Refined_FSP_%", np.nan)
    bas  = row.get("Basic_FSP_%", np.nan)
    gci  = row.get("GCI", np.nan)
    bits = [
        f"**{name}** — Finish: **{int(fin) if not pd.isna(fin) else '—'}**, Time: **{rt:.2f}s**",
        f"Late profile: Refined FSP {ref:.1f}% | Basic FSP {bas:.1f}%; GCI {gci:.2f}",
        f"Race shape (SPI median): {spi_med:.1f}%."
    ]
    return "  \n".join(bits)

# ===================
# UI: Inputs & Flow
# ===================
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input type", ["Upload file (CSV/XLSX)", "Manual"], index=0)
    # distance entry: allow any distance (e.g., 1160) — grid logic uses nearest 200 for labels only
    distance_m = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=10)
    n_horses = None
    if source == "Manual":
        n_horses = st.number_input("Number of runners", min_value=2, max_value=20, value=10, step=1)

df_raw = None

try:
    if source == "Upload file (CSV/XLSX)":
        uploaded = st.file_uploader("Upload CSV/XLSX (columns: Horse, Race Time, 800-400, 400-Finish[, Finish_Pos])", type=["csv","xlsx"])
        if uploaded is None:
            st.info("Upload a file or switch to Manual input.")
            st.stop()
        df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        st.success("File loaded.")
    else:
        # Manual grid
        st.markdown("### Manual entry")
        st.caption("Enter per-runner: Horse, Finish_Pos, Race Time (s or MM:SS.ms), 800-400 (s), 400-Finish (s).")
        # Build a blank frame
        df_raw = pd.DataFrame({
            "Horse": ["" for _ in range(n_horses)],
            "Finish_Pos": [np.nan for _ in range(n_horses)],
            "Race Time": [np.nan for _ in range(n_horses)],
            "800-400": [np.nan for _ in range(n_horses)],
            "400-Finish": [np.nan for _ in range(n_horses)],
        })
        # Editable grid
        df_raw = st.data_editor(df_raw, use_container_width=True, num_rows="dynamic")
        st.info("Tip: Distance labels (for pace view) count down from the race trip; calculations use exact distance.")
except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw table preview")
st.dataframe(df_raw.head(12), use_container_width=True)
_dbg("Raw columns", list(df_raw.columns))

# Detect & convert 200m TPD tables if user uploaded those
df_norm = df_raw.copy()
df_norm.columns = [_norm_header(c) for c in df_norm.columns]
looks_200m = _has_200m_headers(df_raw) or _has_200m_headers(df_norm)

try:
    if looks_200m and source == "Upload file (CSV/XLSX)":
        df_conv_input = df_raw.copy()
        df_conv_input.columns = df_norm.columns
        work = convert_200m_table_to_app(df_conv_input)
        st.info("Detected 200m TPD-style paste — converted to app schema.")
    else:
        work = df_raw.rename(columns={
            "Race time": "Race Time", "Race_Time": "Race Time", "RaceTime": "Race Time",
            "800_400": "800-400", "400_Finish": "400-Finish",
            "Horse Name": "Horse", "Finish": "Finish_Pos", "Placing": "Finish_Pos"
        })
except Exception as e:
    st.error("Conversion to app schema failed.")
    if DEBUG: st.exception(e)
    st.stop()

# Required columns
_required = ["Horse", "Race Time", "800-400", "400-Finish"]
_missing = [c for c in _required if c not in work.columns]
if _missing:
    st.error("Missing required columns: " + ", ".join(_missing))
    st.stop()

# Normalize & parse
df = work.copy()
df["RaceTime_s"] = df["Race Time"].apply(parse_race_time)
for col in ["800-400", "400-Finish", "Finish_Pos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# If Finish_Pos missing, infer by time
if ("Finish_Pos" not in df.columns) or df["Finish_Pos"].isna().all():
    if df["RaceTime_s"].notna().any():
        df["Finish_Pos"] = df["RaceTime_s"].rank(method="min").astype("Int64")

st.subheader("Converted table (ready for analysis)")
st.dataframe(df.head(12), use_container_width=True)

# 200m vs 400m split sanity (scale times if needed)
try:
    med_mid = float(np.nanmedian(df["800-400"]))
    med_fin = float(np.nanmedian(df["400-Finish"]))
except Exception:
    med_mid = med_fin = np.nan

scale = 1.0
# If both medians look like single 200m times (<~18s), scale by 2 to get 400m
if (not pd.isna(med_mid)) and (not pd.isna(med_fin)) and (med_mid < 18.0) and (med_fin < 18.0):
    scale = 2.0
df["800-400"]    = df["800-400"] * scale
df["400-Finish"] = df["400-Finish"] * scale

# ===================
# Analysis Pipeline
# ===================
try:
    metrics = compute_metrics(df, distance_m=distance_m)
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# GCI v3.3
gci_df, ctx = compute_gci_v33(metrics, distance_m=distance_m)

# Sleepers (KickQ + Balanced)
sleep_df, qinfo = flag_sleepers_v33_balanced(gci_df, distance_m=distance_m, ctx=ctx)

# ===================
# Outputs
# ===================
st.subheader("Sectional Metrics + GCI v3.3")
disp_cols = ["Horse","Finish_Pos","RaceTime_s","Basic_FSP_%","Refined_FSP_%","SPI_%",
             "Kick01","GCI","T","LQ","OPp","CMp","LTp","SS","EFF"]
disp = round_display(gci_df[disp_cols].copy()).sort_values(["Finish_Pos"], na_position="last")
st.dataframe(disp, use_container_width=True)

st.caption(
    "Kick01 is anchored (98→104 mapping) for cross-race comparability. "
    "GCI v3.3 uses sectionals only (mid-race speed percentile, Kick01), "
    "with sectional leader-tax and chase merit."
)

# Pace Curves
st.subheader("Pace Curves — Field Average (black) + Top 8 by Finish")
avg_mid = gci_df["Mid400_Speed"].mean()
avg_fin = gci_df["Final400_Speed"].mean()
top8 = gci_df.sort_values("Finish_Pos").head(8)

fig, ax = plt.subplots()
x_vals = [1, 2]
ax.plot(x_vals, [avg_mid, avg_fin], marker="o", linewidth=3, color="black", label="Average (Field)")
for _, row in top8.iterrows():
    ax.plot(x_vals, [row["Mid400_Speed"], row["Final400_Speed"]], marker="o", linewidth=2, label=str(row.get("Horse","Runner")))
ax.set_xticks([1, 2]); ax.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
ax.set_ylabel("Speed (m/s)"); ax.set_title("Average vs Top 8 Pace Curves")
ax.grid(True, linestyle="--", alpha=0.3)
fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
st.pyplot(fig)

# Race context
st.subheader("Race Shape / Context")
st.write(
    f"**SPI median:** {ctx['spi_median']:.2f}%  |  "
    f"**Early heat (0–1):** {ctx['early_heat']:.2f}  |  "
    f"**Pressure ratio′ (mid_pct≥0.65):** {ctx['pressure_ratio_prime']:.2f}  |  "
    f"**Field size:** {ctx['field_size']}"
)

# Sleepers
st.subheader("Sleepers (KickQ + Balanced)")
sleep_disp_cols = ["Horse","Finish_Pos","RaceTime_s","Kick_abs","KickQ","mid_pct",
                   "deficit_s","kick_comp","chase_comp","time_comp","sleeper01","Sleeper","Sleeper_Tier"]
sleep_disp = round_display(sleep_df[sleep_disp_cols].copy()).sort_values(
    ["Sleeper","sleeper01","Finish_Pos"], ascending=[False, False, True]
)
st.dataframe(sleep_disp, use_container_width=True)
st.caption(
    "Sleepers use quantile-aware **KickQ** for display/scoring with dual gates (abs Kick line or top-quartile KickQ), "
    "trip-aware deficit windows, tiers, heat bonus, and field-size guards. "
    f"KickQ highs this race: Refined hi≈{qinfo['hi_ref']:.2f}, Basic hi≈{qinfo['hi_bas']:.2f}."
)

# Runner-by-runner summaries
st.subheader("Runner-by-runner summaries")
ordered = gci_df.sort_values("Finish_Pos", na_position="last")
for _, row in ordered.iterrows():
    st.markdown(runner_summary(row, ctx["spi_median"]))
    st.markdown("---")

# Download
st.subheader("Download")
out = sleep_df.merge(gci_df[["Horse","GCI"]], on="Horse", how="left", suffixes=("",""))
out = round_display(out)
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download full metrics as CSV", data=csv_bytes, file_name="race_edge_v33_metrics.csv", mime="text/csv")

st.caption(
    "Legend: Basic FSP = Final400 / Race Avg; Refined FSP = Final400 / Mid400; "
    "SPI = Mid400 / Race Avg; Kick01 (anchored) for GCI; KickQ (quantile-aware) for Sleepers/Display; "
    "GCI v3.3 applies distance-normalised, sectional pressure context."
)
