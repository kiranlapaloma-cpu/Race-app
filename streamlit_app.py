import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================================
# App / Branding
# ================================
APP_DIR = Path(__file__).resolve().parent
CANDIDATE_LOGO_PATHS = [
    APP_DIR / "assets" / "logos.png",
    APP_DIR / "assets" / "logo.png",
    APP_DIR / "logos.png",
    APP_DIR / "logo.png",
]
LOGO_PATH = next((p for p in CANDIDATE_LOGO_PATHS if p.exists()), None)

icon = str(LOGO_PATH) if LOGO_PATH and Path(LOGO_PATH).exists() else "🏇"
st.set_page_config(page_title="The Sharpest Edge", page_icon=icon, layout="wide")
if LOGO_PATH and Path(LOGO_PATH).exists():
    st.image(str(LOGO_PATH), width=220)

st.title("🏇 The Sharpest Edge")
st.caption("Upload CSV or use Manual input with countdown 200 m segments. Calculates SPI / Basic FSP / Refined FSP, flags Sleepers, computes GCI v3.1, and plots pace curves.")

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

# =========================================================
# Time parsing (seconds, M:SS.ms, M:SS:ms like 01:37:620, H:MM:SS(.ms))
# =========================================================
_M_SS_MS_RE   = re.compile(r"^(?P<m>\d{1,2}):(?P<s>\d{2}):(?P<ms>\d{2,3})$")      # 01:37:620
_M_SS_DMS_RE  = re.compile(r"^(?P<m>\d{1,2}):(?P<s>\d{2}\.\d+)$")                  # 1:12.45
_H_MM_SS_RE   = re.compile(r"^(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)$")    # 0:01:12.45

def parse_time_any(val):
    """Return seconds (float) from seconds, M:SS.ms, M:SS:ms, or H:MM:SS(.ms)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    s = s.replace("：", ":")                         # normalize unicode colon
    s = re.sub(r"[^\d:\.\s]", "", s)                # keep digits/colon/dot/space
    s = re.sub(r"\s+", "", s)

    # plain seconds?
    try:
        return float(s)
    except Exception:
        pass

    m = _M_SS_MS_RE.match(s)
    if m:
        mm, ss, ms = int(m.group("m")), int(m.group("s")), int(m.group("ms"))
        if ss < 60 and ms < 1000:
            return mm * 60 + ss + ms / 1000.0

    m = _M_SS_DMS_RE.match(s)
    if m:
        mm, sec = int(m.group("m")), float(m.group("s"))
        return mm * 60 + sec

    m = _H_MM_SS_RE.match(s)
    if m:
        hh, mm, sec = int(m.group("h")), int(m.group("m")), float(m.group("s"))
        return hh * 3600 + mm * 60 + sec

    return np.nan

# =========================================================
# Core metric helpers
# =========================================================
def compute_metrics(df, distance_m=1400.0):
    """Assumes df has numeric RaceTime_s, 800-400, 400-Finish columns; optional positions."""
    out = df.copy()

    for c in ["RaceTime_s", "800-400", "400-Finish"]:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")

    # Optional positions if present (used for EPI/Pos_Change; OK if absent)
    for c in ["200_Pos", "400_Pos", "800_Pos", "1000_Pos", "Finish_Pos"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    with np.errstate(divide='ignore', invalid='ignore'):
        out["Race_AvgSpeed"]  = distance_m / out["RaceTime_s"]
        out["Mid400_Speed"]   = 400.0 / out["800-400"]
        out["Final400_Speed"] = 400.0 / out["400-Finish"]

        out["SPI_%"]         = (out["Mid400_Speed"]   / out["Race_AvgSpeed"]) * 100.0
        out["Basic_FSP_%"]   = (out["Final400_Speed"] / out["Race_AvgSpeed"]) * 100.0
        out["Refined_FSP_%"] = (out["Final400_Speed"] / out["Mid400_Speed"]) * 100.0

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # EPI (optional; only if positions exist)
    if ("200_Pos" in out.columns) and ("400_Pos" in out.columns):
        out["EPI"] = out["200_Pos"] * 0.6 + out["400_Pos"] * 0.4
    elif ("1000_Pos" in out.columns) and ("800_Pos" in out.columns):
        out["EPI"] = out["1000_Pos"] * 0.6 + out["800_Pos"] * 0.4
    elif "400_Pos" in out.columns:
        out["EPI"] = out["400_Pos"]
    else:
        out["EPI"] = np.nan

    # Pos change (optional)
    if ("400_Pos" in out.columns) and ("Finish_Pos" in out.columns):
        out["Pos_Change"] = out["400_Pos"] - out["Finish_Pos"]
    else:
        out["Pos_Change"] = np.nan

    return out

def round_display(df):
    df = df.copy()
    for c in ["Basic_FSP_%", "Refined_FSP_%", "SPI_%", "Race_AvgSpeed", "Mid400_Speed", "Final400_Speed"]:
        if c in df.columns:
            df.loc[:, c] = df[c].round(2)
    for c in ["RaceTime_s", "EPI"]:
        if c in df.columns:
            df.loc[:, c] = df[c].round(3 if c == "RaceTime_s" else 1)
    if "Pos_Change" in df.columns:
        df.loc[:, "Pos_Change"] = df["Pos_Change"].round(0).astype("Int64")
    return df

# =========================================================
# Sleeper flag
# =========================================================
def flag_sleepers(df):
    """
    Flags 'Sleeper' if:
      - Refined_FSP_% >= 102, OR
      - Pos_Change >= 3 (when available)
    And (if Finish_Pos present) finished outside top 3.
    Works even without positions (then uses the first rule only).
    """
    out = df.copy()
    out["Sleeper"] = False
    cond = (out["Refined_FSP_%"] >= 102.0) | (out["Pos_Change"].fillna(0) >= 3)
    if "Finish_Pos" in out.columns:
        cond = cond & (out["Finish_Pos"] > 3)
    out.loc[cond, "Sleeper"] = True
    return out

# =========================================================
# Pressure context + GCI v3.1
# =========================================================
def _to01(val, lo, hi):
    if pd.isna(val) or hi == lo:
        return 0.0
    return float(min(1.0, max(0.0, (val - lo) / (hi - lo))))

def compute_pressure_context(metrics_df):
    spi_med = metrics_df["SPI_%"].median()
    early_heat = 0.0 if pd.isna(spi_med) else max(0.0, min(1.0, (spi_med - 100.0) / 3.0))  # ~103 → 1.0
    epi_series = metrics_df["EPI"] if "EPI" in metrics_df.columns else pd.Series(dtype=float)
    if epi_series.empty:
        pressure_ratio = 0.0
        field_size = int(metrics_df.shape[0])
    else:
        valid_epi = epi_series.dropna()
        field_size = int(metrics_df.shape[0])
        pressure_ratio = 0.0 if valid_epi.empty else float((valid_epi <= 3.0).mean())
    return {"early_heat": early_heat, "pressure_ratio": pressure_ratio, "field_size": field_size, "spi_median": spi_med}

def _distance_bucket(distance_m: float) -> str:
    if distance_m <= 1400:
        return "SPRINT"
    if distance_m < 1800:
        return "MILE"
    if distance_m < 2200:
        return "MIDDLE"
    return "STAY"

def distance_profile(distance_m: float) -> dict:
    b = _distance_bucket(distance_m)
    if b == "SPRINT":
        return dict(bucket=b, wT=0.22, wPACE=0.38, wSS=0.20, wEFF=0.20, lq_ref_w=0.70, lq_basic_w=0.30, ss_lo=99.0, ss_hi=105.0, lq_floor=0.35)
    if b == "STAY":
        return dict(bucket=b, wT=0.30, wPACE=0.28, wSS=0.30, wEFF=0.12, lq_ref_w=0.50, lq_basic_w=0.50, ss_lo=97.5, ss_hi=104.5, lq_floor=0.25)
    if b == "MIDDLE":
        return dict(bucket=b, wT=0.27, wPACE=0.32, wSS=0.28, wEFF=0.13, lq_ref_w=0.60, lq_basic_w=0.40, ss_lo=98.0, ss_hi=105.0, lq_floor=0.30)
    return dict(bucket="MILE", wT=0.25, wPACE=0.35, wSS=0.25, wEFF=0.15, lq_ref_w=0.60, lq_basic_w=0.40, ss_lo=98.0, ss_hi=105.0, lq_floor=0.30)

def compute_gci_v31(row, ctx, distance_m: float, winner_time_s=None):
    prof = distance_profile(float(distance_m))
    reasons = [f"{prof['bucket'].title()} weighting"]

    refined = row.get("Refined_FSP_%", np.nan)
    basic   = row.get("Basic_FSP_%",   np.nan)
    spi     = row.get("SPI_%",         np.nan)
    epi     = row.get("EPI",           np.nan)
    rtime   = row.get("RaceTime_s",    np.nan)

    early_heat   = float(ctx.get("early_heat", 0.0))
    pressure_rat = float(ctx.get("pressure_ratio", 0.0))
    spi_median   = ctx.get("spi_median", np.nan)

    fast_early  = (not pd.isna(spi_median)) and (spi_median >= 103)
    sprint_home = (not pd.isna(spi_median)) and (spi_median <= 97)

    # T: Time vs winner
    T = 0.0
    if (winner_time_s is not None) and (not pd.isna(rtime)):
        deficit = rtime - winner_time_s
        if deficit <= 0.30:
            T = 1.0; reasons.append("≤0.30s off winner")
        elif deficit <= 0.60:
            T = 0.7; reasons.append("0.31–0.60s off winner")
        elif deficit <= 1.00:
            T = 0.4; reasons.append("0.61–1.00s off winner")
        else:
            T = 0.2

    # LQ: Late quality (distance-aware blend)
    def map_pct(x): return max(0.0, min(1.0, (x - 98.0) / 6.0))  # 98→0, 104→1
    LQ = 0.0
    if not pd.isna(refined) and not pd.isna(basic):
        LQ = prof["lq_ref_w"] * map_pct(refined) + prof["lq_basic_w"] * map_pct(basic)
        if refined >= 103:
            reasons.append("strong late profile")
        elif refined >= 101.5:
            reasons.append("useful late profile")

    # OP: On-pace merit (with pressure gates)
    OP = 0.0
    if not pd.isna(epi) and not pd.isna(basic):
        on_speed = (epi <= 2.5)
        handy    = (epi <= 3.5)
        if handy and basic >= 99:
            OP = 0.5
        if on_speed and basic >= 100:
            OP = max(OP, 0.7)
        if on_speed and fast_early and (early_heat >= 0.7) and (pressure_rat >= 0.35) and basic >= 100:
            OP = max(OP, 1.0); reasons.append("on-speed under genuine heat & pressure")

    # Leader Tax (applies only when we can infer early lead & soft conditions)
    LT = 0.0
    if not pd.isna(epi) and epi <= 2.0:
        soft_early = (early_heat < 0.5)
        low_press  = (pressure_rat < 0.30)
        weak_late  = (pd.isna(refined) or refined < 100.0) or (pd.isna(basic) or basic < 100.0)
        if soft_early: LT += 0.25
        if low_press:  LT += 0.20
        if weak_late:  LT += 0.20
        if sprint_home: LT += 0.15
        LT = min(0.60, LT)
        if LT > 0: reasons.append(f"leader tax applied ({LT:.2f})")

    # SS: Sustained speed
    if pd.isna(spi) or pd.isna(basic):
        SS = 0.0
    else:
        mean_sb = (spi + basic) / 2.0
        SS = _to01(mean_sb, prof["ss_lo"], prof["ss_hi"])
        if mean_sb >= (prof["ss_hi"] - 2.0):
            reasons.append("strong sustained speed")

    # EFF: Efficiency
    if pd.isna(refined):
        EFF = 0.0
    else:
        EFF = max(0.0, 1.0 - abs(refined - 100.0) / 8.0)
        if 99 <= refined <= 103:
            reasons.append("efficient sectional profile")

    wT, wPACE, wSS, wEFF = prof["wT"], prof["wPACE"], prof["wSS"], prof["wEFF"]
    PACE = max(LQ, OP) * (1.0 - LT)

    if LQ < prof["lq_floor"]:
        PACE = min(PACE, 0.60)

    score01 = min(1.0, (wT * T) + (wPACE * PACE) + (wSS * SS) + (wEFF * EFF))
    score10 = round(10.0 * score01, 2)
    return score10, reasons

# =========================================================
# Manual-mode grid utilities (countdown 200 m segments)
# =========================================================
def make_countdown_headers(distance_m: int):
    """Return segment headers in countdown order, e.g. 1400 -> ['1200m','1000m','800m','600m','400m','200m','Finish']"""
    if distance_m % 200 != 0 or distance_m < 800:
        raise ValueError("Distance must be a multiple of 200 and at least 800 m.")
    headers = []
    for d in range(distance_m - 200, 0, -200):
        headers.append(f"{d}m")
    headers.append("Finish")
    return headers  # each header is a 200 m segment time

def build_manual_frame(n_rows: int, seg_headers: list, keep: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create an empty manual-entry frame with Horse + seg columns (+ optional Finish_Pos). Preserve overlapping data from `keep`."""
    cols = ["Horse"] + seg_headers + ["Finish_Pos"]
    df = pd.DataFrame({c: [np.nan]*n_rows for c in cols})
    df["Horse"] = ""
    if keep is not None:
        for c in set(keep.columns) & set(cols):
            n = min(n_rows, len(keep))
            df.loc[:n-1, c] = keep.loc[:n-1, c].values
    return df

def segments_to_mid_final_400(seg_cols: list[str]) -> tuple[list[str], list[str]]:
    """Given countdown segment column names, return two lists of column names to sum:
       mid400 uses the 3rd and 4th from the end; final400 uses last two."""
    if len(seg_cols) < 4:
        return [], []
    final400_cols = seg_cols[-2:]            # e.g., ['200m','Finish']
    mid400_cols   = seg_cols[-4:-2]          # e.g., ['600m','400m']
    return mid400_cols, final400_cols

# ===================
# Sidebar & data source
# ===================
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input type", ["Upload CSV", "Manual input"], index=1)

    distance_m = st.number_input("Race distance (m)", min_value=800, max_value=4000, value=1200, step=200, help="Multiples of 200 only")
    num_horses = st.number_input("Number of horses (manual)", min_value=1, max_value=30, value=8, step=1, disabled=(source != "Manual input"))

    st.caption("Manual mode shows countdown 200 m segments. Upload mode expects your CSV schema (Race Time, 800-400, 400-Finish).")

# ===================
# Collect data (Upload vs Manual)
# ===================
df_raw = None
mode_banner = "🖐️ Manual mode: enter 200 m **segment times** in countdown order." if source == "Manual input" else "📄 Upload CSV mode: provide Race Time, 800-400, 400-Finish."
st.info(mode_banner)

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    df_raw = pd.read_csv(uploaded)
    st.success("File loaded.")

else:
    # ----- Manual mode: build dynamic grid from distance & number of horses -----
    try:
        seg_headers = make_countdown_headers(int(distance_m))
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Persist / rebuild editor frame when controls change
    key_rows = ("manual_rows" not in st.session_state) or (st.session_state["manual_rows"] != int(num_horses))
    key_cols = ("manual_cols" not in st.session_state) or (st.session_state["manual_cols"] != tuple(seg_headers))

    if key_rows or key_cols or ("manual_df" not in st.session_state):
        keep = st.session_state.get("manual_df")
        st.session_state["manual_df"] = build_manual_frame(int(num_horses), seg_headers, keep=keep)
        st.session_state["manual_rows"] = int(num_horses)
        st.session_state["manual_cols"] = tuple(seg_headers)

    st.subheader("Manual input (countdown 200 m segments)")
    st.write(f"Columns: {' | '.join(seg_headers)}  — enter **segment times** (seconds or M:SS.ms / M:SS:ms).")
    manual_df = st.data_editor(
        st.session_state["manual_df"],
        width="stretch",
        num_rows="dynamic",
        key="manual_editor",
        column_config={
            "Horse": st.column_config.TextColumn(required=True),
            **{h: st.column_config.TextColumn(help=f"200 m segment time for {h}") for h in seg_headers},
            "Finish_Pos": st.column_config.NumberColumn(format="%d", help="Optional"),
        },
    )

    # Clean blank rows (no Horse)
    df_raw = manual_df.copy()
    df_raw["Horse"] = df_raw["Horse"].astype(str).str.strip()
    nonempty = df_raw["Horse"].ne("")
    if not nonempty.any():
        st.warning("Enter at least one horse row.")
        st.stop()
    df_raw = df_raw.loc[nonempty].reset_index(drop=True)
    st.success("Manual data captured.")

# ===================
# Normalize to analysis schema
# ===================
st.subheader("Raw table preview")
st.dataframe(df_raw.head(12), width="stretch")
_dbg("Raw columns", list(df_raw.columns))

if source == "Upload CSV":
    # Map common variants
    df = df_raw.rename(columns={
        "Race time": "Race Time", "Race_Time": "Race Time", "RaceTime": "Race Time",
        "800_400": "800-400", "400_Finish": "400-Finish",
        "Horse Name": "Horse", "Finish": "Finish_Pos", "Placing": "Finish_Pos"
    }).copy()

    required = ["Horse", "Race Time", "800-400", "400-Finish"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Missing required columns in CSV: " + ", ".join(missing))
        st.stop()

    # Parse times safely
    df["RaceTime_s"] = df["Race Time"].apply(parse_time_any)
    for col in ["800-400", "400-Finish"]:
        df[col] = pd.to_numeric(df[col].apply(parse_time_any), errors="coerce")

    # Optional numeric fields
    if "Finish_Pos" in df.columns:
        df["Finish_Pos"] = pd.to_numeric(df["Finish_Pos"], errors="coerce").astype("Int64")

else:
    # Manual: derive RaceTime_s, 800-400, 400-Finish from segment columns
    df = df_raw.copy()

    # Use the exact countdown order we generated for the grid
    desired_order = list(st.session_state.get("manual_cols", []))
    if not desired_order:
        desired_order = [c for c in df.columns if c not in ("Horse", "Finish_Pos")]

    # Keep only existing columns, preserving countdown order
    seg_cols = [c for c in desired_order if c in df.columns]

    # Strong safeguard
    if not seg_cols:
        st.error("No segment columns found. Please enter 200 m times in the manual editor.")
        st.stop()

    # Parse each segment to seconds
    for c in seg_cols:
        df[c] = pd.to_numeric(df[c].apply(parse_time_any), errors="coerce")

    # RaceTime_s = sum of all segment times (allow partial rows but require at least 1 value)
    df["RaceTime_s"] = df[seg_cols].sum(axis=1, min_count=1)

    # mid- and final-400 sums from tail segments
    mid_cols, fin_cols = segments_to_mid_final_400(seg_cols)
    if not mid_cols or not fin_cols:
        st.error("Distance must be at least 800 m to compute Mid400 and Final400.")
        st.stop()

    df["800-400"]    = df[mid_cols].sum(axis=1, min_count=len(mid_cols))
    df["400-Finish"] = df[fin_cols].sum(axis=1, min_count=len(fin_cols))

    # Optional Finish_Pos
    if "Finish_Pos" in df.columns:
        df["Finish_Pos"] = pd.to_numeric(df["Finish_Pos"], errors="coerce").astype("Int64")

    # Alias for downstream compatibility
    if "Race Time" not in df.columns and "RaceTime_s" in df.columns:
        df["Race Time"] = df["RaceTime_s"]

st.subheader("Converted table (ready for analysis)")
st.dataframe(df.head(12), width="stretch")
_dbg("Dtypes", df.dtypes)

# ===================
# Analysis: metrics + Sleepers + GCI
# ===================
try:
    metrics = compute_metrics(df, distance_m=float(distance_m))
    # Winner time (best Finish_Pos if provided, else fastest RaceTime_s)
    winner_time = None
    if "Finish_Pos" in df.columns and df["Finish_Pos"].notna().any():
        try:
            winner_time = metrics.loc[metrics["Finish_Pos"].idxmin(), "RaceTime_s"]
        except Exception:
            winner_time = None
    if winner_time is None and metrics["RaceTime_s"].notna().any():
        winner_time = metrics["RaceTime_s"].min()

    # Pressure context & GCI
    ctx = compute_pressure_context(metrics)
    gci_scores, gci_reasons = [], []
    for _, r in metrics.iterrows():
        gci, why = compute_gci_v31(r, ctx, distance_m=float(distance_m), winner_time_s=winner_time)
        gci_scores.append(gci)
        gci_reasons.append("; ".join(why))
    metrics["GCI"] = gci_scores
    metrics["GCI_Reasons"] = gci_reasons
    metrics["Group_Candidate"] = metrics["GCI"] >= 7.0

    # Sleepers
    metrics = flag_sleepers(metrics)

except Exception as e:
    st.error("Metric/GCI computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# If Finish_Pos missing, rank by time for display order
if "Finish_Pos" not in metrics.columns or metrics["Finish_Pos"].isna().all():
    metrics["Finish_Pos"] = metrics["RaceTime_s"].rank(method="min").astype("Int64")

# ===================
# Outputs
# ===================
st.subheader("Sectional Metrics")
disp = round_display(metrics.copy()).sort_values("Finish_Pos", na_position="last")
st.dataframe(disp, width="stretch")

st.subheader("Pace Curves — Field Average (black) + Top 8 by Finish")
avg_mid = metrics["Mid400_Speed"].mean()
avg_fin = metrics["Final400_Speed"].mean()
top8 = metrics.sort_values("Finish_Pos").head(8).copy()
top8["HorseShort"] = top8["Horse"].astype(str).str.slice(0, 20)

fig, ax = plt.subplots()
x_vals = [1, 2]
ax.plot(x_vals, [avg_mid, avg_fin], marker="o", linewidth=3, color="black", label="Average (Field)")
for _, row in top8.iterrows():
    ax.plot(x_vals, [row["Mid400_Speed"], row["Final400_Speed"]], marker="o", linewidth=2, label=row["HorseShort"])
ax.set_xticks([1, 2]); ax.set_xticklabels(["Mid 400", "Final 400"])
ax.set_ylabel("Speed (m/s)"); ax.set_title("Average vs Top 8 Pace Curves")
ax.grid(True, linestyle="--", alpha=0.3)
fig.subplots_adjust(bottom=0.22)
fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
st.pyplot(fig)

st.subheader("Insights")
ctx_spi = ctx.get("spi_median", np.nan)
if pd.isna(ctx_spi):
    st.write("SPI median: —  |  Early heat: —  |  Pressure ratio (EPI≤3.0): —  |  Field size: ", ctx.get("field_size", 0))
else:
    st.write(
        f"**SPI median:** {ctx_spi:.1f}%  |  "
        f"**Early heat (0–1):** {ctx['early_heat']:.2f}  |  "
        f"**Pressure ratio (EPI≤3.0):** {ctx['pressure_ratio']:.2f}  |  "
        f"**Field size:** {ctx['field_size']}"
    )

st.subheader("Sleepers")
sleepers = disp[disp["Sleeper"] == True][["Horse","Finish_Pos","Refined_FSP_%","Pos_Change"]] if 'disp' in locals() else pd.DataFrame()
if sleepers.empty:
    st.write("No clear sleepers flagged under current thresholds.")
else:
    st.dataframe(sleepers, width="stretch")

st.subheader("Potential Group-class candidates (GCI v3.1)")
cands = metrics.loc[metrics["Group_Candidate"]].copy().sort_values(["GCI","Finish_Pos"], ascending=[False, True])
if cands.empty:
    st.write("No horses met the Group-class threshold today.")
else:
    st.dataframe(
        round_display(cands[["Horse","Finish_Pos","RaceTime_s","Refined_FSP_%","Basic_FSP_%","SPI_%","Pos_Change","EPI","GCI","GCI_Reasons"]]),
        width="stretch"
    )

# ===================
# Download
# ===================
st.subheader("Download")
csv_bytes = disp.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics as CSV", data=csv_bytes, file_name="race_sectional_metrics.csv", mime="text/csv")

st.caption(
    "Manual mode uses countdown 200 m segment times. We derive RaceTime (seconds), Mid400 and Final400, then compute SPI / Basic FSP / Refined FSP, flag Sleepers, and compute GCI v3.1. "
    "Upload mode expects: Horse, Race Time, 800-400, 400-Finish (Finish_Pos optional). "
    "EPI/position fields are optional; when absent, GCI ignores on-pace/leader adjustments."
)
