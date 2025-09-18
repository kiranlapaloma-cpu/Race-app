import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================================
# Branding / Logo (robust loader)
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
    st.image(str(LOGO_PATH), width=250)

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
# Helpers: parsing, metrics, GCI v3.1, narratives
# =========================================================
def parse_race_time(val):
    """
    Accepts:
      - seconds as float/int:  "72.45"
      - M:SS(.ms):            "1:12.45"
      - H:MM:SS(.ms):         "0:01:12.45"
      - M:SS:ms (TPD-style):  "01:24:510"  -> 84.510 seconds
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()

    # plain seconds?
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
            a, b, c = parts

            # Heuristic for M:SS:ms (no dot in last piece, <=3 digits, a<60, b<60)
            if (c.replace(" ", "").isdigit() and "." not in c and len(c) <= 3
                and b.replace(" ", "").isdigit() and int(b) < 60
                and a.replace(" ", "").isdigit() and int(a) < 60):
                return int(a) * 60 + int(b) + (int(c) / 1000.0)

            # Otherwise treat as H:MM:SS(.ms)
            return int(a) * 3600 + int(b) * 60 + float(c)

    except Exception:
        return np.nan

    return np.nan

def compute_metrics(df, distance_m=1400.0):
    out = df.copy()

    # Ensure numeric for required inputs
    for c in ["RaceTime_s", "800-400", "400-Finish"]:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")

    with np.errstate(divide='ignore', invalid='ignore'):
        out["Race_AvgSpeed"]  = distance_m / out["RaceTime_s"]
        out["Mid400_Speed"]   = 400.0 / out["800-400"]
        out["Final400_Speed"] = 400.0 / out["400-Finish"]

        out["Basic_FSP_%"]   = (out["Final400_Speed"] / out["Race_AvgSpeed"]) * 100.0
        out["Refined_FSP_%"] = (out["Final400_Speed"] / out["Mid400_Speed"]) * 100.0
        out["SPI_%"]         = (out["Mid400_Speed"]   / out["Race_AvgSpeed"]) * 100.0

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Early Position Index (fallbacks + common alternates)
    def _pick(series, *names):
        for n in names:
            if n in out.columns:
                return pd.to_numeric(out[n], errors="coerce")
        return None

    p200  = _pick(out, "200_Pos", "Pos200", "200Pos", "P200")
    p400  = _pick(out, "400_Pos", "Pos400", "400Pos", "P400")
    p800  = _pick(out, "800_Pos", "Pos800", "800Pos", "P800")
    p1000 = _pick(out, "1000_Pos","Pos1000","1000Pos","P1000")

    if (p200 is not None) and (p400 is not None):
        out["EPI"] = p200 * 0.6 + p400 * 0.4
    elif (p1000 is not None) and (p800 is not None):
        out["EPI"] = p1000 * 0.6 + p800 * 0.4
    elif p400 is not None:
        out["EPI"] = p400
    else:
        out["EPI"] = np.nan

    # Late position change (400m to finish)
    if ("400_Pos" in out.columns) and ("Finish_Pos" in out.columns):
        out["Pos_Change"] = pd.to_numeric(out["400_Pos"], errors="coerce") - pd.to_numeric(out["Finish_Pos"], errors="coerce")
    elif "Finish_Pos" in out.columns:
        out["Pos_Change"] = np.nan

    return out

def flag_sleepers(df):
    out = df.copy()
    out["Sleeper"] = False
    cond = (out["Refined_FSP_%"] >= 102.0) | (out["Pos_Change"].fillna(0) >= 3)
    if "Finish_Pos" in out.columns:
        cond = cond & (out["Finish_Pos"] > 3)
    out.loc[cond, "Sleeper"] = True
    return out

def round_display(df):
    for c in ["Basic_FSP_%", "Refined_FSP_%", "SPI_%", "Race_AvgSpeed", "Mid400_Speed", "Final400_Speed"]:
        if c in df.columns:
            df[c] = df[c].round(2)
    if "EPI" in df.columns:
        df["EPI"] = df["EPI"].round(1)
    if "Pos_Change" in df.columns:
        df["Pos_Change"] = df["Pos_Change"].round(0).astype("Int64")
    if "RaceTime_s" in df.columns:
        df["RaceTime_s"] = df["RaceTime_s"].round(3)
    return df

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
    return {
        "early_heat": early_heat,
        "pressure_ratio": pressure_ratio,
        "field_size": field_size,
        "spi_median": spi_med
    }

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
        return dict(bucket=b, wT=0.22, wPACE=0.38, wSS=0.20, wEFF=0.20,
                    lq_ref_w=0.70, lq_basic_w=0.30, ss_lo=99.0, ss_hi=105.0, lq_floor=0.35)
    if b == "STAY":
        return dict(bucket=b, wT=0.30, wPACE=0.28, wSS=0.30, wEFF=0.12,
                    lq_ref_w=0.50, lq_basic_w=0.50, ss_lo=97.5, ss_hi=104.5, lq_floor=0.25)
    if b == "MIDDLE":
        return dict(bucket=b, wT=0.27, wPACE=0.32, wSS=0.28, wEFF=0.13,
                    lq_ref_w=0.60, lq_basic_w=0.40, ss_lo=98.0, ss_hi=105.0, lq_floor=0.30)
    return dict(bucket="MILE", wT=0.25, wPACE=0.35, wSS=0.25, wEFF=0.15,
                lq_ref_w=0.60, lq_basic_w=0.40, ss_lo=98.0, ss_hi=105.0, lq_floor=0.30)

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
        if deficit <= 0.30: T = 1.0; reasons.append("≤0.30s off winner")
        elif deficit <= 0.60: T = 0.7; reasons.append("0.31–0.60s off winner")
        elif deficit <= 1.00: T = 0.4; reasons.append("0.61–1.00s off winner")
        else: T = 0.2

    # LQ: Late quality (distance-aware blend)
    def map_pct(x): return max(0.0, min(1.0, (x - 98.0) / 6.0))  # 98→0, 104→1
    LQ = 0.0
    if not pd.isna(refined) and not pd.isna(basic):
        LQ = prof["lq_ref_w"] * map_pct(refined) + prof["lq_basic_w"] * map_pct(basic)
        if refined >= 103: reasons.append("strong late profile")
        elif refined >= 101.5: reasons.append("useful late profile")

    # OP: On-pace merit (with pressure gates)
    OP = 0.0
    if not pd.isna(epi) and not pd.isna(basic):
        on_speed = (epi <= 2.5)
        handy    = (epi <= 3.5)
        if handy and basic >= 99: OP = 0.5
        if on_speed and basic >= 100: OP = max(OP, 0.7)
        if on_speed and fast_early and (early_heat >= 0.7) and (pressure_rat >= 0.35) and basic >= 100:
            OP = max(OP, 1.0); reasons.append("on-speed under genuine heat & pressure")

    # Leader Tax
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
        if 99 <= refined <= 103: reasons.append("efficient sectional profile")

    wT, wPACE, wSS, wEFF = prof["wT"], prof["wPACE"], prof["wSS"], prof["wEFF"]
    PACE = max(LQ, OP) * (1.0 - LT)

    if LQ < prof["lq_floor"]:
        PACE = min(PACE, 0.60)

    score01 = min(1.0, (wT * T) + (wPACE * PACE) + (wSS * SS) + (wEFF * EFF))
    score10 = round(10.0 * score01, 2)
    return score10, reasons

def style_from_epi(epi):
    if pd.isna(epi): return "position unknown"
    if epi <= 2.0:   return "on-speed/leader"
    if epi <= 5.0:   return "handy/midfield"
    return "backmarker"

def kick_from_refined(refined):
    if pd.isna(refined):     return "—"
    if refined >= 103.0:     return "accelerated strongly late"
    if refined >= 100.0:     return "kept building late"
    if refined >= 97.0:      return "flattened late"
    return "faded late"

def pace_note(spi_med):
    if pd.isna(spi_med): return "pace uncertain"
    if spi_med >= 103:   return "fast early / tough late"
    if spi_med <= 97:    return "slow early / sprint-home"
    return "even tempo"

def distance_hint(refined, pos_change, epi):
    if pd.isna(refined) or pd.isna(pos_change) or pd.isna(epi): return ""
    if refined >= 102 and pos_change >= 2: return "likely to appreciate a little further."
    if refined < 98 and epi <= 2.5:        return "may prefer slightly shorter or a softer mid-race."
    if refined >= 100 and epi >= 5:        return "effective if they can be ridden a touch closer."
    return ""

def runner_summary(row, spi_med):
    name = str(row.get("Horse", ""))
    fin  = row.get("Finish_Pos", np.nan)
    rt   = row.get("RaceTime_s", np.nan)
    epi  = row.get("EPI", np.nan)
    ref  = row.get("Refined_FSP_%", np.nan)
    spi  = row.get("SPI_%", np.nan)
    pc   = row.get("Pos_Change", np.nan)
    slp  = bool(row.get("Sleeper", False))
    gci  = row.get("GCI", np.nan)
    gcand= bool(row.get("Group_Candidate", False))

    fin_txt = f"{int(fin)}" if not pd.isna(fin) else "—"
    rt_txt  = f"{rt:.2f}s" if not pd.isna(rt) else "—"
    ref_txt = f"{ref:.1f}%" if not pd.isna(ref) else "—"
    pc_txt  = f"{int(pc)}" if not pd.isna(pc) else "—"
    gci_txt = f"{gci:.1f}" if not pd.isna(gci) else "—"

    bits = [
        f"**{name}** — Finish: **{fin_txt}**, Time: **{rt_txt}**",
        f"Style: {style_from_epi(epi)} (EPI {epi:.1f} if known); Race shape: {pace_note(spi_med)}.",
        f"Late profile: {kick_from_refined(ref)} (Refined FSP {ref_txt}); "
        f"Pos change 400–Finish: {pc_txt}; GCI: {gci_txt}."
    ]
    dh = distance_hint(ref, pc, epi)
    if dh: bits.append(f"Note: {dh}")
    tag = []
    if slp: tag.append("Sleeper")
    if gcand: tag.append("Group candidate")
    if tag: bits[0] += f" **[{', '.join(tag)}]**"
    return "  \n".join(bits)

# ===================
# UI: Inputs & Flow (CSV or Manual only)
# ===================
st.title("🏇 The Sharpest Edge")
st.caption("CSV upload or manual input. URL and paste ingestion removed.")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input type", ["Upload CSV", "Manual input"], index=0)
    distance_m = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=50)
    show_example = st.checkbox("Prefill manual input with 6 example rows", value=False)

def _empty_manual_frame(n_rows: int = 8) -> pd.DataFrame:
    """Blank manual-entry frame with the exact columns the analysis expects."""
    cols = [
        "Horse", "Finish_Pos", "Race Time", "800-400", "400-Finish",
        "200_Pos", "400_Pos", "800_Pos", "1000_Pos"
    ]
    df = pd.DataFrame({c: [np.nan]*n_rows for c in cols})
    df["Horse"] = ""
    return df

df_raw = None

try:
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV (columns: Horse, Finish_Pos (opt), Race Time, 800-400, 400-Finish, 200_Pos/400_Pos/800_Pos/1000_Pos (opt))",
            type=["csv"]
        )
        if uploaded is None:
            st.info("Upload a CSV or switch to Manual input.")
            st.stop()
        df_raw = pd.read_csv(uploaded)
        st.success("File loaded.")
    else:
        # ---------- Manual input ----------
        st.subheader("Manual input")
        st.write("Time fields accept seconds (e.g., 72.45) or formats like 1:12.45 or 01:24:510.")
        n_rows = st.number_input("Rows", min_value=1, max_value=30, value=6, step=1)
        if "manual_df" not in st.session_state or st.session_state.get("manual_df_rows", 0) != n_rows:
            st.session_state["manual_df"] = _empty_manual_frame(n_rows)
            st.session_state["manual_df_rows"] = n_rows

        if show_example and st.session_state["manual_df"]["Horse"].eq("").all():
            ex = _empty_manual_frame(6)
            ex.loc[0, ["Horse","Race Time","800-400","400-Finish","200_Pos","400_Pos"]] = ["Runner A","01:24:510","25.2","23.8",2,3]
            ex.loc[1, ["Horse","Race Time","800-400","400-Finish","200_Pos","400_Pos"]] = ["Runner B","01:24:640","25.3","23.9",3,4]
            ex.loc[2, ["Horse","Race Time","800-400","400-Finish","200_Pos","400_Pos"]] = ["Runner C","01:24:680","25.7","24.0",6,6]
            ex.loc[3, ["Horse","Race Time","800-400","400-Finish","200_Pos","400_Pos"]] = ["Runner D","01:24:850","25.4","24.2",1,2]
            ex.loc[4, ["Horse","Race Time","800-400","400-Finish","200_Pos","400_Pos"]] = ["Runner E","01:25:180","26.0","24.3",8,7]
            ex.loc[5, ["Horse","Race Time","800-400","400-Finish","200_Pos","400_Pos"]] = ["Runner F","01:25:230","25.9","24.2",5,5]
            st.session_state["manual_df"] = ex

        manual_df = st.data_editor(
            st.session_state["manual_df"],
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Horse": st.column_config.TextColumn(required=True, help="Runner name"),
                "Finish_Pos": st.column_config.NumberColumn(format="%d", help="Optional (we infer from time if missing)"),
                "Race Time": st.column_config.TextColumn(help="Seconds or M:SS(.ms) or M:SS:ms"),
                "800-400": st.column_config.TextColumn(help="Seconds for 800→400 segment"),
                "400-Finish": st.column_config.TextColumn(help="Seconds for 400→Finish segment"),
                "200_Pos": st.column_config.NumberColumn(format="%.1f", help="Optional"),
                "400_Pos": st.column_config.NumberColumn(format="%.1f", help="Optional"),
                "800_Pos": st.column_config.NumberColumn(format="%.1f", help="Optional"),
                "1000_Pos": st.column_config.NumberColumn(format="%.1f", help="Optional"),
            },
            key="manual_editor",
        )

        # Clean blank rows (no Horse & no times)
        df_raw = manual_df.copy()
        df_raw["Horse"] = df_raw["Horse"].astype(str).str.strip()
        nonempty = (
            df_raw["Horse"].ne("") |
            df_raw["Race Time"].astype(str).str.strip().ne("") |
            df_raw["800-400"].astype(str).str.strip().ne("") |
            df_raw["400-Finish"].astype(str).str.strip().ne("")
        )
        df_raw = df_raw.loc[nonempty].reset_index(drop=True)

        if df_raw.empty:
            st.info("Enter at least one row (Horse + times) above.")
            st.stop()
        st.success("Manual data captured.")

except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw table preview")
st.dataframe(df_raw.head(12), use_container_width=True)
_dbg("Raw columns", list(df_raw.columns))
_dbg("Raw dtypes", df_raw.dtypes)

# --- Final normalization so the rest of the app works the same ---
df = df_raw.rename(columns={
    "Race time": "Race Time", "Race_Time": "Race Time", "RaceTime": "Race Time",
    "800_400": "800-400", "400_Finish": "400-Finish",
    "Horse Name": "Horse", "Finish": "Finish_Pos", "Placing": "Finish_Pos"
}).copy()

_required = ["Horse", "Race Time", "800-400", "400-Finish"]
_missing = [c for c in _required if c not in df.columns]
if _missing:
    st.error(
        "Missing required columns: " + ", ".join(_missing) +
        "\n\nRequired: Horse, Race Time, 800-400, 400-Finish\n"
        "Optional: Finish_Pos, 200_Pos, 400_Pos, 800_Pos, 1000_Pos"
    )
    _dbg("Workframe columns", list(df.columns))
    st.stop()

# Parse times safely
df["RaceTime_s"] = df["Race Time"].apply(parse_race_time)

# Safety net: fix any absurd values (>10 minutes) as M:SS:ms if possible
mask = df["RaceTime_s"] > 600
if mask.any():
    def _m_ss_ms_fallback(x):
        p = str(x).strip().split(":")
        if len(p) == 3 and p[2].isdigit() and len(p[2]) <= 3:
            return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 1000.0
        return np.nan
    df.loc[mask, "RaceTime_s"] = df.loc[mask, "Race Time"].apply(_m_ss_ms_fallback).fillna(df.loc[mask, "RaceTime_s"])

for col in ["800-400", "400-Finish"]:
    df[col] = pd.to_numeric(df[col].apply(parse_race_time), errors="coerce")

# Optional numeric fields
for col in ["200_Pos", "400_Pos", "800_Pos", "1000_Pos", "Finish_Pos", "Pos200", "Pos400", "Pos800", "Pos1000", "P200", "P400", "P800", "P1000"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# If Finish_Pos missing, infer by time (best-effort)
if ("Finish_Pos" not in df.columns) or df["Finish_Pos"].isna().all():
    if df["RaceTime_s"].notna().any():
        df["Finish_Pos"] = df["RaceTime_s"].rank(method="min")
    df["Finish_Pos"] = df["Finish_Pos"].astype("Int64")

st.subheader("Converted table (ready for analysis)")
st.dataframe(df.head(12), use_container_width=True)
_dbg("Normalized dtypes", df.dtypes)

# ===================
# Analysis Pipeline
# ===================
try:
    metrics = compute_metrics(df, distance_m=distance_m)
    metrics = flag_sleepers(metrics)
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

for drop_col in ["Jockey", "Trainer"]:
    if drop_col in metrics.columns:
        metrics = metrics.drop(columns=[drop_col])

winner_time = None
try:
    if "Finish_Pos" in metrics.columns and "RaceTime_s" in metrics.columns and not metrics["Finish_Pos"].isna().all():
        winner_time = metrics.loc[metrics["Finish_Pos"].idxmin(), "RaceTime_s"]
except Exception as e:
    if DEBUG: st.exception(e)

ctx = compute_pressure_context(metrics)
spi_median = ctx["spi_median"]

try:
    gci_scores, gci_reasons = [], []
    for _, r in metrics.iterrows():
        gci, why = compute_gci_v31(r, ctx, distance_m=distance_m, winner_time_s=winner_time)
        gci_scores.append(gci)
        gci_reasons.append("; ".join(why))
    metrics["GCI"] = gci_scores
    metrics["GCI_Reasons"] = gci_reasons
    metrics["Group_Candidate"] = metrics["GCI"] >= 7.0
except Exception as e:
    st.error("GCI computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# ===================
# Outputs
# ===================
try:
    st.subheader("Sectional Metrics")
    disp = round_display(metrics.copy()).sort_values("Finish_Pos", na_position="last")
    st.dataframe(disp, use_container_width=True)
except Exception as e:
    st.error("Displaying the metrics table failed.")
    if DEBUG: st.exception(e)

try:
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
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
    ax.set_ylabel("Speed (m/s)"); ax.set_title("Average vs Top 8 Pace Curves")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.subplots_adjust(bottom=0.22)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
    st.pyplot(fig)
except Exception as e:
    st.error("Plotting pace curves failed.")
    if DEBUG: st.exception(e)

try:
    st.subheader("Insights")
    if pd.isna(spi_median):
        st.write("Insufficient data to infer pace context.")
    else:
        st.write(
            f"**SPI median:** {spi_median:.1f}%  |  "
            f"**Early heat (0–1):** {ctx['early_heat']:.2f}  |  "
            f"**Pressure ratio (EPI≤3.0):** {ctx['pressure_ratio']:.2f}  |  "
            f"**Field size:** {ctx['field_size']}"
        )
except Exception as e:
    st.error("Rendering insights failed.")
    if DEBUG: st.exception(e)

try:
    st.subheader("Sleepers")
    sleepers = disp[disp["Sleeper"] == True][["Horse","Finish_Pos","Refined_FSP_%","Pos_Change"]] if 'disp' in locals() else pd.DataFrame()
    if sleepers.empty:
        st.write("No clear sleepers flagged under current thresholds.")
    else:
        st.dataframe(sleepers, use_container_width=True)
except Exception as e:
    st.error("Rendering sleepers table failed.")
    if DEBUG: st.exception(e)

try:
    st.subheader("Potential Group-class candidates (GCI v3.1)")
    cands = metrics.loc[metrics["Group_Candidate"]].copy().sort_values(["GCI","Finish_Pos"], ascending=[False, True])
    if cands.empty:
        st.write("No horses met the Group-class threshold today.")
    else:
        st.dataframe(
            round_display(cands[["Horse","Finish_Pos","RaceTime_s","Refined_FSP_%","Basic_FSP_%","SPI_%","Pos_Change","EPI","GCI","GCI_Reasons"]]),
            use_container_width=True
        )
except Exception as e:
    st.error("Rendering candidates table failed.")
    if DEBUG: st.exception(e)

try:
    st.subheader("Runner-by-runner summaries")
    ordered = metrics.sort_values("Finish_Pos", na_position="last")
    for _, row in ordered.iterrows():
        st.markdown(runner_summary(row, spi_median))
        st.markdown("---")
except Exception as e:
    st.error("Rendering runner summaries failed.")
    if DEBUG: st.exception(e)

try:
    st.subheader("Download")
    csv_bytes = disp.to_csv(index=False).encode("utf-8") if 'disp' in locals() else b""
    st.download_button("Download metrics as CSV", data=csv_bytes, file_name="race_sectional_metrics.csv", mime="text/csv", disabled=(csv_bytes==b""))
except Exception as e:
    st.error("Preparing download failed.")
    if DEBUG: st.exception(e)

st.caption(
    "Legend: Basic FSP = Final400 / Race Avg; Refined FSP = Final400 / Mid400; "
    "SPI = Mid400 / Race Avg; EPI = early positioning index (lower = closer to lead); "
    "Pos_Change = gain from 400m to finish. GCI v3.1 applies distance-normalised, pace-pressure context."
)
