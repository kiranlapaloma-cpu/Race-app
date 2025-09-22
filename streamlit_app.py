# streamlit_app.py
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================
# App config (no logo / no download button)
# =========================================
st.set_page_config(page_title="Race Edge — PI v2.4-B+ & GPI v0.95", layout="wide")

# ============
# Debug toggle
# ============
with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# =====================
# Helpers & small utils
# =====================
def parse_race_time(val):
    """Parse seconds or 'M:SS.ms' / 'MM:SS.ms'."""
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except Exception:
        pass
    s = str(val).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                m, sec = parts
                return int(m) * 60 + float(sec)
            if len(parts) == 3:
                h, m, sec = parts
                return int(h)*3600 + int(m)*60 + float(sec)
    except Exception:
        return np.nan
    return np.nan

def to_pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(x)
    if sigma <= 0 or np.isnan(sigma):
        sigma = 1.0
    return (x - med) / sigma

def winsorize(s: pd.Series, p_lo=5, p_hi=95) -> pd.Series:
    lo, hi = np.nanpercentile(s, [p_lo, p_hi])
    return s.clip(lo, hi)

def relu(x):
    return np.maximum(0.0, x)

def sec_per_length_for_distance(distance_m: float) -> float:
    if distance_m <= 1600:
        return 0.14
    if distance_m <= 2000:
        return 0.16
    return 0.17

# =========================
# Distance-aware weightings
# =========================
def pi_weights(distance_m: float):
    # v2.4-B+ weights (sum to 1)
    if distance_m <= 1200:
        return dict(F200=0.08, tsSPI=0.30, Accel=0.20, Grind=0.42)
    if distance_m <= 1600:
        return dict(F200=0.10, tsSPI=0.34, Accel=0.24, Grind=0.32)
    if distance_m <= 2000:
        return dict(F200=0.10, tsSPI=0.36, Accel=0.22, Grind=0.32)
    return dict(F200=0.08, tsSPI=0.38, Accel=0.18, Grind=0.36)

# ==================================
# Manual input grid helpers (200m UI)
# ==================================
def round_up_200(x):
    x = int(x)
    return x if x % 200 == 0 else x + (200 - x % 200)

def build_manual_template(distance_m: int, n_horses: int) -> pd.DataFrame:
    """Build a grid with per-horse columns and descending distance rows."""
    rows = list(range(distance_m, 0, -200))
    # ensure we end at 0 (if not exact multiple)
    if rows[-1] != 0:
        rows.append(0)
    base = pd.DataFrame({"Marker_m": rows})
    # For each horse, we want first-200 split (0–100 & 100–200), mid windows, last 100/200, etc.
    # But to keep consistent with your CSV schema, we capture per-100m times if user has them;
    # in manual mode we’ll ask for 200m segment times from the end (Finish_200) and first 200 (F200_100 + F200_200).
    # For simplicity, collect positions at each marker too.
    for i in range(1, n_horses+1):
        base[f"H{i}_Pos"] = np.nan
        base[f"H{i}_Time"] = np.nan  # segment time between this marker and next (200m step)
    return base

def manual_compute_longform_to_wide(man_df: pd.DataFrame, horse_names: list, distance_m: int) -> pd.DataFrame:
    """
    Convert the manual long grid into a wide race table aligned with the app schema expectations.
    Assumptions:
      - Marker_m goes e.g., 1400,1200,...,0
      - For each Hk_Time at row r, that's the segment time from Marker_m[r] to next lower marker (200m).
      - Positions Hk_Pos are position-at-marker (lower = closer to lead). We’ll pick needed markers.
    """
    df = man_df.copy()
    # Build per-horse per-section dictionaries
    out_rows = []
    for idx, name in enumerate(horse_names, start=1):
        hpos = df[f"H{idx}_Pos"].values
        hseg = df[f"H{idx}_Time"].values  # 200m segment times between row and next row

        # Collect 100m splits are not available; we will approximate:
        # First 200m = sum of first segment (distance_m → distance_m-200)
        first200_time = np.nan
        if len(hseg) >= 1:
            first200_time = hseg[0]

        # Last 200m = segment from 200→0 (which is the last non-null if grid is exact)
        last200_time = np.nan
        # find last defined segment
        last_defined = None
        for t in hseg:
            if not pd.isna(t):
                last_defined = t
        last200_time = last_defined

        # Mid-400 (800→400) time (needs four 100m or two 200m chunks).
        # With 200m manual, mid-400 is the two segments centered near mid. We derive indexes:
        # From markers: distance_m, distance_m-200, ..., 400, 200, 0
        # The segments are: [distance_m→distance_m-200], ..., [800→600], [600→400], [400→200], [200→0]
        # Mid-400 = (800→600) + (600→400) = segments at indices where the upper marker equals 800 and 600 respectively.
        def seg_time_for_span(start_m, end_m):
            # sum of segments covering [start_m → end_m] in steps of 200 downward
            if start_m <= end_m: 
                return np.nan
            # index of row with Marker_m == start_m
            try:
                start_idx = int(np.where(df["Marker_m"].values == start_m)[0][0])
            except Exception:
                return np.nan
            # number of segments = (start_m - end_m) / 200
            count = int((start_m - end_m) // 200)
            times = []
            for k in range(count):
                if start_idx + k < len(hseg):
                    times.append(hseg[start_idx + k])
            if len(times) != count or any(pd.isna(t) for t in times):
                return np.nan
            return float(np.nansum(times))

        mid400_time = seg_time_for_span(800, 400)  # works for trips ≥ 800m
        fin400_time = seg_time_for_span(400, 0)

        # Race time = sum of all defined segments
        race_time = float(np.nansum([t for t in hseg if not pd.isna(t)])) if np.isfinite(np.nansum(hseg)) else np.nan

        # Positions at 400, 200 (if present)
        def pos_at(marker):
            try:
                return float(df.loc[df["Marker_m"] == marker, f"H{idx}_Pos"].values[0])
            except Exception:
                return np.nan

        pos_400 = pos_at(400)
        pos_200 = pos_at(200)
        finish_pos = pos_at(0)

        out_rows.append(dict(
            Horse=name,
            Finish_Pos=finish_pos,
            RaceTime_s=race_time,
            # Construct fields consistent with uploaded schema expectations:
            **{
              "800-400": mid400_time if pd.notna(mid400_time) else np.nan,
              "400-Finish": fin400_time if pd.notna(fin400_time) else np.nan,
              "200_Time": np.nan,   # not available at 100m granularity (we’ll derive F200 differently)
              "100_Time": np.nan,
              "300_Time": np.nan,
              "Finish_Time": np.nan,  # no 100m
              "200_Pos": pos_200,
              "400_Pos": pos_400
            }
        ))
    return pd.DataFrame(out_rows)

# =========================================
# Core metric builders (sectionals, PI, GPI)
# =========================================
def build_sectionals(df: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """
    Compute F200, tsSPI, Accel, Grind using available columns.
    Supports:
      • 100m-split CSVs (expect 100_Time, 200_Time, 300_Time, Finish_Time)
      • 200m manual (no 100m splits) — will approximate:
            F200 from first 200 segment,
            Grind from last 200 segment (instead of last 100),
            Accel left NaN (no 300→200 detail),
    """
    out = df.copy()

    # race avg speed
    if "RaceTime_s" not in out.columns and "Race Time" in out.columns:
        out["RaceTime_s"] = out["Race Time"].apply(parse_race_time)
    # guard
    out["Race_AvgSpeed"] = distance_m / out["RaceTime_s"]

    # tsSPI uses "800-400" & "400-Finish" like your app
    out["Mid400_Speed"] = np.where(out["800-400"].notna(), 400.0 / out["800-400"], np.nan)
    out["Final400_Speed"] = np.where(out["400-Finish"].notna(), 400.0 / out["400-Finish"], np.nan)
    out["tsSPI"] = out["Mid400_Speed"] / out["Race_AvgSpeed"]

    # F200, Accel, Grind:
    has_100m = all(c in out.columns for c in ["100_Time","200_Time","300_Time","Finish_Time"])
    if has_100m:
        # First 200 uses 100_Time + 200_Time
        out["F200"] = (200.0 / (out["200_Time"] + out["100_Time"])) / out["Race_AvgSpeed"]
        # Accel = speed(200–100)/speed(300–200)
        sp_200_100 = 100.0 / out["200_Time"]
        sp_300_200 = 100.0 / out["300_Time"]
        out["Accel"] = sp_200_100 / sp_300_200
        # Grind: last 100 vs race avg (consistent with earlier trials)
        out["Grind"] = (100.0 / out["Finish_Time"]) / out["Race_AvgSpeed"]
    else:
        # Manual 200m approximation
        # F200: use first 200m segment (derive from constructed manual)
        # We stored no explicit seg time column; manual converter inserted only aggregate windows.
        # So in manual mode, we’ll estimate F200 from RaceTime_s proportion if first 200 present.
        # Prefer: if we have a separate column 'First200_s' (not present here), fallback:
        out["F200"] = np.nan
        out["Accel"] = np.nan  # no 300→200 lens
        # Grind: last 200 segment vs race avg (closest available)
        out["Grind"] = np.where(out["400-Finish"].notna(), (200.0 / (out["400-Finish"] / 2.0)) / out["Race_AvgSpeed"], np.nan)

    # Replace inf
    out = out.replace([np.inf, -np.inf], np.nan)

    return out

def compute_PI(df_sec: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """PI v2.4-B+: winsorize → weighted sum → robust-z + logistic to 0–10."""
    out = df_sec.copy()
    for col in ["F200","tsSPI","Accel","Grind"]:
        if col in out.columns:
            out[col] = winsorize(out[col])

    w = pi_weights(distance_m)
    # raw composite per horse (renormalize if some parts missing)
    vals = []
    for _, r in out.iterrows():
        parts, ws = [], []
        for k, ww in w.items():
            v = r.get(k, np.nan)
            if not pd.isna(v):
                parts.append(v * ww)
                ws.append(ww)
        vals.append(np.sum(parts)/np.sum(ws) if ws else np.nan)
    out["PI_raw"] = vals

    # robust z + logistic
    z = robust_z(out["PI_raw"])
    k = 0.9
    out["PI"] = 10.0 * (1.0 / (1.0 + np.exp(-k * z)))
    return out

def compute_GPI(df_sec_pi: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """GPI v0.95 — balance + separation + bonuses + anchors → 0–10."""
    out = df_sec_pi.copy()

    # Percentiles of components (within race)
    out["p_F200"]  = to_pct_rank(out["F200"])
    out["p_tsSPI"] = to_pct_rank(out["tsSPI"])
    out["p_Accel"] = to_pct_rank(out["Accel"])
    out["p_Grind"] = to_pct_rank(out["Grind"])
    out["p_PI"]    = to_pct_rank(out["PI"])

    # Balance (breadth) with soft spread penalty
    pF = out["p_F200"]; pM = out["p_tsSPI"]; pA = out["p_Accel"]; pG = out["p_Grind"]
    min4 = pd.concat([pF, pM, pA, pG], axis=1).min(axis=1)
    max4 = pd.concat([pF, pM, pA, pG], axis=1).max(axis=1)
    spread = max4 - min4
    BAL_star = relu(min4 - 0.50) / 0.50
    PEN = 1.0 - relu(spread - 0.25) / 0.25
    BALp = BAL_star * PEN

    # Separation (tsSPI vs best of Accel/Grind)
    S1 = 2.0 * relu(pM - 0.50)
    S2 = 2.0 * relu(pd.concat([pA, pG], axis=1).max(axis=1) - 0.50)
    SEPP = (S1 + S2) / 2.0

    # Dual-weapon
    DWB = np.where((pA >= 0.75) & (pG >= 0.75), 0.10,
          np.where((pA >= 0.65) & (pG >= 0.65), 0.05, 0.0))

    # Stress: above race medians on all four
    meds = pd.Series([pF.median(), pM.median(), pA.median(), pG.median()], index=["pF","pM","pA","pG"])
    STRESS = np.where((pF >= meds["pF"]) & (pM >= meds["pM"]) & (pA >= meds["pA"]) & (pG >= meds["pG"]), 0.05, 0.0)

    # Anchors (distance-aware)
    spl = sec_per_length_for_distance(distance_m)
    # winner time from Finish_Pos (lowest = winner)
    winner_time = out.loc[out["Finish_Pos"].astype("Int64").idxmin(), "RaceTime_s"] if "RaceTime_s" in out.columns else np.nan
    L_behind = (out["RaceTime_s"] - winner_time) / spl if pd.notna(winner_time) else np.nan
    L_max = np.nanmax(L_behind.values) if isinstance(L_behind, pd.Series) else 0.0

    WINB = np.where(out["Finish_Pos"] == 1, np.minimum(0.10, 0.02*np.log1p(max(L_max, 0.0))), 0.0)
    PLCB = np.where((out["Finish_Pos"] > 1) & (L_behind <= 1.0), np.maximum(0.0, 0.03 - 0.02*L_behind), 0.0)

    # Field size guard
    N = out.shape[0]
    guard = 0.8 if N <= 6 else 1.0
    BALp = BALp * guard
    SEPP = SEPP * guard
    DWB  = DWB * guard
    STRESS = STRESS * guard

    # One-trick penalty
    GPI01 = (0.15*out["p_PI"] + 0.35*BALp + 0.30*SEPP + 0.10*DWB + 0.10*STRESS + WINB + PLCB)
    one_trick = (spread > 0.35).astype(float) * 0.03
    GPI01 = GPI01 - one_trick
    out["GPI"] = (np.clip(GPI01, 0, 1) * 10.0).round(3)

    return out

# ==================
# UI: Inputs & Flow
# ==================
st.title("🏇 Race Edge — PI v2.4-B+ & GPI v0.95")
st.caption("Upload CSV/XLSX (100m splits preferred) or use Manual mode (200m grid). The app computes sectionals, PI (0–10), and GPI (0–10).")

with st.sidebar:
    mode = st.radio("Data Source", ["Upload file", "Manual input"], index=0)
    dist_in = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=10)
    # round-up policy applies only to manual grid row count
    rounded_dist = round_up_200(dist_in) if mode == "Manual input" else int(dist_in)

df_input = None

try:
    if mode == "Upload file":
        uploaded = st.file_uploader("Upload CSV/XLSX (any jurisdiction)", type=["csv", "xlsx"])
        if uploaded is None:
            st.info("Upload a file or switch to Manual input.")
            st.stop()
        if uploaded.name.lower().endswith(".csv"):
            df_input = pd.read_csv(uploaded)
        else:
            df_input = pd.read_excel(uploaded)
        st.success("File loaded.")

    else:
        st.subheader("Manual Input (200m grid)")
        n_horses = st.sidebar.number_input("Number of horses", min_value=2, max_value=24, value=8, step=1)
        st.sidebar.caption(f"Grid uses distance rows from {rounded_dist} → 0 in 200m steps.")
        # Horse names
        st.markdown("Enter horse names (one per line):")
        names_text = st.text_area("Horse names", value="\n".join([f"Horse {i}" for i in range(1, n_horses+1)]), height=150)
        horse_names = [n.strip() for n in names_text.splitlines() if n.strip()]
        if len(horse_names) != n_horses:
            st.warning("Please provide exactly the same number of horse names as 'Number of horses'.")
            st.stop()

        grid = build_manual_template(rounded_dist, n_horses)
        st.markdown("Fill **position** (1=leader) and **segment time (s)** for each 200m segment downward.")
        st.dataframe(grid, use_container_width=True)
        csv = st.text_area("Paste your filled grid here (CSV with same columns) to run analysis.", height=120)
        if not csv.strip():
            st.stop()
        try:
            man_df = pd.read_csv(io.StringIO(csv))
        except Exception:
            st.error("Could not parse the pasted CSV grid. Ensure headers and numeric values are valid.")
            st.stop()
        # convert manual grid to wide table
        df_input = manual_compute_longform_to_wide(man_df, horse_names, rounded_dist)
        # Inject minimal columns needed downstream to mimic uploaded schema
        df_input["Race Time"] = df_input["RaceTime_s"]
        # Ensure Finish_Pos numeric Int
        if "Finish_Pos" in df_input.columns:
            df_input["Finish_Pos"] = pd.to_numeric(df_input["Finish_Pos"], errors="coerce").astype("Int64")

except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw table preview")
st.dataframe(df_input.head(12), use_container_width=True)
_dbg("Raw columns", list(df_input.columns))

# ===================
# Analysis Pipeline
# ===================
distance_m = float(dist_in)
try:
    work = df_input.copy()
    if "RaceTime_s" not in work.columns:
        if "Race Time" in work.columns:
            work["RaceTime_s"] = work["Race Time"].apply(parse_race_time)
        else:
            # If only manual
            pass

    # Ensure numeric for windows if present
    for c in ["800-400","400-Finish","100_Time","200_Time","300_Time","Finish_Time","Finish_Pos"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # If Finish_Pos missing, approximate by RaceTime_s
    if ("Finish_Pos" not in work.columns) or work["Finish_Pos"].isna().all():
        if "RaceTime_s" in work.columns and work["RaceTime_s"].notna().any():
            work["Finish_Pos"] = work["RaceTime_s"].rank(method="min").astype("Int64")

    # Build sectionals (F200, tsSPI, Accel, Grind)
    sec = build_sectionals(work, distance_m)
    # PI v2.4-B+
    pi_df = compute_PI(sec, distance_m)
    # GPI v0.95
    gpi_df = compute_GPI(pi_df, distance_m)

except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# ===================
# Outputs
# ===================
st.subheader("Sectional Metrics (PI & GPI)")
disp_cols = ["Horse","Finish_Pos","RaceTime_s","F200","tsSPI","Accel","Grind","PI","GPI"]
present = [c for c in disp_cols if c in gpi_df.columns]
disp = gpi_df[present].copy()

# Present sectionals as percentages where appropriate
for c in ["F200","tsSPI","Accel","Grind"]:
    if c in disp.columns:
        disp[c] = (disp[c] * 100.0).round(2)

disp["PI"] = disp["PI"].round(3)
if "GPI" in disp.columns:
    disp["GPI"] = disp["GPI"].round(3)

disp = disp.sort_values(["PI","Finish_Pos"], ascending=[False, True])
st.dataframe(disp, use_container_width=True)

# =========
# Visuals
# =========
# 1) Sectional Shape Map (Kick vs Grind scatter with arrows/labels kept simple)
st.subheader("Sectional Shape Map — Kick (Accel) vs Grind")
fig, ax = plt.subplots()
x = gpi_df["Accel"]*100.0
y = gpi_df["Grind"]*100.0
ax.scatter(x, y, s=60, alpha=0.9)
# Median lines
if x.notna().any():
    ax.axvline(x.median(), linestyle="--", alpha=0.4)
if y.notna().any():
    ax.axhline(y.median(), linestyle="--", alpha=0.4)
# Annotate with arrows (simple offset)
for _, r in gpi_df.iterrows():
    if pd.isna(r.get("Accel")) or pd.isna(r.get("Grind")): 
        continue
    ax.annotate(str(r.get("Horse","")),
                xy=(r["Accel"]*100.0, r["Grind"]*100.0),
                xytext=(5, 5), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5))
ax.set_xlabel("Kick — Acceleration (%)")
ax.set_ylabel("Grind — Late (%)")
ax.set_title("Kick vs Grind")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# 2) Pace Curves — Average (black) + Top 8 by finish
st.subheader("Pace Curves — Field Average (black) + Top 8 by Finish")
avg_mid = gpi_df["Mid400_Speed"].mean()
avg_fin = gpi_df["Final400_Speed"].mean()
top8 = gpi_df.sort_values("Finish_Pos").head(8)

fig2, ax2 = plt.subplots()
x_vals = [1, 2]
ax2.plot(x_vals, [avg_mid, avg_fin], marker="o", linewidth=3, color="black", label="Average (Field)")
for _, row in top8.iterrows():
    ax2.plot(x_vals, [row["Mid400_Speed"], row["Final400_Speed"]], marker="o", linewidth=2, label=str(row.get("Horse", "Runner")))
ax2.set_xticks([1, 2]); ax2.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
ax2.set_ylabel("Speed (m/s)"); ax2.set_title("Average vs Top 8 Pace Curves")
ax2.grid(True, linestyle="--", alpha=0.3)
fig2.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
st.pyplot(fig2)

# 3) Interactive Radar (one-horse-at-a-time — keep simple with a selectbox)
st.subheader("Per-Horse Radar (F200 / tsSPI / Kick / Grind)")
radar_df = gpi_df[["Horse","F200","tsSPI","Accel","Grind"]].dropna()
if not radar_df.empty:
    pick = st.selectbox("Pick a horse", radar_df["Horse"].tolist())
    row = radar_df[radar_df["Horse"]==pick].iloc[0]
    labels = ["F200","tsSPI","Kick","Grind"]
    values = [float(row["F200"])*100.0, float(row["tsSPI"])*100.0, float(row["Accel"])*100.0, float(row["Grind"])*100.0]
    # close the polygon
    L = labels + [labels[0]]
    V = values + [values[0]]
    ang = np.linspace(0, 2*np.pi, len(L), endpoint=False)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, polar=True)
    ax3.plot(ang, V, linewidth=2)
    ax3.fill(ang, V, alpha=0.15)
    ax3.set_xticks(ang[:-1])
    ax3.set_xticklabels(labels)
    ax3.set_title(f"Sectional Radar — {pick}")
    st.pyplot(fig3)
else:
    st.info("Radar requires non-missing F200, tsSPI, Accel, Grind.")

# 4) Top-8 PI bar chart
st.subheader("Top 8 by PI (0–10)")
bar8 = gpi_df.sort_values("PI", ascending=False).head(8)[["Horse","PI"]]
fig4, ax4 = plt.subplots()
ax4.barh(bar8["Horse"].iloc[::-1], bar8["PI"].iloc[::-1])
ax4.set_xlabel("PI (0–10)")
ax4.set_title("Top 8 — Performance Index")
st.pyplot(fig4)

st.caption(
    "Notes: PI v2.4-B+ uses winsorized sectionals (F200 / tsSPI / Accel / Grind) with distance-aware weights, "
    "then robust z + logistic scaling to 0–10. GPI v0.95 blends balance, separation, dual-weapon and sanity anchors "
    "to estimate group potential on a 0–10 scale."
)
