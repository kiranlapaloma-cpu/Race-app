# streamlit_app.py
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# App config (no favicon)
# =========================
st.set_page_config(page_title="Race Edge — PI v2.4-B+ & GPI v0.95", layout="wide")

# =========================
# Sidebar / Debug
# =========================
with st.sidebar:
    DEBUG = st.checkbox("Debug mode", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# =========================
# Parsing & normalization
# =========================
def parse_race_time(val):
    """
    Robust parser for race times. Accepts:
      • seconds:         73.30
      • M:SS.ms:         1:49.540   (minutes : seconds.milliseconds)
      • M:SS:ms|cs:      1:49:540   (minutes : seconds : milli/centi/tenths)
      • vendor numerics: 7330 (centiseconds), 73300 (milliseconds)
    Returns seconds (float).
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()

    # Clock-like formats
    if ":" in s:
        parts = s.split(":")
        # M:SS.ms  (e.g. "1:49.540")
        if len(parts) == 2:
            m, sec = parts
            try:
                return int(m) * 60 + float(sec)
            except Exception:
                return np.nan

        # M:SS:ms|cs  (e.g. "1:49:540" → 1min 49.540s)
        if len(parts) == 3:
            m_str, ss_str, frac_str = parts
            try:
                m = int(m_str)
                ss = int(ss_str)
                if len(frac_str) >= 3:
                    frac = int(frac_str) / 1000.0   # milliseconds
                elif len(frac_str) == 2:
                    frac = int(frac_str) / 100.0    # centiseconds
                else:
                    frac = int(frac_str) / 10.0     # tenths
                return m * 60 + ss + frac
            except Exception:
                return np.nan

    # Pure numeric (seconds / centiseconds / milliseconds)
    try:
        x = float(s)
    except Exception:
        return np.nan

    if 5000 <= x < 20000:        # centiseconds
        return x / 100.0
    if 50000 <= x < 200000:      # milliseconds
        return x / 1000.0
    return x

def normalize_time_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Heuristic conversion for vendor columns carrying centiseconds/milliseconds.
    If a column median is too large for seconds, downscale to seconds.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median()
        # typical seconds medians for 100–400m windows are ~6–27
        if 100 <= med < 20000:        # likely centiseconds
            out[c] = out[c] / 100.0
        elif 10000 <= med < 200000:   # likely milliseconds
            out[c] = out[c] / 1000.0
    return out

def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(x)
    if sigma <= 0 or np.isnan(sigma):
        sigma = 1.0
    return (x - med) / sigma

def winsorize(s: pd.Series, p_lo=5, p_hi=95) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    lo, hi = np.nanpercentile(s, [p_lo, p_hi])
    return s.clip(lo, hi)

def to_pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def relu(x):
    return np.maximum(0.0, x)

def sec_per_length_for_distance(distance_m: float) -> float:
    if distance_m <= 1600:
        return 0.14
    if distance_m <= 2000:
        return 0.16
    return 0.17

# =========================
# PI weights (distance-aware)
# =========================
def pi_weights(distance_m: float):
    if distance_m <= 1200:
        return dict(F200=0.08, tsSPI=0.30, Accel=0.20, Grind=0.42)
    if distance_m <= 1600:
        return dict(F200=0.10, tsSPI=0.34, Accel=0.24, Grind=0.32)
    if distance_m <= 2000:
        return dict(F200=0.10, tsSPI=0.36, Accel=0.22, Grind=0.32)
    return dict(F200=0.08, tsSPI=0.38, Accel=0.18, Grind=0.36)

# =========================
# Manual input scaffolding
# =========================
def round_up_200(x):
    x = int(x)
    return x if x % 200 == 0 else x + (200 - x % 200)

def build_manual_template(distance_m: int, n_horses: int) -> pd.DataFrame:
    rows = list(range(distance_m, 0, -200))
    if rows[-1] != 0:
        rows.append(0)
    base = pd.DataFrame({"Marker_m": rows})
    for i in range(1, n_horses+1):
        base[f"H{i}_Pos"] = np.nan
        base[f"H{i}_Time"] = np.nan  # segment time (s) per 200m
    return base

def manual_compute_longform_to_wide(man_df: pd.DataFrame, horse_names: list, distance_m: int) -> pd.DataFrame:
    """
    Convert manual 200m grid into wide schema.
    """
    df = man_df.copy()
    out_rows = []
    for idx, name in enumerate(horse_names, start=1):
        hpos = pd.to_numeric(df.get(f"H{idx}_Pos"), errors="coerce")
        hseg = pd.to_numeric(df.get(f"H{idx}_Time"), errors="coerce")

        race_time = float(np.nansum([t for t in hseg if not pd.isna(t)])) if len(hseg) else np.nan

        def seg_time_for_span(start_m, end_m):
            try:
                start_idx = int(np.where(df["Marker_m"].values == start_m)[0][0])
            except Exception:
                return np.nan
            cnt = int((start_m - end_m) // 200)
            ts = []
            for k in range(cnt):
                if start_idx + k < len(hseg):
                    ts.append(hseg[start_idx + k])
            if len(ts) != cnt or any(pd.isna(t) for t in ts):
                return np.nan
            return float(np.nansum(ts))

        mid400_time = seg_time_for_span(800, 400)
        fin400_time = seg_time_for_span(400, 0)

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
            **{
                "800-400": mid400_time if pd.notna(mid400_time) else np.nan,
                "400-Finish": fin400_time if pd.notna(fin400_time) else np.nan,
                "200_Pos": pos_200,
                "400_Pos": pos_400,
                # manual has no 100m splits
                "100_Time": np.nan, "200_Time": np.nan, "300_Time": np.nan, "Finish_Time": np.nan
            }
        ))
    return pd.DataFrame(out_rows)

# =========================
# Sectionals → PI → GPI
# =========================
def build_sectionals(df: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """
    Compute F200, tsSPI, Accel, Grind.
    Works with 100m-split CSVs and manual 200m derived windows.
    """
    out = df.copy()

    # Race time (seconds) — always prefer "Race Time" string when present
    if "Race Time" in out.columns:
        out["RaceTime_s"] = out["Race Time"].apply(parse_race_time)
    elif "RaceTime_s" in out.columns:
        out["RaceTime_s"] = pd.to_numeric(out["RaceTime_s"], errors="coerce")
    else:
        # last resort: sum any *_Time columns if they exist
        tcols = [c for c in out.columns if c.endswith("_Time") and c.split("_")[0].isdigit()]
        if tcols:
            out["RaceTime_s"] = out[tcols].sum(axis=1)

    # Belt-and-braces: downscale vendor numerics if they sneaked in
    med_rt = pd.to_numeric(out["RaceTime_s"], errors="coerce").median()
    if 5000 <= med_rt < 20000:
        out["RaceTime_s"] = out["RaceTime_s"] / 100.0
    elif 50000 <= med_rt < 200000:
        out["RaceTime_s"] = out["RaceTime_s"] / 1000.0

    out["Race_AvgSpeed"] = distance_m / out["RaceTime_s"]

    # Normalize window units (vendor centiseconds/milliseconds → seconds)
    out = normalize_time_columns(out, ["800-400", "400-Finish", "100_Time", "200_Time", "300_Time", "Finish_Time"])

    # tsSPI (mid-400 vs race average)
    out["Mid400_Speed"] = np.where(out["800-400"].notna(), 400.0 / out["800-400"], np.nan)
    out["Final400_Speed"] = np.where(out["400-Finish"].notna(), 400.0 / out["400-Finish"], np.nan)
    out["tsSPI"] = out["Mid400_Speed"] / out["Race_AvgSpeed"]

    # F200 from the START (dynamic *_Time columns)
    def _first200_cols(distance_m: float, cols: list[str]):
        a = f"{int(distance_m)-100}_Time"
        b = f"{int(distance_m)-200}_Time"
        if a in cols and b in cols:
            return a, b
        # fallback: two largest *_Time numeric headers
        nums = sorted([int(c.split("_")[0]) for c in cols if c.endswith("_Time") and c.split("_")[0].isdigit()], reverse=True)
        if len(nums) >= 2:
            return f"{nums[0]}_Time", f"{nums[1]}_Time"
        return None, None

    cols = out.columns.tolist()
    c1, c2 = _first200_cols(distance_m, cols)
    if c1 and c2:
        out["F200"] = (200.0 / (out[c1] + out[c2])) / out["Race_AvgSpeed"]
    else:
        out["F200"] = np.nan  # manual without explicit first 200

    # Accel (kick): 200–100 vs 300–200 near the finish if available
    if {"200_Time", "300_Time"}.issubset(cols):
        sp_200_100 = 100.0 / out["200_Time"]
        sp_300_200 = 100.0 / out["300_Time"]
        out["Accel"] = sp_200_100 / sp_300_200
    else:
        out["Accel"] = np.nan

    # Grind: last 100 vs race average; else approximate from last 200
    if "Finish_Time" in cols:
        out["Grind"] = (100.0 / out["Finish_Time"]) / out["Race_AvgSpeed"]
    else:
        out["Grind"] = np.where(out["400-Finish"].notna(),
                                (200.0 / (out["400-Finish"] / 2.0)) / out["Race_AvgSpeed"], np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def pi_weights(distance_m: float):
    if distance_m <= 1200:
        return dict(F200=0.08, tsSPI=0.30, Accel=0.20, Grind=0.42)
    if distance_m <= 1600:
        return dict(F200=0.10, tsSPI=0.34, Accel=0.24, Grind=0.32)
    if distance_m <= 2000:
        return dict(F200=0.10, tsSPI=0.36, Accel=0.22, Grind=0.32)
    return dict(F200=0.08, tsSPI=0.38, Accel=0.18, Grind=0.36)

def compute_PI(df_sec: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """PI v2.4-B+: winsorize → weighted sum → robust-z + logistic to 0–10."""
    out = df_sec.copy()
    for col in ["F200", "tsSPI", "Accel", "Grind"]:
        if col in out.columns:
            out[col] = winsorize(out[col])

    w = pi_weights(distance_m)

    vals = []
    for _, r in out.iterrows():
        parts, ws = [], []
        for k, ww in w.items():
            v = r.get(k, np.nan)
            if not pd.isna(v):
                parts.append(v * ww)
                ws.append(ww)
        vals.append(np.sum(parts) / np.sum(ws) if ws else np.nan)
    out["PI_raw"] = vals

    z = robust_z(out["PI_raw"])
    k = 0.9
    out["PI"] = 10.0 * (1.0 / (1.0 + np.exp(-k * z)))
    return out

def compute_GPI(df_sec_pi: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """
    GPI v0.95 — “Group potential” from sectional balance & separation, with anchors.
    Output 0–10 added to table only (no visuals).
    """
    out = df_sec_pi.copy()

    out["p_F200"]  = to_pct_rank(out["F200"])
    out["p_tsSPI"] = to_pct_rank(out["tsSPI"])
    out["p_Accel"] = to_pct_rank(out["Accel"])
    out["p_Grind"] = to_pct_rank(out["Grind"])
    out["p_PI"]    = to_pct_rank(out["PI"])

    pF, pM, pA, pG = out["p_F200"], out["p_tsSPI"], out["p_Accel"], out["p_Grind"]
    min4 = pd.concat([pF, pM, pA, pG], axis=1).min(axis=1)
    max4 = pd.concat([pF, pM, pA, pG], axis=1).max(axis=1)
    spread = max4 - min4

    BAL_star = relu(min4 - 0.50) / 0.50
    PEN = 1.0 - relu(spread - 0.25) / 0.25
    BALp = BAL_star * PEN

    S1 = 2.0 * relu(out["p_tsSPI"] - 0.50)
    S2 = 2.0 * relu(pd.concat([out["p_Accel"], out["p_Grind"]], axis=1).max(axis=1) - 0.50)
    SEPP = (S1 + S2) / 2.0

    DWB = np.where((out["p_Accel"] >= 0.75) & (out["p_Grind"] >= 0.75), 0.10,
          np.where((out["p_Accel"] >= 0.65) & (out["p_Grind"] >= 0.65), 0.05, 0.0))

    meds = {
        "pF": pF.median(), "pM": pM.median(), "pA": pA.median(), "pG": pG.median()
    }
    STRESS = np.where((pF >= meds["pF"]) & (pM >= meds["pM"]) & (pA >= meds["pA"]) & (pG >= meds["pG"]), 0.05, 0.0)

    spl = sec_per_length_for_distance(distance_m)
    if "Finish_Pos" in out.columns and out["Finish_Pos"].notna().any():
        try:
            idx_w = out["Finish_Pos"].astype("Int64").idxmin()
            winner_time = out.loc[idx_w, "RaceTime_s"]
        except Exception:
            winner_time = np.nan
    else:
        winner_time = np.nan

    L_behind = (out["RaceTime_s"] - winner_time) / spl if pd.notna(winner_time) else np.nan
    L_max = np.nanmax(L_behind.values) if isinstance(L_behind, pd.Series) else 0.0

    WINB = np.where(out["Finish_Pos"] == 1, np.minimum(0.10, 0.02 * np.log1p(max(L_max, 0.0))), 0.0)
    PLCB = np.where((out["Finish_Pos"] > 1) & (L_behind <= 1.0), np.maximum(0.0, 0.03 - 0.02 * L_behind), 0.0)

    N = out.shape[0]
    guard = 0.8 if N <= 6 else 1.0
    BALp *= guard; SEPP *= guard; DWB *= guard; STRESS *= guard

    GPI01 = (0.15 * out["p_PI"] + 0.35 * BALp + 0.30 * SEPP + 0.10 * DWB + 0.10 * STRESS + WINB + PLCB)
    one_trick = (spread > 0.35).astype(float) * 0.03
    GPI01 = GPI01 - one_trick

    out["GPI"] = (np.clip(GPI01, 0, 1) * 10.0).round(3)
    return out

# =========================
# UI & ingest
# =========================
st.title("🏇 Race Edge — PI v2.4-B+ & GPI v0.95")
st.caption("Upload CSV/XLSX (100m splits preferred) or use Manual mode (200m grid). Calculates sectionals, PI (0–10), and GPI (0–10).")

with st.sidebar:
    mode = st.radio("Data Source", ["Upload file", "Manual input"], index=0)
    dist_in = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=10)
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
        df_input = manual_compute_longform_to_wide(man_df, horse_names, rounded_dist)
        df_input["Race Time"] = df_input["RaceTime_s"]  # convenience for downstream

except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw / Converted Table")
st.dataframe(df_input.head(12), use_container_width=True)
_dbg("Raw columns", list(df_input.columns))

# =========================
# Analysis pipeline
# =========================
distance_m = float(dist_in)
try:
    work = df_input.copy()

    # numeric casting for windows if present
    for c in ["800-400", "400-Finish", "100_Time", "200_Time", "300_Time", "Finish_Time", "Finish_Pos"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # If Finish_Pos missing, approximate by time
    if ("Finish_Pos" not in work.columns) or work["Finish_Pos"].isna().all():
        if "Race Time" in work.columns:
            rt = work["Race Time"].apply(parse_race_time)
            work["Finish_Pos"] = rt.rank(method="min").astype("Int64")
        elif "RaceTime_s" in work.columns:
            work["Finish_Pos"] = work["RaceTime_s"].rank(method="min").astype("Int64")

    # Sectionals
    sec = build_sectionals(work, distance_m)
    # PI
    pi_df = compute_PI(sec, distance_m)
    # GPI
    gpi_df = compute_GPI(pi_df, distance_m)

except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# =========================
# Outputs
# =========================
st.subheader("Sectional Metrics (PI & GPI)")
disp_cols = ["Horse", "Finish_Pos", "RaceTime_s", "F200", "tsSPI", "Accel", "Grind", "PI", "GPI"]
present = [c for c in disp_cols if c in gpi_df.columns]
disp = gpi_df[present].copy()

# display sectionals as %
for c in ["F200", "tsSPI", "Accel", "Grind"]:
    if c in disp.columns:
        disp[c] = (disp[c] * 100.0).round(2)
if "PI" in disp.columns:
    disp["PI"] = disp["PI"].round(3)
if "GPI" in disp.columns:
    disp["GPI"] = disp["GPI"].round(3)

disp = disp.sort_values(["PI", "Finish_Pos"], ascending=[False, True])
st.dataframe(disp, use_container_width=True)

# ==============
# Visuals
# ==============
# 1) Shape map — Kick vs Grind
st.subheader("Sectional Shape Map — Kick (Accel) vs Grind")
fig, ax = plt.subplots()
x = gpi_df["Accel"] * 100.0
y = gpi_df["Grind"] * 100.0
ax.scatter(x, y, s=60, alpha=0.9)
if x.notna().any():
    ax.axvline(x.median(), linestyle="--", alpha=0.4)
if y.notna().any():
    ax.axhline(y.median(), linestyle="--", alpha=0.4)
for _, r in gpi_df.iterrows():
    if pd.isna(r.get("Accel")) or pd.isna(r.get("Grind")):
        continue
    ax.annotate(str(r.get("Horse", "")),
                xy=(r["Accel"] * 100.0, r["Grind"] * 100.0),
                xytext=(5, 5), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5))
ax.set_xlabel("Kick — Acceleration (%)")
ax.set_ylabel("Grind — Late (%)")
ax.set_title("Kick vs Grind")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# 2) Pace Curves — Field Average + Top 8 by Finish
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

# 3) Per-Horse Radar
st.subheader("Per-Horse Radar (F200 / tsSPI / Kick / Grind)")
radar_df = gpi_df[["Horse", "F200", "tsSPI", "Accel", "Grind"]].dropna()
if not radar_df.empty:
    pick = st.selectbox("Pick a horse", radar_df["Horse"].tolist())
    row = radar_df[radar_df["Horse"] == pick].iloc[0]
    labels = ["F200", "tsSPI", "Kick", "Grind"]
    values = [float(row["F200"]) * 100.0, float(row["tsSPI"]) * 100.0,
              float(row["Accel"]) * 100.0, float(row["Grind"]) * 100.0]
    L = labels + [labels[0]]
    V = values + [values[0]]
    ang = np.linspace(0, 2 * np.pi, len(L), endpoint=False)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, polar=True)
    ax3.plot(ang, V, linewidth=2)
    ax3.fill(ang, V, alpha=0.15)
    ax3.set_xticks(ang[:-1]); ax3.set_xticklabels(labels)
    ax3.set_title(f"Sectional Radar — {pick}")
    st.pyplot(fig3)
else:
    st.info("Radar requires non-empty data to display.")

# 4) Top-8 PI bar
st.subheader("Top 8 by PI (0–10)")
bar8 = gpi_df.sort_values("PI", ascending=False).head(8)[["Horse", "PI"]]
fig4, ax4 = plt.subplots()
ax4.barh(bar8["Horse"].iloc[::-1], bar8["PI"].iloc[::-1])
ax4.set_xlabel("PI (0–10)")
ax4.set_title("Top 8 — Performance Index")
st.pyplot(fig4)

st.caption(
    "PI v2.4-B+: winsorized sectionals (F200 / tsSPI / Accel / Grind) with distance-aware weights, "
    "robust z and logistic scaling to 0–10. GPI v0.95 blends balance, separation, dual-weapon and "
    "sanity anchors (winner/place guards) to estimate group potential (0–10)."
)
