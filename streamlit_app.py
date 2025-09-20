import io
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ======================
# App config (minimal)
# ======================
st.set_page_config(page_title="Race Edge — Sectionals", layout="wide")

# -------------------
# Sidebar controls
# -------------------
with st.sidebar:
    st.title("Race Edge")
    MODE = st.radio("Input mode", ["Upload CSV", "Manual"], index=0)
    distance_m_input = st.number_input("Race distance (m)", min_value=800, max_value=4000, value=1400, step=10)
    # Manual grid uses 200m cells → round UP to nearest 200
    distance_m_rounded = int(ceil(distance_m_input / 200.0) * 200)
    num_horses = st.number_input("Number of runners (manual only)", min_value=2, max_value=24, value=10, step=1)
    st.caption("Manual grid shows 200 m segments counting DOWN from the race distance.")
    DEBUG = st.checkbox("Debug", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔎 {label}")
        if obj is not None:
            st.write(obj)

# ======================
# Parsing & helpers
# ======================
def parse_race_time(val):
    """Parse 'MM:SS.ms' or seconds to float seconds."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:
                m, sec, ms = parts
                return int(m) * 60 + int(sec) + int(ms) / 1000.0
            if len(parts) == 2:
                m, sec = parts
                return int(m) * 60 + float(sec)
        return float(s)
    except Exception:
        return np.nan

def percent_rank(series):
    """0..1 percentile ranks; returns NaN if all NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([np.nan]*len(s), index=s.index)
    return s.rank(pct=True)

def compute_from_200m_segments(row, seg_cols, distance_m, race_avg_speed):
    """
    seg_cols: ordered first→last 200m segments (countdown labels), each is time (s) for that 200m.
    Returns: race_time_s, F200%, tsSPI%, Grind%, Kick%
    """
    seg_times = pd.to_numeric(row[seg_cols], errors="coerce").values.astype(float)
    seg_times = np.where(seg_times > 0, seg_times, np.nan)

    race_time_s = np.nansum(seg_times)
    if (pd.isna(race_avg_speed) or race_avg_speed <= 0) and race_time_s > 0:
        race_avg_speed = distance_m / race_time_s

    # F200: first 200m
    t_f200 = seg_times[0] if len(seg_times) >= 1 else np.nan
    f200_speed = (200.0 / t_f200) if (pd.notna(t_f200) and t_f200 > 0) else np.nan
    F200_pct = (f200_speed / race_avg_speed) * 100 if (pd.notna(f200_speed) and race_avg_speed > 0) else np.nan

    # tsSPI: exclude first 200m and last 400m → [1:-2]
    if len(seg_times) >= 4:
        mid = seg_times[1:-2]
        if np.isfinite(mid).any():
            t_mid = np.nansum(mid)
            dist_mid = 200.0 * np.isfinite(mid).sum()
            s_mid = dist_mid / t_mid if (t_mid and t_mid > 0) else np.nan
            tsSPI_pct = (s_mid / race_avg_speed) * 100 if (pd.notna(s_mid) and race_avg_speed > 0) else np.nan
        else:
            tsSPI_pct = np.nan
    else:
        tsSPI_pct = np.nan

    # Grind: 600→200 (last two segments before the final)
    if len(seg_times) >= 3:
        grind_window = seg_times[-3:-1]
        if np.isfinite(grind_window).any():
            t_grind = np.nansum(grind_window)
            s_grind = 400.0 / t_grind if (t_grind and t_grind > 0) else np.nan
            Grind_pct = (s_grind / race_avg_speed) * 100 if (pd.notna(s_grind) and race_avg_speed > 0) else np.nan
        else:
            Grind_pct = np.nan
    else:
        Grind_pct = np.nan

    # Kick: last 200m
    if len(seg_times) >= 1 and pd.notna(seg_times[-1]) and seg_times[-1] > 0:
        t_kick = seg_times[-1]
        s_kick = 200.0 / t_kick
        Kick_pct = (s_kick / race_avg_speed) * 100 if race_avg_speed > 0 else np.nan
    else:
        Kick_pct = np.nan

    # Caps (sane bounds)
    if pd.notna(F200_pct): F200_pct = min(F200_pct, 120.0)
    if pd.notna(Kick_pct): Kick_pct = min(Kick_pct, 120.0)

    return race_time_s, F200_pct, tsSPI_pct, Grind_pct, Kick_pct

def compute_from_existing_schema(df, distance_m):
    """
    For CSVs that already have app-like columns:
      'Race Time', '800-400', '400-Finish', '200_Time', 'Horse', 'Finish_Pos', and optional weight aliases.
    """
    out = df.copy()

    # Parse race time
    if "Race Time" in out.columns:
        out["RaceTime_s"] = pd.to_numeric(out["Race Time"].apply(parse_race_time), errors="coerce")
    else:
        out["RaceTime_s"] = np.nan

    # Average race speed
    out["Race_AvgSpeed"] = distance_m / out["RaceTime_s"]

    # Mid & Final 400 speeds
    for c in ["800-400", "400-Finish", "200_Time"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["Mid400_Speed"]   = 400.0 / out["800-400"]
    out["Final400_Speed"] = 400.0 / out["400-Finish"]

    # Sectional %s
    out["Basic_FSP%"]   = (out["Final400_Speed"] / out["Race_AvgSpeed"]) * 100
    out["Refined_FSP%"] = (out["Final400_Speed"] / out["Mid400_Speed"]) * 100
    out["SPI%"]         = (out["Mid400_Speed"] / out["Race_AvgSpeed"]) * 100

    # tsSPI proxy (we lack all 100m splits here) → use SPI% as proxy
    out["tsSPI%"] = out["SPI%"]

    # Gate speed F200%
    if "200_Time" in out.columns:
        out["F200%"] = (200.0 / out["200_Time"]) / out["Race_AvgSpeed"] * 100
    else:
        out["F200%"] = np.nan

    # Kick (anchored)
    out["Kick%"] = 0.70 * out["Refined_FSP%"] + 0.30 * out["Basic_FSP%"]

    # Grind
    out["Grind%"] = (out["Mid400_Speed"] / out["Race_AvgSpeed"]) * 100

    # Optional: infer finish pos if missing
    if ("Finish_Pos" not in out.columns) or out["Finish_Pos"].isna().all():
        if out["RaceTime_s"].notna().any():
            out["Finish_Pos"] = out["RaceTime_s"].rank(method="min").astype("Int64")

    return out

# ======================
# PI v3.3 (execution today)
# ======================
def compute_pi_v33(df_metrics):
    """
    Requires: F200%, tsSPI%, Grind%, Kick%, Finish_Pos (optional).
    Returns: df with PI_v3.2 (base), PI_v3.3 (winner-protected), race_shape.
    """
    M = df_metrics.copy()

    # Percentiles
    for col in ["F200%", "tsSPI%", "Grind%", "Kick%"]:
        M[f"{col}_pct"] = percent_rank(M[col])

    # Race shape by tsSPI median
    spi_median = float(np.nanmedian(M["tsSPI%"].values.astype(float)))
    if spi_median >= 103:
        shape = "fast"
        base_w = {"F200%": 0.10, "tsSPI%": 0.25, "Grind%": 0.35, "Kick%": 0.30}
    elif spi_median <= 97:
        shape = "slow"
        base_w = {"F200%": 0.10, "tsSPI%": 0.15, "Grind%": 0.25, "Kick%": 0.50}
    else:
        shape = "even"
        base_w = {"F200%": 0.15, "tsSPI%": 0.20, "Grind%": 0.30, "Kick%": 0.35}

    # Base PI with ≥3 valid components (per-horse renormalize)
    pi_base = []
    for _, r in M.iterrows():
        valid = {}
        for m in ["F200%", "tsSPI%", "Grind%", "Kick%"]:
            v = r.get(f"{m}_pct", np.nan)
            if pd.notna(v) and np.isfinite(v):
                valid[m] = float(v)
        if len(valid) < 3:
            pi_base.append(np.nan)
            continue
        total_w = sum(base_w[m] for m in valid.keys())
        score = sum((base_w[m] / total_w) * valid[m] for m in valid.keys())
        pi_base.append(score)
    M["PI_v3.2"] = pi_base

    # Minimal winner protection (tiny, conditional)
    def efficiency_index_simple(kick, spi):
        if pd.isna(kick) or pd.isna(spi):
            return 0.0
        return max(0.0, 1.0 - (abs(kick - 100.0) + abs(spi - 100.0)) / 10.0)

    M["PI_v3.3"] = M["PI_v3.2"].copy()
    if "Finish_Pos" in M.columns and M["Finish_Pos"].notna().any():
        try:
            w_idx = M["Finish_Pos"].astype("float").idxmin()
            pi_w = M.at[w_idx, "PI_v3.2"]
            pi_med = np.nanmedian(M["PI_v3.2"].values)
            kick_w = M.at[w_idx, "Kick%"]
            spi_w = M.at[w_idx, "tsSPI%"]
            fast_early = (spi_median >= 103)
            eligible = (
                pd.notna(pi_w) and pi_w < pi_med and
                pd.notna(kick_w) and pd.notna(spi_w) and
                abs(kick_w - 100.0) <= 4.0 and abs(spi_w - 100.0) <= 4.0
            )
            if fast_early and (kick_w < 100.0):
                eligible = False
            if eligible:
                EI = efficiency_index_simple(kick_w, spi_w)
                uplift = min(0.04 * EI, 0.025)
                M.at[w_idx, "PI_v3.3"] = pi_w + uplift
        except Exception:
            pass

    return M, shape

# ======================
# Ability Index (AI) — position-agnostic
# ======================
def compute_PI_core(df_metrics):
    """PI without winner protection (no Finish_Pos influence)."""
    M = df_metrics.copy()
    for col in ["F200%", "tsSPI%", "Grind%", "Kick%"]:
        M[f"{col}_pct"] = percent_rank(M[col])

    spi_med = float(np.nanmedian(M["tsSPI%"].values.astype(float)))
    if spi_med >= 103:
        w = {"F200%":0.10, "tsSPI%":0.25, "Grind%":0.35, "Kick%":0.30}
    elif spi_med <= 97:
        w = {"F200%":0.10, "tsSPI%":0.15, "Grind%":0.25, "Kick%":0.50}
    else:
        w = {"F200%":0.15, "tsSPI%":0.20, "Grind%":0.30, "Kick%":0.35}

    pi_core = []
    for _, r in M.iterrows():
        valid = {m: float(r.get(f"{m}_pct", np.nan)) for m in ["F200%","tsSPI%","Grind%","Kick%"]
                 if pd.notna(r.get(f"{m}_pct", np.nan))}
        if len(valid) < 3:
            pi_core.append(np.nan); continue
        tw = sum(w[m] for m in valid)
        pi_core.append(sum((w[m]/tw)*valid[m] for m in valid))
    M["PI_core"] = pi_core
    shape = "fast" if spi_med >= 103 else ("slow" if spi_med <= 97 else "even")
    return M, shape

def compute_AI(df_metrics, shape, weights_kg=None):
    """
    AI = PI_core + bias (race-shape) + distance (fit) + weight.
    No use of Finish_Pos; fully position-agnostic.
    """
    M = df_metrics.copy()
    if "PI_core" not in M.columns:
        M, _ = compute_PI_core(M)

    # Medians for race-relative logic
    K_med = float(np.nanmedian(M["Kick%"]))
    G_med = float(np.nanmedian(M["Grind%"]))
    TS_med = float(np.nanmedian(M["tsSPI%"]))

    # Optional weight context
    if weights_kg is not None:
        Wt = pd.to_numeric(weights_kg, errors="coerce")
        Wt_avg = float(np.nanmean(Wt))
    else:
        Wt = pd.Series([np.nan]*len(M), index=M.index)
        Wt_avg = None

    def bias_adj(k, g):
        dK, dG = k - K_med, g - G_med
        if shape == "slow":     # sprint-home → reward above-median Kick
            if dK >= 6: return 0.09
            if dK >= 3: return 0.06
            if dK <= -3: return -0.02
            return 0.0
        if shape == "fast":     # fast-early → reward above-median Grind
            if dG >= 6: return 0.09
            if dG >= 3: return 0.06
            if dG <= -3: return -0.02
            return 0.0
        # even
        bonus = 0.0
        if dK >= 3: bonus += 0.03
        if dG >= 3: bonus += 0.03
        if dK <= -3 or dG <= -3: bonus -= 0.01
        return max(-0.06, min(0.06, bonus))

    def dist_adj(k, ts, f200):
        # Sprint-aware via medians embedded in cutoffs
        if (k >= max(103, K_med + 3) and ts >= TS_med):
            return 0.06  # wants further
        if (k <= min(98, K_med - 3) and pd.notna(f200) and f200 >= 110):
            return 0.04  # wants shorter
        if abs(ts - TS_med) <= 1:
            return 0.0   # trip about right
        return 0.02 if (k >= K_med and ts >= TS_med) else 0.0

    def weight_adj(w_kg):
        if pd.isna(w_kg) or Wt_avg is None:
            return 0.0
        W = 0.02 * ((w_kg - Wt_avg) / 2.5)  # +0.02 AI per +2.5kg
        return max(-0.06, min(0.06, W))

    AI_vals = []
    for i, r in M.iterrows():
        PIc = float(r.get("PI_core", np.nan))
        if not pd.notna(PIc):
            AI_vals.append(np.nan); continue
        k, g, ts = float(r["Kick%"]), float(r["Grind%"]), float(r["tsSPI%"])
        f2 = float(r.get("F200%", np.nan))
        a = PIc + bias_adj(k, g) + dist_adj(k, ts, f2) + weight_adj(Wt.loc[i])
        AI_vals.append(max(0.0, min(1.0, a)))

    M["AI"] = AI_vals
    M["AI_pct"] = percent_rank(M["AI"])
    return M

# ======================
# Input & build workframe
# ======================
st.title("🏇 Race Edge — Sectional Analysis (PI & AI)")

df_input = None
distance_for_calc = distance_m_input  # will override in CSV mode if detectable

if MODE == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (e.g., columns: Horse, Race Time, 800-400, 400-Finish, 200_Time, optional weights).", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or switch to Manual mode.")
        st.stop()

    raw = pd.read_csv(uploaded)
    st.subheader("Raw preview")
    st.dataframe(raw.head(12), use_container_width=True)

    # If the file contains 100m splits, you can extend this to compute exact tsSPI. Here we assume app schema.
    metrics0 = compute_from_existing_schema(raw, distance_m_input)

    # Weight column aliases
    weight_aliases = {"weight", "horse weight", "carried_kg", "wt", "weight_kg"}
    weight_col = None
    for c in metrics0.columns:
        if str(c).strip().lower() in weight_aliases:
            weight_col = c
            break
    if weight_col:
        metrics0.rename(columns={weight_col: "Horse Weight"}, inplace=True)

    distance_for_calc = distance_m_input  # keep user input unless you derive from splits

    df_input = pd.DataFrame()
    df_input["Horse"] = metrics0["Horse"] if "Horse" in metrics0.columns else [f"Runner {i+1}" for i in range(len(metrics0))]
    df_input["Finish_Pos"] = pd.to_numeric(metrics0.get("Finish_Pos", np.nan), errors="coerce")
    df_input["RaceTime_s"] = pd.to_numeric(metrics0.get("RaceTime_s", np.nan), errors="coerce")
    df_input["Horse Weight"] = pd.to_numeric(metrics0.get("Horse Weight", np.nan), errors="coerce")
    for col in ["F200%", "tsSPI%", "Grind%", "Kick%"]:
        df_input[col] = pd.to_numeric(metrics0.get(col, np.nan), errors="coerce")

else:
    # -------- Manual mode: 200m grid with countdown labels + Weight --------
    st.subheader("Manual 200m split entry")
    st.caption("Enter **seconds** for each 200m segment (counting down). Last column is **200→Finish**. "
               "Leave Finish_Pos blank to infer by total time. Add **Weight (kg)** if known.")

    N = distance_m_rounded // 200
    rems = list(range(distance_m_rounded, 0, -200))  # e.g., 1200, 1000, ..., 200
    seg_labels = [f"{rems[i]}→{(rems[i]-200) if (rems[i]-200)>0 else 'Finish'}" for i in range(N)]

    base = pd.DataFrame({
        "Horse": [f"Runner {i+1}" for i in range(num_horses)],
        "Finish_Pos": [np.nan] * num_horses,
        "Horse Weight": [np.nan] * num_horses,
    })
    for lbl in seg_labels:
        base[lbl] = np.nan

    edited = st.data_editor(
        base,
        use_container_width=True,
        num_rows="dynamic",
        key="manual_editor",
        column_config={
            "Finish_Pos": st.column_config.NumberColumn("Finish_Pos", help="Optional — if blank, inferred by total time."),
            "Horse Weight": st.column_config.NumberColumn("Weight (kg)", min_value=40.0, max_value=70.0, step=0.5, help="Carried weight in kg (optional)."),
        }
    )

    st.caption(f"Distance rounded up to **{distance_m_rounded} m** for manual grid.")
    distance_for_calc = distance_m_rounded

    # Build metrics from manual rows
    rows = []
    for _, r in edited.iterrows():
        horse = str(r.get("Horse", "")).strip() or "Runner"
        times = pd.to_numeric(r[seg_labels], errors="coerce")
        race_time = times.sum(skipna=True)
        race_avg_speed = (distance_m_rounded / race_time) if (pd.notna(race_time) and race_time > 0) else np.nan

        race_time_s, F200_pct, tsSPI_pct, Grind_pct, Kick_pct = compute_from_200m_segments(
            r, seg_labels, distance_m_rounded, race_avg_speed
        )
        rows.append({
            "Horse": horse,
            "Finish_Pos": pd.to_numeric(r.get("Finish_Pos", np.nan), errors="coerce"),
            "Horse Weight": pd.to_numeric(r.get("Horse Weight", np.nan), errors="coerce"),
            "RaceTime_s": race_time_s,
            "F200%": F200_pct,
            "tsSPI%": tsSPI_pct,
            "Grind%": Grind_pct,
            "Kick%": Kick_pct,
        })
    df_input = pd.DataFrame(rows)

    if df_input["Finish_Pos"].isna().all() and df_input["RaceTime_s"].notna().any():
        df_input["Finish_Pos"] = df_input["RaceTime_s"].rank(method="min").astype("Int64")

# Guard numeric bounds
for col in ["F200%", "tsSPI%", "Grind%", "Kick%"]:
    if col in df_input.columns:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")
df_input["F200%"] = df_input["F200%"].clip(upper=120)
df_input["Kick%"] = df_input["Kick%"].clip(upper=120)

# ======================
# Compute PI & AI
# ======================
pi_df, race_shape = compute_pi_v33(df_input)

# Find weight col for AI (already normalized to "Horse Weight" if present)
weights_series = pi_df.get("Horse Weight", None)

pi_core_df, shape_from_core = compute_PI_core(pi_df)
ai_enriched = compute_AI(pi_core_df, shape_from_core, weights_kg=weights_series)

# Merge AI & PI_core back
pi_df = pi_df.join(ai_enriched[["PI_core", "AI", "AI_pct"]])

# ======================
# Display: table with rank toggle & legend
# ======================
st.subheader("Sectional Metrics (ranked view)")

rank_key = st.radio("Rank by", ["PI v3.3 (execution today)", "AI (latent ability shown)"], index=0)
rank_col = "PI_v3.3" if rank_key.startswith("PI") else "AI"

disp_cols = ["Horse", "Finish_Pos", "Horse Weight", "F200%", "tsSPI%", "Grind%", "Kick%", "PI_core", "PI_v3.3", "AI"]
table = pi_df[disp_cols].copy()
table["Δ (AI–PI)"] = (table["AI"] - table["PI_v3.3"]).round(3)
table = table.sort_values(rank_col, ascending=False, na_position="last").round(3)

st.dataframe(table, use_container_width=True)

with st.expander("How to read PI vs AI"):
    st.markdown(
        "- **PI v3.3** = *on-the-day sectional performance* (how well they executed in this race).\n"
        "- **AI** = *position-agnostic latent ability* (PI_core + small adjustments for race-shape bias, distance fit, and weight).\n"
        "- **Δ (AI–PI)** (within a race):\n"
        "  - **Positive** → hidden ability shown (upgrade next time).\n"
        "  - **Near zero** → fair reflection.\n"
        "  - **Negative** → setup beneficiary (be cautious).\n"
        "- Rules of thumb: |Δ| < **0.02** normal; **0.02–0.05** noteworthy; > **0.05** strong."
    )

# ======================
# Kick vs Grind chart (bubble = tsSPI, color = PI/AI)
# ======================
st.subheader("Kick vs Grind (size = tsSPI, color = PI or AI)")

plot_df = table.dropna(subset=["Kick%", "Grind%", "tsSPI%", "PI_v3.3", "AI"]).copy()
if plot_df.empty:
    st.info("Not enough data to plot.")
else:
    # Bubble sizes: scale tsSPI% to 50..800 pts^2
    ts_min, ts_max = float(plot_df["tsSPI%"].min()), float(plot_df["tsSPI%"].max())
    if ts_max > ts_min:
        plot_df["size_pts2"] = 50.0 + (plot_df["tsSPI%"] - ts_min) * (800.0 - 50.0) / (ts_max - ts_min)
    else:
        plot_df["size_pts2"] = 300.0

    color_series = plot_df["AI"] if rank_col == "AI" else plot_df["PI_v3.3"]

    fig, ax = plt.subplots()
    sc = ax.scatter(
        plot_df["Kick%"], plot_df["Grind%"],
        s=plot_df["size_pts2"], c=color_series,
        alpha=0.7, edgecolors="none"
    )

    ax.set_xlabel("Kick%  (200 → Finish vs race avg)")
    ax.set_ylabel("Grind% (600 → 200 vs race avg)")
    ax.set_title("Sectional profile: finish vs late drive (bubble = sustain)")

    # 100 reference lines + race medians
    try:
        ax.axvline(x=100.0, linestyle="--", alpha=0.25)
        ax.axhline(y=100.0, linestyle="--", alpha=0.25)
        x_med = float(np.nanmedian(plot_df["Kick%"]))
        y_med = float(np.nanmedian(plot_df["Grind%"]))
        ax.axvline(x=x_med, linestyle=":", alpha=0.25)
        ax.axhline(y=y_med, linestyle=":", alpha=0.25)
    except Exception:
        pass

    # Annotate top by selected rank
    top_n = plot_df.sort_values(rank_col, ascending=False).head(8)
    for _, r in top_n.iterrows():
        ax.annotate(str(r["Horse"])[:12], (r["Kick%"], r["Grind%"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("AI" if rank_col == "AI" else "PI v3.3")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    st.caption("Reading guide: → right = stronger Kick; ↑ up = stronger Grind; bigger bubble = stronger tsSPI (sustain); color = selected rank metric (PI or AI).")

# ======================
# Chart breakdown — distance-aware thresholds (SA sprint-aware)
# ======================
st.subheader("Chart breakdown — distance-aware reads")

if plot_df.empty:
    st.info("No chart data available for breakdown.")
else:
    # Determine race distance
    race_distance = int(distance_for_calc)

    is_sprint = (race_distance <= 1200)
    k_med   = float(np.nanmedian(plot_df["Kick%"]))   if is_sprint else None
    ts_med  = float(np.nanmedian(plot_df["tsSPI%"]))  if is_sprint else None

    if not is_sprint:
        if race_distance <= 1400:
            K_HI, G_HI, TS_HI = 104.0, 102.0, 101.0
        elif race_distance < 1800:
            K_HI, G_HI, TS_HI = 103.0, 102.0, 101.0
        elif race_distance < 2200:
            K_HI, G_HI, TS_HI = 102.0, 101.0, 100.5
        else:
            K_HI, G_HI, TS_HI = 101.5, 101.0, 100.0
        K_LO, G_LO, TS_LO = 98.0, 98.0, 99.0

    def tag_and_distance_note(row):
        k   = float(row["Kick%"]); g = float(row["Grind%"]); ts = float(row["tsSPI%"])
        f2  = float(row.get("F200%", np.nan))
        pi  = float(row["PI_v3.3"]); ai = float(row["AI"])

        # Race-shape tags
        if is_sprint:
            if k >= k_med + 3: k_tag = "showed late kick"
            elif k <= k_med - 3: k_tag = "flattened late"
            else: k_tag = "average late burst"

            if g >= 102: g_tag = "applied strong mid-race pressure"
            elif g <= 98: g_tag = "couldn’t hold mid-late pace"
            else: g_tag = "kept pace evenly"

            if "late kick" in k_tag and g >= 100:
                tag = "closer with late power"
            elif g >= 103 and k < (k_med + 1):
                tag = "true grinder"
            elif 98 <= k <= 102 and 98 <= g <= 102:
                tag = "balanced profile"
            else:
                tag = "mixed sectional profile"

            if (k >= k_med + 3 and ts >= ts_med) or (ts >= ts_med + 1 and k >= k_med):
                dist = "scope to improve over further"
            elif (k <= k_med - 3 and (not np.isnan(f2) and f2 >= 110)) or (ts <= ts_med - 1 and k < k_med):
                dist = "profile sharper at shorter"
            elif abs(ts - ts_med) <= 1:
                dist = "trip looks about right"
            else:
                dist = "distance flexible"

        else:
            if k >= K_HI and g >= 100:
                tag = "closer with late power"
            elif g >= G_HI and k < (K_HI - 1):
                tag = "true grinder"
            elif 98 <= k <= 102 and 98 <= g <= 102:
                tag = "balanced profile"
            else:
                tag = "mixed sectional profile"

            if (k >= K_HI and ts >= TS_HI) or (ts >= TS_HI + 0.5 and k >= 100):
                dist = "scope to improve over further"
            elif (k <= K_LO and (not np.isnan(f2) and f2 >= 110)) or (ts <= TS_LO and k < 100):
                dist = "profile sharper at shorter"
            elif 98 <= ts <= 102:
                dist = "trip looks about right"
            else:
                dist = "distance flexible"

        # PI/AI gloss
        gloss = "sectional standout" if ai >= np.nanpercentile(plot_df["AI"], 75) else (
                "solid merit" if ai >= np.nanpercentile(plot_df["AI"], 25) else "sectional underperformer")
        return tag, f"{dist} — {gloss}", k, g, ts, pi, ai

    # Build breakdown lines (sorted by chosen rank metric)
    rows = []
    for _, r in plot_df.sort_values(rank_col, ascending=False).iterrows():
        tag, dist_note, k, g, ts, p, a = tag_and_distance_note(r)
        rows.append(f"- **{r['Horse']}** — {tag}; {dist_note}. "
                    f"(Kick {k:.1f}, Grind {g:.1f}, tsSPI {ts:.1f}, PI {p:.3f}, AI {a:.3f})")

    for line in rows:
        st.markdown(line)

# Footer
st.caption(
    "Core metrics: F200% (break), tsSPI% (sustain excluding first 200 & last 400 where available), "
    "Grind% (600→200), Kick% (final 200). "
    "PI v3.3 = execution today (with tiny winner protection). "
    "AI = position-agnostic latent ability (bias, distance fit, weight)."
)
