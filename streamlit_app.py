import io
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ======================
# App config (no logo)
# ======================
st.set_page_config(page_title="Race Edge — Sectionals", layout="wide")

# -------------------
# Sidebar controls
# -------------------
with st.sidebar:
    st.title("Race Edge")
    MODE = st.radio("Input mode", ["Upload CSV", "Manual"], index=0)
    distance_m_input = st.number_input("Race distance (m)", min_value=800, max_value=4000, value=1400, step=10)
    # Round UP to nearest 200m for manual grid & any 200m logic
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
# Helpers
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
        return pd.Series([np.nan] * len(s), index=s.index)
    return s.rank(pct=True)

def compute_from_200m_segments(row, seg_cols, distance_m, race_avg_speed):
    """
    seg_cols: ordered first→last 200m segments (countdown labels), each is time (s) for that 200m.
    """
    seg_times = pd.to_numeric(row[seg_cols], errors="coerce").values.astype(float)
    seg_times = np.where(seg_times > 0, seg_times, np.nan)  # guard

    race_time_s = np.nansum(seg_times)
    if np.isnan(race_avg_speed) and race_time_s > 0:
        race_avg_speed = distance_m / race_time_s

    # F200: first 200m segment (index 0)
    t_f200 = seg_times[0] if len(seg_times) >= 1 else np.nan
    f200_speed = (200.0 / t_f200) if (pd.notna(t_f200) and t_f200 > 0) else np.nan
    F200_pct = (f200_speed / race_avg_speed) * 100 if (pd.notna(f200_speed) and pd.notna(race_avg_speed) and race_avg_speed > 0) else np.nan

    # tsSPI: exclude FIRST 200m and LAST 400m → [1:-2]
    if len(seg_times) >= 4:
        mid = seg_times[1:-2]
        if np.isfinite(mid).any():
            t_mid = np.nansum(mid)
            dist_mid = 200.0 * np.isfinite(mid).sum()
            s_mid = dist_mid / t_mid if t_mid and t_mid > 0 else np.nan
            tsSPI_pct = (s_mid / race_avg_speed) * 100 if (pd.notna(s_mid) and pd.notna(race_avg_speed) and race_avg_speed > 0) else np.nan
        else:
            tsSPI_pct = np.nan
    else:
        tsSPI_pct = np.nan

    # Grind: 600→200 before finish → last two segments before the final (indices -3 and -2)
    if len(seg_times) >= 3:
        grind_window = seg_times[-3:-1]
        if np.isfinite(grind_window).any():
            t_grind = np.nansum(grind_window)
            s_grind = 400.0 / t_grind if t_grind and t_grind > 0 else np.nan
            Grind_pct = (s_grind / race_avg_speed) * 100 if (pd.notna(s_grind) and pd.notna(race_avg_speed) and race_avg_speed > 0) else np.nan
        else:
            Grind_pct = np.nan
    else:
        Grind_pct = np.nan

    # Kick: last 200m
    if len(seg_times) >= 1 and pd.notna(seg_times[-1]) and seg_times[-1] > 0:
        t_kick = seg_times[-1]
        s_kick = 200.0 / t_kick
        Kick_pct = (s_kick / race_avg_speed) * 100 if (pd.notna(race_avg_speed) and race_avg_speed > 0) else np.nan
    else:
        Kick_pct = np.nan

    # Caps
    if pd.notna(F200_pct):
        F200_pct = min(F200_pct, 120.0)
    if pd.notna(Kick_pct):
        Kick_pct = min(Kick_pct, 120.0)

    return race_time_s, F200_pct, tsSPI_pct, Grind_pct, Kick_pct

def compute_from_100m_csv(df, distance_m):
    """
    Expects split columns like '100_Time', '200_Time', ..., 'Finish_Time' (seconds).
    'Horse' optional; 'Finish_Pos' optional; 'Race Time' optional.
    """
    out = df.copy()
    split_cols = [c for c in out.columns if c.endswith("_Time") and c[0].isdigit()]

    def split_val(c):
        try:
            return int(c.split("_")[0])
        except Exception:
            return None

    split_cols = sorted([c for c in split_cols if split_val(c) is not None], key=lambda x: split_val(x))
    for c in split_cols + (["Finish_Time"] if "Finish_Time" in out.columns else []):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "Race Time" in out.columns:
        out["RaceTime_s"] = pd.to_numeric(out["Race Time"].apply(parse_race_time), errors="coerce")
    else:
        time_cols = split_cols + (["Finish_Time"] if "Finish_Time" in out.columns else [])
        out["RaceTime_s"] = out[time_cols].sum(axis=1, skipna=True)

    out["Race_AvgSpeed"] = distance_m / out["RaceTime_s"]

    # F200% = 100 + 200
    if "100_Time" in out.columns and "200_Time" in out.columns:
        t0_200 = out["100_Time"] + out["200_Time"]
        out["F200%"] = (200.0 / t0_200) / out["Race_AvgSpeed"] * 100
    else:
        out["F200%"] = np.nan

    # tsSPI%: 300 → (max-400)
    max_split = max([split_val(c) for c in split_cols]) if split_cols else 0
    mid_cols = [c for c in split_cols if 300 <= split_val(c) <= max_split - 400]
    if mid_cols:
        t_mid = out[mid_cols].sum(axis=1, skipna=True)
        dist_mid = 100.0 * len(mid_cols)
        s_mid = dist_mid / t_mid
        out["tsSPI%"] = (s_mid / out["Race_AvgSpeed"]) * 100
    else:
        out["tsSPI%"] = np.nan

    # Grind%: last 600→200 (400m total) = max-600..max-300
    grind_starts = [max_split - 600, max_split - 500, max_split - 400, max_split - 300]
    grind_cols = [f"{d}_Time" for d in grind_starts if f"{d}_Time" in out.columns]
    if grind_cols:
        t_grind = out[grind_cols].sum(axis=1, skipna=True)
        s_grind = 400.0 / t_grind
        out["Grind%"] = (s_grind / out["Race_AvgSpeed"]) * 100
    else:
        out["Grind%"] = np.nan

    # Kick%: last 200m = (max-100) + Finish
    kick_cols = []
    if f"{max_split - 100}_Time" in out.columns:
        kick_cols.append(f"{max_split - 100}_Time")
    if "Finish_Time" in out.columns:
        kick_cols.append("Finish_Time")
    if kick_cols:
        t_kick = out[kick_cols].sum(axis=1, skipna=True)
        s_kick = 200.0 / t_kick
        out["Kick%"] = (s_kick / out["Race_AvgSpeed"]) * 100
    else:
        out["Kick%"] = np.nan

    # Caps
    out["F200%"] = out["F200%"].clip(upper=120)
    out["Kick%"] = out["Kick%"].clip(upper=120)

    return out

def compute_pi_v33(df_metrics):
    """
    PI v3.3: Kick-forward + minimal winner protection.
    Requires: F200%, tsSPI%, Grind%, Kick%, Finish_Pos.
    """
    M = df_metrics.copy()

    # Percentiles
    for col in ["F200%", "tsSPI%", "Grind%", "Kick%"]:
        M[f"{col}_pct"] = percent_rank(M[col])

    # Race shape by tsSPI median
    spi_median = np.nanmedian(M["tsSPI%"].values.astype(float))
    if spi_median >= 103:
        shape = "fast"
        base_w = {"F200%": 0.10, "tsSPI%": 0.25, "Grind%": 0.35, "Kick%": 0.30}
    elif spi_median <= 97:
        shape = "slow"
        base_w = {"F200%": 0.10, "tsSPI%": 0.15, "Grind%": 0.25, "Kick%": 0.50}
    else:
        shape = "even"
        base_w = {"F200%": 0.15, "tsSPI%": 0.20, "Grind%": 0.30, "Kick%": 0.35}

    # Compute base PI with ≥3 valid components (per-horse renormalize)
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

    # Winner-only minimal protection (sectionals-first, conditional)
    def efficiency_index_simple(kick, spi):
        if pd.isna(kick) or pd.isna(spi):
            return 0.0
        return max(0.0, 1.0 - (abs(kick - 100.0) + abs(spi - 100.0)) / 10.0)

    M["PI_v3.3"] = M["PI_v3.2"].copy()
    if "Finish_Pos" in M.columns and M["Finish_Pos"].notna().any():
        try:
            w_idx = M["Finish_Pos"].astype("float").idxmin()  # winner row
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

def narrative_for_runner(row, spi_median):
    """One-liner from PI: race-shape note + distance hint."""
    name = str(row.get("Horse", "")).strip() or "Runner"
    pi = row.get("PI_v3.3", np.nan)
    kick = row.get("Kick%", np.nan)
    grind = row.get("Grind%", np.nan)
    spi = row.get("tsSPI%", np.nan)
    f200 = row.get("F200%", np.nan)
    finish = row.get("Finish_Pos", np.nan)

    notes = []

    # Race-shape tags
    if pd.notna(kick) and pd.notna(pi) and pi >= 0.5 and kick >= 103:
        notes.append("powerful late burst — thrives with pace on")
    elif pd.notna(grind) and pd.notna(kick) and grind >= 100 and kick < 101:
        notes.append("ground it out — better in truly run races")
    elif pd.notna(spi) and 98 <= spi <= 102:
        notes.append("balanced, efficient profile")
    else:
        notes.append("solid sectional profile")

    # Distance hints
    hint = None
    if pd.notna(kick) and pd.notna(spi):
        if kick >= 103 and spi >= 100:
            hint = "could improve over a little further"
        elif kick < 98 and pd.notna(f200) and f200 >= 110:
            hint = "may be better dropping slightly in trip"
        elif 98 <= spi <= 102:
            hint = "trip looks about right"
    if hint:
        notes.append(hint)

    fin_txt = f"{int(finish)}" if pd.notna(finish) else "—"
    pi_txt = f"{pi:.3f}" if pd.notna(pi) else "—"
    return f"**{name}** (Finish {fin_txt}, PI {pi_txt}) — " + "; ".join(notes) + "."

# ======================
# Input & Build Workframe
# ======================
st.title("🏇 Race Edge — Sectional Analysis (PI v3.3)")

df_input = None
if MODE == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV with 100m splits (e.g., 100_Time ... Finish_Time).", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or switch to Manual mode.")
        st.stop()
    raw = pd.read_csv(uploaded)
    st.subheader("Raw preview")
    st.dataframe(raw.head(12), use_container_width=True)

    # Detect distance from last split if possible
    split_cols = [c for c in raw.columns if c.endswith("_Time") and c[0].isdigit()]

    def _sv(c):
        try:
            return int(c.split("_")[0])
        except:
            return None

    last_split = max([_sv(c) for c in split_cols]) if split_cols else None
    distance_m = distance_m_input
    if last_split and "Finish_Time" in raw.columns:
        distance_m = last_split + 100
    st.caption(f"Using distance: **{int(distance_m)} m**")

    metrics0 = compute_from_100m_csv(raw, distance_m=distance_m)

    if ("Finish_Pos" not in metrics0.columns) or metrics0["Finish_Pos"].isna().all():
        metrics0["Finish_Pos"] = metrics0["RaceTime_s"].rank(method="min").astype("Int64")

    df_input = pd.DataFrame()
    if "Horse" in metrics0.columns:
        df_input["Horse"] = metrics0["Horse"]
    elif "Horse" in raw.columns:
        df_input["Horse"] = raw["Horse"]
    else:
        df_input["Horse"] = [f"Runner {i+1}" for i in range(len(metrics0))]

    df_input["Finish_Pos"] = pd.to_numeric(metrics0["Finish_Pos"], errors="coerce")
    df_input["RaceTime_s"] = pd.to_numeric(metrics0["RaceTime_s"], errors="coerce")
    df_input["F200%"] = pd.to_numeric(metrics0["F200%"], errors="coerce")
    df_input["tsSPI%"] = pd.to_numeric(metrics0["tsSPI%"], errors="coerce")
    df_input["Grind%"] = pd.to_numeric(metrics0["Grind%"], errors="coerce")
    df_input["Kick%"] = pd.to_numeric(metrics0["Kick%"], errors="coerce")

else:
    # -------- Manual mode: 200m grid with countdown labels --------
    st.subheader("Manual 200m split entry")
    st.caption("Enter **seconds** for each 200m segment. Last column is **200→Finish**. Leave Finish_Pos blank to infer by total time.")

    # Build countdown labels
    N = distance_m_rounded // 200
    rems = list(range(distance_m_rounded, 0, -200))  # e.g., 1200, 1000, ..., 200
    seg_labels = [f"{rems[i]}→{(rems[i]-200) if (rems[i]-200)>0 else 'Finish'}" for i in range(N)]

    base = pd.DataFrame({
        "Horse": [f"Runner {i+1}" for i in range(num_horses)],
        "Finish_Pos": [np.nan] * num_horses
    })
    for lbl in seg_labels:
        base[lbl] = np.nan

    edited = st.data_editor(
        base,
        use_container_width=True,
        num_rows="dynamic",
        key="manual_editor",
        column_config={
            "Finish_Pos": st.column_config.NumberColumn("Finish_Pos", help="Optional — if left blank, inferred by total time.")
        }
    )

    st.caption(f"Distance rounded up to **{distance_m_rounded} m** for manual grid.")
    # Convert manual to metric frame
    rows = []
    for _, r in edited.iterrows():
        horse = str(r.get("Horse", "")).strip() or "Runner"
        seg_cols = seg_labels
        times = pd.to_numeric(r[seg_cols], errors="coerce")
        race_time = times.sum(skipna=True)
        race_avg_speed = (distance_m_rounded / race_time) if (pd.notna(race_time) and race_time > 0) else np.nan
        race_time_s, F200_pct, tsSPI_pct, Grind_pct, Kick_pct = compute_from_200m_segments(
            r, seg_cols, distance_m_rounded, race_avg_speed
        )
        finish_pos = pd.to_numeric(r.get("Finish_Pos", np.nan), errors="coerce")

        rows.append({
            "Horse": horse,
            "Finish_Pos": finish_pos,
            "RaceTime_s": race_time_s,
            "F200%": F200_pct,
            "tsSPI%": tsSPI_pct,
            "Grind%": Grind_pct,
            "Kick%": Kick_pct,
        })

    df_input = pd.DataFrame(rows)

    # Infer finish pos if missing
    if df_input["Finish_Pos"].isna().all() and df_input["RaceTime_s"].notna().any():
        df_input["Finish_Pos"] = df_input["RaceTime_s"].rank(method="min").astype("Int64")

# ======================
# Compute PI v3.3
# ======================
metrics = df_input.copy()

# Guard: ensure numeric and cap
for col in ["F200%", "tsSPI%", "Grind%", "Kick%"]:
    metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
metrics["F200%"] = metrics["F200%"].clip(upper=120)
metrics["Kick%"] = metrics["Kick%"].clip(upper=120)

pi_df, race_shape = compute_pi_v33(metrics)

# Build display table (ranked by PI v3.3 desc)
disp_cols = ["Horse", "Finish_Pos", "F200%", "tsSPI%", "Grind%", "Kick%", "PI_v3.3"]
table = pi_df[disp_cols].copy()
table = table.sort_values("PI_v3.3", ascending=False, na_position="last")

# Next-Up flag vs winner's PI_v3.3
winner_mask = (pi_df["Finish_Pos"] == pi_df["Finish_Pos"].min())
winner_pi = float(pi_df.loc[winner_mask, "PI_v3.3"].values[0]) if winner_mask.any() else np.nan
table["Next-Up?"] = (table["PI_v3.3"] >= winner_pi) & (table["Finish_Pos"] > 1)

st.subheader("Sectional Metrics (ranked by PI v3.3)")
st.dataframe(table.round(3), use_container_width=True)

# Race-shape headline
st.caption(f"Race shape inferred from tsSPI median: **{race_shape.upper()}**")

# ======================
# Narratives
# ======================
st.subheader("Runner-by-runner narratives")
spi_med_for_notes = np.nanmedian(pi_df["tsSPI%"].values.astype(float))
# Merge back the full metric rows aligned to the sorted table order
merged = table.merge(pi_df, on=disp_cols, how="left")
for _, row in merged.iterrows():
    st.markdown(narrative_for_runner(row, spi_med_for_notes))
    st.markdown("---")

# ======================
# Quick visual: Kick vs PI
# ======================
st.subheader("Kick vs PI (quick view)")
fig, ax = plt.subplots()
plot_df = table.dropna(subset=["Kick%", "PI_v3.3"]).copy()
ax.scatter(plot_df["Kick%"], plot_df["PI_v3.3"])
for _, r in plot_df.iterrows():
    ax.annotate(str(r["Horse"])[:12], (r["Kick%"], r["PI_v3.3"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Kick% (200→Finish vs race avg)")
ax.set_ylabel("PI v3.3")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# Footer
st.caption(
    "Core metrics: F200% (break), tsSPI% (sustain, excl. first 200 & last 400), "
    "Grind% (600→200 before finish), Kick% (last 200). "
    "PI v3.3 = Kick-forward composite with tiny conditional winner protection. Horses ranked by PI."
)
