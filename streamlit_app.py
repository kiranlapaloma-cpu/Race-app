import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Race Analysis by Kiran", layout="wide")

# ---------- Helpers ----------

def parse_race_time(val):
    """Parse 'Race Time' which may be seconds (float) or 'MM:SS:ms' (string)."""
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except Exception:
        pass
    try:
        parts = str(val).strip().split(":")
        if len(parts) == 3:
            m, s, ms = parts
            return int(m) * 60 + int(s) + int(ms) / 1000.0
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return np.nan
    except Exception:
        return np.nan

def compute_metrics(df, distance_m=1400.0, apply_wind=False, wind_speed_kph=0.0, wind_dir="None"):
    out = df.copy()
    out["Race_AvgSpeed"]  = distance_m / out["RaceTime_s"]
    out["Mid400_Speed"]   = 400.0 / out["800-400"]
    out["Final400_Speed"] = 400.0 / out["400-Finish"]

    # Optional small wind tweak to Final-400 only
    if apply_wind and wind_speed_kph and wind_dir in {"Tail", "Head"}:
        k = 0.0045
        adj = k * float(wind_speed_kph)
        if wind_dir == "Tail":
            out["Final400_Speed"] *= (1.0 + adj)
        elif wind_dir == "Head":
            out["Final400_Speed"] *= (1.0 - adj)

    out["Basic_FSP_%"]   = (out["Final400_Speed"] / out["Race_AvgSpeed"]) * 100.0
    out["Refined_FSP_%"] = (out["Final400_Speed"] / out["Mid400_Speed"]) * 100.0
    out["SPI_%"]         = (out["Mid400_Speed"]   / out["Race_AvgSpeed"]) * 100.0

    # Early Position Index (fallbacks)
    if ("200_Pos" in out.columns) and ("400_Pos" in out.columns):
        out["EPI"] = out["200_Pos"] * 0.6 + out["400_Pos"] * 0.4
    elif ("1000_Pos" in out.columns) and ("800_Pos" in out.columns):
        out["EPI"] = out["1000_Pos"] * 0.6 + out["800_Pos"] * 0.4
    elif "400_Pos" in out.columns:
        out["EPI"] = out["400_Pos"]
    else:
        out["EPI"] = np.nan

    # Late position change (400m to finish)
    if ("400_Pos" in out.columns) and ("Finish_Pos" in out.columns):
        out["Pos_Change"] = out["400_Pos"] - out["Finish_Pos"]
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

def apply_pace_scenario(df, scenario="Even"):
    out = df.copy()
    if scenario == "Slow Early / Sprint-Home":
        out["800-400"]    = out["800-400"] * 1.05
        out["400-Finish"] = out["400-Finish"] * 0.95
    elif scenario == "Fast Early / Tough Late":
        out["800-400"]    = out["800-400"] * 0.94
        out["400-Finish"] = out["400-Finish"] * 1.04
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

# ---------- NEW: Natural-language summary generator ----------

def style_from_epi(epi):
    if pd.isna(epi):
        return "position unknown"
    if epi <= 2.0:
        return "on-speed/leader"
    if epi <= 5.0:
        return "handy/midfield"
    return "backmarker"

def kick_from_refined(refined):
    if pd.isna(refined):
        return "—"
    if refined >= 103.0:
        return "accelerated strongly late"
    if refined >= 100.0:
        return "kept building late"
    if refined >= 97.0:
        return "flattened late"
    return "faded late"

def pace_note(spi_med):
    if pd.isna(spi_med):
        return "pace uncertain"
    if spi_med >= 103:
        return "fast early / tough late"
    if spi_med <= 97:
        return "slow early / sprint-home"
    return "even tempo"

def distance_hint(refined, pos_change, epi):
    # Simple distance guidance
    if pd.isna(refined) or pd.isna(pos_change) or pd.isna(epi):
        return ""
    if refined >= 102 and pos_change >= 2:
        return "likely to appreciate a little further."
    if refined < 98 and epi <= 2.5:
        return "may prefer slightly shorter or a softer mid-race."
    if refined >= 100 and epi >= 5:
        return "effective if they can be ridden a touch closer."
    return ""

def runner_summary(row, spi_med):
    name = str(row.get("Horse", ""))
    fin  = row.get("Finish_Pos", np.nan)
    rt   = row.get("RaceTime_s", np.nan)
    epi  = row.get("EPI", np.nan)
    ref  = row.get("Refined_FSP_%", np.nan)
    bas  = row.get("Basic_FSP_%", np.nan)
    spi  = row.get("SPI_%", np.nan)
    pc   = row.get("Pos_Change", np.nan)
    slp  = bool(row.get("Sleeper", False))

    style = style_from_epi(epi)
    kick  = kick_from_refined(ref)
    pnote = pace_note(spi_med)
    dhint = distance_hint(ref, pc, epi)

    fin_txt = f"{int(fin)}" if not pd.isna(fin) else "—"
    rt_txt  = f"{rt:.2f}s" if not pd.isna(rt) else "—"
    epi_txt = f"{epi:.1f}" if not pd.isna(epi) else "—"
    ref_txt = f"{ref:.1f}%" if not pd.isna(ref) else "—"
    spi_txt = f"{spi:.1f}%" if not pd.isna(spi) else "—"
    pc_txt  = f"{int(pc)}" if not pd.isna(pc) else "—"

    sleeper_tag = " **(Sleeper)**" if slp else ""
    bits = [
        f"**{name}** — Finish: **{fin_txt}**, Time: **{rt_txt}**{sleeper_tag}",
        f"Style: {style} (EPI {epi_txt}); Race shape: {pnote} (SPI median).",
        f"Late profile: {kick} (Refined FSP {ref_txt}); Pos change 400–Finish: {pc_txt}.",
    ]
    if dhint:
        bits.append(f"Note: {dhint}")
    return "  \n".join(bits)

# ---------- UI ----------

st.title("🏇 Race Analysis by Kiran")
st.caption("Upload a race CSV/XLSX, compute FSP, Refined FSP, SPI, EPI, sleepers, simulate pace/wind, and read per-runner summaries.")

with st.sidebar:
    st.header("Race Inputs")
    uploaded = st.file_uploader("Upload Race File (CSV or XLSX)", type=["csv", "xlsx"])
    distance_m = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=50)
    wind_dir = st.selectbox("Wind Direction (optional)", ["None", "Head", "Tail"])
    wind_speed = st.number_input("Wind Speed (kph, optional)", min_value=0.0, max_value=80.0, value=0.0, step=1.0)
    apply_wind = st.checkbox("Apply wind adjustment to Final 400", value=False)

    st.divider()
    st.header("Pace Scenario")
    pace_scenario = st.selectbox("Scenario", ["Even", "Slow Early / Sprint-Home", "Fast Early / Tough Late"])

if uploaded is None:
    st.info("Upload a race file to begin.")
    st.stop()

# Load + standardize headers
try:
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

raw = raw.rename(columns={
    "Race time": "Race Time",
    "Race_Time": "Race Time",
    "RaceTime": "Race Time",
    "800_400": "800-400",
    "400_Finish": "400-Finish",
    "Horse Name": "Horse",
    "Finish": "Finish_Pos",
    "Placing": "Finish_Pos"
})

df = raw.copy()
df["RaceTime_s"] = df["Race Time"].apply(parse_race_time)
for col in ["800-400", "400-Finish", "200_Pos", "400_Pos", "800_Pos", "1000_Pos", "Finish_Pos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Apply pace scenario & compute metrics
df_scn = apply_pace_scenario(df, scenario=pace_scenario)
metrics = compute_metrics(df_scn, distance_m=distance_m, apply_wind=apply_wind, wind_speed_kph=wind_speed, wind_dir=wind_dir)
metrics = flag_sleepers(metrics)

# Drop Jockey/Trainer if present
for drop_col in ["Jockey", "Trainer"]:
    if drop_col in metrics.columns:
        metrics = metrics.drop(columns=[drop_col])

# ---------- Table ----------
st.subheader("Sectional Metrics")
disp = round_display(metrics.copy()).sort_values("Finish_Pos", na_position="last")
st.dataframe(disp, use_container_width=True)

# ---------- Charts ----------
st.subheader("Pace Curve — Field Average")
avg_mid = metrics["Mid400_Speed"].mean()
avg_fin = metrics["Final400_Speed"].mean()
fig1, ax1 = plt.subplots()
ax1.plot([1, 2], [avg_mid, avg_fin], marker="o")
ax1.set_xticks([1, 2])
ax1.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
ax1.set_ylabel("Speed (m/s)")
ax1.set_title("Average Pace Curve")
st.pyplot(fig1)

st.subheader("Pace Curve — First 8 Finishers")
top8 = metrics.sort_values("Finish_Pos").head(8)
fig2, ax2 = plt.subplots()
colors = plt.cm.tab10.colors
x_vals = [1, 2]
for idx, (_, row) in enumerate(top8.iterrows()):
    y_vals = [row["Mid400_Speed"], row["Final400_Speed"]]
    color = colors[idx % len(colors)]
    ax2.plot(x_vals, y_vals, marker="o", label=str(row["Horse"]), color=color, linewidth=2)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
ax2.set_ylabel("Speed (m/s)")
ax2.set_title("Per-Horse Pace Curves (Top 8 by Finish)")
ax2.grid(True, linestyle="--", alpha=0.3)
fig2.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
st.pyplot(fig2)

# ---------- Insights & Sleepers ----------
st.subheader("Insights")
spi_med = metrics["SPI_%"].median()
if pd.isna(spi_med):
    st.write("Insufficient data to infer race shape.")
else:
    if spi_med >= 103:
        race_shape = "Fast Early / Tough Late"
    elif spi_med <= 97:
        race_shape = "Slow Early / Sprint-Home"
    else:
        race_shape = "Even"
    st.write(f"**Race shape (by SPI median):** {race_shape} (median SPI {spi_med:.1f}%)")

sleepers = disp[disp["Sleeper"] == True][["Horse","Finish_Pos","Refined_FSP_%","Pos_Change"]]
st.subheader("Sleepers")
if sleepers.empty:
    st.write("No clear sleepers flagged under current thresholds.")
else:
    st.dataframe(sleepers, use_container_width=True)

# ---------- NEW: Runner-by-runner summaries ----------
st.subheader("Runner-by-runner summaries")
ordered = metrics.sort_values("Finish_Pos", na_position="last")
for _, row in ordered.iterrows():
    st.markdown(runner_summary(row, spi_med))
    st.markdown("---")

# ---------- Download ----------
st.subheader("Download")
csv_bytes = disp.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics as CSV", data=csv_bytes, file_name="race_sectional_metrics.csv", mime="text/csv")

st.caption(
    "Legend: Basic FSP = Final400 / Race Avg; Refined FSP = Final400 / Mid400; "
    "SPI = Mid400 / Race Avg; EPI = early positioning index (lower = closer to lead); "
    "Pos_Change = gain from 400m to finish."
)
