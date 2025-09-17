import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Race Sectional & Energy Model", layout="wide")

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

def detect_halved_times(df, sectional_cols=("800-400", "400-Finish")):
    flag = False
    for col in sectional_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if (s > 0).any() and (s.dropna() < 10.5).any():
                flag = True
                break
    return flag

def compute_metrics(df, distance_m=1400.0, apply_wind=False, wind_speed_kph=0.0, wind_dir="None"):
    out = df.copy()
    out["Race_AvgSpeed"]  = distance_m / out["RaceTime_s"]
    out["Mid400_Speed"]   = 400.0 / out["800-400"]
    out["Final400_Speed"] = 400.0 / out["400-Finish"]

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

    if ("200_Pos" in out.columns) and ("400_Pos" in out.columns):
        out["EPI"] = out["200_Pos"] * 0.6 + out["400_Pos"] * 0.4
    elif "400_Pos" in out.columns:
        out["EPI"] = out["400_Pos"]
    else:
        out["EPI"] = np.nan

    if ("400_Pos" in out.columns) and ("Finish_Pos" in out.columns):
        out["Pos_Change"] = out["400_Pos"] - out["Finish_Pos"]
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

# ---------- UI ----------

st.title("🏇 Sectional Energy Model — Race Analyzer")

with st.sidebar:
    uploaded = st.file_uploader("Upload Race File (CSV or XLSX)", type=["csv", "xlsx"])
    distance_m = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=50)
    wind_dir = st.selectbox("Wind Direction", ["None", "Head", "Tail"])
    wind_speed = st.number_input("Wind Speed (kph)", min_value=0.0, max_value=80.0, value=0.0, step=1.0)
    apply_wind = st.checkbox("Apply wind adjustment to Final 400", value=False)

if uploaded is None:
    st.info("Upload a race file to begin.")
    st.stop()

# Load
if uploaded.name.lower().endswith(".csv"):
    raw = pd.read_csv(uploaded)
else:
    raw = pd.read_excel(uploaded)

raw = raw.rename(columns={
    "Race time": "Race Time",
    "Race_Time": "Race Time",
    "RaceTime": "Race Time",
    "800_400": "800-400",
    "400_Finish": "400-Finish",
    "Horse Name": "Horse",
    "Finish": "Finish_Pos"
})

df = raw.copy()
df["RaceTime_s"] = df["Race Time"].apply(parse_race_time)
for col in ["800-400", "400-Finish"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
for col in ["200_Pos", "400_Pos", "Finish_Pos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if detect_halved_times(df):
    st.warning("⚠️ Sectional times look halved. Tick box to fix.")
    if st.checkbox("Fix halved times (×2)", value=True):
        for col in ["800-400", "400-Finish"]:
            if col in df.columns:
                df[col] = df[col] * 2
        df["RaceTime_s"] = df["RaceTime_s"] * 2

metrics = compute_metrics(df, distance_m=distance_m, apply_wind=apply_wind,
                          wind_speed_kph=wind_speed, wind_dir=wind_dir)
metrics = flag_sleepers(metrics)
disp = round_display(metrics).sort_values("Finish_Pos", na_position="last")

st.subheader("Sectional Metrics")
st.dataframe(disp, use_container_width=True)

st.subheader("Pace Curve (Field Averages)")
avg_mid = metrics["Mid400_Speed"].mean()
avg_fin = metrics["Final400_Speed"].mean()
fig1, ax1 = plt.subplots()
ax1.plot([1, 2], [avg_mid, avg_fin], marker="o")
ax1.set_xticks([1, 2])
ax1.set_xticklabels(["Mid 400", "Final 400"])
ax1.set_ylabel("Speed (m/s)")
st.pyplot(fig1)

sleepers = disp[disp["Sleeper"] == True][["Horse","Finish_Pos","Refined_FSP_%","Pos_Change"]]
st.subheader("Sleepers")
st.write(sleepers if not sleepers.empty else "No clear sleepers flagged.")
