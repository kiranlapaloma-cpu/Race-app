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
    elif ("1000_Pos" in out.columns) and ("800_Pos" in out.columns):
        out["EPI"] = out["1000_Pos"] * 0.6 + out["800_Pos"] * 0.4
    elif "400_Pos" in out.columns:
        out["EPI"] = out["400_Pos"]
    else:
        out["EPI"] = np.nan

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

# ---------- Pace-aware Group-Class Index (GCI v2) ----------

def _to01(val, lo, hi):
    if pd.isna(val):
        return 0.0
    if hi == lo:
        return 0.0
    return float(min(1.0, max(0.0, (val - lo) / (hi - lo))))

def compute_gci_v2(row, spi_median, winner_time_s=None):
    reasons = []
    refined = row.get("Refined_FSP_%", np.nan)
    basic   = row.get("Basic_FSP_%",   np.nan)
    spi     = row.get("SPI_%",         np.nan)
    epi     = row.get("EPI",           np.nan)
    rtime   = row.get("RaceTime_s",    np.nan)

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

    # LK: Late kick (pace-adjusted)
    LK_raw = 0.0
    if not pd.isna(refined):
        if refined >= 106: LK_raw = 1.0; reasons.append("elite late kick (RefFSP≥106)")
        elif refined >= 104: LK_raw = 0.8; reasons.append("strong late kick (104–106)")
        elif refined >= 102: LK_raw = 0.6; reasons.append("good late kick (102–104)")
        elif refined >= 100: LK_raw = 0.4; reasons.append("useful late work (100–102)")
        else: LK_raw = 0.2
    LK = LK_raw * (0.7 if sprint_home else 1.0)

    # OP: On-pace merit
    OP = 0.0
    if not pd.isna(epi) and not pd.isna(basic):
        if fast_early and epi <= 2.5 and basic >= 98:
            OP = 1.0; reasons.append("on-speed under pressure (fast early)")
        elif not fast_early and epi <= 3.5 and basic >= 99:
            OP = 0.7; reasons.append("efficient on-speed/handy")

    P = max(LK, OP)

    # SS: Sustained speed
    if pd.isna(spi) or pd.isna(basic):
        SS = 0.0
    else:
        mean_sb = (spi + basic) / 2.0
        SS = _to01(mean_sb, 98.0, 105.0)
        if mean_sb >= 103:
            reasons.append("strong sustained speed")

    # EFF: Efficiency (around 100% Refined FSP)
    if pd.isna(refined):
        EFF = 0.0
    else:
        EFF = max(0.0, 1.0 - abs(refined - 100.0) / 8.0)
        if 99 <= refined <= 103:
            reasons.append("efficient sectional profile")

    wT, wP, wSS, wEFF = 0.30, 0.30, 0.25, 0.15
    score01 = (wT * T) + (wP * P) + (wSS * SS) + (wEFF * EFF)
    score10 = round(10.0 * score01, 2)
    return score10, reasons

# ---------- Runner-by-runner summaries ----------

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
    gci  = row.get("GCI", np.nan)
    gcand= bool(row.get("Group_Candidate", False))

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
    gci_txt = f"{gci:.1f}" if not pd.isna(gci) else "—"

    tags = []
    if slp:  tags.append("Sleeper")
    if gcand: tags.append("Group candidate")
    tag_str = f" **[{', '.join(tags)}]**" if tags else ""

    bits = [
        f"**{name}** — Finish: **{fin_txt}**, Time: **{rt_txt}**{tag_str}",
        f"Style: {style} (EPI {epi_txt}); Race shape: {pnote}.",
        f"Late profile: {kick} (Refined FSP {ref_txt}); Pos change 400–Finish: {pc_txt}; GCI: {gci_txt}.",
    ]
    if dhint:
        bits.append(f"Note: {dhint}")
    return "  \n".join(bits)

# ---------- UI ----------

st.title("🏇 Race Analysis by Kiran")
st.caption("Upload a race CSV/XLSX, compute FSP, Refined FSP, SPI, EPI, sleepers, simulate pace/wind, and read per-runner summaries. GCI v2 is pace-aware.")

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
    "Placing": "Finish_Pos",
    "Drawn": "Draw",  # normalize common variants
    "Barrier": "Draw"
})

df = raw.copy()
df["RaceTime_s"] = df["Race Time"].apply(parse_race_time)
for col in ["800-400", "400-Finish", "200_Pos", "400_Pos", "800_Pos", "1000_Pos", "Finish_Pos", "Draw"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col != "Horse" else df[col]

# Apply pace scenario & compute metrics
df_scn = apply_pace_scenario(df, scenario=pace_scenario)
metrics = compute_metrics(df_scn, distance_m=distance_m, apply_wind=apply_wind, wind_speed_kph=wind_speed, wind_dir=wind_dir)
metrics = flag_sleepers(metrics)

# Drop Jockey/Trainer if present
for drop_col in ["Jockey", "Trainer"]:
    if drop_col in metrics.columns:
        metrics = metrics.drop(columns=[drop_col])

# ---- Group-class scoring (pace-aware v2) ----
winner_time = None
if "Finish_Pos" in metrics.columns and "RaceTime_s" in metrics.columns and not metrics["Finish_Pos"].isna().all():
    try:
        winner_time = metrics.loc[metrics["Finish_Pos"].idxmin(), "RaceTime_s"]
    except Exception:
        winner_time = None

spi_median = metrics["SPI_%"].median()
gci_scores, gci_reasons = [], []
for _, r in metrics.iterrows():
    gci, why = compute_gci_v2(r, spi_median, winner_time_s=winner_time)
    gci_scores.append(gci)
    gci_reasons.append("; ".join(why))
metrics["GCI"] = gci_scores
metrics["GCI_Reasons"] = gci_reasons
metrics["Group_Candidate"] = metrics["GCI"] >= 7.0

# ---------- Table (Horse/Draw frozen workaround + search) ----------
st.subheader("Sectional Metrics")
# Quick filter
col_search1, col_search2 = st.columns([1, 3])
with col_search1:
    q = st.text_input("Filter by horse name (contains):", value="")
if q:
    filt = metrics["Horse"].astype(str).str.contains(q, case=False, na=False) if "Horse" in metrics.columns else pd.Series([True]*len(metrics))
    metrics_disp = metrics.loc[filt].copy()
else:
    metrics_disp = metrics.copy()

# Reorder so Horse/Draw are first; then duplicate at right for readability while scrolled
left_cols = [c for c in ["Horse", "Draw"] if c in metrics_disp.columns]
other_cols = [c for c in metrics_disp.columns if c not in left_cols]
disp = metrics_disp[left_cols + other_cols].copy()
# Append duplicates at the end (if present)
if "Horse" in metrics_disp.columns:
    disp["Horse↔"] = metrics_disp["Horse"]
if "Draw" in metrics_disp.columns:
    disp["Draw↔"] = metrics_disp["Draw"]

disp = round_display(disp).sort_values("Finish_Pos", na_position="last")
st.dataframe(disp, use_container_width=True)

# ---------- Combined Pace Chart (avg + first 8 finishers) ----------
st.subheader("Combined Pace Curve — Average (black) + First 8 Finishers")
avg_mid = metrics["Mid400_Speed"].mean()
avg_fin = metrics["Final400_Speed"].mean()
top8 = metrics.sort_values("Finish_Pos").head(8)

fig, ax = plt.subplots()
x_vals = [1, 2]
# Average in thick black
ax.plot(x_vals, [avg_mid, avg_fin], marker="o", color="black", linewidth=3, label="Field Average")

# First 8 finishers in distinct colours
colors = plt.cm.tab10.colors
for idx, (_, row) in enumerate(top8.iterrows()):
    y_vals = [row["Mid400_Speed"], row["Final400_Speed"]]
    ax.plot(x_vals, y_vals, marker="o", linewidth=2, color=colors[idx % len(colors)], label=str(row["Horse"]))

ax.set_xticks([1, 2])
ax.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
ax.set_ylabel("Speed (m/s)")
ax.set_title("Pace Curves")
ax.grid(True, linestyle="--", alpha=0.3)
fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.18))
st.pyplot(fig)

# ---------- Insights & Sleepers ----------
st.subheader("Insights")
if pd.isna(spi_median):
    st.write("Insufficient data to infer race shape.")
else:
    if spi_median >= 103:
        race_shape = "Fast Early / Tough Late"
    elif spi_median <= 97:
        race_shape = "Slow Early / Sprint-Home"
    else:
        race_shape = "Even"
    st.write(f"**Race shape (by SPI median):** {race_shape} (median SPI {spi_median:.1f}%)")

sleepers = disp[disp["Sleeper"] == True][["Horse","Finish_Pos","Refined_FSP_%","Pos_Change"]] if "Sleeper" in disp.columns else pd.DataFrame()
st.subheader("Sleepers")
if sleepers is None or sleepers.empty:
    st.write("No clear sleepers flagged under current thresholds.")
else:
    st.dataframe(sleepers, use_container_width=True)

# ---------- Group-class candidates ----------
st.subheader("Potential Group-class candidates (GCI v2)")
cands = metrics.loc[metrics["Group_Candidate"]].copy().sort_values(["GCI","Finish_Pos"], ascending=[False, True])
if cands.empty:
    st.write("No horses met the Group-class threshold today.")
else:
    st.dataframe(
        round_display(cands[["Horse","Finish_Pos","RaceTime_s","Refined_FSP_%","Basic_FSP_%","SPI_%","Pos_Change","EPI","GCI","GCI_Reasons"]]),
        use_container_width=True
    )

# ---------- Runner-by-runner summaries ----------
st.subheader("Runner-by-runner summaries")
ordered = metrics.sort_values("Finish_Pos", na_position="last")
for _, row in ordered.iterrows():
    st.markdown(runner_summary(row, spi_median))
    st.markdown("---")

# ---------- Download ----------
st.subheader("Download")
csv_bytes = disp.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics as CSV", data=csv_bytes, file_name="race_sectional_metrics.csv", mime="text/csv")

st.caption(
    "Legend: Basic FSP = Final400 / Race Avg; Refined FSP = Final400 / Mid400; "
    "SPI = Mid400 / Race Avg; EPI = early positioning index (lower = closer to lead); "
    "Pos_Change = gain from 400m to finish. GCI v2 is pace-aware (≥7.0 flags candidates)."
)
