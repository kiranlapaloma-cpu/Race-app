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

def compute_metrics(df, distance_m=1400.0):
    out = df.copy()
    out["Race_AvgSpeed"]  = distance_m / out["RaceTime_s"]
    out["Mid400_Speed"]   = 400.0 / out["800-400"]
    out["Final400_Speed"] = 400.0 / out["400-Finish"]

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

# ---------- Pace-Pressure Context & GCI v3 ----------

def _to01(val, lo, hi):
    """Clamp-linear map val in [lo,hi] to [0,1]."""
    if pd.isna(val):
        return 0.0
    if hi == lo:
        return 0.0
    return float(min(1.0, max(0.0, (val - lo) / (hi - lo))))

def compute_pressure_context(metrics_df):
    """
    Returns dict with:
      - early_heat (0..1): higher if SPI median >> 100 (fast early),
      - pressure_ratio (0..1): share of field that raced handy/on-speed (EPI ≤ 3.0),
      - field_size (int), spi_median (float)
    """
    ctx = {}
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
    ctx.update({
        "early_heat": early_heat,
        "pressure_ratio": pressure_ratio,
        "field_size": field_size,
        "spi_median": spi_med
    })
    return ctx

def compute_gci_v3(row, ctx, winner_time_s=None):
    """
    Pace-pressure adjusted Group-Class Index on 0–10 scale.
    Returns (score_0_10, reasons_list)
    """
    reasons = []
    refined = row.get("Refined_FSP_%", np.nan)
    basic   = row.get("Basic_FSP_%",   np.nan)
    spi     = row.get("SPI_%",         np.nan)
    epi     = row.get("EPI",           np.nan)
    rtime   = row.get("RaceTime_s",    np.nan)

    early_heat   = float(ctx.get("early_heat", 0.0))        # 0..1
    pressure_rat = float(ctx.get("pressure_ratio", 0.0))    # 0..1
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

    # Late Quality (LQ) 0..1 from refined/basic
    def map_pct(x): return max(0.0, min(1.0, (x - 98.0) / 6.0))  # 98→0, 104→1
    LQ = 0.0
    if not pd.isna(refined) and not pd.isna(basic):
        LQ = 0.6 * map_pct(refined) + 0.4 * map_pct(basic)
        if refined >= 103: reasons.append("strong late profile")
        elif refined >= 101.5: reasons.append("useful late profile")

    # On-pace Merit (OP) with pressure gates
    OP = 0.0
    if not pd.isna(epi) and not pd.isna(basic):
        on_speed = (epi <= 2.5)
        handy    = (epi <= 3.5)
        if handy and basic >= 99:
            OP = 0.5
        if on_speed and basic >= 100:
            OP = max(OP, 0.7)
        if on_speed and fast_early and (early_heat >= 0.7) and (pressure_rat >= 0.35) and basic >= 100:
            OP = max(OP, 1.0)
            reasons.append("on-speed under genuine heat & pressure")

    # Leader Tax (LT)
    LT = 0.0
    if not pd.isna(epi) and epi <= 2.0:
        soft_early = (early_heat < 0.5)   # SPI ≲ ~101.5
        low_press  = (pressure_rat < 0.30)
        weak_late  = (pd.isna(refined) or refined < 100.0) or (pd.isna(basic) or basic < 100.0)
        if soft_early: LT += 0.25
        if low_press:  LT += 0.20
        if weak_late:  LT += 0.20
        if sprint_home: LT += 0.15
        LT = min(0.60, LT)
        if LT > 0:
            reasons.append(f"leader tax applied ({LT:.2f})")

    # Sustained Speed (SS)
    if pd.isna(spi) or pd.isna(basic):
        SS = 0.0
    else:
        mean_sb = (spi + basic) / 2.0
        SS = _to01(mean_sb, 98.0, 105.0)
        if mean_sb >= 103:
            reasons.append("strong sustained speed")

    # Efficiency around 100% Refined FSP (EFF)
    if pd.isna(refined):
        EFF = 0.0
    else:
        EFF = max(0.0, 1.0 - abs(refined - 100.0) / 8.0)
        if 99 <= refined <= 103:
            reasons.append("efficient sectional profile")

    # Combine pillars
    wT, wPACE, wSS, wEFF = 0.25, 0.35, 0.25, 0.15
    PACE = max(LQ, OP) * (1.0 - LT)
    if LQ < 0.30:
        PACE = min(PACE, 0.60)  # Late-quality floor

    score01 = (wT * T) + (wPACE * PACE) + (wSS * SS) + (wEFF * EFF)
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
    # kept for narrative flavour only
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
st.caption("Upload a race CSV/XLSX, compute FSP, Refined FSP, SPI, EPI, sleepers, and read per-runner summaries. GCI v3 uses pace-pressure context. (Wind inputs removed; single combined pace chart.)")

with st.sidebar:
    st.header("Race Inputs")
    uploaded = st.file_uploader("Upload Race File (CSV or XLSX)", type=["csv", "xlsx"])
    distance_m = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=50)

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

# Compute metrics
metrics = compute_metrics(df, distance_m=distance_m)
metrics = flag_sleepers(metrics)

# Drop Jockey/Trainer if present
for drop_col in ["Jockey", "Trainer"]:
    if drop_col in metrics.columns:
        metrics = metrics.drop(columns=[drop_col])

# ---- Group-class scoring (v3 with pace-pressure context) ----
winner_time = None
if "Finish_Pos" in metrics.columns and "RaceTime_s" in metrics.columns and not metrics["Finish_Pos"].isna().all():
    try:
        winner_time = metrics.loc[metrics["Finish_Pos"].idxmin(), "RaceTime_s"]
    except Exception:
        winner_time = None

ctx = compute_pressure_context(metrics)
spi_median = ctx["spi_median"]
gci_scores, gci_reasons = [], []
for _, r in metrics.iterrows():
    gci, why = compute_gci_v3(r, ctx, winner_time_s=winner_time)
    gci_scores.append(gci)
    gci_reasons.append("; ".join(why))
metrics["GCI"] = gci_scores
metrics["GCI_Reasons"] = gci_reasons
metrics["Group_Candidate"] = metrics["GCI"] >= 7.0

# ---------- Table ----------
st.subheader("Sectional Metrics")
disp = round_display(metrics.copy()).sort_values("Finish_Pos", na_position="last")
st.dataframe(disp, use_container_width=True)

# ---------- Combined Pace Chart (Average + Top 8) ----------
st.subheader("Pace Curves — Field Average (black) + Top 8 by Finish")

avg_mid = metrics["Mid400_Speed"].mean()
avg_fin = metrics["Final400_Speed"].mean()
top8 = metrics.sort_values("Finish_Pos").head(8)

fig, ax = plt.subplots()
x_vals = [1, 2]

# Field average in black (thicker line)
ax.plot(x_vals, [avg_mid, avg_fin], marker="o", linewidth=3, color="black", label="Average (Field)")

# Top 8 finishers in distinctive colours
for _, row in top8.iterrows():
    ax.plot(
        x_vals,
        [row["Mid400_Speed"], row["Final400_Speed"]],
        marker="o",
        linewidth=2,
        label=str(row.get("Horse", "Runner"))
    )

ax.set_xticks([1, 2])
ax.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
ax.set_ylabel("Speed (m/s)")
ax.set_title("Average vs Top 8 Pace Curves")
ax.grid(True, linestyle="--", alpha=0.3)

# Legend below the chart
fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
st.pyplot(fig)

# ---------- Insights & Sleepers ----------
st.subheader("Insights")
if pd.isna(spi_median):
    st.write("Insufficient data to infer pace context.")
else:
    st.write(f"**SPI median:** {spi_median:.1f}%  |  "
             f"**Early heat (0–1):** {ctx['early_heat']:.2f}  |  "
             f"**Pressure ratio (EPI≤3.0):** {ctx['pressure_ratio']:.2f}  |  "
             f"**Field size:** {ctx['field_size']}")

sleepers = disp[disp["Sleeper"] == True][["Horse","Finish_Pos","Refined_FSP_%","Pos_Change"]]
st.subheader("Sleepers")
if sleepers.empty:
    st.write("No clear sleepers flagged under current thresholds.")
else:
    st.dataframe(sleepers, use_container_width=True)

# ---------- Group-class candidates ----------
st.subheader("Potential Group-class candidates (GCI v3)")
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
    "Pos_Change = gain from 400m to finish. GCI v3 applies pace-pressure context."
)
