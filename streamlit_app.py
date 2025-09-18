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
st.caption("Upload CSV or use Manual input with countdown 200 m segments. Calculates SPI / Basic FSP / Refined FSP and pace curves.")

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
# Time parsing (handles seconds, M:SS.ms, M:SS:ms like 01:37:620, H:MM:SS(.ms))
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
# Metric helpers
# =========================================================
def compute_metrics(df, distance_m=1400.0):
    """Assumes df has numeric RaceTime_s, 800-400, 400-Finish columns."""
    out = df.copy()

    for c in ["RaceTime_s", "800-400", "400-Finish"]:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")

    with np.errstate(divide='ignore', invalid='ignore'):
        out["Race_AvgSpeed"]  = distance_m / out["RaceTime_s"]
        out["Mid400_Speed"]   = 400.0 / out["800-400"]
        out["Final400_Speed"] = 400.0 / out["400-Finish"]

        out["SPI_%"]         = (out["Mid400_Speed"]   / out["Race_AvgSpeed"]) * 100.0
        out["Basic_FSP_%"]   = (out["Final400_Speed"] / out["Race_AvgSpeed"]) * 100.0
        out["Refined_FSP_%"] = (out["Final400_Speed"] / out["Mid400_Speed"]) * 100.0

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

def round_display(df):
    df = df.copy()
    for c in ["Basic_FSP_%", "Refined_FSP_%", "SPI_%", "Race_AvgSpeed", "Mid400_Speed", "Final400_Speed"]:
        if c in df.columns:
            df.loc[:, c] = df[c].round(2)
    if "RaceTime_s" in df.columns:
        df.loc[:, "RaceTime_s"] = df["RaceTime_s"].round(3)
    return df

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
    # Segment columns are everything except 'Horse' and 'Finish_Pos'
    seg_cols = [c for c in df.columns if c not in ("Horse", "Finish_Pos")]

    # Parse each segment to seconds
    for c in seg_cols:
        df[c] = pd.to_numeric(df[c].apply(parse_time_any), errors="coerce")

    # RaceTime_s = sum of segments
    df["RaceTime_s"] = df[seg_cols].sum(axis=1, min_count=len(seg_cols))

    # Build mid- and final-400 sums from tail segments
    mid_cols, fin_cols = segments_to_mid_final_400(seg_cols)
    if not mid_cols or not fin_cols:
        st.error("Distance must be at least 800 m to compute Mid400 and Final400.")
        st.stop()

    df["800-400"]    = df[mid_cols].sum(axis=1, min_count=2)
    df["400-Finish"] = df[fin_cols].sum(axis=1, min_count=2)

    # Optional Finish_Pos
    if "Finish_Pos" in df.columns:
        df["Finish_Pos"] = pd.to_numeric(df["Finish_Pos"], errors="coerce").astype("Int64")

    # >>> KEY FIX: alias RaceTime_s to 'Race Time' to avoid KeyError downstream
    if "Race Time" not in df.columns and "RaceTime_s" in df.columns:
        df["Race Time"] = df["RaceTime_s"]

st.subheader("Converted table (ready for analysis)")
st.dataframe(df.head(12), width="stretch")
_dbg("Dtypes", df.dtypes)

# ===================
# Analysis
# ===================
try:
    metrics = compute_metrics(df, distance_m=float(distance_m))
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG: st.exception(e)
    st.stop()

# If Finish_Pos missing, rank by time for display order
if "Finish_Pos" not in metrics.columns or metrics["Finish_Pos"].isna().all():
    metrics["Finish_Pos"] = metrics["RaceTime_s"].rank(method="min").astype("Int64")

st.subheader("Sectional Metrics")
disp = round_display(metrics.copy()).sort_values("Finish_Pos", na_position="last")
st.dataframe(disp, width="stretch")

# ===================
# Pace Curves
# ===================
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

# ===================
# Download
# ===================
st.subheader("Download")
csv_bytes = disp.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics as CSV", data=csv_bytes, file_name="race_sectional_metrics.csv", mime="text/csv")

st.caption(
    "Manual mode uses countdown 200 m segment times. We derive RaceTime (seconds), Mid400 and Final400, then compute SPI / Basic FSP / Refined FSP. "
    "Upload mode expects: Horse, Race Time, 800-400, 400-Finish (Finish_Pos optional)."
)
