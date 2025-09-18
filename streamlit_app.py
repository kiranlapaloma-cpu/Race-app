import io
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

st.set_page_config(
    page_title="The Sharpest Edge",
    page_icon=str(LOGO_PATH) if LOGO_PATH else None,
    layout="wide",
)

if LOGO_PATH:
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
# Helpers: parsing, headers, metrics, GCI v3.1
# =========================================================
def parse_race_time(val):
    """Parse Race Time as seconds or 'MM:SS.ms' (e.g., 1:12.49)."""
    if pd.isna(val):
        return np.nan
    # already seconds?
    try:
        return float(val)
    except Exception:
        pass
    s = str(val).strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            m, sec, ms = parts
            return int(m) * 60 + int(sec) + int(ms) / 1000.0
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except Exception:
        return np.nan
    return np.nan

def _norm_header(s: str) -> str:
    """Lowercase, strip spaces/commas to match variants like '1,000 m' → '1000m'."""
    s = str(s).strip().lower()
    s = s.replace(",", "").replace(" ", "")
    return s

def _has_200m_headers(df: pd.DataFrame) -> bool:
    norm_cols = {_norm_header(c) for c in df.columns}
    keys = {"1000m", "800m", "600m", "400m", "200m"}
    return len(keys & norm_cols) > 0

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

    score01 = (wT * T) + (wPACE * PACE) + (wSS * SS) + (wEFF * EFF)
    score10 = round(10.0 * score01, 2)
    return score10, reasons

# Narrative helpers
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

    bits = [
        f"**{name}** — Finish: **{int(fin) if not pd.isna(fin) else '—'}**, Time: **{rt:.2f}s**",
        f"Style: {style_from_epi(epi)} (EPI {epi:.1f} if known); Race shape: {pace_note(spi_med)}.",
        f"Late profile: {kick_from_refined(ref)} (Refined FSP {ref:.1f}% if known); "
        f"Pos change 400–Finish: {int(pc) if not pd.isna(pc) else '—'}; "
        f"GCI: {gci:.1f} if known."
    ]
    dh = distance_hint(ref, pc, epi)
    if dh: bits.append(f"Note: {dh}")
    tag = []
    if slp: tag.append("Sleeper")
    if gcand: tag.append("Group candidate")
    if tag: bits[0] += f" **[{', '.join(tag)}]**"
    return "  \n".join(bits)

# ===================================================
# TPD 200m converter & paste handling (with spill/fused)
# ===================================================
TIME_RE = re.compile(r"(\d+:\d{2}\.\d{1,3}|\d+\.\d{1,3})")

def _last_float(cell):
    if pd.isna(cell): return np.nan
    m = re.findall(r"\d+\.\d{1,3}", str(cell))
    return float(m[-1]) if m else np.nan

def _first_text(cell):
    if pd.isna(cell): return ""
    return str(cell).splitlines()[0].strip()

def _first_int_and_time(cell):
    """
    Handles fused “pos+time” like '158.37' -> (1, 58.37)
    Also handles '9 1:00.09' -> (9, 60.09), 'NR' -> (nan, nan)
    """
    if pd.isna(cell):
        return np.nan, np.nan
    s = str(cell).strip()
    if s.upper() == "NR":
        return np.nan, np.nan
    # fused 1–2 digits + float
    m = re.match(r"^(\d{1,2})(\d+\.\d{1,3})$", s)
    if m:
        return int(m.group(1)), float(m.group(2))
    # general case: find first integer and first time token
    m_time = TIME_RE.findall(s)
    m_pos  = re.findall(r"\b\d+\b", s)
    pos = int(m_pos[0]) if m_pos else np.nan
    if m_time:
        t = m_time[0]
        if ":" in t:
            mm, rest = t.split(":")
            return pos, int(mm) * 60 + float(rest)
        return pos, float(t)
    return pos, np.nan

def convert_200m_table_to_app(df):
    """
    df is expected to have normalized headers (lowercase, no commas/spaces), e.g.:
    'horse', 'result'/'finish'/'time', '1000m','800m','600m','400m','200m'
    """
    cols = {str(c).lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            n = n.lower()
            if n in cols:
                return cols[n]
        return None

    col_horse  = pick("horse", "runner", "name")
    col_result = pick("result", "finish", "time")

    col_800    = pick("800m")
    col_600    = pick("600m")
    col_400    = pick("400m")
    col_200    = pick("200m")

    out = pd.DataFrame()
    out["Horse"] = (df[col_horse].apply(_first_text) if col_horse else df.iloc[:, 0].apply(_first_text))

    if col_result:
        ft = df[col_result].apply(_first_int_and_time)
        out["Finish_Pos"] = ft.apply(lambda x: x[0])
        out["Race Time"]  = ft.apply(lambda x: x[1])
    else:
        out["Finish_Pos"] = np.nan
        out["Race Time"]  = np.nan

    seg_800 = df[col_800].apply(_last_float) if col_800 else np.nan
    seg_600 = df[col_600].apply(_last_float) if col_600 else np.nan
    seg_400 = df[col_400].apply(_last_float) if col_400 else np.nan
    seg_200 = df[col_200].apply(_last_float) if col_200 else np.nan

    out["800-400"]    = seg_800 + seg_600        # 800→600 + 600→400
    out["400-Finish"] = seg_400 + seg_200        # 400→200 + 200→Finish

    for c in ["Finish_Pos", "Race Time", "800-400", "400-Finish"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ---------- Unsquasher for TPD pastes (handles “spill” and fused tokens) ----------
def _looks_squashed_tpd(txt: str) -> bool:
    first_nonempty = next((ln for ln in txt.splitlines() if ln.strip()), "")
    return all(k in first_nonempty for k in ["800m", "600m", "400m", "200m"])

def _unsquash_tpd(txt: str) -> str:
    """
    Convert the 'squashed' TPD paste into TSV:
    Result | No.(Draw) | Horse | 800m | 600m | 400m | 200m | Finish

    Handles:
      • Fused tokens like '158.37' -> '58.37' (pos handled later)
      • 'Number spill' of floats sitting after the header; distributes 5 per runner.
    """
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return txt

    float_pat = re.compile(r"\d+\.\d{1,3}")

    def _defuse(tok: str) -> str:
        m = re.match(r"^(\d{1,2})(\d+\.\d{1,3})$", tok)
        return m.group(2) if m else tok

    # 1) locate header line (contains all labels)
    hdr_idx = 0
    for i, ln in enumerate(lines):
        if all(k in ln for k in ["800m", "600m", "400m", "200m", "Finish"]):
            hdr_idx = i
            break

    header_line = lines[hdr_idx]
    rows = lines[hdr_idx + 1:]

    # 2) try to harvest a "spill" of floats from the tail of the header line
    #    i.e., after the last label 'Finish'
    spill = float_pat.findall(header_line.split("Finish")[-1])

    # or, if the first row after header is just a block of floats, use that instead
    if not spill and rows:
        tokens = rows[0].split()
        if tokens and all(float_pat.search(t) or t in {"--", "—"} for t in tokens):
            spill = float_pat.findall(rows[0])
            rows = rows[1:]

    header = ["Result", "No.(Draw)", "Horse", "800m", "600m", "400m", "200m", "Finish"]
    out = ["\t".join(header)]

    # 3) helper to distribute 5 floats per runner
    def get_segments(i):
        if not spill:
            return [""] * 5
        start = i * 5
        end = start + 5
        if end <= len(spill):
            return spill[start:end]
        return [""] * 5

    # 4) parse runner rows
    for i, ln in enumerate(rows):
        toks = [_defuse(t) for t in ln.strip().split()]
        if not toks:
            continue

        # Result token: may be 'NR', '1:12.49', '58.37', '158.37' etc
        result = toks.pop(0) if toks else ""
        # No.(Draw): like 3(8)
        nd = toks.pop(0) if toks and "(" in toks[0] else ""
        # Horse = the rest (trainer/jockey usually included; fine)
        horse = " ".join(toks).strip()

        segs = get_segments(i)
        out.append("\t".join([result, nd, horse] + segs))

    return "\n".join(out)

# ===================
# UI: Inputs & Flow
# ===================
st.title("🏇 The Sharpest Edge")
st.caption("Upload CSV/XLSX, paste a TPD-style sectional table, or load from a URL with static HTML. 200m splits are auto-converted to 400m metrics and GCI v3.1 is computed.")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input type", ["Upload file", "Web URL", "Paste table"], index=0)
    distance_m = st.number_input("Race Distance (m)", min_value=800, max_value=4000, value=1400, step=50)

df_raw = None

try:
    if source == "Upload file":
        uploaded = st.file_uploader("Upload CSV/XLSX (any jurisdiction)", type=["csv", "xlsx"])
        if uploaded is None:
            st.info("Upload a file or switch to 'Web URL' or 'Paste table'.")
            st.stop()
        df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        st.success("File loaded.")

    elif source == "Web URL":
        url = st.text_input("Enter a page URL (static HTML table) or a direct CSV/XLSX link")
        if not url:
            st.stop()
        try:
            if url.lower().endswith(".csv"):
                df_raw = pd.read_csv(url)
                st.success("Loaded CSV from URL.")
            elif url.lower().endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(url)
                st.success("Loaded Excel from URL.")
            else:
                tables = pd.read_html(url)  # only static tables
                if not tables:
                    raise ValueError("No tables in static HTML.")
                df_raw = max(tables, key=lambda t: t.shape[0] * t.shape[1])
                st.success("Loaded HTML table from URL.")
        except Exception as e:
            st.error("Could not extract a table from that URL (likely rendered by JavaScript).")
            if DEBUG: st.exception(e)
            st.info("Use 'Paste table' (bookmarklet) or provide a direct CSV/XLSX link.")
            st.stop()

    else:
        # ---------- Paste table (robust for compact/fused TPD text) ----------
        st.markdown("Run your bookmarklet → a white box appears → Select All → Copy → paste **everything** below.")
        pasted = st.text_area("Paste table here (raw)", height=280)
        if not pasted.strip():
            st.stop()

        if _looks_squashed_tpd(pasted):
            tsv = _unsquash_tpd(pasted)
            df_raw = pd.read_csv(io.StringIO(tsv), sep="\t")
            st.info("Detected compact TPD paste. Rebuilt into a proper table automatically.")
        else:
            # regular TSV/CSV/FWF paste
            if "\t" in pasted:
                df_raw = pd.read_csv(io.StringIO(pasted), sep="\t")
            else:
                try:
                    df_raw = pd.read_csv(io.StringIO(pasted))
                except Exception:
                    df_raw = pd.read_fwf(io.StringIO(pasted))
            st.success("Parsed pasted table.")

except Exception as e:
    st.error("Input parsing failed.")
    if DEBUG: st.exception(e)
    st.stop()

st.subheader("Raw table preview")
st.dataframe(df_raw.head(12), use_container_width=True)
_dbg("Raw columns", list(df_raw.columns))

# Normalize headers for detection/mapping
df_norm = df_raw.copy()
df_norm.columns = [_norm_header(c) for c in df_norm.columns]
looks_200m = _has_200m_headers(df_raw) or _has_200m_headers(df_norm)
_dbg("Looks like 200m table?", looks_200m)

try:
    if looks_200m:
        # Align names to normalized ones before conversion
        df_conv_input = df_raw.copy()
        df_conv_input.columns = df_norm.columns
        work = convert_200m_table_to_app(df_conv_input)
    else:
        # Fall back to app schema if user already provides it
        work = df_raw.rename(columns={
            "Race time": "Race Time", "Race_Time": "Race Time", "RaceTime": "Race Time",
            "800_400": "800-400", "400_Finish": "400-Finish",
            "Horse Name": "Horse", "Finish": "Finish_Pos", "Placing": "Finish_Pos"
        })
except Exception as e:
    st.error("Conversion to app schema failed.")
    if DEBUG: st.exception(e)
    st.stop()

# Hard guard
_required = ["Horse", "Race Time", "800-400", "400-Finish"]
_missing = [c for c in _required if c not in work.columns]
if _missing:
    st.error(
        "I couldn't produce the columns the analysis needs.\n\n"
        "Missing: " + ", ".join(_missing) +
        "\n\nTips:\n"
        "• Paste the actual table text (not a screenshot).\n"
        "• Parser handles compact TPD pastes, fused position+time tokens, and header spills.\n"
        "• Check the Raw preview above to see how headers came through."
    )
    _dbg("Workframe columns", list(work.columns))
    st.stop()

# --- Final normalization so the rest of the app works the same ---
df = work.copy()
df["RaceTime_s"] = df["Race Time"].apply(parse_race_time)
for col in ["800-400", "400-Finish", "200_Pos", "400_Pos", "800_Pos", "1000_Pos", "Finish_Pos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# If Finish_Pos missing, infer by time (best-effort)
if ("Finish_Pos" not in df.columns) or df["Finish_Pos"].isna().all():
    if df["RaceTime_s"].notna().any():
        df["Finish_Pos"] = df["RaceTime_s"].rank(method="min").astype("Int64")

st.subheader("Converted table (ready for analysis)")
st.dataframe(df.head(12), use_container_width=True)

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

# Drop optional columns that clutter
for drop_col in ["Jockey", "Trainer"]:
    if drop_col in metrics.columns:
        metrics = metrics.drop(columns=[drop_col])

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
    gci, why = compute_gci_v31(r, ctx, distance_m=distance_m, winner_time_s=winner_time)
    gci_scores.append(gci)
    gci_reasons.append("; ".join(why))
metrics["GCI"] = gci_scores
metrics["GCI_Reasons"] = gci_reasons
metrics["Group_Candidate"] = metrics["GCI"] >= 7.0

# ===================
# Outputs
# ===================
st.subheader("Sectional Metrics")
disp = round_display(metrics.copy()).sort_values("Finish_Pos", na_position="last")
st.dataframe(disp, use_container_width=True)

st.subheader("Pace Curves — Field Average (black) + Top 8 by Finish")
avg_mid = metrics["Mid400_Speed"].mean()
avg_fin = metrics["Final400_Speed"].mean()
top8 = metrics.sort_values("Finish_Pos").head(8)

fig, ax = plt.subplots()
x_vals = [1, 2]
ax.plot(x_vals, [avg_mid, avg_fin], marker="o", linewidth=3, color="black", label="Average (Field)")
for _, row in top8.iterrows():
    ax.plot(x_vals, [row["Mid400_Speed"], row["Final400_Speed"]], marker="o", linewidth=2, label=str(row.get("Horse", "Runner")))
ax.set_xticks([1, 2]); ax.set_xticklabels(["Mid 400 (800→400)", "Final 400 (400→Finish)"])
ax.set_ylabel("Speed (m/s)"); ax.set_title("Average vs Top 8 Pace Curves")
ax.grid(True, linestyle="--", alpha=0.3)
fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
st.pyplot(fig)

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

st.subheader("Sleepers")
sleepers = disp[disp["Sleeper"] == True][["Horse","Finish_Pos","Refined_FSP_%","Pos_Change"]]
if sleepers.empty:
    st.write("No clear sleepers flagged under current thresholds.")
else:
    st.dataframe(sleepers, use_container_width=True)

st.subheader("Potential Group-class candidates (GCI v3.1)")
cands = metrics.loc[metrics["Group_Candidate"]].copy().sort_values(["GCI","Finish_Pos"], ascending=[False, True])
if cands.empty:
    st.write("No horses met the Group-class threshold today.")
else:
    st.dataframe(
        round_display(cands[["Horse","Finish_Pos","RaceTime_s","Refined_FSP_%","Basic_FSP_%","SPI_%","Pos_Change","EPI","GCI","GCI_Reasons"]]),
        use_container_width=True
    )

st.subheader("Runner-by-runner summaries")
ordered = metrics.sort_values("Finish_Pos", na_position="last")
for _, row in ordered.iterrows():
    st.markdown(runner_summary(row, spi_median))
    st.markdown("---")

st.subheader("Download")
csv_bytes = disp.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics as CSV", data=csv_bytes, file_name="race_sectional_metrics.csv", mime="text/csv")

st.caption(
    "Legend: Basic FSP = Final400 / Race Avg; Refined FSP = Final400 / Mid400; "
    "SPI = Mid400 / Race Avg; EPI = early positioning index (lower = closer to lead); "
    "Pos_Change = gain from 400m to finish. GCI v3.1 applies distance-normalised, pace-pressure context."
)
