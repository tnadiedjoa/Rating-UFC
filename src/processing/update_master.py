"""
update_master.py -- Rebuild the UFC master dataset from all sources

Main function: update()
  Loads Kaggle 1 + Kaggle 2 + ufc_scraped_data.csv + fighter details,
  deduplicates and enriches, then saves data/processed/ufc_master_enriched.csv.

  The master is always rebuilt from scratch (fast — only CSV operations).
  Only the scraping step (scrape_since in ingest_data.py) is incremental.

Returns (master DataFrame, n_total_fights)
"""

import os
import re
import unicodedata
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file's location)
# ---------------------------------------------------------------------------

# src/processing/update_master.py  -> 3 dirname -> repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.dirname(_THIS_DIR))

RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
KAGGLE1_DIR = os.path.join(RAW_DIR, "kaggle_1")
KAGGLE2_DIR = os.path.join(RAW_DIR, "kaggle_2")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MASTER_PATH = os.path.join(PROC_DIR, "ufc_master_enriched.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# "X of Y" columns to split (R_ / B_ prefix added dynamically)
OF_COLS = [
    "SIG_STR", "TOTAL_STR", "TD",
    "HEAD", "BODY", "LEG",
    "DISTANCE", "CLINCH", "GROUND",
]

# Columns to keep from Kaggle 2
KAGGLE2_KEEP = [
    "RedFighter", "BlueFighter", "Date",
    "Gender", "WeightClass",
    "RedOdds", "BlueOdds", "RedExpectedValue", "BlueExpectedValue",
    "RedDecOdds", "BlueDecOdds", "RSubOdds", "BSubOdds", "RKOOdds", "BKOOdds",
    "Finish", "FinishDetails", "FinishRound", "FinishRoundTime", "TotalFightTimeSecs",
    "RMatchWCRank", "BMatchWCRank", "EmptyArena",
]

# Columns to keep from raw_fighter_details
FIGHTER_KEEP = [
    "fighter_name",
    "Height_cms", "Weight_lbs", "Reach_cms",
    "Stance", "DOB",
    "SLpM", "Str_Acc", "SApM", "Str_Def",
    "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
]


# ===========================================================================
# Helpers
# ===========================================================================

def normalize_name(name):
    """Lowercase, strip, normalize apostrophes and accents."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    # Typographic apostrophes -> simple apostrophe
    name = name.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    # Remove diacritics
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name


def parse_fight_key(r, b, date):
    """
    Unique key for a fight: sorted name pair + date.
    Returned format: "name1|name2|YYYY-MM-DD"
    """
    r_n = normalize_name(r)
    b_n = normalize_name(b)
    pair = "|".join(sorted([r_n, b_n]))
    date_str = ""
    if date is not None and not (isinstance(date, float) and pd.isna(date)):
        if isinstance(date, str) and date.strip():
            try:
                date_str = pd.to_datetime(date, dayfirst=False).strftime("%Y-%m-%d")
            except Exception:
                date_str = date.strip()[:10]
        elif hasattr(date, "strftime"):
            try:
                date_str = date.strftime("%Y-%m-%d")
            except Exception:
                pass
    return f"{pair}|{date_str}"


def parse_name_key(r, b):
    """Secondary key without date (fallback for fights with unknown date)."""
    r_n = normalize_name(r)
    b_n = normalize_name(b)
    return "|".join(sorted([r_n, b_n]))


def split_of(value):
    """
    Converts "X of Y" to dict {"_landed": X, "_attempted": Y}.
    Returns 0/0 on failure.
    """
    if not isinstance(value, str):
        return {"_landed": 0, "_attempted": 0}
    m = re.match(r"(\d+)\s+of\s+(\d+)", value.strip(), re.IGNORECASE)
    if m:
        return {"_landed": int(m.group(1)), "_attempted": int(m.group(2))}
    return {"_landed": 0, "_attempted": 0}


def time_to_seconds(t):
    """
    Converts "M:SS" (or "H:MM:SS") to seconds (int).
    Also accepts numeric values.
    """
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return 0
    if isinstance(t, (int, float)):
        return int(t)
    parts = str(t).strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return 0


def pct_to_float(p):
    """
    Converts "47%" to 0.47.
    Returns NaN on failure.
    """
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return float("nan")
    if isinstance(p, (int, float)):
        return float(p) if float(p) <= 1.0 else float(p) / 100.0
    s = str(p).strip().replace("%", "")
    try:
        return float(s) / 100.0
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Physical conversion helpers
# ---------------------------------------------------------------------------

def _height_to_cms(h):
    """Converts '5\\' 11\"' to centimeters (float)."""
    if not isinstance(h, str) or not h.strip():
        return float("nan")
    m = re.match(r"(\d+)'\s*(\d*)", h.strip())
    if m:
        feet   = int(m.group(1))
        inches = int(m.group(2)) if m.group(2) else 0
        return round((feet * 12 + inches) * 2.54, 1)
    return float("nan")


def _weight_to_lbs(w):
    """Converts '185 lbs.' to float."""
    if not isinstance(w, str) or not w.strip():
        return float("nan")
    m = re.match(r"([\d.]+)\s*lbs", w.strip(), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return float("nan")


def _reach_to_cms(r):
    """Converts '72\"' to centimeters (float)."""
    if not isinstance(r, str) or not r.strip():
        return float("nan")
    m = re.match(r"([\d.]+)", r.strip())
    if m:
        return round(float(m.group(1)) * 2.54, 1)
    return float("nan")


# ===========================================================================
# Common cleaning for a fight DataFrame
# ===========================================================================

def clean_fight_df(df):
    """
    Applies to a raw fight DataFrame:
      - Renames columns with trailing period (R_SIG_STR. -> R_SIG_STR)
      - Splits "X of Y" for each OF_COLS column for R and B sides
      - Converts CTRL -> CTRL_sec (seconds)
      - Converts SIG_STR_pct and TD_pct to float [0..1]

    Modifies the DataFrame in place and returns it.
    """
    df = df.copy()

    for side in ("R", "B"):
        # -- Split "X of Y" -------------------------------------------------
        for col in OF_COLS:
            raw_col = f"{side}_{col}"
            # Kaggle1 stores some columns with a trailing period
            alt_col = raw_col + "."
            if alt_col in df.columns and raw_col not in df.columns:
                df.rename(columns={alt_col: raw_col}, inplace=True)

            if raw_col in df.columns:
                parsed = df[raw_col].apply(split_of)
                df[f"{raw_col}_landed"]    = parsed.apply(lambda x: x["_landed"])
                df[f"{raw_col}_attempted"] = parsed.apply(lambda x: x["_attempted"])
                df.drop(columns=[raw_col], inplace=True)

        # -- CTRL -> CTRL_sec -----------------------------------------------
        ctrl_col = f"{side}_CTRL"
        if ctrl_col in df.columns:
            df[f"{side}_CTRL_sec"] = df[ctrl_col].apply(time_to_seconds)
            df.drop(columns=[ctrl_col], inplace=True)

        # -- SIG_STR_pct -> float -------------------------------------------
        for pct_col in (f"{side}_SIG_STR_pct", f"{side}_TD_pct"):
            if pct_col in df.columns:
                df[pct_col] = df[pct_col].apply(pct_to_float)

    return df


# ===========================================================================
# Loading data sources
# ===========================================================================

def load_kaggle1():
    """
    Loads raw_total_fight_data.csv (separator ';').
    Cleans, renames. Returns an empty DataFrame if the file is missing.
    """
    path = os.path.join(KAGGLE1_DIR, "raw_total_fight_data.csv")
    if not os.path.exists(path):
        print(f"[update_master] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, sep=";", low_memory=False)

    # Normalize fighter column casing
    rename_map = {}
    for old, new in [("R_fighter", "R_Fighter"), ("B_fighter", "B_Fighter")]:
        if old in df.columns:
            rename_map[old] = new
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Clean fights
    df = clean_fight_df(df)

    # Win columns
    if "Winner" in df.columns:
        winner = df["Winner"].str.strip().str.lower()
        r_fighter = df["R_Fighter"].str.strip().str.lower() if "R_Fighter" in df.columns else pd.Series("", index=df.index)
        b_fighter = df["B_Fighter"].str.strip().str.lower() if "B_Fighter" in df.columns else pd.Series("", index=df.index)
        # Winner can be a fighter name or "red"/"blue"
        df["R_Win"] = ((winner == r_fighter) | (winner == "red")).astype(int)
        df["B_Win"] = ((winner == b_fighter) | (winner == "blue")).astype(int)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

    df["source"] = "kaggle1"
    return df.reset_index(drop=True)


def load_kaggle2():
    """
    Loads ufc-master.csv from Kaggle 2.
    Keeps only useful columns, renames fighter columns.
    Returns an empty DataFrame if the file is missing.
    """
    path = os.path.join(KAGGLE2_DIR, "ufc-master.csv")
    if not os.path.exists(path):
        print(f"[update_master] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Keep only available columns from KAGGLE2_KEEP
    keep = [c for c in KAGGLE2_KEEP if c in df.columns]
    df = df[keep].copy()

    # Rename
    df.rename(columns={
        "RedFighter":  "R_Fighter",
        "BlueFighter": "B_Fighter",
        "Date":        "date",
    }, inplace=True)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

    return df.reset_index(drop=True)


def load_fighter_details():
    """
    Loads raw_fighter_details.csv.
    Converts height, weight, reach and DOB.
    Returns an empty DataFrame if the file is missing.
    """
    path = os.path.join(KAGGLE1_DIR, "raw_fighter_details.csv")
    if not os.path.exists(path):
        print(f"[update_master] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Physical conversions
    if "Height" in df.columns:
        df["Height_cms"] = df["Height"].apply(_height_to_cms)
    if "Weight" in df.columns:
        df["Weight_lbs"] = df["Weight"].apply(_weight_to_lbs)
    if "Reach" in df.columns:
        df["Reach_cms"]  = df["Reach"].apply(_reach_to_cms)

    # DOB -> datetime
    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce", format="%b %d, %Y")

    # Keep only useful columns
    keep = [c for c in FIGHTER_KEEP if c in df.columns]
    df = df[keep].copy()

    return df.reset_index(drop=True)


def load_scraped():
    """
    Loads ufc_scraped_data.csv (the complete accumulated scraped dataset).
    Applies the same cleaning as Kaggle 1.
    Returns an empty DataFrame if the file is missing.
    """
    path = os.path.join(RAW_DIR, "ufc_scraped_data.csv")

    if not os.path.exists(path):
        print(f"[update_master] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Clean fights
    df = clean_fight_df(df)

    # Win columns from Status
    if "R_Status" in df.columns:
        df["R_Win"] = (df["R_Status"].str.strip().str.upper() == "W").astype(int)
    if "B_Status" in df.columns:
        df["B_Win"] = (df["B_Status"].str.strip().str.upper() == "W").astype(int)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

    df["source"] = "scraped"
    return df.reset_index(drop=True)


# ===========================================================================
# Enrichment
# ===========================================================================

def _add_keys(df):
    """Adds _fight_key and _name_key columns to the DataFrame."""
    df = df.copy()
    df["_fight_key"] = df.apply(
        lambda row: parse_fight_key(
            row.get("R_Fighter", ""),
            row.get("B_Fighter", ""),
            row.get("date", None),
        ),
        axis=1,
    )
    df["_name_key"] = df.apply(
        lambda row: parse_name_key(
            row.get("R_Fighter", ""),
            row.get("B_Fighter", ""),
        ),
        axis=1,
    )
    return df


def deduplicate(df):
    """
    Deduplicates a fight DataFrame on fight_key.
    The first occurrence (most reliable source first) is kept.
    """
    if df.empty:
        return df
    df = _add_keys(df)
    df = df.drop_duplicates(subset=["_fight_key"], keep="first")
    df = df.drop(columns=["_fight_key", "_name_key"])
    return df.reset_index(drop=True)


def enrich_with_kaggle2(df_fights, df_k2):
    """
    Joins the main DataFrame with Kaggle 2 on the fight key.
    Attempt 1: fight_key (names + date).
    Attempt 2 (fallback): name_key (names only, without date).
    Kaggle 2 columns do not replace already-present columns.
    """
    if df_k2.empty or df_fights.empty:
        return df_fights

    df_k2 = df_k2.copy()
    df_k2 = _add_keys(df_k2)

    # Remove duplicate keys in k2
    df_k2_fk = df_k2.drop_duplicates(subset=["_fight_key"], keep="first")
    df_k2_nk = df_k2.drop_duplicates(subset=["_name_key"],  keep="first")

    # Columns to bring (avoid duplicates with df_fights)
    shared_id_cols = {"R_Fighter", "B_Fighter", "date", "_fight_key", "_name_key"}
    new_cols = [
        c for c in df_k2.columns
        if c not in shared_id_cols and c not in df_fights.columns
    ]
    if not new_cols:
        return df_fights

    df_fights = _add_keys(df_fights)

    # -- Main join by fight_key --------------------------------------
    merged = df_fights.merge(
        df_k2_fk[["_fight_key"] + new_cols],
        on="_fight_key",
        how="left",
    )

    # -- Fallback by name_key for unmatched rows -----------------
    sentinel_col = new_cols[0]
    missing_idx = merged.index[merged[sentinel_col].isna()]

    if len(missing_idx) > 0:
        fallback_left = merged.loc[missing_idx, ["_name_key"]].copy()
        fallback = fallback_left.merge(
            df_k2_nk[["_name_key"] + new_cols],
            on="_name_key",
            how="left",
        )
        for col in new_cols:
            merged.loc[missing_idx, col] = fallback[col].values

    merged.drop(columns=["_fight_key", "_name_key"], inplace=True)
    return merged.reset_index(drop=True)


def enrich_with_fighters(df_fights, df_fighters):
    """
    Joins physical attributes on R_Fighter and B_Fighter.
    Physical columns are prefixed with R_ or B_.
    """
    if df_fighters.empty or df_fights.empty:
        return df_fights

    phys_cols = [c for c in df_fighters.columns if c != "fighter_name"]
    df_fights = df_fights.copy()

    for side, col_name in [("R", "R_Fighter"), ("B", "B_Fighter")]:
        if col_name not in df_fights.columns:
            continue

        renamed = {c: f"{side}_{c}" for c in phys_cols}
        df_side = df_fighters.rename(columns={"fighter_name": col_name, **renamed})

        # If physical columns already exist, drop them to avoid _x/_y suffixes
        existing = [c for c in renamed.values() if c in df_fights.columns]
        if existing:
            df_fights.drop(columns=existing, inplace=True)

        df_fights = df_fights.merge(df_side, on=col_name, how="left")

    return df_fights.reset_index(drop=True)


# ===========================================================================
# Kaggle download
# ===========================================================================

def download_kaggle_datasets():
    """
    Downloads both datasets via the Kaggle API.
    Does not raise exceptions if Kaggle is inaccessible or not configured.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()

        os.makedirs(KAGGLE1_DIR, exist_ok=True)
        os.makedirs(KAGGLE2_DIR, exist_ok=True)

        print("[update_master] Downloading Kaggle 1 (rajeevw/ufcdata)...")
        api.dataset_download_files(
            "rajeevw/ufcdata", path=KAGGLE1_DIR, unzip=True, quiet=False
        )

        print("[update_master] Downloading Kaggle 2 (mdabbert/ultimate-ufc-dataset)...")
        api.dataset_download_files(
            "mdabbert/ultimate-ufc-dataset", path=KAGGLE2_DIR, unzip=True, quiet=False
        )

        print("[update_master] Downloads complete.")

    except ImportError:
        print("[update_master] Kaggle module not installed -- download skipped.")
    except Exception as exc:
        print(f"[update_master] Kaggle error (offline mode or invalid key): {exc}")


# ===========================================================================
# Main function
# ===========================================================================

def update():
    """
    Rebuilds the UFC master dataset from all sources.

    Sources:
      - Kaggle 1 (fight stats) + scraped data (ufc_scraped_data.csv) -> base fights
      - Kaggle 2 (betting odds, metadata) -> enrichment
      - Fighter details (physical attributes) -> enrichment

    The master is always rebuilt from scratch to guarantee completeness.
    Only the scraping step (scrape_since) is incremental.

    Returns
    -------
    tuple (df_master: pd.DataFrame, n_total_fights: int)
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    # 1. Re-download Kaggle datasets --------------------------------------
    download_kaggle_datasets()

    # 2. Load all sources --------------------------------------------------
    print("[update_master] Loading Kaggle 1...")
    df_k1 = load_kaggle1()
    print(f"  {len(df_k1)} fight(s) from Kaggle 1")

    print("[update_master] Loading Kaggle 2...")
    df_k2 = load_kaggle2()
    print(f"  {len(df_k2)} fight(s) from Kaggle 2")

    print("[update_master] Loading physical attributes...")
    df_fighters = load_fighter_details()
    print(f"  {len(df_fighters)} fighter profile(s)")

    print("[update_master] Loading scraped data...")
    df_scraped = load_scraped()
    print(f"  {len(df_scraped)} scraped fight(s)")

    # 3. Assemble and deduplicate ------------------------------------------
    print("[update_master] Assembling and enriching...")

    frames = [f for f in [df_k1, df_scraped] if not f.empty]
    if not frames:
        print("[update_master] No fight data found.")
        return pd.DataFrame(), 0

    df_master = pd.concat(frames, ignore_index=True, sort=False)
    df_master = deduplicate(df_master)
    print(f"  {len(df_master)} unique fight(s) (Kaggle 1 + scraped)")

    # 4. Enrich ------------------------------------------------------------
    df_master = enrich_with_kaggle2(df_master, df_k2)
    df_master = enrich_with_fighters(df_master, df_fighters)

    # 5. Save --------------------------------------------------------------
    df_master.to_csv(MASTER_PATH, index=False)
    print(
        f"[update_master] Master saved: {MASTER_PATH} "
        f"({len(df_master)} fights)"
    )

    return df_master, len(df_master)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, n = update()
    print(f"Master shape: {df.shape}  |  Total: {n}")
