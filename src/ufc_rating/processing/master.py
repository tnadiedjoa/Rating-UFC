"""
Build the master fight table: one row per UFC fight, clean types, odds attached.

Sources
  - data/raw/ufcstats/master.csv : Kaggle mirror of ufcstats.com (fights, totals,
    results, fighter profiles). Also includes other promotions; only UFC events
    are kept.
  - data/raw/scraped/fights.csv  : optional output of our own scraper, same layout.
  - data/raw/odds/ufc-master.csv : betting odds and official ranks at fight time
    (2010 onwards), joined on fighter names and date.

Corner convention
  ``r_`` is the fighter listed first on ufcstats.com. Since about 2010 this is
  the red corner. Before that, ufcstats lists the winner first in every
  fight, so the ``r_`` side is NOT a real corner. Anything that learns from the r/b orientation must randomise it
  (see features.build_matchups).
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ufc_rating.config import (
    DIVISIONS, MASTER_CSV, ODDS_CSV, SCRAPED_CSV, UFCSTATS_CSV,
)

# canonical stat name -> column suffix in the ufcstats mirror
STAT_COLUMNS = {
    "kd": "total_kd",
    "sig_landed": "total_sig_landed",
    "sig_att": "total_sig_atmp",
    "str_landed": "total_total_str_landed",
    "str_att": "total_total_str_atmp",
    "td_landed": "total_td_success",
    "td_att": "total_td_atmp",
    "sub_att": "total_sub_att",
    "rev": "total_rev",
    "ctrl_sec": "total_ctrl_seconds",
    "head_landed": "total_sig_str_landed_head",
    "head_att": "total_sig_str_atmp_head",
    "body_landed": "total_sig_str_landed_body",
    "body_att": "total_sig_str_atmp_body",
    "leg_landed": "total_sig_str_landed_leg",
    "leg_att": "total_sig_str_atmp_leg",
    "distance_landed": "total_sig_str_landed_distance",
    "distance_att": "total_sig_str_atmp_distance",
    "clinch_landed": "total_sig_str_landed_clinch",
    "clinch_att": "total_sig_str_atmp_clinch",
    "ground_landed": "total_sig_str_landed_ground",
    "ground_att": "total_sig_str_atmp_ground",
}
STATS = list(STAT_COLUMNS)

# ufcstats did not record control time before UFC 21 (July 1999): it shows 0:00
# for both fighters. Double zeros before 2000 are treated as missing.
CTRL_TRACKED_FROM = pd.Timestamp("2000-01-01")


# ===========================================================================
# Parsing helpers
# ===========================================================================

def is_ufc_event(name) -> bool:
    """UFC numbered events, Fight Nights, TUF finales, Noche UFC... but not Road to UFC."""
    if not isinstance(name, str):
        return False
    if re.match(r"(?i)road to ufc", name):
        return False
    return bool(re.search(r"\bUFC\b", name)) or name.startswith(("The Ultimate Fighter", "Ortiz vs"))


_DIVISION_KEYWORDS = [
    "light heavyweight", "heavyweight", "middleweight", "welterweight",
    "lightweight", "featherweight", "bantamweight", "flyweight", "strawweight",
]


def parse_division(weight_class) -> Optional[str]:
    """
    Map a raw weight-class label to one of the twelve UFC divisions.
    Catch weight, open weight and old tournament labels give None.
    """
    if not isinstance(weight_class, str):
        return None
    label = weight_class.lower()
    if "super heavyweight" in label:
        return None
    for keyword in _DIVISION_KEYWORDS:
        if re.search(rf"\b{keyword}\b", label):
            division = keyword.title()
            if "women" in label:
                division = f"Women's {division}"
            return division if division in DIVISIONS else None
    return None


def round_lengths(time_format) -> list:
    """'3 Rnd (5-5-5)' -> [5, 5, 5] (minutes). 'No Time Limit' -> []."""
    if not isinstance(time_format, str):
        return []
    match = re.search(r"\(([\d\-]+)\)", time_format)
    return [int(x) for x in match.group(1).split("-")] if match else []


def mmss_to_seconds(value) -> float:
    if not isinstance(value, str):
        return np.nan
    match = re.match(r"^(\d+):(\d{2})$", value.strip())
    return int(match.group(1)) * 60 + int(match.group(2)) if match else np.nan


def fight_seconds(time_format, finish_round, finish_time) -> float:
    """Total fight duration: full rounds before the last one + time in the last one."""
    last = mmss_to_seconds(finish_time)
    if np.isnan(last) or pd.isna(finish_round):
        return np.nan
    lengths = round_lengths(time_format)
    previous = int(finish_round) - 1
    if previous <= 0:
        return float(last)
    if previous <= len(lengths):
        return float(sum(lengths[:previous]) * 60 + last)
    return float(previous * 300 + last)


def height_to_cm(value) -> float:
    """'5\\' 11"' -> 180.3"""
    if not isinstance(value, str):
        return np.nan
    match = re.match(r"(\d+)'\s*(\d*)", value.strip())
    if not match:
        return np.nan
    inches = int(match.group(1)) * 12 + int(match.group(2) or 0)
    return round(inches * 2.54, 1)


def method_group(method) -> str:
    """Collapse ufcstats methods into KO/TKO, Submission, Decision, Other."""
    if not isinstance(method, str):
        return "Other"
    if "KO" in method:          # 'KO/TKO', "TKO - Doctor's Stoppage"
        return "KO/TKO"
    if method.startswith("Submission"):
        return "Submission"
    if method.startswith("Decision"):
        return "Decision"
    return "Other"              # DQ, Overturned, Could Not Continue, Other


def normalize_name(name) -> str:
    """Lowercase, strip accents and punctuation: used to match names across sources."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name.lower())
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[.'’`-]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def american_to_prob(odds) -> np.ndarray:
    """American odds -> implied probability (vig included)."""
    o = np.asarray(pd.to_numeric(pd.Series(odds), errors="coerce"), dtype=float)
    prob = np.full(o.shape, np.nan)
    positive, negative = o > 0, o < 0
    prob[positive] = 100 / (o[positive] + 100)
    prob[negative] = -o[negative] / (-o[negative] + 100)
    return prob


# ===========================================================================
# Loading
# ===========================================================================

def load_fights(path: Path) -> pd.DataFrame:
    """Read a file in the ufcstats-mirror layout and keep UFC events only."""
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["event_name"].map(is_ufc_event)]
    return to_canonical(raw)


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and type the mirror columns into the master schema."""
    df = pd.DataFrame({
        "fight_id": raw["fight_id"].values,
        "event_id": raw["event_id"].values,
        "event_name": raw["event_name"].values,
        "date": pd.to_datetime(raw["event_date"]).values,
        "weight_class": raw["weight_class"].values,
        "title_fight": pd.to_numeric(raw["title_fight"], errors="coerce").fillna(0).astype(int).values,
        "time_format": raw["time_format"].values,
        "method": raw["method"].values,
        "finish_round": pd.to_numeric(raw["finish_round"], errors="coerce").values,
        "finish_time": raw["finish_time"].values,
    })
    df["division"] = df["weight_class"].map(parse_division)
    df["method_group"] = df["method"].map(method_group)
    df["scheduled_rounds"] = [len(round_lengths(f)) or 1 for f in df["time_format"]]
    df["fight_seconds"] = [
        fight_seconds(f, r, t)
        for f, r, t in zip(df["time_format"], df["finish_round"], df["finish_time"])
    ]

    status = raw["result_status"].values
    winner = raw["winner_id"].values
    r_id = raw["r_fighter_id"].values
    df["outcome"] = np.select(
        [status == "draw", status == "no_contest", winner == r_id],
        ["draw", "nc", "r"],
        default="b",
    )

    for side in ("r", "b"):
        df[f"{side}_id"] = raw[f"{side}_fighter_id"].values
        df[f"{side}_name"] = raw[f"{side}_fighter_name"].values
        df[f"{side}_height_cm"] = raw[f"{side}_height"].map(height_to_cm).values
        df[f"{side}_reach_cm"] = (pd.to_numeric(raw[f"{side}_reach_inches"], errors="coerce") * 2.54).round(1).values
        df[f"{side}_stance"] = raw[f"{side}_stance"].values
        df[f"{side}_dob"] = pd.to_datetime(raw[f"{side}_dob"], errors="coerce").values
        for stat, suffix in STAT_COLUMNS.items():
            df[f"{side}_{stat}"] = pd.to_numeric(raw[f"{side}_{suffix}"], errors="coerce").values

    untracked = (df["date"] < CTRL_TRACKED_FROM) & (df["r_ctrl_sec"] == 0) & (df["b_ctrl_sec"] == 0)
    df.loc[untracked, ["r_ctrl_sec", "b_ctrl_sec"]] = np.nan
    return df


def load_odds(path: Path) -> pd.DataFrame:
    """Odds and official ranks, keyed by normalised names and date."""
    raw = pd.read_csv(path, low_memory=False)
    return pd.DataFrame({
        "odds_date": pd.to_datetime(raw["date"]),
        "odds_r": raw["R_fighter"].map(normalize_name),
        "odds_b": raw["B_fighter"].map(normalize_name),
        "R_odds": pd.to_numeric(raw["R_odds"], errors="coerce"),
        "B_odds": pd.to_numeric(raw["B_odds"], errors="coerce"),
        "R_rank": pd.to_numeric(raw["R_match_weightclass_rank"], errors="coerce"),
        "B_rank": pd.to_numeric(raw["B_match_weightclass_rank"], errors="coerce"),
    })


# ===========================================================================
# Odds join
# ===========================================================================

def attach_odds(fights: pd.DataFrame, odds: pd.DataFrame, max_days: int = 1) -> pd.DataFrame:
    """
    Add r_odds, b_odds (American) and r_rank, b_rank (official rank at fight
    time, 0 = champion, NaN = unranked) to each fight.

    Fights are matched on the unordered pair of normalised names and a date
    within ``max_days`` (the sources disagree by a day on some overseas
    cards). When the odds source lists the fighters the other way round, its
    columns are swapped so that r_odds always belongs to r_name.
    """
    fights = fights.copy()
    left = pd.DataFrame({
        "fight_id": fights["fight_id"],
        "date": fights["date"],
        "r_norm": fights["r_name"].map(normalize_name),
        "b_norm": fights["b_name"].map(normalize_name),
    })
    left["pair"] = [" | ".join(sorted(p)) for p in zip(left["r_norm"], left["b_norm"])]
    odds = odds.copy()
    odds["pair"] = [" | ".join(sorted(p)) for p in zip(odds["odds_r"], odds["odds_b"])]

    merged = left.merge(odds, on="pair", how="inner")
    merged["gap"] = (merged["odds_date"] - merged["date"]).dt.days.abs()
    merged = (merged[merged["gap"] <= max_days]
              .sort_values("gap")
              .drop_duplicates("fight_id"))

    same = merged["odds_r"] == merged["r_norm"]
    merged["r_odds"] = np.where(same, merged["R_odds"], merged["B_odds"])
    merged["b_odds"] = np.where(same, merged["B_odds"], merged["R_odds"])
    merged["r_rank"] = np.where(same, merged["R_rank"], merged["B_rank"])
    merged["b_rank"] = np.where(same, merged["B_rank"], merged["R_rank"])

    extra = merged.set_index("fight_id")[["r_odds", "b_odds", "r_rank", "b_rank"]]
    for col in extra.columns:
        fights[col] = fights["fight_id"].map(extra[col])
    return fights


# ===========================================================================
# Main entry point
# ===========================================================================

def build_master(
    ufcstats_csv: Path = UFCSTATS_CSV,
    scraped_csv: Path = SCRAPED_CSV,
    odds_csv: Path = ODDS_CSV,
    out_path: Optional[Path] = MASTER_CSV,
) -> pd.DataFrame:
    """
    Merge all sources into the master table and save it to ``out_path``.
    Fights present in both the mirror and our scrape are kept once (mirror first).
    """
    frames = [load_fights(ufcstats_csv)]
    if scraped_csv is not None and Path(scraped_csv).exists():
        frames.append(load_fights(scraped_csv))
    fights = (pd.concat(frames, ignore_index=True)
              .drop_duplicates("fight_id", keep="first"))

    if odds_csv is not None and Path(odds_csv).exists():
        fights = attach_odds(fights, load_odds(odds_csv))

    fights = fights.sort_values(["date", "fight_id"]).reset_index(drop=True)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fights.to_csv(out_path, index=False)
    return fights


def read_master(path: Path = MASTER_CSV) -> pd.DataFrame:
    """Load a saved master table with its date columns parsed."""
    return pd.read_csv(path, low_memory=False, parse_dates=["date", "r_dob", "b_dob"])
