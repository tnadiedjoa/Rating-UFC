"""
Build the master fight table (one row per UFC fight, clean types, odds
attached) and the round table (one row per round of those fights).

Sources
  - data/raw/ufcstats/master.csv : Kaggle mirror of ufcstats.com (fights, totals,
    results, judges' scores, bonuses, fighter profiles). Also includes other
    promotions; only UFC events are kept.
  - data/raw/ufcstats/round.csv  : the same mirror, round by round.
  - data/raw/scraped/            : our own scraper (events newer than the mirror),
    same layouts.
  - data/raw/odds/ufc-master.csv : betting odds and official ranks at fight time
    (2010 to March 2026), joined on fighter names and date.
  - data/raw/bestfightodds/odds.csv : closing odds of the later events.
  - data/raw/wikipedia/rankings.csv : weekly official rankings (2018 onwards);
    they replace the ranks above where available (see attach_official_ranks).

Corner convention
  ``r_`` is the fighter listed first on ufcstats.com. Since about 2010 this is
  the red corner. Before that, ufcstats lists the winner first in every
  fight, so the ``r_`` side is NOT a real corner. Anything that learns from the r/b orientation must randomise it
  (see features.build_matchups).
"""

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ufc_rating.config import (
    DIVISIONS, MASTER_CSV, ODDS_CSV, ROUNDS_CSV, SCRAPED_CSV, SCRAPED_ROUNDS_CSV,
    UFCSTATS_CSV, UFCSTATS_ROUNDS_CSV, WIKI_RANKINGS_CSV, WIKI_RECORDS_CSV, BFO_ODDS_CSV,
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

# canonical stat name -> column suffix in the mirror's round.csv
ROUND_STAT_COLUMNS = {
    "kd": "kd",
    "sig_landed": "sig_landed",
    "sig_att": "sig_atmp",
    "str_landed": "total_str_landed",
    "str_att": "total_str_atmp",
    "td_landed": "td_success",
    "td_att": "td_atmp",
    "sub_att": "sub_att",
    "rev": "rev",
    **{f"{zone}_{kind}": f"sig_str_{'landed' if kind == 'landed' else 'atmp'}_{zone}"
       for zone in ("head", "body", "leg", "distance", "clinch", "ground")
       for kind in ("landed", "att")},
}

N_JUDGES = 3

# Placeholders that ranking tables sometimes show instead of a fighter
NOT_A_NAME = {"vacant", "tba", "tbd", "to be announced", "to be determined"}

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


def parse_scorecards(details, outcome) -> list:
    """
    Judges' scores of a decision, as [(judge, r_points, b_points), ...].

    ufcstats writes each score as "loser - winner", whichever side won (on
    unanimous decisions the second number is the larger one in all but a
    handful of cards), so the winner's side tells which number belongs to
    which fighter. Draws cannot be oriented and give []. Some early cards
    have no judge name: the name is then None.
    """
    if not isinstance(details, str) or outcome not in ("r", "b"):
        return []
    cards = []
    for judge, loser, winner in re.findall(r"(?:([^\d.]+?)\s+)?(\d+)\s*-\s*(\d+)", details):
        loser, winner = int(loser), int(winner)
        r_points, b_points = (winner, loser) if outcome == "r" else (loser, winner)
        cards.append((judge.strip() or None, r_points, b_points))
    return cards


def _clean_judge_names(cards: list) -> list:
    """
    Some cards start with a note ('Point Deducted: Illegal Knee by Menne Tony
    Weeks 28 - 29'): the note is removed by keeping the known judge name the
    text ends with (known = seen without a note elsewhere), else None.
    """
    known = {name for card in cards for name, _, _ in card if name and ":" not in name}
    cleaned = []
    for card in cards:
        fixed = []
        for name, r_points, b_points in card:
            if name and ":" in name:
                endings = [k for k in known if name.endswith(" " + k)]
                name = max(endings, key=len) if endings else None
            fixed.append((name, r_points, b_points))
        cleaned.append(fixed)
    return cleaned


# Letters that Unicode decomposition does not reduce to a plain one ('Błachowicz')
_LETTERS = str.maketrans({"ł": "l", "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "đ": "d", "ı": "i", "ð": "d", "þ": "th"})


def normalize_name(name) -> str:
    """Lowercase, strip accents and punctuation: used to match names across sources."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name.lower()).translate(_LETTERS)
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
    def optional(column):   # descriptive columns that older scrapes may lack
        if column not in raw:
            return None
        # the mirror keeps some line breaks and runs of spaces from the web pages
        return raw[column].str.replace(r"\s+", " ", regex=True).str.strip().values

    df = pd.DataFrame({
        "fight_id": raw["fight_id"].values,
        "event_id": raw["event_id"].values,
        "event_name": raw["event_name"].values,
        "date": pd.to_datetime(raw["event_date"]).values,
        "location": optional("event_location"),
        "weight_class": raw["weight_class"].values,
        "title_fight": pd.to_numeric(raw["title_fight"], errors="coerce").fillna(0).astype(int).values,
        "time_format": raw["time_format"].values,
        "method": raw["method"].values,
        "finish_round": pd.to_numeric(raw["finish_round"], errors="coerce").values,
        "finish_time": raw["finish_time"].values,
        "referee": optional("referee"),
        "details": optional("details"),
        "bonuses": optional("bonuses"),
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

    # Judges' scores, oriented to the r/b sides (decisions only)
    cards = [parse_scorecards(d, o) if m == "Decision" else []
             for d, o, m in zip(df["details"], df["outcome"], df["method_group"])]
    cards = _clean_judge_names(cards)
    for j in range(N_JUDGES):
        df[f"judge{j + 1}_name"] = [c[j][0] if len(c) > j else None for c in cards]
        df[f"judge{j + 1}_r_score"] = [c[j][1] if len(c) > j else np.nan for c in cards]
        df[f"judge{j + 1}_b_score"] = [c[j][2] if len(c) > j else np.nan for c in cards]

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


def load_rounds(path: Path) -> pd.DataFrame:
    """Read a file in the layout of the mirror's round.csv into the canonical schema."""
    raw = pd.read_csv(path, low_memory=False)
    df = pd.DataFrame({"fight_id": raw["fight_id"].values,
                       "round": pd.to_numeric(raw["round_no"], errors="coerce").values})
    for side in ("r", "b"):
        df[f"{side}_id"] = raw[f"{side}_id"].values
        for stat, suffix in ROUND_STAT_COLUMNS.items():
            df[f"{side}_{stat}"] = pd.to_numeric(raw[f"{side}_{suffix}"], errors="coerce").values
        df[f"{side}_ctrl_sec"] = raw[f"{side}_ctrl"].map(mmss_to_seconds).values
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
        "odds_source": "ultimate-ufc-dataset",
    })


def load_bfo_odds(path: Path) -> pd.DataFrame:
    """bestfightodds.com consensus closing odds (ingest.bestfightodds), same keys as load_odds."""
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "odds_date": pd.to_datetime(raw["date"]),
        "odds_r": raw["fighter_1"].map(normalize_name),
        "odds_b": raw["fighter_2"].map(normalize_name),
        "R_odds": pd.to_numeric(raw["odds_1"], errors="coerce"),
        "B_odds": pd.to_numeric(raw["odds_2"], errors="coerce"),
        "R_rank": np.nan,
        "B_rank": np.nan,
        "odds_source": "bestfightodds",
    })


# ===========================================================================
# Odds join
# ===========================================================================

def name_similarity(a: str, b: str) -> float:
    """
    Similarity (0 to 1) of two normalised names, tolerant to spacing
    ('dooho choi' / 'doo ho choi') and word order ('liu ce' / 'ce liu').
    """
    if not a or not b:
        return 0.0
    squashed = SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()
    reordered = SequenceMatcher(None, "".join(sorted(a.split())), "".join(sorted(b.split()))).ratio()
    return max(squashed, reordered)


def _loose_matches(left: pd.DataFrame, odds: pd.DataFrame, max_days: int,
                   min_similarity: float = 0.8, min_each: float = 0.5) -> pd.DataFrame:
    """Second pass of attach_odds: best-scoring pairs on the same dates, each odds row used once."""
    candidates = []
    for fight in left.itertuples(index=False):
        near = odds[(odds["odds_date"] - fight.date).abs().dt.days <= max_days]
        for row in near.itertuples(index=False):
            straight = (name_similarity(fight.r_norm, row.odds_r), name_similarity(fight.b_norm, row.odds_b))
            crossed = (name_similarity(fight.r_norm, row.odds_b), name_similarity(fight.b_norm, row.odds_r))
            best = max(straight, crossed, key=sum)
            if sum(best) / 2 >= min_similarity and min(best) >= min_each:
                candidates.append((sum(best) / 2, row.source_order, fight.fight_id, row.odds_row))
    used_fights, used_rows, keep = set(), set(), []
    for score, _, fight_id, odds_row in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if fight_id not in used_fights and odds_row not in used_rows:
            used_fights.add(fight_id)
            used_rows.add(odds_row)
            keep.append((fight_id, odds_row))
    if not keep:
        return pd.DataFrame()
    pairs = pd.DataFrame(keep, columns=["fight_id", "odds_row"])
    return pairs.merge(left, on="fight_id").merge(odds, on="odds_row")


def attach_odds(fights: pd.DataFrame, odds: pd.DataFrame, max_days: int = 1) -> pd.DataFrame:
    """
    Add r_odds, b_odds (American) and r_rank, b_rank (official rank at fight
    time, 0 = champion, NaN = unranked) to each fight.

    Fights are matched on the unordered pair of normalised names and a date
    within ``max_days`` (the sources disagree by a day on some overseas
    cards). Fights left unmatched get a second chance against the odds rows
    of the same dates, with names compared loosely (spelling, spacing, word
    order): a pair is accepted when both names are close on average
    (``min_similarity``) and neither is far off. When the odds source lists
    the fighters the other way round, its columns are swapped so that r_odds
    always belongs to r_name. When ``odds`` stacks several sources, the first
    listed wins.

    Odds and ranks are matched separately, each on the rows that have them:
    a source can list a fight without its odds (or ranks) while another has
    them.
    """
    fights = fights.copy()
    left = pd.DataFrame({
        "fight_id": fights["fight_id"],
        "date": fights["date"],
        "r_norm": fights["r_name"].map(normalize_name),
        "b_norm": fights["b_name"].map(normalize_name),
    })
    left["pair"] = [" | ".join(sorted(p)) for p in zip(left["r_norm"], left["b_norm"])]
    odds = odds.copy().reset_index(drop=True)
    odds["pair"] = [" | ".join(sorted(p)) for p in zip(odds["odds_r"], odds["odds_b"])]
    odds["odds_row"] = odds.index
    odds["source_order"] = pd.Categorical(odds["odds_source"], pd.unique(odds["odds_source"])).codes

    priced = odds[odds["R_odds"].notna() & odds["B_odds"].notna()]
    ranked = odds[odds["R_rank"].notna() | odds["B_rank"].notna()]
    for rows, pairs in ((priced, [("r_odds", "R_odds", "B_odds"), ("b_odds", "B_odds", "R_odds")]),
                        (ranked, [("r_rank", "R_rank", "B_rank"), ("b_rank", "B_rank", "R_rank")])):
        merged = _match_pairs(left, rows, max_days)
        # orientation from both names, as in the loose pass
        same = [name_similarity(r, o) + name_similarity(b, ob) >= name_similarity(r, ob) + name_similarity(b, o)
                for r, b, o, ob in zip(merged["r_norm"], merged["b_norm"], merged["odds_r"], merged["odds_b"])]
        merged = merged.set_index("fight_id")
        for column, own, other in pairs:
            values = pd.Series(np.where(same, merged[own], merged[other]), index=merged.index)
            fights[column] = fights["fight_id"].map(values)
        if rows is priced:
            fights["odds_source"] = fights["fight_id"].map(merged["odds_source"])
    return fights


def _match_pairs(left: pd.DataFrame, odds: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """attach_odds matching: exact name pairs first, then the loose second pass."""
    merged = left.merge(odds, on="pair", how="inner")
    merged["gap"] = (merged["odds_date"] - merged["date"]).dt.days.abs()
    merged = (merged[merged["gap"] <= max_days]
              .sort_values(["source_order", "gap"])
              .drop_duplicates("fight_id"))
    loose = _loose_matches(left[~left["fight_id"].isin(merged["fight_id"])],
                           odds[~odds["odds_row"].isin(merged["odds_row"])], max_days)
    columns = ["fight_id", "r_norm", "b_norm", "odds_r", "odds_b", "R_odds", "B_odds", "R_rank", "B_rank",
               "odds_source"]
    return pd.concat([merged[columns]] + ([loose[columns]] if len(loose) else []), ignore_index=True)


# ===========================================================================
# Official rankings join
# ===========================================================================

def match_ranked_fighters(rankings: pd.DataFrame, fights: pd.DataFrame, pages: Optional[dict] = None,
                          window_days: int = 730, min_similarity: float = 0.85) -> pd.Series:
    """
    ufcstats fighter_id of each ranked name, found in this order:

    1. the Wikipedia page the name links to, when ``pages`` ({page title:
       fighter_id}, from the professional records) knows it; this follows
       name changes ('Katlyn Chookagian' -> Katlyn Cerminara). A link that
       contradicts the displayed name (another fighter of the division has
       exactly that name) is a markup error and is ignored;
    2. a fighter with the same normalised name who fought in the division
       within ``window_days`` of the snapshot (else at any time in the
       division, else in any division);
    3. the most similar name (``name_similarity`` >= ``min_similarity``)
       among the fighters of the division within ``window_days``;
    4. a fighter of the division whose name is a shorter form of the ranked
       name, when only one fits;
    5. a fighter of the division with the same last name and first initial
       ('Steve' for 'Stephen'), when only one fits.

    Unmatched names get None.
    """
    parts = [pd.DataFrame({"fighter_id": fights[f"{side}_id"], "name": fights[f"{side}_name"].map(normalize_name),
                           "division": fights["division"], "date": fights["date"]})
             for side in ("r", "b")]
    appearances = pd.concat(parts, ignore_index=True)
    by_name = dict(tuple(appearances.groupby("name")))
    by_division = dict(tuple(appearances.groupby("division")))
    pages = pages or {}
    has_page = "page" in rankings

    cache, ids = {}, []
    for row in rankings.itertuples():
        page = row.page if has_page and isinstance(row.page, str) else None
        key = (page, normalize_name(row.fighter), row.division, row.date.year)
        if key not in cache:
            by_link, exact_near, by_exact_name = pages.get(page), None, None
            group = by_name.get(key[1])
            if group is not None:
                same_division = group[group["division"] == row.division]
                near = same_division[(same_division["date"] - row.date).abs().dt.days <= window_days]
                if len(near):
                    exact_near = near["fighter_id"].mode().iloc[0]
                for pool in (near, same_division, group):
                    if len(pool):
                        by_exact_name = pool["fighter_id"].mode().iloc[0]
                        break
            if by_link is not None and exact_near is not None and exact_near != by_link:
                cache[key] = exact_near
            else:
                cache[key] = by_link if by_link is not None else by_exact_name
            if cache[key] is None and row.division in by_division:
                division = by_division[row.division]
                near = division[(division["date"] - row.date).abs().dt.days <= window_days]
                names = near.drop_duplicates("name")
                if len(names):
                    scores = names["name"].map(lambda n: name_similarity(key[1], n))
                    if scores.max() >= min_similarity:
                        cache[key] = names.loc[scores.idxmax(), "fighter_id"]
                    else:
                        # 4. a shorter form of the name ('Diego Ferreira' for 'Carlos Diego Ferreira'),
                        #    only when a single fighter of the division fits
                        words = set(key[1].split())
                        subset = names[names["name"].map(lambda n: 0 < len(set(n.split())) and set(n.split()) <= words)]
                        if subset["fighter_id"].nunique() == 1:
                            cache[key] = subset["fighter_id"].iloc[0]
                        elif key[1]:
                            # 5. same last name and first initial ('Steve Erceg' for 'Stephen Erceg')
                            first, last = key[1][0], key[1].split()[-1]
                            same_last = names[names["name"].map(
                                lambda n: bool(n) and n[0] == first and n.split()[-1] == last)]
                            if same_last["fighter_id"].nunique() == 1:
                                cache[key] = same_last["fighter_id"].iloc[0]
        ids.append(cache[key])
    return pd.Series(ids, index=rankings.index, dtype=object)


def read_official_rankings(path: Path, fights: pd.DataFrame,
                           records_csv: Optional[Path] = WIKI_RECORDS_CSV) -> pd.DataFrame:
    """The weekly Wikipedia snapshots with a ``fighter_id`` column (see match_ranked_fighters)."""
    rankings = pd.read_csv(path, parse_dates=["date"])
    rankings = rankings[~rankings["fighter"].str.lower().isin(NOT_A_NAME)].reset_index(drop=True)
    pages = {}
    if records_csv is not None and Path(records_csv).exists():
        records = pd.read_csv(records_csv, usecols=["fighter_id", "wiki_title"]).drop_duplicates()
        pages = dict(zip(records["wiki_title"], records["fighter_id"]))
    rankings["fighter_id"] = match_ranked_fighters(rankings, fights, pages)
    return rankings


def attach_official_ranks(fights: pd.DataFrame, rankings: pd.DataFrame, max_age_days: int = 14) -> pd.DataFrame:
    """
    Official rank of each fighter just before the fight (0 = champion or
    interim champion, NaN = unranked), from the last weekly snapshot dated
    before the fight day, if it is at most ``max_age_days`` old.

    r_rank / b_rank: the media-panel rankings (from the snapshots where they
    exist, else the odds dataset's ranks). r_meta_rank / b_meta_rank: the
    Meta UFC Rankings, published from June 2026.
    """
    fights = fights.copy()
    for system, prefix in (("media", ""), ("meta", "meta_")):
        snaps = rankings[(rankings["system"] == system) & rankings["fighter_id"].notna()]
        if snaps.empty:
            if system == "meta":
                fights["r_meta_rank"] = fights["b_meta_rank"] = np.nan
            continue
        dates = np.sort(snaps["date"].unique())
        position = np.searchsorted(dates, fights["date"].values, side="left") - 1
        snap_date = pd.Series(pd.NaT, index=fights.index, dtype="datetime64[ns]")
        valid = position >= 0
        snap_date[valid] = dates[position[valid]]
        fresh = (fights["date"] - snap_date).dt.days <= max_age_days
        lookup = snaps.drop_duplicates(["date", "division", "fighter_id"]).set_index(
            ["date", "division", "fighter_id"])["rank"]
        # snapshot divisions with a name linked to no fighter: absence there does not prove "unranked"
        unmatched = rankings[(rankings["system"] == system) & rankings["fighter_id"].isna()]
        incomplete = set(zip(unmatched["date"], unmatched["division"]))
        in_incomplete = pd.Series([(d, v) in incomplete for d, v in zip(snap_date, fights["division"])],
                                  index=fights.index)
        for side in ("r", "b"):
            key = pd.MultiIndex.from_arrays([snap_date, fights["division"], fights[f"{side}_id"]])
            rank = pd.Series(lookup.reindex(key).values, index=fights.index).where(fresh)
            column = f"{side}_{prefix}rank"
            if system == "media" and column in fights:
                # a fresh snapshot decides (NaN = unranked), unless the fighter is absent from a
                # snapshot that has unlinked names; otherwise keep the odds dataset's rank
                decides = fresh & (rank.notna() | ~in_incomplete)
                fights[column] = np.where(decides, rank, fights[column])
            else:
                fights[column] = rank
    return fights


# ===========================================================================
# Main entry point
# ===========================================================================

def build_master(
    ufcstats_csv: Path = UFCSTATS_CSV,
    scraped_csv: Path = SCRAPED_CSV,
    odds_csv: Path = ODDS_CSV,
    out_path: Optional[Path] = MASTER_CSV,
    rankings_csv: Optional[Path] = WIKI_RANKINGS_CSV,
    bfo_csv: Optional[Path] = BFO_ODDS_CSV,
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

    odds = [load(path) for load, path in ((load_odds, odds_csv), (load_bfo_odds, bfo_csv))
            if path is not None and Path(path).exists()]
    if odds:
        fights = attach_odds(fights, pd.concat(odds, ignore_index=True))
    if rankings_csv is not None and Path(rankings_csv).exists():
        fights = attach_official_ranks(fights, read_official_rankings(rankings_csv, fights))

    fights = fights.sort_values(["date", "fight_id"]).reset_index(drop=True)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fights.to_csv(out_path, index=False)
    return fights


def read_master(path: Path = MASTER_CSV) -> pd.DataFrame:
    """Load a saved master table with its date columns parsed."""
    return pd.read_csv(path, low_memory=False, parse_dates=["date", "r_dob", "b_dob"])


def build_rounds(
    master: pd.DataFrame,
    ufcstats_rounds_csv: Path = UFCSTATS_ROUNDS_CSV,
    scraped_rounds_csv: Path = SCRAPED_ROUNDS_CSV,
    out_path: Optional[Path] = ROUNDS_CSV,
) -> pd.DataFrame:
    """
    Round-by-round stats of the fights in ``master``, with the date and the
    round's duration in seconds (full length, or the finish time for the
    last round). The r/b sides are those of the master table.
    """
    frames = [load_rounds(p) for p in (ufcstats_rounds_csv, scraped_rounds_csv)
              if p is not None and Path(p).exists()]
    if not frames:
        return pd.DataFrame(columns=["fight_id", "round"])
    rounds = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(["fight_id", "round"], keep="first"))

    fights = master.set_index("fight_id")
    rounds = rounds[rounds["fight_id"].isin(fights.index)].copy()
    same_sides = rounds["r_id"].values == fights.loc[rounds["fight_id"], "r_id"].values
    if not same_sides.all():
        raise ValueError(f"{(~same_sides).sum()} rounds list the fighters in another order than the fights")

    info = fights.loc[rounds["fight_id"], ["date", "time_format", "finish_round", "finish_time"]]
    rounds.insert(1, "date", info["date"].values)
    lengths = [round_lengths(f) for f in info["time_format"]]
    last = info["finish_round"].to_numpy()
    finish = info["finish_time"].map(mmss_to_seconds).to_numpy()
    rounds["seconds"] = [
        fin if rnd == last_rnd else (60.0 * lens[int(rnd) - 1] if int(rnd) <= len(lens) else 300.0)
        for rnd, last_rnd, fin, lens in zip(rounds["round"], last, finish, lengths)
    ]
    rounds = rounds.sort_values(["date", "fight_id", "round"]).reset_index(drop=True)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        rounds.to_csv(out_path, index=False)
    return rounds


def read_rounds(path: Path = ROUNDS_CSV) -> pd.DataFrame:
    """Load a saved round table with its date column parsed."""
    return pd.read_csv(path, low_memory=False, parse_dates=["date"])
