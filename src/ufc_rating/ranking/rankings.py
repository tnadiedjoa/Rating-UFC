"""
Division rankings from the three methods, and the official UFC ranks used
to check them.

Only active fighters are ranked: at least ``min_fights`` UFC fights and a
fight in the last ``active_days`` before the reference date. A fighter's
division is the one of their most recent fight with a known division.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ufc_rating.config import DIVISIONS
from ufc_rating.ranking.round_robin import round_robin_scores
from ufc_rating.ranking.weighted import weighted_scores

ACTIVE_DAYS = 730
MIN_FIGHTS = 5


def eligible(profiles: pd.DataFrame, division: str, as_of: pd.Timestamp,
             active_days: int = ACTIVE_DAYS, min_fights: int = MIN_FIGHTS) -> pd.DataFrame:
    """Active fighters of ``division`` (exact match: 'Heavyweight' excludes 'Light Heavyweight')."""
    recent = profiles["last_fight"] >= pd.Timestamp(as_of) - pd.Timedelta(days=active_days)
    mask = (profiles["division"] == division) & recent & (profiles["n_fights"] >= min_fights)
    return profiles[mask].copy()


def division_ranking(
    profiles: pd.DataFrame,
    division: str,
    model,
    features: List[str],
    as_of: pd.Timestamp,
    weights: Optional[Dict[str, float]] = None,
    active_days: int = ACTIVE_DAYS,
    min_fights: int = MIN_FIGHTS,
) -> pd.DataFrame:
    """
    Rank the active fighters of one division by the three methods.
    Rows are sorted by the consensus (mean of the three ranks).
    """
    pool = eligible(profiles, division, as_of, active_days, min_fights)
    if len(pool) < 2:   # a ranking needs at least two fighters
        return pd.DataFrame()

    pool["Elo"] = pool["elo"]
    pool["Weighted"] = weighted_scores(pool, weights)
    pool["Model"] = round_robin_scores(pool, model, features)
    for method in ("Elo", "Weighted", "Model"):
        pool[f"{method} rank"] = pool[method].rank(ascending=False, method="min").astype(int)
    pool["Consensus"] = pool[["Elo rank", "Weighted rank", "Model rank"]].mean(axis=1)
    pool = pool.sort_values(["Consensus", "Model rank"]).reset_index(drop=True)

    record = (pool["record_wins"].astype(int).astype(str) + "-"
              + pool["record_losses"].astype(int).astype(str)
              + np.where(pool["record_draws"] > 0, "-" + pool["record_draws"].astype(int).astype(str), ""))
    table = pd.DataFrame({
        "division": division,
        "fighter_id": pool["fighter_id"],
        "Fighter": pool["fighter_name"],
        "UFC record": record,
        "Last fight": pool["last_fight"].dt.date,
        "Elo": pool["Elo"].round(0).astype(int),
        "Weighted": pool["Weighted"].round(3),
        "Model": pool["Model"].round(3),
        "Elo rank": pool["Elo rank"],
        "Weighted rank": pool["Weighted rank"],
        "Model rank": pool["Model rank"],
        "Consensus": pool["Consensus"].round(1),
    })
    table.index = table.index + 1
    return table


def all_rankings(profiles: pd.DataFrame, model, features: List[str], as_of: pd.Timestamp,
                 weights: Optional[Dict[str, float]] = None, **kwargs) -> pd.DataFrame:
    """division_ranking() for the twelve divisions, stacked, with a 'rank' column."""
    tables = []
    for division in DIVISIONS:
        table = division_ranking(profiles, division, model, features, as_of, weights, **kwargs)
        if not table.empty:
            tables.append(table.rename_axis("rank").reset_index())
    if not tables:
        raise ValueError("No division has two eligible fighters: loosen active_days or min_fights.")
    return pd.concat(tables, ignore_index=True)


def method_agreement(rankings: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the three methods, per division."""
    rows = []
    for division, table in rankings.groupby("division", sort=False):
        row = {"division": division, "fighters": len(table)}
        for a, b in (("Elo", "Weighted"), ("Elo", "Model"), ("Weighted", "Model")):
            row[f"{a} vs {b}"] = spearmanr(table[f"{a} rank"], table[f"{b} rank"])[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("division").round(2)


# ---------------------------------------------------------------------------
# Official UFC rankings (from the odds dataset: rank of each fighter at fight time)
# ---------------------------------------------------------------------------

def latest_official_ranks(master: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """
    Most recent official rank of each fighter (0 = champion), taken from
    their fights in the last ``window_days`` before the end of the odds data.

    Ranks are observed at different dates within the window, so two fighters
    can share a rank: the comparison is indicative, not a single snapshot.
    """
    parts = []
    for side in ("r", "b"):
        part = master[["date", "division", f"{side}_id", f"{side}_name", f"{side}_rank"]].copy()
        part.columns = ["date", "division", "fighter_id", "fighter_name", "official_rank"]
        parts.append(part)
    ranks = pd.concat(parts).dropna(subset=["official_rank", "division"])
    ranks = ranks[ranks["date"] >= ranks["date"].max() - pd.Timedelta(days=window_days)]
    ranks = ranks.sort_values("date").groupby("fighter_id").tail(1)
    return ranks.rename(columns={"date": "rank_date"}).reset_index(drop=True)


def compare_with_official(rankings: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """
    Per division, Spearman correlation between each method's rank and the
    official rank, over the fighters present in both.
    """
    merged = rankings.merge(official[["fighter_id", "official_rank", "division"]],
                            on=["fighter_id", "division"], how="inner")
    rows = []
    for division, table in merged.groupby("division", sort=False):
        if len(table) < 5:
            continue
        row = {"division": division, "ranked fighters compared": len(table)}
        for method in ("Elo", "Weighted", "Model"):
            row[method] = spearmanr(table[f"{method} rank"], table["official_rank"])[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("division").round(2)
