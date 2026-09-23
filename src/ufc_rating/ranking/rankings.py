"""
Division rankings, and the official UFC ranks used to check them.

Fighters are ranked by the model: in a virtual round-robin tournament, the
stats model predicts every pairing of the division (ranking.round_robin), and
a fighter's score is their mean win probability. Their Elo rating is shown
alongside as a measure of their record: Elo rewards who a fighter beat.
ranking.backtest checks both against the fights between ranked fighters.

Only active fighters are ranked: at least ``min_fights`` UFC fights and a
fight in the last ``active_days`` before the reference date. A fighter's
division is the one of their most recent fight with a known division.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ufc_rating.config import DIVISIONS
from ufc_rating.ranking.round_robin import round_robin_scores

ACTIVE_DAYS = 730
MIN_FIGHTS = 5
METHODS = ("Model", "Elo")


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
    active_days: int = ACTIVE_DAYS,
    min_fights: int = MIN_FIGHTS,
) -> pd.DataFrame:
    """
    Rank the active fighters of one division by the model's round-robin
    score, with their Elo rating and Elo rank alongside.
    """
    pool = eligible(profiles, division, as_of, active_days, min_fights)
    if len(pool) < 2:   # a ranking needs at least two fighters
        return pd.DataFrame()

    pool["Model"] = round_robin_scores(pool, model, features)
    pool["Elo"] = pool["elo"]
    for method in METHODS:
        pool[f"{method} rank"] = pool[method].rank(ascending=False, method="min").astype(int)
    pool = pool.sort_values(["Model rank", "Elo rank"]).reset_index(drop=True)

    record = (pool["record_wins"].astype(int).astype(str) + "-"
              + pool["record_losses"].astype(int).astype(str)
              + np.where(pool["record_draws"] > 0, "-" + pool["record_draws"].astype(int).astype(str), ""))
    table = pd.DataFrame({
        "division": division,
        "fighter_id": pool["fighter_id"],
        "Fighter": pool["fighter_name"],
        "UFC record": record,
        "Last fight": pool["last_fight"].dt.date,
        "Model": pool["Model"].round(3),
        "Elo": pool["Elo"].round(0).astype(int),
        "Model rank": pool["Model rank"],
        "Elo rank": pool["Elo rank"],
    })
    table.index = table.index + 1
    return table


def all_rankings(profiles: pd.DataFrame, model, features: List[str], as_of: pd.Timestamp,
                 **kwargs) -> pd.DataFrame:
    """division_ranking() for the twelve divisions, stacked, with a 'rank' column."""
    tables = []
    for division in DIVISIONS:
        table = division_ranking(profiles, division, model, features, as_of, **kwargs)
        if not table.empty:
            tables.append(table.rename_axis("rank").reset_index())
    if not tables:
        raise ValueError("No division has two eligible fighters: loosen active_days or min_fights.")
    return pd.concat(tables, ignore_index=True)


def method_agreement(rankings: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the model and Elo ranks, per division."""
    rows = [{"division": division, "fighters": len(table),
             "Model vs Elo": spearmanr(table["Model rank"], table["Elo rank"])[0]}
            for division, table in rankings.groupby("division", sort=False)]
    return pd.DataFrame(rows).set_index("division").round(2)


# ---------------------------------------------------------------------------
# Official UFC rankings (weekly snapshots from Wikipedia, see ingest.wikipedia)
# ---------------------------------------------------------------------------

def latest_official_ranks(official: pd.DataFrame, system: str = "media",
                          as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    The last official rankings published on or before ``as_of`` (default:
    the latest), from the weekly snapshots of
    ``processing.master.read_official_rankings``. ``system``: 'media' (the
    media-panel rankings) or 'meta' (the Meta UFC Rankings, from June 2026).
    Returns fighter_id, fighter, division, official_rank (0 = champion) and
    rank_date; names that match no ufcstats fighter are dropped.
    """
    snaps = official[(official["system"] == system) & official["fighter_id"].notna()]
    if as_of is not None:
        snaps = snaps[snaps["date"] <= pd.Timestamp(as_of)]
    if snaps.empty:
        raise ValueError(f"No '{system}' rankings snapshot available")
    latest = snaps[snaps["date"] == snaps["date"].max()]
    return (latest.rename(columns={"rank": "official_rank", "date": "rank_date"})
            [["fighter_id", "fighter", "division", "official_rank", "rank_date"]]
            .reset_index(drop=True))


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
        for method in METHODS:
            row[method] = spearmanr(table[f"{method} rank"], table["official_rank"])[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("division").round(2)
