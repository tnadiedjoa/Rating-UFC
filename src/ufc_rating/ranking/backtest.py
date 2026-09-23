"""
Do the rankings predict the fights?

A ranking claims that a fighter is better than the fighters below. The fights
between two ranked fighters of the same division test that claim: the
better-ranked fighter should win.

The test period is replayed event by event. The day before each event, the
rankings are rebuilt from the fights known at that date, with the stats model
trained on the fights before the test period. The model ranking and the Elo
ranking then designate a favourite in every fight between two fighters they
rank, and so do the official UFC rankings (the ranks published before the
fight) and the betting market (the closing odds).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ufc_rating.processing.features import appearances, fighter_states, profiles_as_of
from ufc_rating.processing.master import american_to_prob
from ufc_rating.ranking.rankings import all_rankings

# Predictor -> (r column, b column, lower value is better). The rank columns
# of our methods are filled by ranking_picks(); the others come from the master table.
PREDICTORS = {
    "Official rankings": ("r_rank", "b_rank", True),
    "Model": ("r_Model rank", "b_Model rank", True),
    "Elo": ("r_Elo rank", "b_Elo rank", True),
    "Betting favourite": ("r_market", "b_market", False),
}
OUR_METHODS = ["Model rank", "Elo rank"]


def rankings_before_events(
    master: pd.DataFrame,
    elo_history: pd.DataFrame,
    rounds: Optional[pd.DataFrame],
    model,
    features: List[str],
    start: pd.Timestamp,
    **kwargs,
) -> pd.DataFrame:
    """
    The division rankings as they stood the day before each event date from
    ``start`` on, stacked, with the event ``date``. ``model`` must be trained
    on fights before ``start``.
    """
    states = fighter_states(master, elo_history, rounds)
    apps = appearances(master)
    tables = []
    for date in np.sort(master.loc[master["date"] >= pd.Timestamp(start), "date"].unique()):
        as_of = pd.Timestamp(date) - pd.Timedelta(days=1)
        profiles = profiles_as_of(states, apps, as_of)
        tables.append(all_rankings(profiles, model, features, as_of, **kwargs)
                      .assign(date=pd.Timestamp(date)))
    return pd.concat(tables, ignore_index=True)


def _favourite_won(r_value, b_value, r_won, lower_is_better: bool) -> np.ndarray:
    """1 if the predictor's favourite won, 0 if it lost, 0.5 on a tie, NaN without a value."""
    r_value, b_value = np.asarray(r_value, dtype=float), np.asarray(b_value, dtype=float)
    r_favoured = b_value - r_value if lower_is_better else r_value - b_value
    return np.where(np.isnan(r_favoured), np.nan,
                    np.where(r_favoured == 0, 0.5, ((r_favoured > 0) == np.asarray(r_won)).astype(float)))


def ranking_picks(master: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    One row per decided fight of the replayed period between two fighters
    who both held an official rank in the fight's division, with each
    predictor's verdict (see _favourite_won). ``covered`` marks the fights
    where our rankings also rank both fighters in that division (at least
    five UFC fights and a fight in the last two years).
    """
    fights = master[(master["date"] >= history["date"].min()) & master["outcome"].isin(["r", "b"])
                    & master["r_rank"].notna() & master["b_rank"].notna()].copy()
    ours = history.drop_duplicates(["date", "division", "fighter_id"]).set_index(
        ["date", "division", "fighter_id"])[OUR_METHODS]
    for side in ("r", "b"):
        key = pd.MultiIndex.from_arrays([fights["date"], fights["division"], fights[f"{side}_id"]])
        ranks = ours.reindex(key)
        for method in OUR_METHODS:
            fights[f"{side}_{method}"] = ranks[method].to_numpy()
        fights[f"{side}_market"] = american_to_prob(fights[f"{side}_odds"])

    picks = fights[["fight_id", "date", "division", "r_name", "b_name", "outcome",
                    "r_rank", "b_rank", "title_fight"]].reset_index(drop=True)
    picks["covered"] = fights[[f"{s}_{m}" for s in ("r", "b") for m in OUR_METHODS]].notna().all(axis=1).to_numpy()
    r_won = (fights["outcome"] == "r").to_numpy()
    for name, (r_col, b_col, lower) in PREDICTORS.items():
        picks[name] = _favourite_won(fights[r_col], fights[b_col], r_won, lower)
    return picks


def backtest_summary(picks: pd.DataFrame, reference: str = "Official rankings") -> pd.DataFrame:
    """
    For each predictor, on the fights covered by our rankings: the share of
    fights whose favourite won, and the paired comparison with ``reference``
    on the fights where the two favour different fighters (two-sided sign
    test: p is the chance of a split at least this uneven if both were
    equally good).
    """
    covered = picks[picks["covered"]]
    rows = []
    for name in PREDICTORS:
        verdict = covered[name]
        row = {"predictor": name, "fights": int(verdict.notna().sum()), "accuracy": verdict.mean()}
        if name != reference:
            ref = covered[reference]
            split = verdict.isin([0.0, 1.0]) & ref.isin([0.0, 1.0]) & (verdict != ref)
            won = int((verdict[split] == 1.0).sum())
            row.update({f"disagreements with {reference.lower()}": int(split.sum()),
                        "won by this predictor": won,
                        "p-value": binomtest(won, int(split.sum())).pvalue if split.any() else np.nan})
        rows.append(row)
    return pd.DataFrame(rows).set_index("predictor")


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Readable version of backtest_summary()."""
    out = pd.DataFrame(index=summary.index)
    out["Fights"] = summary["fights"].astype(int)
    out["Favourite won"] = (summary["accuracy"] * 100).map("{:.1f}%".format)
    split = [c for c in summary.columns if c.startswith("disagreements with")][0]
    out["Picks a different fighter than the official rankings"] = summary[split].map(
        lambda v: "" if pd.isna(v) else f"{int(v)}")
    out["... and is right"] = summary["won by this predictor"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    out["p-value"] = summary["p-value"].map(lambda v: "" if pd.isna(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}"))
    return out.rename_axis(None)
