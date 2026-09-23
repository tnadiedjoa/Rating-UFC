"""
Do the rankings predict the fights?

A ranking claims that a fighter is better than the fighters below. The fights
between two ranked fighters of the same division test that claim: the
better-ranked fighter should win.

The fights are replayed event by event. The day before each event, the
rankings are rebuilt from the fights known at that date, with a stats model
trained on earlier fights only. The model ranking and the Elo ranking then
designate a favourite in every fight between two fighters they rank, and so
do the official UFC rankings (the ranks published before the fight) and the
betting market (the closing odds).

Two replays:
  rankings_before_events()  the test period, with the model trained before it
  walk_forward_rankings()   every season since 2013, with the model retrained
                            (and retuned) on the fights before each season
"""

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ufc_rating.models.training import train_models
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
FIRST_SEASON = 2013   # the official rankings start in February 2013


def ranked_fight_dates(master: pd.DataFrame, start: pd.Timestamp,
                       end: Optional[pd.Timestamp] = None) -> np.ndarray:
    """Event dates in [start, end) with at least one fight between two officially ranked fighters."""
    mask = (master["date"] >= pd.Timestamp(start)) & master["r_rank"].notna() & master["b_rank"].notna()
    if end is not None:
        mask &= master["date"] < pd.Timestamp(end)
    return np.sort(master.loc[mask, "date"].unique())


def _replay(states: pd.DataFrame, apps: pd.DataFrame, dates: Iterable, model,
            features: List[str], **kwargs) -> List[pd.DataFrame]:
    """The division rankings the day before each of ``dates``, with the event ``date``."""
    tables = []
    for date in dates:
        as_of = pd.Timestamp(date) - pd.Timedelta(days=1)
        tables.append(all_rankings(profiles_as_of(states, apps, as_of), model, features, as_of, **kwargs)
                      .assign(date=pd.Timestamp(date)))
    return tables


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
    The division rankings as they stood the day before each event from
    ``start`` on (the events with a fight between two ranked fighters),
    stacked, with the event ``date``. ``model`` must be trained on fights
    before ``start``.
    """
    states, apps = fighter_states(master, elo_history, rounds), appearances(master)
    dates = ranked_fight_dates(master, start)
    return pd.concat(_replay(states, apps, dates, model, features, **kwargs), ignore_index=True)


def walk_forward_rankings(
    master: pd.DataFrame,
    elo_history: pd.DataFrame,
    rounds: Optional[pd.DataFrame],
    matchups: pd.DataFrame,
    model_name: str,
    features: List[str],
    first_season: int = FIRST_SEASON,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    rankings_before_events() over every season from ``first_season`` on.
    Before each season, the ``model_name`` model is tuned and fitted on the
    matchups dated before 1 January of that season (training.train_models),
    and used for the whole season.
    """
    states, apps = fighter_states(master, elo_history, rounds), appearances(master)
    tables = []
    for season in range(first_season, master["date"].max().year + 1):
        start, end = pd.Timestamp(season, 1, 1), pd.Timestamp(season + 1, 1, 1)
        dates = ranked_fight_dates(master, start, end)
        if len(dates) == 0:
            continue
        past = matchups[matchups["date"] < start]
        model = train_models(past, features, names=[model_name], verbose=False)[model_name]
        tables += _replay(states, apps, dates, model, features, **kwargs)
        if verbose:
            print(f"  {season}: model fitted on {len(past):,} fights, {len(dates)} events replayed")
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


PERIODS = {"2013-2017": (2013, 2017), "2018-2021": (2018, 2021), "2022-2026": (2022, 2026)}


def accuracy_by_period(picks: pd.DataFrame, periods=None) -> pd.DataFrame:
    """
    Share of the covered fights won by each predictor's favourite, per period
    of seasons (label -> (first, last season)) and over all of them, with the
    number of fights.
    """
    covered = picks[picks["covered"]]
    year = covered["date"].dt.year
    groups = {label: covered[(year >= first) & (year <= last)]
              for label, (first, last) in (periods or PERIODS).items()}
    groups["All seasons"] = covered
    table = pd.DataFrame({label: group[list(PREDICTORS)].mean() for label, group in groups.items()}).T
    table["fights"] = [len(group) for group in groups.values()]
    return table
