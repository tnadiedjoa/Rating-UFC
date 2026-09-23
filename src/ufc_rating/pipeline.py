"""
End-to-end pipeline: data update -> master table -> features -> models -> rankings.

Every step writes its output to data/processed/, which the analysis notebooks
read. Run it from the command line:

    python -m ufc_rating.pipeline              # refresh Kaggle data, then run
    python -m ufc_rating.pipeline --offline    # use the versioned snapshot
    python -m ufc_rating.pipeline --scrape     # also try ufcstats.com directly
"""

import argparse
from typing import Dict, Optional

import pandas as pd
from sklearn.base import clone

from ufc_rating import config
from ufc_rating.ingest.kaggle_sources import download_sources
from ufc_rating.ingest.ufcstats import BotChallengeError, scrape_since
from ufc_rating.models.training import (
    compare_with_market, evaluate, predict, save_models, temporal_split, train_models,
)
from ufc_rating.processing.features import (
    ODDS_FEATURES, STATS_FEATURES, build_matchups, current_profiles,
)
from ufc_rating.processing.master import build_master
from ufc_rating.ranking.elo import compute_elo
from ufc_rating.ranking.rankings import all_rankings


def update_data(refresh: bool = True, scrape: bool = False) -> pd.DataFrame:
    """Refresh the raw sources, optionally scrape newer events, rebuild the master table."""
    if refresh:
        print("Refreshing Kaggle sources...")
        download_sources()
    if scrape:
        known = build_master(out_path=None)["date"].max().date()
        print(f"Scraping ufcstats.com for events after {known}...")
        try:
            scrape_since(known)
        except (BotChallengeError, ConnectionError, ValueError) as exc:
            print(f"  Scraping skipped: {exc}")
    master = build_master()
    print(f"Master table: {len(master):,} UFC fights, "
          f"{master['date'].min().date()} to {master['date'].max().date()}")
    return master


def build_features(master: pd.DataFrame):
    """Elo history, matchup vectors for the models and current fighter profiles."""
    elo_history = compute_elo(master)
    matchups = build_matchups(master, elo_history=elo_history)
    profiles = current_profiles(master, elo_history=elo_history)
    elo_history.to_csv(config.ELO_HISTORY_CSV, index=False)
    matchups.to_csv(config.MATCHUPS_CSV, index=False)
    profiles.to_csv(config.PROFILES_CSV, index=False)
    print(f"Matchups: {len(matchups):,} fights with two experienced fighters, "
          f"{len(STATS_FEATURES)} stat features (+ odds)")
    return elo_history, matchups, profiles


def fit_models(matchups: pd.DataFrame) -> Dict:
    """Train both model families, score them, save models and test predictions."""
    train, val, test = temporal_split(matchups)
    print("Stats-only models:")
    stats_models = train_models(train, STATS_FEATURES)
    print("Stats + odds models:")
    odds_models = train_models(train, ODDS_FEATURES)

    val_scores = evaluate(stats_models, val, STATS_FEATURES)
    best = val_scores.drop(index="Elo only")["log_loss"].idxmin()

    # The rankings describe fighters today: the selected model, with its tuned
    # hyperparameters, is refitted on every fight (train + validation + test).
    ranking_model = clone(stats_models[best]).fit(
        matchups[STATS_FEATURES].to_numpy(dtype=float), matchups["a_wins"].to_numpy())
    results = {
        "split": {name: (part["date"].min(), part["date"].max(), len(part))
                  for name, part in (("train", train), ("validation", val), ("test", test))},
        "validation": val_scores,
        "test_stats": evaluate(stats_models, test, STATS_FEATURES),
        "test_odds": evaluate(odds_models, test, ODDS_FEATURES),
        "test_market": compare_with_market(stats_models, odds_models, test,
                                           STATS_FEATURES, ODDS_FEATURES),
        "best_stats_model": best,
    }

    predictions = test[["fight_id", "date", "division", "a_name", "b_name", "a_is_r",
                        "a_wins", "a_odds_prob", "delta_elo"]].copy()
    for name, proba in predict(stats_models, test, STATS_FEATURES).items():
        predictions[f"{name} (stats)"] = proba
    for name, proba in predict(odds_models, test, ODDS_FEATURES).items():
        predictions[f"{name} (stats + odds)"] = proba
    predictions.to_csv(config.PREDICTIONS_CSV, index=False)

    save_models({"stats": stats_models, "odds": odds_models, "ranking_model": ranking_model,
                 "stats_features": STATS_FEATURES, "odds_features": ODDS_FEATURES,
                 "results": results}, config.MODELS_PKL)
    return {"stats": stats_models, "odds": odds_models, "ranking_model": ranking_model,
            "results": results}


def rank_fighters(profiles: pd.DataFrame, model, as_of: pd.Timestamp,
                  weights: Optional[Dict[str, float]] = None, **kwargs) -> pd.DataFrame:
    rankings = all_rankings(profiles, model, STATS_FEATURES, as_of, weights, **kwargs)
    rankings.to_csv(config.RANKINGS_CSV, index=False)
    return rankings


def run(refresh: bool = True, scrape: bool = False) -> Dict:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master = update_data(refresh, scrape)
    elo_history, matchups, profiles = build_features(master)
    fitted = fit_models(matchups)
    best = fitted["results"]["best_stats_model"]
    rankings = rank_fighters(profiles, fitted["ranking_model"], master["date"].max())
    print(f"Rankings written to {config.RANKINGS_CSV} (round-robin model: {best})")
    return {"master": master, "matchups": matchups, "profiles": profiles,
            "elo_history": elo_history, "rankings": rankings, **fitted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UFC rating pipeline.")
    parser.add_argument("--offline", action="store_true",
                        help="do not download the Kaggle sources, use data/raw as is")
    parser.add_argument("--scrape", action="store_true",
                        help="also scrape ufcstats.com for events newer than the data")
    args = parser.parse_args()
    run(refresh=not args.offline, scrape=args.scrape)


if __name__ == "__main__":
    main()
