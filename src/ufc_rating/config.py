"""
Project-wide paths and constants.

Paths resolve relative to the repository root, so the package is meant to be
installed in editable mode (``pip install -e .``) from a clone of the repo.
Set the ``UFC_RATING_DATA`` environment variable to use another data folder.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("UFC_RATING_DATA", ROOT_DIR / "data"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw inputs (see README, "Data sources")
UFCSTATS_CSV = RAW_DIR / "ufcstats" / "master.csv"        # Kaggle mirror of ufcstats.com, CC0
UFCSTATS_ROUNDS_CSV = RAW_DIR / "ufcstats" / "round.csv"   # same mirror, round-by-round stats
ODDS_CSV = RAW_DIR / "odds" / "ufc-master.csv"             # betting odds + official ranks, CC BY 4.0
SCRAPED_CSV = RAW_DIR / "scraped" / "fights.csv"           # our scraper: events newer than the mirror
SCRAPED_ROUNDS_CSV = RAW_DIR / "scraped" / "rounds.csv"    # our scraper: their round-by-round stats
WIKI_RANKINGS_CSV = RAW_DIR / "wikipedia" / "rankings.csv" # official rankings, weekly, CC BY-SA 4.0
WIKI_RECORDS_CSV = RAW_DIR / "wikipedia" / "records.csv"   # professional records, CC BY-SA 4.0
BFO_ODDS_CSV = RAW_DIR / "bestfightodds" / "odds.csv"      # closing odds after the odds dataset ends

# Pipeline outputs (regenerated, not versioned)
MASTER_CSV = PROCESSED_DIR / "master.csv"
ROUNDS_CSV = PROCESSED_DIR / "rounds.csv"
MATCHUPS_CSV = PROCESSED_DIR / "matchups.csv"
PROFILES_CSV = PROCESSED_DIR / "profiles.csv"
ELO_HISTORY_CSV = PROCESSED_DIR / "elo_history.csv"
ABLATION_CSV = PROCESSED_DIR / "feature_ablation.csv"
PREDICTIONS_CSV = PROCESSED_DIR / "test_predictions.csv"
RANKINGS_CSV = PROCESSED_DIR / "rankings.csv"
RANKING_BACKTEST_CSV = PROCESSED_DIR / "ranking_backtest.csv"
MODELS_PKL = PROCESSED_DIR / "models.joblib"

SEED = 42

# The twelve current UFC divisions, lightest first within each gender.
MEN_DIVISIONS = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]
WOMEN_DIVISIONS = [
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
]
DIVISIONS = MEN_DIVISIONS + WOMEN_DIVISIONS
