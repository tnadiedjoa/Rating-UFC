# UFC Fighter Rating

**Predicting UFC fights and ranking fighters from three decades of fight data, without leaking the future.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Data: CC BY-SA 4.0](https://img.shields.io/badge/data-CC%20BY--SA%204.0-lightgrey.svg)](dataset/README.md#sources-and-licence)
[![Tests](https://github.com/nadiedjoa-24/Rating-UFC/actions/workflows/ci.yml/badge.svg)](https://github.com/nadiedjoa-24/Rating-UFC/actions/workflows/ci.yml)

A complete pipeline over every UFC fight from UFC 1 (November 1993) to September 2026. It gathers fight statistics, betting odds, official rankings and professional records from several sources into one clean, documented set of tables refreshed every week, builds leakage-free fighter features, trains four fight-outcome models, and ranks the active fighters of each division with three independent methods: a tuned Elo rating, a weighted statistical score and a virtual round-robin tournament simulated by the model.

---

## Results

### Predicting fights

Test period: the 982 most recent decided fights (June 2024 to September 2026), never used for training or model selection. Fighter A is drawn at random in each fight, so a coin flip scores 50%.

| Model (fighter statistics only) | Accuracy | AUC | Log loss |
|---|:---:|:---:|:---:|
| **Logistic regression** (selected on validation) | **65.8%** | **0.715** | **0.626** |
| SVM (linear) | 66.0% | 0.715 | 0.625 |
| Random forest | 65.1% | 0.702 | 0.640 |
| XGBoost | 64.7% | 0.706 | 0.634 |
| Elo rating alone | 56.1% | 0.591 | 0.681 |

Against the betting market, on the 911 test fights that have odds:

| Predictor | Accuracy | AUC | Log loss |
|---|:---:|:---:|:---:|
| Betting favourite (market) | 69.9% | 0.758 | 0.586 |
| Logistic regression, statistics only | 66.0% | 0.715 | 0.626 |
| Logistic regression, statistics + odds | 69.5% | 0.758 | 0.587 |

- Public fight statistics predict the winner two times out of three.
- The market remains the reference. Adding the statistics to the odds brings the model level with the bookmakers, not above them: the statistics hold no information the market has not already priced in.
- The statistics-only model is underconfident on the test period (when it gives a fighter 65%, that fighter wins about 74% of the time): its picks and their order are right, its probabilities too cautious ([notebook 03](notebooks/03_models_and_rankings.ipynb)).

### Division rankings (as of 19 September 2026)

Top 3 by consensus of the three methods, among fighters with at least five UFC fights and a fight in the last two years (UFC record in brackets):

| Division | 1 | 2 | 3 |
|---|---|---|---|
| Flyweight | Joshua Van (11-1) | Tatsuro Taira (8-2) | Manel Kape (8-3) |
| Bantamweight | Sean O'Malley (12-3) | Umar Nurmagomedov (8-2) | Petr Yan (12-4) |
| Featherweight | Alexander Volkanovski (15-3) | Jean Silva (7-1) | Aljamain Sterling (18-5) |
| Lightweight | Ilia Topuria (9-1) | Quillan Salkilld (6-0) | Charles Oliveira (25-11) |
| Welterweight | Islam Makhachev (18-1) | Sean Brady (9-2) | Shavkat Rakhmonov (7-0) |
| Middleweight | Khamzat Chimaev (9-1) | Dricus Du Plessis (10-1) | Kamaru Usman (16-4) |
| Light Heavyweight | Carlos Ulberg (10-1) | Navajo Stirling (6-0) | Magomed Ankalaev (13-2-1) |
| Heavyweight | Jon Jones (22-1) | Ciryl Gane (11-2) | Tom Aspinall (8-1) |
| Women's Strawweight | Tatiana Suarez (9-1) | Fatima Kline (4-1) | Denise Gomes (7-2) |
| Women's Flyweight | Valentina Shevchenko (15-3-1) | Erin Blanchfield (8-1) | Zhang Weili (10-3) |
| Women's Bantamweight | Luana Santos (6-1) | Ailin Perez (6-1) | Joselyne Edwards (9-4) |

Full top 10 of every division, with each method's rank: [UFC_Pipeline.ipynb](UFC_Pipeline.ipynb). Women's featherweight has too few active fighters to be ranked.

Of the three methods, Elo is by far the closest to the official rankings: mean Spearman correlation 0.84 with the media panel and 0.80 with the Meta UFC Rankings, against 0.55 to 0.57 for the model round-robin and about 0.4 for the weighted score. The official rankings reward who a fighter beat, and Elo is the only method that knows the opponents.

## Findings

**The "red corner" of old fights is the winner.** On ufcstats.com the winner is listed first in every fight before 2010. A model trained on the raw red/blue sides learns "red wins" from the early years. The pipeline draws fighter A at random in each fight (fixed seed), which gives a 50/50 target in every era.

**Career rates computed on a few minutes are noise.** A 20-second knockout debut reads as 15 strikes landed per minute. Every ratio is shrunk towards the UFC average with a prior worth about one average fight.

**A strong feature is not always a useful one.** The share of rounds a fighter has won is, on its own, the most informative feature of all, ahead of Elo. Yet adding the round, judges and bonus features to the model changes its log loss by less than a thousandth in a rolling-origin evaluation: they repeat what career win rates and per-minute statistics already say. They stay in the published tables, not in the model ([notebook 02](notebooks/02_feature_engineering.ipynb)).

**The new data improved Elo instead.** With K = 80 instead of the textbook 32, and split or majority decisions counting half (the judges themselves disagreed), the Elo-only log loss drops from 0.684 to 0.678 on the fights before the test period, a larger gain than any feature group ([notebook 03](notebooks/03_models_and_rankings.ipynb)).

**Judges do not count strikes.** Since 2010, 21% of decisions went to the fighter who landed fewer significant strikes, and 38% of split decisions ([notebook 01](notebooks/01_exploration.ipynb)).

**Some data would leak the future.** Wikipedia has a page for about 80% of the fighters who debuted in the 2010s but far fewer recent ones: having a page depends on how the career went. Their professional records are therefore in the published tables but not in the models.

## The data

The tables behind the project are published in [`dataset/`](dataset/) and can be reused on their own. As of 19 September 2026: **8,905 fights**, **20,904 rounds**, **2,760 fighters**, **427 weekly snapshots** of the official rankings and **46,223 professional fights** of 1,791 fighters.

| File | One row per | Content |
|---|---|---|
| [`fights.csv`](dataset/fights.csv) | UFC fight | result and method, 22 fight totals per fighter, the three judges' scores, referee, bonuses, closing odds, official ranks at fight time |
| [`rounds.csv`](dataset/rounds.csv) | round of a fight | the same statistics round by round, with the round's duration |
| [`fighters.csv`](dataset/fighters.csv) | fighter | profile, UFC record, current Elo rating, professional record |
| [`rankings.csv`](dataset/rankings.csv) | fighter in a weekly ranking | the official UFC rankings since 2018: the media panel, and the Meta UFC Rankings since June 2026 |
| [`records.csv`](dataset/records.csv) | professional fight | the whole career of fighters with a Wikipedia record, inside and outside the UFC |
| [`elo.csv`](dataset/elo.csv) | fighter in a fight | Elo rating before and after every fight |

The [data card](dataset/README.md) describes every file, its coverage (generated from the data at each update) and the points to know before using it.

### How it is built and checked

- **One key across sources.** Statistics, judges' cards, odds, official rankings and professional records all join on the ufcstats.com ids of fighters, fights and events. Names are matched once, in the pipeline, including name changes and spelling variants (99.9% of ranked names are linked to a fighter; the loosest odds matches were reviewed by hand).
- **Known traps handled.** Before 2010 ufcstats lists the *winner first* in every fight, so its red/blue sides leak the result; draws are not wins; judges' scores are written "loser - winner" and are re-oriented to the two fighters; homonyms are kept apart by id.
- **Checked.** Round-by-round statistics add up to the fight totals for every fight that has both; the parsers are tested on real pages and markup of every source; the latest scraped events were compared field by field with an independent dataset (100% agreement); where two sources give the official rank at fight time, they agree exactly for 82-85% of fighters and within one place for about 90%.
- **Kept up to date** every week (see [Weekly update](#weekly-update)).

### Sources and licence

| Data | Source | Licence |
|---|---|---|
| Fights, rounds, profiles, judges, bonuses | [ufcstats.com](http://ufcstats.com), via the Kaggle mirror [UFC Datasets 1994-2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025) and this repository's scraper for the latest events | CC0 (mirror) |
| Odds to March 2026, ranks 2010-2017 | [Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) | CC BY 4.0 |
| Odds it misses since 2023, later odds | [bestfightodds.com](https://www.bestfightodds.com) (median over the sportsbooks) | factual data, source credited |
| Official rankings, professional records | [English Wikipedia](https://en.wikipedia.org/wiki/UFC_rankings), through its API | CC BY-SA 4.0 |

Because they include material from Wikipedia, the published tables are under **CC BY-SA 4.0**. The code is under the [MIT License](LICENSE).

### Loading the tables

```python
import pandas as pd

base = "https://raw.githubusercontent.com/nadiedjoa-24/Rating-UFC/main/dataset/"
fights = pd.read_csv(base + "fights.csv", parse_dates=["date"])
fighters = pd.read_csv(base + "fighters.csv")

# Every fight of a fighter, with the opponent and the result
jones = fighters.loc[fighters["fighter"] == "Jon Jones", "fighter_id"].item()
fights[(fights["r_fighter_id"] == jones) | (fights["b_fighter_id"] == jones)][
    ["date", "event", "r_fighter", "b_fighter", "outcome", "method"]]
```

## How it works

```
ufcstats.com (Kaggle mirror + scraper) ──┐
betting odds (Kaggle + bestfightodds) ───┼─> master and round tables ─> dataset/ (published tables)
Wikipedia (rankings, records) ───────────┘            │
                                                      v
                   fighter states before each date ─> matchups (A vs B) ─> 4 models x 2 feature sets
                                                      │
                   current profiles ─> Elo / weighted score / model round-robin ─> division rankings
```

- **Features without leakage.** For a fight on date *D*, every feature uses only fights on dates strictly before *D*: raw counts are cumulated per fighter, kept once per date and shifted by one date, so even the second bout of a 1990s tournament night does not know the first. `tests/test_features.py` rebuilds the features with the later fights removed and checks that nothing changes.
- **24 differences between fighter A and fighter B**: record (fights, win, finish, KO and submission rates, rate of being finished, recent form, streak, layoff), striking (landed and absorbed per minute, accuracy, defence, knockdowns), grappling (takedowns, accuracy, defence, submission attempts, control time), physical (age, height, reach, stance) and the pre-fight Elo. The "stats + odds" models add the market's log-odds.
- **Models.** A chronological split (70% train, 15% validation, 15% test); logistic regression, SVM, random forest and XGBoost tuned by expanding-window cross-validation on the training period; the validation period picks the model used for the rankings, the test period is scored once. The logistic regression has no intercept, so P(A beats B) = 1 - P(B beats A) exactly.
- **Rankings.** Active fighters (a fight in the last two years, at least five UFC fights) are ranked in their most recent division by Elo, by a weighted score of career statistics (percentiles within the division, configurable weights), and by a round-robin where the model simulates every pair of fighters. Fighters are sorted by the mean of the three ranks.

---

## Installation

```bash
git clone https://github.com/nadiedjoa-24/Rating-UFC.git
cd Rating-UFC
pip install -e ".[notebooks]"
```

Python 3.10 or newer. Development tools: `pip install -e ".[dev,notebooks]"`. Scraper: `pip install -e ".[scrape]"`, then `python -m playwright install chromium`.

## Usage

```bash
python -m ufc_rating.pipeline             # refresh the sources, rebuild the tables, models and rankings
python -m ufc_rating.pipeline --offline   # same, from the versioned snapshots in data/raw
python -m ufc_rating.pipeline --scrape    # also scrape ufcstats.com for events newer than the mirror
```

The refresh downloads the Kaggle sources (with [Kaggle API credentials](https://www.kaggle.com/docs/api)), the Wikipedia rankings and records, and the odds of recent events; a source that cannot be reached is reported and its snapshot is used.

The notebooks show every step. [UFC_Pipeline.ipynb](UFC_Pipeline.ipynb) runs the pipeline and writes `data/processed/`, which the three analysis notebooks read: run it first.

| Notebook | Content |
|---|---|
| [UFC_Pipeline](UFC_Pipeline.ipynb) | Runs the pipeline; the published tables, model scores and division rankings |
| [01_exploration](notebooks/01_exploration.ipynb) | Coverage, the corner artefact, how fights end, bonuses, round by round, the judges, data quality, the betting market, official rankings, professional records |
| [02_feature_engineering](notebooks/02_feature_engineering.ipynb) | One career fight by fight, shrinkage, leakage check, signal of each feature, the round/judges/bonus ablation |
| [03_models_and_rankings](notebooks/03_models_and_rankings.ipynb) | Calibration, confidence, coefficients, model vs market, agreement between rankings and with both official rankings, Elo tuning, Elo through history |

### Weekly update

`scripts/weekly_update.ps1` runs the pipeline with the scraper, then commits and pushes the new data. `scripts/install_weekly_task.ps1` registers it as a Windows scheduled task (every Monday; if the computer is off, it runs at the next start-up). The scraper fetches ufcstats.com pages with random pauses between requests, and renders them in a headless Chromium (Playwright) when the site requires a JavaScript-capable client. Wikipedia is read through its public API, and bestfightodds.com one event page at a time with a pause between requests.

## Tests

```bash
pytest
```

76 tests cover the parsing of every source (ufcstats.com pages archived by the Wayback Machine, Wikipedia markup of every table layout since 2018, bestfightodds.com pages), the matching of names across sources, the orientation of the judges' scores, the absence of leakage in every feature group, Elo, the exact symmetry of the logistic regression, the ranking rules and the dataset export. GitHub Actions runs them on Python 3.10, 3.12 and 3.13, then runs the full pipeline on the versioned data.

## Project structure

```
Rating-UFC/
├── dataset/                      # the published tables and their generated data card
├── UFC_Pipeline.ipynb            # run the pipeline, see the results
├── notebooks/                    # analysis notebooks (read data/processed/)
├── scripts/                      # weekly update and its scheduled task
├── src/ufc_rating/
│   ├── config.py                 # paths and constants
│   ├── pipeline.py               # end-to-end pipeline and CLI
│   ├── dataset.py                # published tables and data card
│   ├── plotting.py               # shared chart style for the notebooks
│   ├── ingest/
│   │   ├── kaggle_sources.py     # Kaggle snapshots
│   │   ├── ufcstats.py           # ufcstats.com scraper (fights, rounds, fighters)
│   │   ├── wikipedia.py          # weekly official rankings, professional records
│   │   └── bestfightodds.py      # closing odds of recent events
│   ├── processing/
│   │   ├── master.py             # master and round tables, odds and rankings matching
│   │   └── features.py           # leakage-free fighter states and matchups
│   ├── models/
│   │   └── training.py           # split, tuning, evaluation, ablation
│   └── ranking/
│       ├── elo.py                # dynamic Elo
│       ├── weighted.py           # weighted statistical score
│       ├── round_robin.py        # model-simulated round-robin
│       └── rankings.py           # division rankings, comparison with official ranks
├── data/
│   ├── raw/                      # versioned source snapshots
│   └── processed/                # pipeline outputs (not versioned)
├── tests/                        # pytest suite and archived HTML fixtures
└── pyproject.toml
```

## Limitations

- **Odds are not complete.** About 94% of the fights since 2010 have closing odds; the gaps are in 2010 and 2023-2024, and bestfightodds.com sometimes lists only part of a card. Market comparisons use only the fights with odds.
- **Official rankings before 2018** only exist at fight time, from the Kaggle odds dataset (2010-2017); the weekly history starts in January 2018.
- **Only UFC fights feed the models.** Debut fights are excluded and newcomers start at Elo 1500; the professional records that would describe them cover notable fighters only (see Findings).
- **Public statistics miss what bookmakers see**: injuries, short-notice replacements, weight cuts, stylistic match-ups. The models do not beat the market and are not meant for betting.
- **The rankings are a statistical view.** The weighted and model rankings ignore the strength of past opponents; Elo accounts for it, but only through UFC fights.
- **The update depends on the sources.** A change in the layout of a source page breaks its parser; the parsers are tested on archived pages so that such a change shows up as a failing test.
