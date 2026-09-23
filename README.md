# UFC Fighter Rating

**Predicting UFC fights and ranking fighters from three decades of fight statistics, without leaking the future.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/nadiedjoa-24/Rating_UFC/actions/workflows/ci.yml/badge.svg)](https://github.com/nadiedjoa-24/Rating_UFC/actions/workflows/ci.yml)

A complete pipeline over every UFC fight from UFC 1 (November 1993) to August 2026:
8,828 fights and 2,738 fighters. It builds leakage-free fighter features, trains
four fight-outcome models and ranks the active fighters of each division with
three independent methods: a dynamic Elo rating, a weighted statistical score
and a virtual round-robin tournament simulated by the model.

---

## Results

### Predicting fights

Test period: the 978 most recent fights (April 2024 to August 2026), never seen
during training or model selection. Fighter A is drawn at random in each fight,
so a coin flip scores 50%.

| Model (fighter statistics only) | Accuracy | AUC | Log loss |
|---|:---:|:---:|:---:|
| **Logistic regression** | **66.0%** | **0.713** | **0.628** |
| SVM | 65.8% | 0.713 | 0.627 |
| Random forest | 64.5% | 0.697 | 0.643 |
| XGBoost | 62.9% | 0.699 | 0.638 |
| Elo rating alone | 54.4% | 0.575 | 0.684 |

Against the betting market, on the 725 test fights that have odds:

| Predictor | Accuracy | AUC | Log loss |
|---|:---:|:---:|:---:|
| Betting favourite (market) | 70.2% | 0.766 | 0.578 |
| Logistic regression, statistics only | 66.9% | 0.723 | 0.623 |
| Logistic regression, statistics + odds | 70.3% | 0.768 | 0.577 |

- Public fight statistics predict the winner two times out of three.
- The market remains the reference. Adding the statistics to the odds brings the
  models level with the bookmakers, not above them: the statistics hold no
  information the market has not already priced in.
- The statistics-only model is underconfident on the test period: when it gives
  a fighter 65%, that fighter wins about 78% of the time. Its picks and their
  order are right, its probabilities are too cautious
  ([notebook 03](notebooks/03_models_and_rankings.ipynb)).

### Division rankings (as of 8 August 2026)

Top 3 by consensus of the three methods, among fighters with at least five UFC
fights and a fight in the last two years (UFC record in brackets):

| Division | 1 | 2 | 3 |
|---|---|---|---|
| Flyweight | Joshua Van (10-1) | Alexandre Pantoja (14-4) | Tatsuro Taira (8-2) |
| Bantamweight | Umar Nurmagomedov (8-1) | Sean O'Malley (12-3) | Mario Bautista (12-3) |
| Featherweight | Alexander Volkanovski (15-3) | Aljamain Sterling (18-5) | Movsar Evloev (10-0) |
| Lightweight | Ilia Topuria (9-1) | Quillan Salkilld (6-0) | Benoit Saint Denis (9-4) |
| Welterweight | Islam Makhachev (17-1) | Shavkat Rakhmonov (7-0) | Sean Brady (9-2) |
| Middleweight | Khamzat Chimaev (9-1) | Dricus Du Plessis (10-1) | Anthony Hernandez (9-3) |
| Light Heavyweight | Navajo Stirling (6-0) | Carlos Ulberg (10-1) | Magomed Ankalaev (13-2-1) |
| Heavyweight | Jon Jones (22-1) | Tom Aspinall (8-1) | Ciryl Gane (11-2) |
| Women's Strawweight | Tatiana Suarez (9-1) | Fatima Kline (4-1) | Gillian Robertson (14-6) |
| Women's Flyweight | Valentina Shevchenko (15-3-1) | Erin Blanchfield (8-1) | Zhang Weili (10-3) |
| Women's Bantamweight | Luana Santos (6-1) | Ailin Perez (6-1) | Joselyne Edwards (9-4) |

Full top 10 of every division, with each method's rank:
[UFC_Pipeline.ipynb](UFC_Pipeline.ipynb). Women's featherweight has too few
active fighters to be ranked.

The three methods agree broadly (Spearman correlation 0.5 to 0.9 depending on
the division). Against the official UFC rankings, Elo is the closest in every
division: the official panel ranks fighters by who they beat, and Elo is the
only method that knows the opponents.

---

## Findings that shaped the pipeline

**The "red corner" of old fights is the winner.** On ufcstats.com the winner is
listed first in every fight before 2010. A model trained on the raw red/blue
sides learns "red wins" from the early years, a rule that has nothing to do with
fighting. The pipeline draws fighter A at random in each fight (fixed seed),
giving a 50/50 target in every era.

**Career rates computed on a few minutes are noise.** A 20-second knockout debut
reads as 15 strikes landed per minute. Every ratio is shrunk towards the UFC
average with a prior worth about one average fight (two for win and finish
rates), so short careers stay close to the average and long careers keep their
own profile.

**Defence says more than offence.** On their own, strikes absorbed per minute
and striking defence predict the winner better than strikes landed. Age is the
single strongest signal: the younger fighter wins more often.

**Elo alone is a weak predictor.** UFC matchmaking pairs fighters with similar
records, so the Elo gap within a fight is small. Elo is still the best of the
three methods at reproducing the official rankings.

---

## How it works

```
Kaggle mirror of ufcstats.com (CC0) ──┐
our ufcstats.com scraper (optional) ──┼─> master table ─> fighter states ─> matchups ─> 4 models x 2 feature sets
betting odds + official ranks ────────┘   (8,828 fights)  (before each date)  (A vs B)          │
                                                                                                 v
                                                  current profiles ─> Elo / Weighted / Model round-robin ─> rankings
```

### Data

| Source | Content | Licence |
|---|---|---|
| [neelagiriaditya/ufc-datasets-1994-2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025) | Kaggle mirror of [ufcstats.com](http://ufcstats.com): results, fight totals, fighter profiles, updated regularly | CC0 |
| [mdabbert/ultimate-ufc-dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) | Betting odds and official ranks at fight time, 2010 to March 2026 | CC BY 4.0 |
| `src/ufc_rating/ingest/ufcstats.py` | Our scraper for events newer than the mirror | |

A snapshot of both Kaggle files is versioned in `data/raw/`, so the project runs
without a Kaggle account. With [Kaggle API credentials](https://www.kaggle.com/docs/api),
the pipeline downloads the latest version first.

Since 2026, ufcstats.com answers automated clients with a JavaScript
"Checking your browser" challenge. The scraper detects it and stops cleanly
instead of trying to get around it; its parsers are tested on archived pages
from the Wayback Machine (`tests/fixtures/`).

Only UFC events are kept (numbered events, Fight Nights, TUF finales, Noche UFC).
Odds are matched to fights on normalised fighter names and a date within one day.

### Features without leakage

For a fight on date *D*, every feature uses only fights on dates strictly before
*D*. Raw counts (fights, wins, minutes, strikes landed and absorbed, takedowns,
control time...) are cumulated per fighter, kept once per date and shifted by
one date, so the second bout of a 1990s tournament night does not know the
result of the first. Rates are computed on the real fight time, not the
scheduled one.

Each fight becomes 24 differences between fighter A and fighter B:

| Group | Features |
|---|---|
| Record | UFC fights, win rate, finish / KO / submission rates, rate of being finished, recent form (last 3), streak, layoff |
| Striking | significant strikes landed and absorbed per minute, accuracy, defence, knockdowns per 15 min |
| Grappling | takedowns per 15 min, takedown accuracy and defence, submission attempts per 15 min, control time share |
| Physical | age, height, reach, stance |
| Rating | pre-fight Elo |

The "stats + odds" models add the log-odds of the market's probability for A
(bookmaker margin removed). Fights with a draw, a no contest or a UFC debutant
are excluded from training. `tests/test_features.py` rebuilds the features with
all fights after 2018 removed and checks that nothing before 2018 changes.

### Models

A chronological split: the oldest 70% of fights for training, the next 15% for
validation, the most recent 15% for the test. Logistic regression, SVM, random
forest and XGBoost are tuned by expanding-window cross-validation on the
training period (log loss), the validation period picks the model used for the
rankings, and the test period is scored once. For the rankings, the selected
model is then refitted on every fight.

Missing differences are imputed with 0 ("no known difference") and features are
scaled without centring. The logistic regression has no intercept, so its
predictions are exactly symmetric: P(A beats B) = 1 - P(B beats A).

### Rankings

Active fighters (a fight in the last 730 days, at least 5 UFC fights) are ranked
in their most recent division by:

- **Elo**: rating updated after every fight since 1993 (K = 32, draw = half a win,
  no contest ignored).
- **Weighted**: career statistics converted to percentiles within the division,
  combined with configurable weights.
- **Model**: every pair of fighters in the division is simulated with the
  statistics-only model; the score is the average win probability.

A division needs at least two eligible fighters to be ranked.

Fighters are sorted by the mean of the three ranks.

---

## Installation

```bash
git clone https://github.com/nadiedjoa-24/Rating_UFC.git
cd Rating_UFC
pip install -e ".[notebooks]"
```

Python 3.10 or newer. For the development tools: `pip install -e ".[dev,notebooks]"`.

## Usage

Run the whole pipeline (about one minute), from the command line:

```bash
python -m ufc_rating.pipeline             # refresh the Kaggle data, then run
python -m ufc_rating.pipeline --offline   # use the versioned snapshot in data/raw
python -m ufc_rating.pipeline --scrape    # also try ufcstats.com for newer events
```

or through the notebooks. [UFC_Pipeline.ipynb](UFC_Pipeline.ipynb) runs every
step, shows the results and writes its outputs to `data/processed/`, which the
three analysis notebooks read. Run it first. The ranking weights, the activity
window and the minimum number of fights are set at its top; it uses the
versioned data unless `REFRESH_DATA = True`.

| Notebook | Content |
|---|---|
| [UFC_Pipeline](UFC_Pipeline.ipynb) | Runs the pipeline; model scores and division rankings |
| [01_exploration](notebooks/01_exploration.ipynb) | Coverage, the corner artefact, how fights end, data quality, the betting market |
| [02_feature_engineering](notebooks/02_feature_engineering.ipynb) | One career fight by fight, shrinkage, leakage check, signal of each feature |
| [03_models_and_rankings](notebooks/03_models_and_rankings.ipynb) | Calibration, confidence, coefficients, model vs market, agreement between rankings and with the official ones, Elo through history |

## Tests

```bash
pytest
```

47 tests cover the parsing of every data source, the scraper (on archived
ufcstats.com pages, including a regression test for its old round-1 bug), the
absence of leakage (a fight's own result and all later fights leave its features
unchanged), the random orientation, Elo, the exact symmetry of the logistic
regression and the ranking rules. GitHub Actions runs them on Python 3.10, 3.12
and 3.13, then runs the full pipeline on the versioned data.

## Project structure

```
Rating_UFC/
├── UFC_Pipeline.ipynb            # run the pipeline, see the results
├── notebooks/                    # analysis notebooks (read data/processed/)
├── src/ufc_rating/
│   ├── config.py                 # paths and constants
│   ├── pipeline.py               # end-to-end pipeline and CLI
│   ├── plotting.py               # shared chart style for the notebooks
│   ├── ingest/
│   │   ├── kaggle_sources.py     # download the Kaggle snapshots
│   │   └── ufcstats.py           # ufcstats.com scraper
│   ├── processing/
│   │   ├── master.py             # master fight table, odds matching
│   │   └── features.py           # leakage-free fighter states and matchups
│   ├── models/
│   │   └── training.py           # split, tuning, evaluation, baselines
│   └── ranking/
│       ├── elo.py                # dynamic Elo
│       ├── weighted.py           # weighted statistical score
│       ├── round_robin.py        # model-simulated round-robin
│       └── rankings.py           # division rankings, comparison with official ranks
├── data/
│   ├── raw/                      # versioned Kaggle snapshots
│   └── processed/                # pipeline outputs (not versioned)
├── tests/                        # pytest suite and archived HTML fixtures
└── pyproject.toml
```

## Limitations

- **The data stops on 8 August 2026.** The CC0 mirror lags ufcstats.com by a few
  weeks, and the scraper that could fill the gap is blocked by ufcstats.com's bot
  challenge. The odds end in March 2026, so the most recent test fights are
  compared with the market only where odds exist.
- **Only UFC fights count.** A fighter's record before the UFC (regional scene,
  other major promotions) is invisible, which is why debut fights are excluded
  and why newcomers start at Elo 1500.
- **Public statistics miss what bookmakers see**: injuries, short-notice
  replacements, weight cuts, stylistic match-ups. The models cannot beat the market
  and are not meant for betting.
- **The rankings are a statistical view, not a replacement for the official
  panel.** The weighted and model rankings ignore the strength of past opponents;
  Elo accounts for it but only through UFC fights.
- **Official ranks** used for validation come from fights up to March 2026 and
  only cover ranked fighters.

## Data attribution

Fight data: [ufcstats.com](http://ufcstats.com), via the Kaggle dataset
[UFC DATASETS 1994-2026](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
by neelagiriaditya (CC0). Odds and official ranks:
[Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)
by mdabbert (CC BY 4.0).

## License

Code under the [MIT License](LICENSE). The data files keep their own licences
(see above).
