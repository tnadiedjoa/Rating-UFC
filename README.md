# UFC Fighter Rating & Ranking System

A complete machine learning pipeline that scrapes UFC fight data, engineers leakage-free features, trains prediction models, and generates fighter rankings by division using three independent methods.

## Overview

This project builds a data-driven UFC ranking system from scratch:

1. **Data Collection** -- Incremental web scraping from [ufcstats.com](http://ufcstats.com) combined with Kaggle datasets (fight details, betting odds, fighter profiles)
2. **Feature Engineering** -- Per-fighter cumulative statistics computed with `expanding().mean().shift(1)` to prevent data leakage. Delta vectors (Red fighter - Blue fighter) capture the relative strength between opponents.
3. **ML Models** -- Logistic Regression, SVM, Random Forest, and XGBoost trained on a strict temporal split (70/15/15) with no random shuffling
4. **Ranking** -- Three complementary methods: weighted stat-based ranking, ML round-robin tournament, and dynamic Elo rating

## Key Results

| Model              | Test Accuracy | Test AUC |
|--------------------|:------------:|:--------:|
| Logistic Regression | 58.9%       | 0.651    |
| SVM                 | 59.1%       | 0.638    |
| Random Forest       | 58.6%       | 0.609    |
| XGBoost             | 58.9%       | 0.611    |
| **Odds Baseline**   | **68.9%**   | **0.734**|

- **8,666 fights** spanning 1994--2026, **2,682 unique fighters**
- **37 features** (35 without betting odds) including striking rates, takedown accuracy, control time, finish rates, age, reach, stance, and rankings
- Elo Top 5: Jon Jones, Georges St-Pierre, Islam Makhachev, Valentina Shevchenko, Khabib Nurmagomedov

## Project Structure

```
Rating_UFC/
|-- UFC_Pipeline.ipynb          # Main pipeline notebook (run this)
|-- generate_notebook.py        # Script to regenerate UFC_Pipeline.ipynb
|-- create_notebooks.py         # Script to generate exploration/FE/ML notebooks
|-- requirements.txt            # Python dependencies
|-- src/
|   |-- ingest/
|   |   |-- ingest_data.py      # UFC Stats web scraper (incremental)
|   |-- processing/
|   |   |-- update_master.py    # Data fusion, deduplication, enrichment
|   |   |-- feature_engineering.py  # Leakage-free feature pipeline
|   |-- models/
|   |   |-- ranking.py          # ML models + 3 ranking methods
|-- data/
|   |-- raw/                    # Raw Kaggle CSVs + scraped data
|   |-- processed/              # Master CSV + feature datasets
|   |-- state/                  # Pipeline state (last run date)
|-- notebooks/                  # Detailed exploration notebooks
    |-- 01_exploration.ipynb
    |-- 02_feature_engineering.ipynb
    |-- 03_ml_models.ipynb
```

## Installation

```bash
git clone https://github.com/nadiedjoa-24/Rating_UFC.git
cd Rating_UFC
pip install -r requirements.txt
```

### Prerequisites

- Python 3.10+
- A [Kaggle API key](https://www.kaggle.com/docs/api) configured at `~/.kaggle/kaggle.json` (for automatic dataset downloads)

## Usage

### Quick Start

Open and run `UFC_Pipeline.ipynb` from top to bottom. It handles everything:

1. **Scrapes** new fights from ufcstats.com (incremental -- only fetches fights after the last run)
2. **Updates** the master dataset by merging Kaggle data + scraped data + fighter profiles
3. **Engineers** features with strict anti-leakage guarantees
4. **Trains** 4 ML models with hyperparameter tuning via `GridSearchCV` + `TimeSeriesSplit`
5. **Generates** rankings by division using all three methods

### Configuration

Edit the top of the notebook to customize:

```python
# Ranking weights (must sum to 1.0)
WEIGHTS = {
    'win_rate':        0.20,
    'finish_rate':     0.15,
    'sig_str_per_min': 0.15,
    'sig_str_acc':     0.10,
    'td_per_min':      0.10,
    'td_acc':          0.10,
    'ctrl_per_min':    0.10,
    'kd_per_min':      0.05,
    'sub_att_per_min': 0.05,
}

# Filter by division (None = all)
WEIGHT_CLASS_FILTER = 'Lightweight'

# Minimum fights for ranking eligibility
MIN_FIGHTS_RANKING = 5
```

### Detailed Notebooks

For deeper analysis, generate and run the standalone notebooks:

```bash
python create_notebooks.py
```

This creates three notebooks in `notebooks/`:
- **01_exploration.ipynb** -- Data distributions, missing values, win methods, betting odds calibration
- **02_feature_engineering.ipynb** -- Step-by-step feature construction with anti-leakage walkthrough
- **03_ml_models.ipynb** -- Model training, comparison, calibration curves, and ranking generation

## Methodology

### Data Leakage Prevention

The core challenge in fight prediction is avoiding data leakage. For a fight at date T:
- All features are computed from fights **strictly before** T
- `expanding().mean().shift(1)` ensures cumulative stats exclude the current fight
- The train/val/test split is purely chronological (no random shuffling)
- Preprocessing (imputation + scaling) is fitted **only** on the training set

### Feature Engineering

Features are built as delta vectors (Red - Blue) from per-fighter cumulative stats:

| Category | Features |
|----------|----------|
| Striking | Sig. strikes/min, striking accuracy, head/body/leg distribution |
| Grappling | Takedowns/min, TD accuracy, control time/min, submission attempts |
| Performance | Win rate, finish rate, KO rate, submission rate, recent form (last 3) |
| Physical | Height delta, reach delta, age delta, stance matchup |
| Context | Weight class, official ranking delta, empty arena flag |
| Betting | Implied probability from American odds (optional) |

### Ranking Methods

1. **Weighted Ranking** -- Normalizes per-fighter career stats (win rate, finish rate, striking, grappling, etc.) and computes a weighted score using configurable weights
2. **ML Round-Robin** -- For each pair of fighters in a division, predicts P(A beats B) using the trained model. A fighter's score = sum of win probabilities against all opponents.
3. **Dynamic Elo** -- Standard Elo rating (K=32, initial=1500) updated chronologically fight by fight. Captures historical skill trajectory.

## Data Sources

- [ufcstats.com](http://ufcstats.com) -- Official UFC statistics (scraped incrementally)
- [rajeevw/ufcdata](https://www.kaggle.com/datasets/rajeevw/ufcdata) -- Fight statistics, fighter physical attributes (height, reach, stance, DOB)
- [mdabbert/ultimate-ufc-dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) -- Fight details, betting odds, official rankings, finish methods

## License

This project is for educational and research purposes.
