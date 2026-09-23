"""
Fight-outcome models: chronological split, tuning, evaluation, baselines.

Every model sees the same antisymmetric inputs (fighter A minus fighter B).
Missing values are imputed with 0, which means "no known difference", and
features are scaled without centring, so swapping A and B flips the sign of
every input. The logistic regression is fitted without an intercept: its
predictions are then exactly symmetric, P(A beats B) = 1 - P(B beats A).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ufc_rating.config import SEED

MODEL_NAMES = ["LogReg", "SVM", "RandomForest", "XGBoost"]


# ===========================================================================
# Split
# ===========================================================================

def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split, cut on event dates so a
    card is never shared between two sets.
    """
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)
    dates = df["date"]

    def cut(frac):
        boundary = dates.iloc[min(int(len(df) * frac), len(df) - 1)]
        return int((dates < boundary).sum())

    i_val, i_test = cut(train_frac), cut(train_frac + val_frac)
    return df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]


# ===========================================================================
# Models
# ===========================================================================

def _pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale", StandardScaler(with_mean=False)),
        ("model", estimator),
    ])


def model_grid(seed: int = SEED) -> Dict[str, Tuple[Pipeline, dict]]:
    """Estimators and their hyperparameter grids."""
    return {
        "LogReg": (
            _pipeline(LogisticRegression(fit_intercept=False, max_iter=2000)),
            {"model__C": [0.001, 0.01, 0.1, 1.0]},
        ),
        "SVM": (
            # Platt scaling on 5 internal folds turns SVM scores into probabilities
            _pipeline(CalibratedClassifierCV(SVC(random_state=seed), method="sigmoid", ensemble=False)),
            {"model__estimator__kernel": ["linear", "rbf"], "model__estimator__C": [0.01, 0.1, 1.0]},
        ),
        "RandomForest": (
            _pipeline(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
            {"model__max_depth": [4, 8], "model__min_samples_leaf": [10, 40]},
        ),
        "XGBoost": (
            _pipeline(XGBClassifier(
                n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=-1, verbosity=0)),
            {"model__max_depth": [2, 3], "model__learning_rate": [0.02, 0.05]},
        ),
    }


def train_models(train: pd.DataFrame, features: List[str], target: str = "a_wins",
                 seed: int = SEED, verbose: bool = True,
                 names: Optional[List[str]] = None) -> Dict[str, Pipeline]:
    """
    Tune each model (all of MODEL_NAMES, or ``names``) with a 5-fold
    expanding-window cross-validation on the training period only
    (TimeSeriesSplit, log-loss), then refit it on the whole training period.
    """
    X, y = train[features].to_numpy(dtype=float), train[target].to_numpy()
    fitted = {}
    grid_by_name = model_grid(seed)
    for name in names or MODEL_NAMES:
        pipeline, grid = grid_by_name[name]
        search = GridSearchCV(pipeline, grid, cv=TimeSeriesSplit(n_splits=5),
                              scoring="neg_log_loss", n_jobs=-1)
        search.fit(X, y)
        fitted[name] = search.best_estimator_
        if verbose:
            params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
            print(f"  {name:<13} CV log-loss {-search.best_score_:.4f}  {params}")
    return fitted


# ===========================================================================
# Evaluation
# ===========================================================================

def scores(y_true, proba) -> dict:
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, proba > 0.5),
        "auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
    }


def elo_probability(df: pd.DataFrame) -> np.ndarray:
    """P(A wins) predicted by the pre-fight Elo ratings alone."""
    return 1.0 / (1.0 + 10 ** (-df["delta_elo"].to_numpy(dtype=float) / 400.0))


def predict(models: Dict[str, Pipeline], df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].to_numpy(dtype=float)
    return pd.DataFrame({name: m.predict_proba(X)[:, 1] for name, m in models.items()}, index=df.index)


def evaluate(models: Dict[str, Pipeline], df: pd.DataFrame, features: List[str],
             target: str = "a_wins") -> pd.DataFrame:
    """Metrics of each model on ``df``, plus the Elo-only baseline."""
    probas = predict(models, df, features)
    probas["Elo only"] = elo_probability(df)
    rows = {name: scores(df[target], probas[name]) for name in probas.columns}
    return pd.DataFrame(rows).T


def compare_with_market(stats_models: Dict[str, Pipeline], odds_models: Dict[str, Pipeline],
                        test: pd.DataFrame, stats_features: List[str], odds_features: List[str],
                        target: str = "a_wins") -> pd.DataFrame:
    """
    Head-to-head on the test fights that have betting odds, so every line of
    the table is scored on exactly the same fights.
    """
    sub = test[test["a_odds_prob"].notna()]
    rows = {"Betting favourite (market)": scores(sub[target], sub["a_odds_prob"]),
            "Elo only": scores(sub[target], elo_probability(sub))}
    for name, proba in predict(stats_models, sub, stats_features).items():
        rows[f"{name} (stats)"] = scores(sub[target], proba)
    for name, proba in predict(odds_models, sub, odds_features).items():
        rows[f"{name} (stats + odds)"] = scores(sub[target], proba)
    return pd.DataFrame(rows).T


def ablation(dev: pd.DataFrame, feature_sets: Dict[str, List[str]], n_folds: int = 5,
             C: float = 0.01, target: str = "a_wins") -> pd.DataFrame:
    """
    Does a feature set beat the first one? Rolling-origin evaluation on the
    train + validation fights (the test set is not involved): the period is
    cut into ``n_folds + 1`` chronological blocks, and each block from the
    second on is predicted by a logistic regression trained on everything
    before it.

    Returns one row per feature set: log loss per fold, mean, and the
    difference with the first set (negative = better) with the number of
    folds where it is better.
    """
    dev = dev.sort_values(["date", "fight_id"]).reset_index(drop=True)
    y = dev[target].to_numpy()
    folds = list(TimeSeriesSplit(n_splits=n_folds).split(dev))
    pipeline = model_grid()["LogReg"][0].set_params(model__C=C)
    losses = {}
    for set_name, features in feature_sets.items():
        X = dev[features].to_numpy(dtype=float)
        losses[set_name] = np.array([
            log_loss(y[test], clone(pipeline).fit(X[train], y[train]).predict_proba(X[test])[:, 1])
            for train, test in folds
        ])
    reference = losses[next(iter(feature_sets))]
    rows = []
    for set_name, values in losses.items():
        rows.append({"features": set_name, "n_features": len(feature_sets[set_name]),
                     **{f"fold {i + 1}": v for i, v in enumerate(values)},
                     "mean log loss": values.mean(),
                     "vs first set": (values - reference).mean(),
                     "folds better": int((values < reference).sum())})
    return pd.DataFrame(rows).set_index("features")


def format_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Readable version of a scores table."""
    out = table.copy()
    out["n"] = out["n"].astype(int)
    out["accuracy"] = (out["accuracy"] * 100).map("{:.1f}%".format)
    for col in ("auc", "log_loss", "brier"):
        out[col] = out[col].map("{:.3f}".format)
    return out.rename(columns={"n": "Fights", "accuracy": "Accuracy", "auc": "AUC",
                               "log_loss": "Log loss", "brier": "Brier"})


# ===========================================================================
# Persistence
# ===========================================================================

def save_models(bundle: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_models(path: Path) -> dict:
    return joblib.load(path)
