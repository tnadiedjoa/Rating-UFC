"""
Model-based ranking: a virtual round-robin tournament.

Every fighter of a division "fights" every other one: the trained model
predicts P(A beats B) from their current profiles. A fighter's score is the
average win probability over all these virtual bouts (0.5 = average).
"""

from typing import List

import numpy as np
import pandas as pd


def win_probability_matrix(pool: pd.DataFrame, model, features: List[str]) -> np.ndarray:
    """
    n x n matrix M where M[i, j] = P(fighter i beats fighter j).

    ``features`` are the model's delta_* inputs; the profile column of each
    one is its name without the prefix. Each pair is predicted in both
    orientations and averaged, so M[i, j] + M[j, i] = 1 for any model.
    """
    columns = [f.removeprefix("delta_") for f in features]
    values = pool[columns].to_numpy(dtype=float)
    n, d = values.shape
    deltas = (values[:, None, :] - values[None, :, :]).reshape(n * n, d)
    raw = model.predict_proba(deltas)[:, 1].reshape(n, n)
    matrix = (raw + (1.0 - raw.T)) / 2.0
    np.fill_diagonal(matrix, np.nan)
    return matrix


def round_robin_scores(pool: pd.DataFrame, model, features: List[str]) -> pd.Series:
    """Mean predicted win probability of each fighter against the rest of ``pool``."""
    if len(pool) < 2:
        return pd.Series(np.nan, index=pool.index)
    matrix = win_probability_matrix(pool, model, features)
    return pd.Series(np.nanmean(matrix, axis=1), index=pool.index)
