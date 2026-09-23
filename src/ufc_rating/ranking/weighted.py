"""
Weighted stat-based ranking.

Each career statistic is converted to a percentile within the pool of
fighters being ranked (so one outlier cannot squash everyone else), then
combined with user-defined weights.
"""

from typing import Dict, Optional

import pandas as pd

# Statistics where a lower value is better: their percentile is inverted.
LOWER_IS_BETTER = {"sapm", "finished_rate", "age", "days_since_last"}

# Keys are columns of features.current_profiles(); weights must sum to 1.
DEFAULT_WEIGHTS = {
    "win_rate":    0.20,   # wins / decided fights
    "finish_rate": 0.15,   # wins by KO/TKO or submission
    "slpm":        0.15,   # significant strikes landed per minute
    "sig_acc":     0.10,   # significant strike accuracy
    "td_per15":    0.10,   # takedowns per 15 minutes
    "td_acc":      0.10,   # takedown accuracy
    "ctrl_pct":    0.10,   # share of fight time in control
    "kd_per15":    0.05,   # knockdowns per 15 minutes
    "sub_per15":   0.05,   # submission attempts per 15 minutes
}


def check_weights(weights: Dict[str, float], columns) -> None:
    unknown = set(weights) - set(columns)
    if unknown:
        raise ValueError(f"Unknown statistics in weights: {sorted(unknown)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0 (got {total:.3f})")


def weighted_scores(pool: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Score in [0, 1] for each fighter of ``pool``: weighted mean of percentiles."""
    weights = weights or DEFAULT_WEIGHTS
    check_weights(weights, pool.columns)
    score = pd.Series(0.0, index=pool.index)
    for column, weight in weights.items():
        ascending = column not in LOWER_IS_BETTER
        score += weight * pool[column].rank(pct=True, ascending=ascending).fillna(0.5)
    return score
