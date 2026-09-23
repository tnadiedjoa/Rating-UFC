"""
Dynamic Elo rating, updated fight by fight in chronological order.

Win = 1, draw = 0.5, no contest = no update. Every fighter starts at 1500.
The history keeps the rating before and after each fight, which is what the
feature pipeline uses (pre-fight Elo is a leakage-free feature).
"""

import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the Elo model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo(master: pd.DataFrame, k: float = K_FACTOR, initial: float = INITIAL_ELO) -> pd.DataFrame:
    """
    Run Elo over the whole master table.

    Returns one row per (fight, fighter): fight_id, date, fighter_id,
    fighter_name, division, elo_before, elo_after. Fights on the same date
    are processed in fight_id order (the true bout order is not in the data;
    it only matters for the one-night tournaments of the 1990s).
    """
    ratings = {}
    rows = []
    fights = master.sort_values(["date", "fight_id"])
    for fight in fights.itertuples(index=False):
        r, b = fight.r_id, fight.b_id
        before_r = ratings.get(r, initial)
        before_b = ratings.get(b, initial)

        if fight.outcome == "nc":
            after_r, after_b = before_r, before_b
        else:
            score_r = {"r": 1.0, "b": 0.0, "draw": 0.5}[fight.outcome]
            exp_r = expected_score(before_r, before_b)
            after_r = before_r + k * (score_r - exp_r)
            after_b = before_b + k * ((1.0 - score_r) - (1.0 - exp_r))
        ratings[r], ratings[b] = after_r, after_b

        for fid, name, before, after in ((r, fight.r_name, before_r, after_r),
                                         (b, fight.b_name, before_b, after_b)):
            rows.append({
                "fight_id": fight.fight_id, "date": fight.date,
                "fighter_id": fid, "fighter_name": name, "division": fight.division,
                "elo_before": before, "elo_after": after,
            })
    return pd.DataFrame(rows)


def peak_elo(history: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """All-time table: highest rating each fighter ever reached, and when."""
    idx = history.groupby("fighter_id")["elo_after"].idxmax()
    peaks = history.loc[idx, ["fighter_name", "elo_after", "date"]]
    peaks = peaks.rename(columns={"fighter_name": "Fighter", "elo_after": "Peak Elo", "date": "Reached on"})
    peaks = peaks.sort_values("Peak Elo", ascending=False).head(top).reset_index(drop=True)
    peaks["Peak Elo"] = peaks["Peak Elo"].round(0).astype(int)
    peaks["Reached on"] = pd.to_datetime(peaks["Reached on"]).dt.date
    peaks.index = peaks.index + 1
    return peaks
