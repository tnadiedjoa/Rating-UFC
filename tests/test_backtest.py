import numpy as np
import pandas as pd
import pytest

from ufc_rating.ranking.backtest import OUR_METHODS, _favourite_won, backtest_summary, ranking_picks


def test_favourite_won_handles_ranks_scores_ties_and_gaps():
    # ranks (lower is better): r is ranked 2 and b 5, r wins -> the favourite won
    assert _favourite_won([2], [5], [True], lower_is_better=True)[0] == 1.0
    assert _favourite_won([2], [5], [False], lower_is_better=True)[0] == 0.0
    # scores (higher is better), tie, missing value
    assert _favourite_won([0.7], [0.3], [False], lower_is_better=False)[0] == 0.0
    assert _favourite_won([3], [3], [True], lower_is_better=True)[0] == 0.5
    assert np.isnan(_favourite_won([np.nan], [1], [True], lower_is_better=True)[0])


def fight(fight_id, date, r, b, outcome, r_rank, b_rank, r_odds=-200, b_odds=170):
    return {"fight_id": fight_id, "date": pd.Timestamp(date), "division": "Lightweight",
            "r_id": r, "b_id": b, "r_name": r.title(), "b_name": b.title(), "outcome": outcome,
            "r_rank": r_rank, "b_rank": b_rank, "r_odds": r_odds, "b_odds": b_odds, "title_fight": 0}


def ranked(date, fighter, rank):
    return {"date": pd.Timestamp(date), "division": "Lightweight", "fighter_id": fighter,
            **{method: rank for method in OUR_METHODS}}


def test_ranking_picks_use_the_ranks_of_the_fight_day_and_keep_ranked_fights_only():
    master = pd.DataFrame([
        fight("f1", "2025-01-10", "ann", "bea", "b", r_rank=3, b_rank=8),       # official favourite loses
        fight("f2", "2025-01-10", "cat", "dia", "r", r_rank=4, b_rank=np.nan),  # dia unranked: left out
        fight("f3", "2025-02-10", "ann", "eve", "r", r_rank=5, b_rank=1),       # eve not in our rankings
        fight("f4", "2024-12-01", "ann", "bea", "r", r_rank=3, b_rank=8),       # before the replay
    ])
    history = pd.DataFrame([
        ranked("2025-01-10", "ann", 4), ranked("2025-01-10", "bea", 2),
        ranked("2025-02-10", "ann", 1),
        ranked("2025-02-10", "bea", 9),   # a later ranking must not be used for f1
    ])
    picks = ranking_picks(master, history).set_index("fight_id")
    assert list(picks.index) == ["f1", "f3"]
    assert picks.loc["f1", "Official rankings"] == 0.0     # ann ranked 3, bea 8, bea won
    assert picks.loc["f1", "Model"] == 1.0                 # our rankings had bea 2nd, ann 4th
    assert picks.loc["f1", "Betting favourite"] == 0.0     # ann was the -200 favourite
    assert picks.loc["f1", "covered"] and not picks.loc["f3", "covered"]


def test_summary_counts_disagreements_with_the_official_rankings():
    picks = pd.DataFrame({"covered": [True] * 5 + [False]})
    picks["Official rankings"] = [1.0, 0.0, 0.0, 1.0, 0.5, 0.0]
    for name in ("Elo", "Betting favourite"):
        picks[name] = picks["Official rankings"]
    picks["Model"] = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    summary = backtest_summary(picks)
    assert summary.loc["Official rankings", "fights"] == 5          # the uncovered fight is left out
    assert summary.loc["Official rankings", "accuracy"] == pytest.approx(2.5 / 5)
    model = summary.loc["Model"]
    # fights 2, 3 (model right) and 4 (official right); fight 5 is a tie for the official rankings
    assert model["disagreements with official rankings"] == 3 and model["won by this predictor"] == 2
    assert summary.loc["Elo", "disagreements with official rankings"] == 0
