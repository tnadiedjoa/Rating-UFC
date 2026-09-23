import numpy as np
import pandas as pd
import pytest

from ufc_rating.processing.features import (
    FIGHTER_FEATURES, PRIORS, PRIOR_MINUTES, STATS_FEATURES,
    build_matchups, current_profiles, pre_fight_features,
)
from conftest import make_master, raw_fight


def history():
    """ann fights four times; the 2020-04-01 card has ann twice (tournament night)."""
    return [
        raw_fight("f1", "2020-01-01", "ann", "bea", result="r",
                  stats={"r_total_sig_landed": 30, "b_total_sig_landed": 10}),
        raw_fight("f2", "2020-02-01", "cat", "ann", result="b"),
        raw_fight("f3", "2020-04-01", "ann", "dia", result="r"),
        raw_fight("f4", "2020-04-01", "ann", "eve", result="b"),
        raw_fight("f5", "2020-06-01", "bea", "cat", result="r"),
    ]


def test_features_only_use_earlier_dates():
    pre = pre_fight_features(make_master(history()))
    ann = pre.loc["ann"]
    assert ann.loc["2020-01-01", "n_fights"] == 0
    assert ann.loc["2020-02-01", "n_fights"] == 1
    # both fights of the tournament night see the same state: 2 prior fights
    assert ann.loc["2020-04-01", "n_fights"] == 2


def test_fight_outcome_and_stats_do_not_leak_into_its_own_features():
    base = make_master(history())
    changed_rows = history()
    changed_rows[2] = raw_fight("f3", "2020-04-01", "ann", "dia", result="b", method="KO/TKO",
                                stats={"r_total_sig_landed": 999})
    changed = make_master(changed_rows)

    before = pre_fight_features(base).loc[("ann", pd.Timestamp("2020-04-01"))]
    after = pre_fight_features(changed).loc[("ann", pd.Timestamp("2020-04-01"))]
    pd.testing.assert_series_equal(before, after)


def test_shrinkage_toward_the_league_average():
    pre = pre_fight_features(make_master(history()))
    # after f1 (15 min, 30 significant strikes landed)
    expected = (30 + PRIORS["slpm"] * PRIOR_MINUTES) / (15 + PRIOR_MINUTES)
    assert pre.loc[("ann", pd.Timestamp("2020-02-01")), "slpm"] == pytest.approx(expected)


def test_streak_and_elo_before_the_fight():
    pre = pre_fight_features(make_master(history()))
    assert pre.loc[("ann", pd.Timestamp("2020-02-01")), "streak"] == 1
    assert pre.loc[("ann", pd.Timestamp("2020-04-01")), "streak"] == 2
    assert pre.loc[("ann", pd.Timestamp("2020-01-01")), "elo"] == 1500
    assert pre.loc[("ann", pd.Timestamp("2020-02-01")), "elo"] > 1500


def test_matchups_orientation_and_filters():
    rows = history() + [
        raw_fight("f6", "2020-07-01", "ann", "bea", result="draw"),
        raw_fight("f7", "2020-08-01", "ann", "cat", result="nc"),
        raw_fight("f8", "2020-09-01", "ann", "new", result="r"),   # debut for 'new'
        raw_fight("f9", "2020-10-01", "ann", "cat", result="b"),
    ]
    matchups = build_matchups(make_master(rows), seed=0)
    ids = set(matchups["fight_id"])
    assert not ids & {"f6", "f7", "f8", "f1"}   # draw, NC, debuts excluded
    assert "f9" in ids
    for feat in FIGHTER_FEATURES:
        np.testing.assert_allclose(matchups[f"delta_{feat}"],
                                   matchups[f"a_{feat}"] - matchups[f"b_{feat}"])
    # a_wins agrees with the corner that actually won
    row = matchups.set_index("fight_id").loc["f9"]
    assert row["a_wins"] == int(row["a_name"] == "Cat")


def test_matchups_are_reproducible_and_balanced(real_master):
    first = build_matchups(real_master, seed=1)
    second = build_matchups(real_master, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert 0.47 < first["a_wins"].mean() < 0.53
    assert 0.47 < first["a_is_r"].mean() < 0.53


def test_no_leakage_from_future_fights(real_master):
    """Features of a fight must not change when every later fight is removed."""
    cutoff = pd.Timestamp("2018-01-01")
    full = build_matchups(real_master).set_index("fight_id")
    truncated = build_matchups(real_master[real_master["date"] <= cutoff]).set_index("fight_id")
    common = truncated.index
    assert len(common) > 3000
    pd.testing.assert_frame_equal(full.loc[common, STATS_FEATURES], truncated[STATS_FEATURES])


def test_current_profiles(real_master):
    profiles = current_profiles(real_master).set_index("fighter_id")
    assert profiles.index.is_unique
    assert (profiles["last_fight"] <= real_master["date"].max()).all()
    assert set(FIGHTER_FEATURES) <= set(profiles.columns)
    assert profiles["division"].dropna().isin(real_master["division"].dropna().unique()).all()
