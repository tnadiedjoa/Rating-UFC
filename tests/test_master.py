import numpy as np
import pandas as pd
import pytest

from ufc_rating.processing.master import (
    american_to_prob, attach_odds, fight_seconds, height_to_cm, is_ufc_event,
    method_group, normalize_name, parse_division, round_lengths,
)
from conftest import make_master, raw_fight


@pytest.mark.parametrize("label, expected", [
    ("Lightweight", "Lightweight"),
    ("Light Heavyweight", "Light Heavyweight"),
    ("Heavyweight", "Heavyweight"),
    ("Women's Strawweight", "Women's Strawweight"),
    ("UFC Women's Bantamweight Title", "Women's Bantamweight"),
    ("Ultimate Fighter 28 Heavyweight Tournament", "Heavyweight"),
    ("Catch Weight", None),
    ("Open Weight", None),
    ("Super Heavyweight", None),
    ("Strawweight", None),          # there is no men's strawweight
    (None, None),
])
def test_parse_division(label, expected):
    assert parse_division(label) == expected


def test_round_lengths_and_fight_seconds():
    assert round_lengths("3 Rnd (5-5-5)") == [5, 5, 5]
    assert round_lengths("1 Rnd + OT (12-3)") == [12, 3]
    assert round_lengths("No Time Limit") == []
    assert fight_seconds("5 Rnd (5-5-5-5-5)", 5, "5:00") == 1500
    assert fight_seconds("3 Rnd (5-5-5)", 2, "1:30") == 390
    assert fight_seconds("1 Rnd + OT (12-3)", 2, "3:00") == 900
    assert fight_seconds("No Time Limit", 1, "12:13") == 733
    assert np.isnan(fight_seconds("3 Rnd (5-5-5)", 1, "--"))


def test_small_parsers():
    assert height_to_cm("5' 11\"") == 180.3
    assert np.isnan(height_to_cm(None))
    assert method_group("TKO - Doctor's Stoppage") == "KO/TKO"
    assert method_group("Submission") == "Submission"
    assert method_group("Decision - Split") == "Decision"
    assert method_group("DQ") == "Other"
    assert normalize_name("José Aldo") == normalize_name("Jose  Aldo")
    assert normalize_name("B.J. Penn") == normalize_name("B J Penn")
    np.testing.assert_allclose(american_to_prob([-200, 100, 300]), [2 / 3, 0.5, 0.25])


def test_ufc_event_filter():
    assert is_ufc_event("UFC 300: Pereira vs. Hill")
    assert is_ufc_event("Noche UFC: Silva vs. Delgado")
    assert is_ufc_event("The Ultimate Fighter: Team Rousey vs. Team Tate Finale")
    assert not is_ufc_event("Road to UFC 3.1 + 3.2")
    assert not is_ufc_event("PRIDE 34: Kamikaze")
    assert not is_ufc_event("Strikeforce - Shamrock vs. Baroni")


def test_outcomes_and_derived_columns():
    master = make_master([
        raw_fight("f1", "2020-01-01", "ann", "bea", result="r"),
        raw_fight("f2", "2020-02-01", "ann", "bea", result="b", method="KO/TKO",
                  finish_round=2, finish_time="0:30"),
        raw_fight("f3", "2020-03-01", "ann", "bea", result="draw"),
        raw_fight("f4", "2020-04-01", "ann", "bea", result="nc", method="Overturned"),
    ]).set_index("fight_id")
    assert master["outcome"].to_dict() == {"f1": "r", "f2": "b", "f3": "draw", "f4": "nc"}
    assert master.loc["f2", "fight_seconds"] == 330
    assert master.loc["f2", "method_group"] == "KO/TKO"
    assert master.loc["f1", "division"] == "Lightweight"
    assert master.loc["f1", "r_reach_cm"] == pytest.approx(177.8)


def test_control_time_untracked_before_2000():
    zero_ctrl = {"r_total_ctrl_seconds": 0, "b_total_ctrl_seconds": 0}
    master = make_master([
        raw_fight("old", "1997-05-01", "ann", "bea", stats=zero_ctrl),
        raw_fight("new", "2005-05-01", "ann", "bea", stats=zero_ctrl),
    ]).set_index("fight_id")
    assert np.isnan(master.loc["old", "r_ctrl_sec"])
    assert master.loc["new", "r_ctrl_sec"] == 0


def test_attach_odds_matches_swapped_names_and_nearby_dates():
    fights = make_master([
        raw_fight("f1", "2020-01-01", "ann", "bea"),
        raw_fight("f2", "2020-06-01", "cat", "dia"),
        raw_fight("f3", "2020-09-01", "eve", "fay"),
    ])
    odds = pd.DataFrame({
        "odds_date": pd.to_datetime(["2020-01-01", "2020-06-02", "2020-09-10"]),
        "odds_r": ["bea", "cat", "eve"],     # f1 listed the other way round
        "odds_b": ["ann", "dia", "fay"],
        "R_odds": [-300, 150, 110], "B_odds": [250, -170, -130],
        "R_rank": [np.nan, 3, 1], "B_rank": [5, np.nan, 2],
    })
    out = attach_odds(fights, odds).set_index("fight_id")
    # f1: odds belong to the swapped fighters
    assert out.loc["f1", "r_odds"] == 250 and out.loc["f1", "b_odds"] == -300
    assert out.loc["f1", "r_rank"] == 5 and np.isnan(out.loc["f1", "b_rank"])
    # f2: one day apart is accepted
    assert out.loc["f2", "r_odds"] == 150
    # f3: nine days apart is a different fight
    assert np.isnan(out.loc["f3", "r_odds"])


def test_real_snapshot_is_consistent(real_master):
    assert real_master["fight_id"].is_unique
    assert real_master["date"].is_monotonic_increasing
    assert set(real_master["outcome"]) == {"r", "b", "draw", "nc"}
    assert real_master["fight_seconds"].notna().all()
    # the 12 divisions cover almost every fight (the rest: catch/open weight)
    assert real_master["division"].notna().mean() > 0.95
