import numpy as np
import pandas as pd
import pytest

from ufc_rating.processing.master import (
    american_to_prob, attach_odds, build_rounds, fight_seconds, height_to_cm, is_ufc_event,
    match_ranked_fighters, method_group, normalize_name, parse_division, parse_scorecards, round_lengths,
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
    assert normalize_name("Jan Błachowicz") == "jan blachowicz"
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
        "odds_source": "dataset",
    })
    out = attach_odds(fights, odds).set_index("fight_id")
    # f1: odds belong to the swapped fighters
    assert out.loc["f1", "r_odds"] == 250 and out.loc["f1", "b_odds"] == -300
    assert out.loc["f1", "r_rank"] == 5 and np.isnan(out.loc["f1", "b_rank"])
    # f2: one day apart is accepted
    assert out.loc["f2", "r_odds"] == 150
    # f3: nine days apart is a different fight
    assert np.isnan(out.loc["f3", "r_odds"])


def test_first_odds_source_wins():
    fights = make_master([raw_fight("f1", "2020-01-01", "ann", "bea")])
    odds = pd.DataFrame({
        "odds_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "odds_r": ["ann", "ann"], "odds_b": ["bea", "bea"],
        "R_odds": [-200, -150], "B_odds": [170, 130],
        "R_rank": np.nan, "B_rank": np.nan,
        "odds_source": ["dataset", "bestfightodds"],
    })
    out = attach_odds(fights, odds).iloc[0]
    assert (out["r_odds"], out["odds_source"]) == (-200, "dataset")
    out = attach_odds(fights, odds.iloc[::-1]).iloc[0]
    assert (out["r_odds"], out["odds_source"]) == (-150, "bestfightodds")


def test_real_snapshot_is_consistent(real_master):
    assert real_master["fight_id"].is_unique
    assert real_master["date"].is_monotonic_increasing
    assert set(real_master["outcome"]) == {"r", "b", "draw", "nc"}
    assert real_master["fight_seconds"].notna().all()
    # the 12 divisions cover almost every fight (the rest: catch/open weight)
    assert real_master["division"].notna().mean() > 0.95


def test_scorecards_are_oriented_with_the_winner():
    details = "Mike Bell 44 - 50. Dave Tirelli 45 - 50. Sal D'amato 48 - 47."
    assert parse_scorecards(details, "r") == [("Mike Bell", 50, 44), ("Dave Tirelli", 50, 45),
                                              ("Sal D'amato", 47, 48)]
    assert parse_scorecards(details, "b")[0] == ("Mike Bell", 44, 50)
    assert parse_scorecards(details, "draw") == []           # cannot be oriented
    assert parse_scorecards("27 - 30. 28 - 29.", "r") == [(None, 30, 27), (None, 29, 28)]


def test_judges_columns_only_for_decisions():
    master = make_master([
        {**raw_fight("f1", "2020-01-01", "ann", "bea", result="b"),
         "details": "Mike Bell 28 - 29. Chris Lee 28 - 29. Tony Weeks 29 - 28."},
        {**raw_fight("f2", "2020-01-01", "cat", "dia", result="r", method="KO/TKO"),
         "details": "Punch to Head At Distance"},
    ])
    f1, f2 = master.iloc[0], master.iloc[1]
    assert (f1["judge1_name"], f1["judge1_r_score"], f1["judge1_b_score"]) == ("Mike Bell", 28, 29)
    assert (f1["judge3_r_score"], f1["judge3_b_score"]) == (29, 28)   # the dissenting judge
    assert f2["judge1_name"] is None and np.isnan(f2["judge1_r_score"])


def test_round_table_durations_and_sides(tmp_path):
    master = make_master([raw_fight("f1", "2020-01-01", "ann", "bea", method="KO/TKO",
                                    finish_round=2, finish_time="2:30")])
    rounds = pd.DataFrame({"fight_id": ["f1", "f1"], "round_no": [1, 2],
                           "r_id": ["ann", "ann"], "b_id": ["bea", "bea"],
                           "r_ctrl": ["1:00", "0:10"], "b_ctrl": ["0:00", "0:00"]})
    for side in ("r", "b"):
        for suffix in ("kd", "sig_landed", "sig_atmp", "total_str_landed", "total_str_atmp",
                       "td_success", "td_atmp", "sub_att", "rev"):
            rounds[f"{side}_{suffix}"] = 1
        for zone in ("head", "body", "leg", "distance", "clinch", "ground"):
            rounds[f"{side}_sig_str_landed_{zone}"] = rounds[f"{side}_sig_str_atmp_{zone}"] = 1
    rounds.to_csv(tmp_path / "round.csv", index=False)

    table = build_rounds(master, tmp_path / "round.csv", None, out_path=None)
    assert list(table["seconds"]) == [300, 150]
    assert list(table["r_ctrl_sec"]) == [60, 10]

    rounds[["r_id", "b_id"]] = rounds[["b_id", "r_id"]].values
    rounds.to_csv(tmp_path / "round.csv", index=False)
    with pytest.raises(ValueError):
        build_rounds(master, tmp_path / "round.csv", None, out_path=None)


def test_odds_come_from_a_source_that_has_them():
    fights = make_master([raw_fight("f1", "2024-06-01", "ann", "bea")])
    odds = pd.DataFrame({
        "odds_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
        "odds_r": ["ann", "bea"], "odds_b": ["bea", "ann"],
        "R_odds": [np.nan, 150], "B_odds": [np.nan, -170],      # first source: fight listed, no odds
        "R_rank": [4, np.nan], "B_rank": [np.nan, np.nan],
        "odds_source": ["dataset", "bestfightodds"],
    })
    out = attach_odds(fights, odds).iloc[0]
    assert (out["r_odds"], out["b_odds"], out["odds_source"]) == (-170, 150, "bestfightodds")
    assert out["r_rank"] == 4


def test_loose_odds_match_keeps_each_price_with_its_fighter():
    # the loose pass pairs the fight crossed; the r fighter must get her own price
    fights = make_master([raw_fight("f1", "2021-01-01", "ariane da silva", "karine silva")])
    fights["r_name"], fights["b_name"] = "Ariane da Silva", "Karine Silva"
    odds = pd.DataFrame({"odds_date": pd.to_datetime(["2021-01-01"]), "odds_r": ["karine silva"],
                         "odds_b": ["ariane lipski"], "R_odds": [-300], "B_odds": [250],
                         "R_rank": np.nan, "B_rank": np.nan, "odds_source": ["dataset"]})
    out = attach_odds(fights, odds).iloc[0]
    assert (out["r_odds"], out["b_odds"]) == (250, -300)


def test_judge_names_lose_the_notes_before_them():
    master = make_master([
        {**raw_fight("f1", "2001-09-28", "ann", "bea", result="r"),
         "details": "Point Deducted: Illegal Knee by Bea Tony Weeks 29 - 27. Chris Lee 28 - 29. Mike Bell 28 - 29."},
        {**raw_fight("f2", "2002-01-01", "cat", "dia", result="r"),
         "details": "Tony Weeks 28 - 29. Chris Lee 28 - 29. Mike Bell 28 - 29."},
    ])
    assert master.loc[master["fight_id"] == "f1", "judge1_name"].item() == "Tony Weeks"


def test_ranked_names_follow_links_but_not_contradicting_ones():
    fights = make_master([raw_fight("f1", "2018-06-01", "dj", "hc", weight_class="Flyweight")])
    fights["r_name"], fights["b_name"] = "Demetrious Johnson", "Henry Cejudo"
    snapshot = pd.DataFrame({"date": pd.to_datetime(["2018-08-05"] * 3), "division": "Flyweight",
                             "fighter": ["Henry Cejudo", "Demetrious Johnson", "Stephen Johnson"],
                             "page": ["Demetrious Johnson", "Demetrious Johnson", None]})
    ids = match_ranked_fighters(snapshot, fights, pages={"Demetrious Johnson": "dj"})
    assert list(ids) == ["hc", "dj", None]   # the wrong link on Cejudo's row is ignored


def test_short_first_names_are_matched_on_the_last_name():
    fights = make_master([raw_fight("f1", "2023-01-01", "se", "ac", weight_class="Flyweight")])
    fights["r_name"], fights["b_name"] = "Steve Erceg", "Alessandro Costa"
    snapshot = pd.DataFrame({"date": pd.to_datetime(["2023-11-01"]), "division": "Flyweight",
                             "fighter": ["Stephen Erceg"], "page": [None]})
    assert list(match_ranked_fighters(snapshot, fights)) == ["se"]
