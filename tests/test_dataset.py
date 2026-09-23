import pandas as pd

from ufc_rating.dataset import dataset_card, export, fighters_table, fights_table
from ufc_rating.ranking.elo import compute_elo
from conftest import make_master, raw_fight


def small_master():
    return make_master([
        {**raw_fight("f1", "2020-01-01", "ann", "bea", result="r"),
         "details": "A B 28 - 29. C D 28 - 29. E F 29 - 28.", "bonuses": "Fight of the Night"},
        raw_fight("f2", "2020-06-01", "bea", "cat", result="draw"),
        raw_fight("f3", "2021-01-01", "ann", "cat", result="b", method="KO/TKO",
                  finish_round=1, finish_time="2:00"),
    ])


def test_fights_table():
    fights = fights_table(small_master()).set_index("fight_id")
    assert fights["winner_id"].tolist()[::2] == ["ann", "cat"] and pd.isna(fights.loc["f2", "winner_id"])
    assert (fights.loc["f1", "judge1_r_score"], fights.loc["f1", "judge1_b_score"]) == (29, 28)
    assert fights.loc["f3", "fight_seconds"] == 120
    assert fights.loc["f1", "bonuses"] == "Fight of the Night"


def test_fighters_table_records_and_elo():
    master = small_master()
    fighters = fighters_table(master, compute_elo(master), records=None).set_index("fighter_id")
    assert fighters.loc["ann", ["ufc_wins", "ufc_losses", "ufc_draws"]].tolist() == [1, 1, 0]
    assert fighters.loc["bea", ["ufc_wins", "ufc_losses", "ufc_draws"]].tolist() == [0, 1, 1]
    assert str(fighters.loc["ann", "first_ufc_fight"]) == "2020-01-01"
    assert fighters.loc["cat", "elo"] > 1500 > fighters.loc["bea", "elo"]


def test_export_writes_every_table_and_a_card(tmp_path):
    master = small_master()
    rounds = pd.DataFrame({"fight_id": ["f1"], "date": [pd.Timestamp("2020-01-01")], "round": [1],
                           "seconds": [300.0], "r_id": ["ann"], "b_id": ["bea"]})
    from ufc_rating.dataset import ROUND_STATS
    for side in ("r", "b"):
        for stat in ROUND_STATS:
            rounds[f"{side}_{stat}"] = 1
    tables = export(master, rounds, compute_elo(master), out_dir=tmp_path,
                    rankings_csv=tmp_path / "none.csv", records_csv=tmp_path / "none.csv")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "README.md", "elo.csv", "fighters.csv", "fights.csv", "rounds.csv"]
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "| `fights.csv` | 3 |" in card
    assert card == dataset_card(tables)


def test_final_elo_on_a_tournament_night():
    # two fights of ann on the same night: her final rating is the one after the second
    master = make_master([raw_fight("f2", "1994-03-11", "ann", "bea", result="r"),
                          raw_fight("f1", "1994-03-11", "cat", "ann", result="b")])
    history = compute_elo(master)
    last = history[history["fighter_id"] == "ann"]["elo_after"].iloc[-1]
    shuffled = history.sample(frac=1, random_state=0)
    fighters = fighters_table(master, shuffled, records=None).set_index("fighter_id")
    assert fighters.loc["ann", "elo"] == round(last, 1)
