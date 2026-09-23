import pytest

from ufc_rating.ranking.elo import compute_elo, expected_score, peak_elo
from conftest import make_master, raw_fight


def test_expected_score():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == pytest.approx(1.0)
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)


def elo_after(rows):
    history = compute_elo(make_master(rows))
    return history.groupby("fighter_id")["elo_after"].last().to_dict()


def test_win_draw_and_no_contest():
    win = elo_after([raw_fight("f1", "2020-01-01", "ann", "bea", result="r")])
    assert win == {"ann": 1516, "bea": 1484}

    draw = elo_after([raw_fight("f1", "2020-01-01", "ann", "bea", result="draw")])
    assert draw == {"ann": 1500, "bea": 1500}

    nc = elo_after([raw_fight("f1", "2020-01-01", "ann", "bea", result="r"),
                    raw_fight("f2", "2020-02-01", "ann", "bea", result="nc")])
    assert nc == {"ann": 1516, "bea": 1484}


def test_rating_points_are_conserved():
    rows = [raw_fight(f"f{i}", f"2020-0{i}-01", a, b, result=res)
            for i, (a, b, res) in enumerate([("ann", "bea", "r"), ("bea", "cat", "b"),
                                             ("cat", "ann", "draw"), ("ann", "bea", "b")], start=1)]
    ratings = elo_after(rows)
    assert sum(ratings.values()) == pytest.approx(1500 * len(ratings))


def test_peak_table():
    history = compute_elo(make_master([raw_fight("f1", "2020-01-01", "ann", "bea", result="r")]))
    peaks = peak_elo(history, top=2)
    assert list(peaks["Fighter"]) == ["Ann", "Bea"]
    assert peaks.loc[1, "Peak Elo"] == 1516
