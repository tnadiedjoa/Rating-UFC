"""
Scraper tests on real ufcstats.com pages archived by the Wayback Machine
(tests/fixtures). No network access.
"""

from datetime import date

import pytest
from bs4 import BeautifulSoup

from ufc_rating.ingest import ufcstats
from conftest import FIXTURES


def soup(name):
    return BeautifulSoup((FIXTURES / name).read_bytes(), "html.parser")


def test_events_list_skips_upcoming_event_and_stops_at_cutoff():
    events = ufcstats.parse_events_list(soup("events_completed.html"),
                                        since=date(2026, 9, 1), today=date(2026, 9, 23))
    # the page opens with the upcoming 26 September card, which must be skipped
    assert [d for _, d in events] == [date(2026, 9, 19), date(2026, 9, 12), date(2026, 9, 5)]
    assert events[0][0].endswith("8a0a35e7c74bebcc")


def test_event_page():
    event = ufcstats.parse_event(soup("event_nurmagomedov_song.html"))
    assert event["event_name"] == "UFC Fight Night: Nurmagomedov vs. Song"
    assert event["event_date"] == "2026-08-29"
    assert event["event_location"] == "Shanghai, China"
    assert len(event["fight_urls"]) == 13


def test_fight_page_reads_fight_totals_not_round_one():
    """Regression test: the first scraper read the per-round tables (round 1 only)."""
    row = ufcstats.parse_fight(soup("fight_holm_aldana.html"),
                               "http://ufcstats.com/fight-details/0005e00b07cee542", {})
    assert row["r_fighter_name"] == "Holly Holm"
    assert row["r_total_sig_landed"] == 154 and row["r_total_sig_atmp"] == 301  # round 1: 21 of 50
    assert row["b_total_sig_landed"] == 69 and row["b_total_sig_atmp"] == 185
    assert row["r_total_total_str_landed"] == 187
    assert row["r_total_td_success"] == 5 and row["r_total_td_atmp"] == 14
    assert row["r_total_ctrl_seconds"] == 316
    assert row["r_total_sig_str_landed_head"] == 81 and row["r_total_sig_str_landed_body"] == 56
    assert row["r_total_sig_str_landed_ground"] == 13


def test_fight_page_metadata():
    row = ufcstats.parse_fight(soup("fight_rosholt_copeland.html"),
                               "http://ufcstats.com/fight-details/0027e179b743c86c", {})
    assert row["fight_id"] == "0027e179b743c86c"
    assert row["method"] == "KO/TKO"          # was never captured by the first scraper
    assert row["finish_round"] == 3 and row["finish_time"] == "3:12"
    assert row["time_format"] == "3 Rnd (5-5-5)"
    assert row["weight_class"] == "Heavyweight"
    assert row["result_status"] == "win"
    assert row["winner_id"] == row["r_fighter_id"] == "91ea901c458e95dd"


def test_fighter_page():
    assert ufcstats.parse_fighter(soup("fighter_holly_holm.html")) == {
        "height": "5' 8\"", "reach_inches": 69.0, "stance": "Southpaw", "dob": "1981-10-17",
    }


def test_bot_challenge_is_detected():
    assert ufcstats.is_bot_challenge(soup("bot_challenge.html"))
    assert not ufcstats.is_bot_challenge(soup("fight_holm_aldana.html"))


def test_fetch_raises_on_bot_challenge(monkeypatch):
    class FakeResponse:
        content = (FIXTURES / "bot_challenge.html").read_bytes()

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(ufcstats.time, "sleep", lambda s: None)
    with pytest.raises(ufcstats.BotChallengeError):
        ufcstats.fetch("http://ufcstats.com/statistics/events/completed", FakeSession())


def test_events_dated_today_are_skipped():
    events = ufcstats.parse_events_list(soup("events_completed.html"),
                                        since=date(2026, 9, 1), today=date(2026, 9, 19))
    assert [d for _, d in events] == [date(2026, 9, 12), date(2026, 9, 5)]
