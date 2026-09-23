"""bestfightodds.com parsers, on a short page in the site's layout (no network access)."""

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from ufc_rating.ingest import bestfightodds

EVENT_PAGE = """
<table class="odds-table odds-table-responsive-header"></table>
<table class="odds-table">
  <thead><tr><th></th><th>Polymarket $20 Bonus</th><th>FanDuel</th><th>Caesars</th><th>BetMGM</th><th>Props</th></tr></thead>
  <tbody>
    <tr><th>41067 Alexandre Pantoja</th><td>+2236 &#9660;</td><td>+130 &#9650;</td><td>+122</td><td>+125</td><td>88</td></tr>
    <tr><th>Joshua van</th><td>-3487</td><td>-154</td><td>-145</td><td>-157</td><td>88</td></tr>
    <tr class="pr"><th>Over 1&#189; rounds</th><td></td><td>-375</td><td></td><td></td><td></td></tr>
    <tr class="pr"><th>Under 1&#189; rounds</th><td></td><td>+240</td><td></td><td></td><td></td></tr>
    <tr><th>Arman Tsarukyan</th><td></td><td>-315</td><td></td><td></td><td></td></tr>
    <tr><th>Mauricio Ruffy</th><td></td><td>+235</td><td></td><td></td><td></td></tr>
  </tbody>
</table>
"""

FIGHTER_PAGE = """
<table class="team-stats-table">
  <tr><th>Matchup</th><th>Open</th><th>Closing range</th><th>Event</th></tr>
  <tr><td><a href="/events/ufc-331-4302">UFC 331</a> Sep 19th 2026</td></tr>
  <tr><td>Alexandre Pantoja</td><td>-185</td><td>+122</td><td><a href="/events/ufc-331-4302">UFC 331</a></td></tr>
  <tr><td><a href="/events/ufc-3895">UFC</a> Dec 7th 2025</td></tr>
</table>
"""


def test_event_odds_skip_exchanges_and_props():
    odds = bestfightodds.parse_event_odds(BeautifulSoup(EVENT_PAGE, "html.parser"))
    assert list(odds["fighter_1"]) == ["Alexandre Pantoja", "Arman Tsarukyan"]
    assert list(odds["fighter_2"]) == ["Joshua van", "Mauricio Ruffy"]
    # median of the three sportsbooks (FanDuel, Caesars, BetMGM), Polymarket left out
    assert list(odds["n_books"]) == [3, 1]
    assert odds.loc[0, "odds_1"] == 125 and odds.loc[0, "odds_2"] == -154
    assert (odds.loc[1, "odds_1"], odds.loc[1, "odds_2"]) == (-315, 235)


def test_american_odds_round_trip():
    from ufc_rating.processing.master import american_to_prob
    for odds in (-315, -154, 125, 235):
        assert bestfightodds._prob_to_american(american_to_prob([odds])[0]) == pytest.approx(odds, abs=1)


def test_fighter_page_event_links():
    links = bestfightodds._event_links(BeautifulSoup(FIGHTER_PAGE, "html.parser"))
    assert links[0] == ("/events/ufc-331-4302", pd.Timestamp("2026-09-19"))
    assert links[-1] == ("/events/ufc-3895", pd.Timestamp("2025-12-07"))
