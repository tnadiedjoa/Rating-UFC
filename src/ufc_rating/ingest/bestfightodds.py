"""
Closing betting odds from bestfightodds.com, for the UFC events that the
odds dataset of the Kaggle snapshot leaves incomplete: it has gaps in 2023
and 2024 and ends in March 2026.

bestfightodds.com compares the lines of many sportsbooks. An event page
shows, for every fight, the last odds each book offered before the fight.
The consensus kept here is the median over the sportsbooks; betting
exchanges and prediction markets (Polymarket, Kalshi...) are left out: their
prices keep trading during and after the fight.

To find an event, the fighter search is used: the page of one of the
event's fighters links to the event of that date.
"""

import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ufc_rating.config import BFO_ODDS_CSV
from ufc_rating.processing.master import american_to_prob, normalize_name

BASE_URL = "https://www.bestfightodds.com"
USER_AGENT = "Rating-UFC/2.0 (https://github.com/nadiedjoa-24/Rating-UFC)"
FIRST_EVENT = date(2023, 1, 1)   # before 2023 the odds dataset of the Kaggle snapshot is nearly complete
EXCHANGES = ("polymarket", "kalshi", "prophetx", "novig", "sporttrade", "betfair")


def _get(session: requests.Session, path: str, pause: float, **params) -> BeautifulSoup:
    time.sleep(pause)
    for attempt in range(3):
        try:
            response = session.get(BASE_URL + path, params=params or None,
                                   headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    raise ConnectionError(f"bestfightodds.com unreachable ({path})")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _american(text: str) -> float:
    match = re.match(r"^\s*([+-]\d+)", text or "")
    return float(match.group(1)) if match else np.nan


def _prob_to_american(p: float) -> float:
    if not 0 < p < 1:
        return np.nan
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def parse_event_odds(soup: BeautifulSoup) -> pd.DataFrame:
    """
    One row per fight of an event page: the two fighters as listed, the
    median sportsbook odds of each (American) and the number of books.
    """
    tables = soup.select("table.odds-table")
    if len(tables) < 2:
        return pd.DataFrame(columns=["fighter_1", "fighter_2", "odds_1", "odds_2", "n_books"])
    table = tables[1]
    books = [th.get_text(" ", strip=True).lower() for th in table.select("thead th")]
    keep = [i for i, book in enumerate(books)
            if i > 0 and book and book != "props" and not any(x in book for x in EXCHANGES)]
    rows = [tr for tr in table.select("tbody tr") if "pr" not in (tr.get("class") or [])]
    fights = []
    for first, second in zip(rows[0::2], rows[1::2]):
        cells = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in (first, second)]
        names = [re.sub(r"^\d+\s+", "", c[0]).strip() for c in cells]
        odds = [[_american(c[i]) if i < len(c) else np.nan for i in keep] for c in cells]
        both = [(a, b) for a, b in zip(*odds) if not (np.isnan(a) or np.isnan(b))]
        if not both:
            continue
        # median of each side's implied probability, over the books that priced both sides
        p1 = float(np.median(american_to_prob([a for a, _ in both])))
        p2 = float(np.median(american_to_prob([b for _, b in both])))
        fights.append({"fighter_1": names[0], "fighter_2": names[1],
                       "odds_1": _prob_to_american(p1), "odds_2": _prob_to_american(p2),
                       "n_books": len(both)})
    return pd.DataFrame(fights, columns=["fighter_1", "fighter_2", "odds_1", "odds_2", "n_books"])


def _event_links(soup: BeautifulSoup) -> list:
    """(event url, date) of every fight listed on a fighter page."""
    links = []
    for row in soup.select("table.team-stats-table tr"):
        link = row.find("a", href=lambda h: h and "/events/" in h)
        when = re.search(r"([A-Z][a-z]{2}) (\d{1,2})(?:st|nd|rd|th) (\d{4})", row.get_text(" ", strip=True))
        if link and when:
            day = pd.to_datetime(f"{when.group(1)} {when.group(2)} {when.group(3)}", format="%b %d %Y")
            links.append((link["href"], day))
    return links


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def find_event(session: requests.Session, fighter: str, day: pd.Timestamp, pause: float = 1.0) -> Optional[str]:
    """URL of the event on ``day`` (one day either way) on the page of ``fighter``."""
    results = _get(session, "/search", pause, query=fighter)
    target = normalize_name(fighter)
    for link in results.select("a[href*='/fighters/']"):
        if normalize_name(link.get_text(strip=True)) == target:
            for url, when in _event_links(_get(session, link["href"], pause)):
                if abs((when - day).days) <= 1:
                    return url
            return None
    return None


def scrape_odds(master: pd.DataFrame, since: date = FIRST_EVENT, out_path: Path = BFO_ODDS_CSV,
                pause: float = 1.0) -> pd.DataFrame:
    """
    Odds of the UFC events of ``master`` from ``since`` on that have at least
    one fight without odds and are not in ``out_path`` yet (an event is read
    once). Appends to the file after each event.
    """
    session = requests.Session()
    known = pd.read_csv(out_path, parse_dates=["date"]) if Path(out_path).exists() else pd.DataFrame()
    done = set(known["event_id"]) if not known.empty else set()
    missing = master.get("r_odds", pd.Series(np.nan, index=master.index)).isna()
    incomplete = master.loc[missing, "event_id"].unique()
    events = (master[(master["date"] >= pd.Timestamp(since)) & master["event_id"].isin(incomplete)
                     & ~master["event_id"].isin(done)]
              .sort_values(["date", "fight_id"]).groupby("event_id", sort=False))
    frames = [known]
    for event_id, fights in events:
        day = fights["date"].iloc[0]
        url = None
        for name in pd.unique(fights[["r_name", "b_name"]].to_numpy().ravel())[:6]:
            url = find_event(session, name, day, pause)
            if url:
                break
        if url is None:
            print(f"  {fights['event_name'].iloc[0]} ({day.date()}): not found")
            continue
        odds = parse_event_odds(_get(session, url, pause))
        odds.insert(0, "event_id", event_id)
        odds.insert(1, "date", day)
        odds["url"] = url
        frames.append(odds)
        print(f"  {fights['event_name'].iloc[0]} ({day.date()}): {len(odds)} fights with odds")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    return pd.concat(frames, ignore_index=True)
