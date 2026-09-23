"""
Incremental scraper for ufcstats.com.

It collects the events completed after a given date and writes one row per
fight in the same column layout as the Kaggle mirror (``data/raw/ufcstats``),
so both sources can be concatenated and deduplicated on ``fight_id``.

Since 2026, ufcstats.com answers automated clients with a JavaScript
"Checking your browser" challenge. The scraper does not try to get around
it: it raises ``BotChallengeError`` and the pipeline carries on with the
Kaggle mirror alone.

The parsers work on BeautifulSoup objects and are tested against archived
pages in ``tests/fixtures``.
"""

import random
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ufc_rating.config import SCRAPED_CSV

BASE_URL = "http://ufcstats.com"
EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"
USER_AGENT = "Rating-UFC research scraper (+https://github.com/nadiedjoa-24/Rating-UFC)"

# Columns of the "Totals" table, in page order after the fighter column.
_TOTALS_FIELDS = ["kd", "sig", "sig_pct", "total_str", "td", "td_pct", "sub_att", "rev", "ctrl"]
# Columns of the "Significant Strikes" table, in page order after the fighter column.
_SIG_FIELDS = ["sig", "sig_pct", "head", "body", "leg", "distance", "clinch", "ground"]


class BotChallengeError(RuntimeError):
    """ufcstats.com served its anti-bot page instead of the requested content."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def fetch(url: str, session: Optional[requests.Session] = None, retries: int = 3) -> BeautifulSoup:
    """GET a page politely (1 to 2.5 s between requests) and parse it."""
    http = session or requests
    last_error = None
    for attempt in range(retries):
        time.sleep(random.uniform(1.0, 2.5))
        try:
            response = http.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(5 * (attempt + 1))
            continue
        soup = BeautifulSoup(response.content, "html.parser")
        if is_bot_challenge(soup):
            raise BotChallengeError(f"ufcstats.com served a bot challenge for {url}")
        return soup
    raise ConnectionError(f"Giving up on {url} after {retries} attempts: {last_error}")


def is_bot_challenge(soup: BeautifulSoup) -> bool:
    title = soup.title.get_text(strip=True) if soup.title else ""
    return title.startswith("Loading") and "Checking your browser" in soup.get_text()


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _text(node) -> str:
    return node.get_text(" ", strip=True) if node is not None else ""


def _id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _parse_date(text: str) -> Optional[date]:
    """'August 29, 2026' (events) or 'Oct 17, 1981' (fighter pages)."""
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _landed_attempted(text: str):
    """'154 of 301' -> (154, 301). Missing values ('---') give (None, None)."""
    match = re.match(r"(\d+)\s+of\s+(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def _to_int(text: str):
    return int(text) if text.strip().isdigit() else None


def _mmss_to_seconds(text: str):
    match = re.match(r"(\d+):(\d{2})$", text.strip())
    return int(match.group(1)) * 60 + int(match.group(2)) if match else None


def _split_cell(td):
    """Each stats cell holds two <p> tags: first-listed fighter, then second."""
    values = [_text(p) for p in td.find_all("p")]
    return (values + ["", ""])[:2]


# ---------------------------------------------------------------------------
# Events list and event pages
# ---------------------------------------------------------------------------

def parse_events_list(soup: BeautifulSoup, since: Optional[date] = None,
                      today: Optional[date] = None) -> list:
    """
    Return [(event_url, event_date), ...] for completed events after ``since``.

    The list is sorted newest first and opens with the next upcoming event.
    Events dated today or later are skipped: a card still in progress has
    fights without results.
    """
    today = today or date.today()
    events = []
    for link in soup.select('a[href*="event-details"]'):
        date_span = link.find_next_sibling("span", class_="b-statistics__date")
        event_date = _parse_date(_text(date_span))
        if event_date is None or event_date >= today:
            continue
        if since is not None and event_date <= since:
            break
        events.append((link["href"].strip(), event_date))
    return events


def parse_event(soup: BeautifulSoup) -> dict:
    """Event name, date, location and the links of its fights."""
    info = {"event_name": _text(soup.select_one("h2.b-content__title"))}
    for item in soup.select("li.b-list__box-list-item"):
        label, _, value = _text(item).partition(":")
        if label.strip() == "Date":
            parsed = _parse_date(value)
            info["event_date"] = parsed.isoformat() if parsed else None
        elif label.strip() == "Location":
            info["event_location"] = value.strip()
    info["fight_urls"] = [
        row["data-link"].strip()
        for row in soup.select("tr.b-fight-details__table-row[data-link]")
        if "fight-details" in row["data-link"]
    ]
    return info


# ---------------------------------------------------------------------------
# Fight pages
# ---------------------------------------------------------------------------

def _stats_table(soup: BeautifulSoup, first_header: str):
    """
    Return the fight-total table whose second header is ``first_header``.

    A fight page has four tables: "Totals", its per-round breakdown,
    "Significant Strikes" and its per-round breakdown. The per-round tables
    carry the ``js-fight-table`` class; the fight totals do not. Picking the
    per-round tables by mistake silently yields round-1 numbers.
    """
    for table in soup.find_all("table"):
        if "js-fight-table" in (table.get("class") or []):
            continue
        headers = [_text(th) for th in table.find_all("th")]
        if len(headers) > 1 and headers[1].lower().startswith(first_header.lower()):
            return table
    return None


def _table_values(table, fields):
    row = table.find("tbody").find("tr")
    cells = row.find_all("td")[1:]
    return {field: _split_cell(cell) for field, cell in zip(fields, cells)}


def parse_fight(soup: BeautifulSoup, fight_url: str, event: dict) -> dict:
    """One fight in the Kaggle-mirror layout (``r_`` = first-listed fighter)."""
    people = soup.select("div.b-fight-details__person")
    if len(people) != 2:
        raise ValueError(f"Expected two fighters on {fight_url}")

    row = {
        "fight_id": _id_from_url(fight_url),
        "event_id": _id_from_url(event.get("event_url", "")) or None,
        "event_name": event.get("event_name"),
        "event_date": event.get("event_date"),
        "event_location": event.get("event_location"),
    }

    statuses = []
    for prefix, person in zip(("r", "b"), people):
        link = person.select_one("h3.b-fight-details__person-name a")
        row[f"{prefix}_fighter_id"] = _id_from_url(link["href"]) if link else None
        row[f"{prefix}_fighter_name"] = _text(person.select_one("h3.b-fight-details__person-name"))
        statuses.append(_text(person.select_one("i.b-fight-details__person-status")))

    if "W" in statuses:
        row["result_status"] = "win"
        row["winner_id"] = row["r_fighter_id"] if statuses[0] == "W" else row["b_fighter_id"]
    else:
        row["result_status"] = "draw" if "D" in statuses else "no_contest"
        row["winner_id"] = None

    title = _text(soup.select_one("i.b-fight-details__fight-title"))
    row["weight_class"] = re.sub(r"\s*(Title\s+)?Bout$", "", title).removeprefix("UFC ").strip()
    row["title_fight"] = int("Title" in title)

    labels = {"Method": "method", "Round": "finish_round", "Time": "finish_time",
              "Time format": "time_format", "Referee": "referee"}
    items = soup.select("i.b-fight-details__text-item_first, i.b-fight-details__text-item")
    for item in items:
        label, _, value = _text(item).partition(":")
        if label.strip() in labels:
            row[labels[label.strip()]] = value.strip()
    row["finish_round"] = _to_int(str(row.get("finish_round", "")))

    totals = _stats_table(soup, "KD")
    strikes = _stats_table(soup, "Sig. str")
    if totals is None or strikes is None:
        raise ValueError(f"Fight totals missing on {fight_url}")
    t = _table_values(totals, _TOTALS_FIELDS)
    s = _table_values(strikes, _SIG_FIELDS)

    for i, prefix in enumerate(("r", "b")):
        row[f"{prefix}_total_kd"] = _to_int(t["kd"][i])
        row[f"{prefix}_total_sig_landed"], row[f"{prefix}_total_sig_atmp"] = _landed_attempted(t["sig"][i])
        (row[f"{prefix}_total_total_str_landed"],
         row[f"{prefix}_total_total_str_atmp"]) = _landed_attempted(t["total_str"][i])
        row[f"{prefix}_total_td_success"], row[f"{prefix}_total_td_atmp"] = _landed_attempted(t["td"][i])
        row[f"{prefix}_total_sub_att"] = _to_int(t["sub_att"][i])
        row[f"{prefix}_total_rev"] = _to_int(t["rev"][i])
        row[f"{prefix}_total_ctrl_seconds"] = _mmss_to_seconds(t["ctrl"][i])
        for zone in ("head", "body", "leg", "distance", "clinch", "ground"):
            (row[f"{prefix}_total_sig_str_landed_{zone}"],
             row[f"{prefix}_total_sig_str_atmp_{zone}"]) = _landed_attempted(s[zone][i])
    return row


# ---------------------------------------------------------------------------
# Fighter pages
# ---------------------------------------------------------------------------

def parse_fighter(soup: BeautifulSoup) -> dict:
    """Physical profile in the Kaggle-mirror formats (height as 5' 8", reach in inches)."""
    fields = {}
    for item in soup.select("ul.b-list__box-list li.b-list__box-list-item"):
        label, _, value = _text(item).partition(":")
        fields[label.strip().upper()] = value.strip()
    reach = re.match(r"(\d+(?:\.\d+)?)", fields.get("REACH", ""))
    dob = _parse_date(fields.get("DOB", ""))
    height = fields.get("HEIGHT", "")
    stance = fields.get("STANCE", "")
    return {
        "height": height if "'" in height else None,
        "reach_inches": float(reach.group(1)) if reach else None,
        "stance": stance or None,
        "dob": dob.isoformat() if dob else None,
    }


# ---------------------------------------------------------------------------
# Incremental scraping
# ---------------------------------------------------------------------------

def scrape_since(since: Optional[date], out_path: Path = SCRAPED_CSV) -> pd.DataFrame:
    """
    Scrape every fight of the events completed after ``since`` and append
    them to ``out_path`` (deduplicated on ``fight_id``). Each event is saved
    as soon as it is scraped; a fight page that cannot be parsed is reported
    and skipped.

    Raises BotChallengeError if ufcstats.com refuses automated access.
    Returns only the newly scraped fights.
    """
    session = requests.Session()
    events = parse_events_list(fetch(EVENTS_URL, session), since=since)
    print(f"  {len(events)} event(s) to scrape since {since}")

    fighter_cache = {}
    scraped = []
    for event_url, event_date in events:
        event = parse_event(fetch(event_url, session))
        event["event_url"] = event_url
        rows = []
        for fight_url in event["fight_urls"]:
            try:
                row = parse_fight(fetch(fight_url, session), fight_url, event)
            except ValueError as exc:
                print(f"    skipped {fight_url}: {exc}")
                continue
            for prefix in ("r", "b"):
                fighter_id = row[f"{prefix}_fighter_id"]
                if fighter_id not in fighter_cache:
                    fighter_url = f"{BASE_URL}/fighter-details/{fighter_id}"
                    fighter_cache[fighter_id] = parse_fighter(fetch(fighter_url, session))
                for key, value in fighter_cache[fighter_id].items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
        print(f"  {event['event_name']} ({event_date}): {len(rows)} fights")
        if rows:
            _append(pd.DataFrame(rows), out_path)
            scraped.extend(rows)
    return pd.DataFrame(scraped)


def _append(new: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        new = pd.concat([pd.read_csv(out_path), new], ignore_index=True)
    new.drop_duplicates("fight_id", keep="last").to_csv(out_path, index=False)
