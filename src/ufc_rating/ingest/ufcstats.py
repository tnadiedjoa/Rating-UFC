"""
Incremental scraper for ufcstats.com.

It collects the events completed after a given date and writes one row per
fight and one row per round, in the same column layouts as the Kaggle mirror
(``data/raw/ufcstats/master.csv`` and ``round.csv``), so both sources can be
concatenated and deduplicated on ``fight_id``.

Pages are fetched with plain HTTP requests and random pauses. ufcstats.com
only serves its content to clients that run its JavaScript page check, so
when a request gets that check instead of the page, a headless Chromium
(Playwright, optional dependency: ``pip install -e ".[scrape]"``) loads the
page once and its session cookie is reused for the following requests.

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

from ufc_rating.config import SCRAPED_CSV, SCRAPED_ROUNDS_CSV

BASE_URL = "http://ufcstats.com"
EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

# Columns of the "Totals" table, in page order after the fighter column.
_TOTALS_FIELDS = ["kd", "sig", "sig_pct", "total_str", "td", "td_pct", "sub_att", "rev", "ctrl"]
# Columns of the "Significant Strikes" table, in page order after the fighter column.
_SIG_FIELDS = ["sig", "sig_pct", "head", "body", "leg", "distance", "clinch", "ground"]
_ZONES = ("head", "body", "leg", "distance", "clinch", "ground")
# Bonus icons on event pages, spelled as in the mirror's ``bonuses`` column.
_BONUS_ICONS = {"fight.png": "Fight of the Night", "perf.png": "Performance of the Night",
                "ko.png": "Knockout of the Night", "sub.png": "Submission of the Night"}


class BotChallengeError(RuntimeError):
    """ufcstats.com served its JavaScript page check and no browser could pass it."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def is_bot_challenge(soup: BeautifulSoup) -> bool:
    title = soup.title.get_text(strip=True) if soup.title else ""
    return title.startswith("Loading") and "Checking your browser" in soup.get_text()


class Client:
    """
    HTTP client with random pauses between requests (1.5 to 4 s, and now and
    then a longer 10 to 30 s break).

    When a response is the site's JavaScript page check, ``browser_session``
    opens the page in headless Chromium, which runs the check, and copies the
    resulting cookies and user agent into the requests session.
    """

    def __init__(self, session: Optional[requests.Session] = None, use_browser: bool = True):
        self.session = session or requests.Session()
        self.use_browser = use_browser
        self.user_agent = USER_AGENT

    def pause(self) -> None:
        time.sleep(random.uniform(1.5, 4.0))
        if random.random() < 0.05:
            time.sleep(random.uniform(10.0, 30.0))

    def request(self, url: str, retries: int = 3) -> BeautifulSoup:
        last_error = None
        for attempt in range(retries):
            self.pause()
            try:
                response = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=20)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(5 * (attempt + 1))
        raise ConnectionError(f"Giving up on {url} after {retries} attempts: {last_error}")

    def browser_session(self, url: str) -> None:
        """Load ``url`` in headless Chromium and reuse its cookies."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise BotChallengeError(
                "ufcstats.com requires a JavaScript-capable client: "
                'pip install -e ".[scrape]" && python -m playwright install chromium'
            ) from None
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            for _ in range(30):                      # the check reloads the page when done
                if "Checking your browser" not in page.content():
                    break
                time.sleep(1)
            self.user_agent = page.evaluate("navigator.userAgent")
            for cookie in page.context.cookies():
                self.session.cookies.set(cookie["name"], cookie["value"],
                                         domain=cookie["domain"], path=cookie["path"])
            browser.close()

    def get(self, url: str) -> BeautifulSoup:
        soup = self.request(url)
        if is_bot_challenge(soup) and self.use_browser:
            self.browser_session(url)
            soup = self.request(url)
        if is_bot_challenge(soup):
            raise BotChallengeError(f"Could not get past the page check for {url}")
        return soup


def fetch(url: str, client: Optional[Client] = None) -> BeautifulSoup:
    """GET and parse one page (see Client)."""
    return (client or Client()).get(url)


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
    """Event name, date, location, the links of its fights and their bonuses."""
    info = {"event_name": _text(soup.select_one("h2.b-content__title"))}
    for item in soup.select("li.b-list__box-list-item"):
        label, _, value = _text(item).partition(":")
        if label.strip() == "Date":
            parsed = _parse_date(value)
            info["event_date"] = parsed.isoformat() if parsed else None
        elif label.strip() == "Location":
            info["event_location"] = value.strip()
    info["fight_urls"], info["bonuses"] = [], {}
    for row in soup.select("tr.b-fight-details__table-row[data-link]"):
        url = row["data-link"].strip()
        if "fight-details" not in url:
            continue
        info["fight_urls"].append(url)
        icons = [img.get("src", "").rsplit("/", 1)[-1] for img in row.find_all("img")]
        bonuses = sorted(_BONUS_ICONS[i] for i in icons if i in _BONUS_ICONS)
        info["bonuses"][url] = ", ".join(bonuses) or None
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


def _row_values(row, fields):
    cells = row.find_all("td")[1:]
    return {field: _split_cell(cell) for field, cell in zip(fields, cells)}


def _table_values(table, fields):
    return _row_values(table.find("tbody").find("tr"), fields)


def _round_tables(soup: BeautifulSoup):
    """The per-round "Totals" and "Significant Strikes" tables (``js-fight-table``)."""
    by_header = {}
    for table in soup.find_all("table"):
        if "js-fight-table" not in (table.get("class") or []):
            continue
        headers = [_text(th) for th in table.find_all("th")]
        if len(headers) > 1:
            by_header["totals" if headers[1] == "KD" else "strikes"] = table
    return by_header.get("totals"), by_header.get("strikes")


def parse_rounds(soup: BeautifulSoup, fight: dict) -> list:
    """
    One row per round in the layout of the mirror's ``round.csv``
    (``r_`` = first-listed fighter, control time as 'm:ss').
    Fights without per-round tables (early UFC events) give [].
    """
    totals, strikes = _round_tables(soup)
    if totals is None or strikes is None:
        return []
    total_rows = [tr for tr in totals.find_all("tr") if tr.find("td")]
    strike_rows = [tr for tr in strikes.find_all("tr") if tr.find("td")]
    rounds = []
    for number, (t_row, s_row) in enumerate(zip(total_rows, strike_rows), start=1):
        t = _row_values(t_row, _TOTALS_FIELDS)
        s = _row_values(s_row, _SIG_FIELDS)
        row = {"fight_id": fight["fight_id"], "round_no": number,
               "r_id": fight["r_fighter_id"], "b_id": fight["b_fighter_id"]}
        for i, prefix in enumerate(("r", "b")):
            row[f"{prefix}_kd"] = _to_int(t["kd"][i])
        for i, prefix in enumerate(("r", "b")):
            row[f"{prefix}_sig_landed"], row[f"{prefix}_sig_atmp"] = _landed_attempted(t["sig"][i])
        for i, prefix in enumerate(("r", "b")):
            (row[f"{prefix}_total_str_landed"],
             row[f"{prefix}_total_str_atmp"]) = _landed_attempted(t["total_str"][i])
        for i, prefix in enumerate(("r", "b")):
            row[f"{prefix}_td_success"], row[f"{prefix}_td_atmp"] = _landed_attempted(t["td"][i])
        for field in ("sub_att", "rev"):
            for i, prefix in enumerate(("r", "b")):
                row[f"{prefix}_{field}"] = _to_int(t[field][i])
        for i, prefix in enumerate(("r", "b")):
            ctrl = t["ctrl"][i]
            row[f"{prefix}_ctrl"] = ctrl if _mmss_to_seconds(ctrl) is not None else None
        for zone in _ZONES:
            for i, prefix in enumerate(("r", "b")):
                (row[f"{prefix}_sig_str_landed_{zone}"],
                 row[f"{prefix}_sig_str_atmp_{zone}"]) = _landed_attempted(s[zone][i])
        rounds.append(row)
    return rounds


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
    # The "Details" paragraph: the finishing technique, or the three judges' scores
    for paragraph in soup.select("p.b-fight-details__text"):
        label, _, value = _text(paragraph).partition(":")
        if label.strip() == "Details":
            row["details"] = value.strip() or None
    row["bonuses"] = event.get("bonuses", {}).get(fight_url)

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
        for zone in _ZONES:
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

def scrape_since(since: Optional[date], out_path: Path = SCRAPED_CSV,
                 rounds_path: Path = SCRAPED_ROUNDS_CSV) -> pd.DataFrame:
    """
    Scrape every fight of the events completed after ``since`` and append
    them to ``out_path``, and their rounds to ``rounds_path`` (deduplicated
    on ``fight_id``). Each event is saved as soon as it is scraped; a fight
    page that cannot be parsed is reported and skipped.

    Raises BotChallengeError if the site's page check cannot be passed.
    Returns only the newly scraped fights.
    """
    client = Client()
    events = parse_events_list(fetch(EVENTS_URL, client), since=since)
    print(f"  {len(events)} event(s) to scrape since {since}")

    fighter_cache = {}
    scraped = []
    for event_url, event_date in events:
        event = parse_event(fetch(event_url, client))
        event["event_url"] = event_url
        rows, rounds = [], []
        for fight_url in event["fight_urls"]:
            try:
                page = fetch(fight_url, client)
                row = parse_fight(page, fight_url, event)
            except ValueError as exc:
                print(f"    skipped {fight_url}: {exc}")
                continue
            rounds.extend(parse_rounds(page, row))
            for prefix in ("r", "b"):
                fighter_id = row[f"{prefix}_fighter_id"]
                if fighter_id not in fighter_cache:
                    fighter_url = f"{BASE_URL}/fighter-details/{fighter_id}"
                    fighter_cache[fighter_id] = parse_fighter(fetch(fighter_url, client))
                for key, value in fighter_cache[fighter_id].items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
        print(f"  {event['event_name']} ({event_date}): {len(rows)} fights")
        if rows:
            _append(pd.DataFrame(rows), out_path, ["fight_id"])
            _append(pd.DataFrame(rounds), rounds_path, ["fight_id", "round_no"])
            scraped.extend(rows)
    return pd.DataFrame(scraped)


def _append(new: pd.DataFrame, out_path: Path, key: list) -> None:
    if new.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        new = pd.concat([pd.read_csv(out_path), new], ignore_index=True)
    new.drop_duplicates(key, keep="last").to_csv(out_path, index=False)
