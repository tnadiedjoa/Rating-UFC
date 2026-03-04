"""
ingest_data.py — Incremental UFC Stats Scraper

Main functions:
  scrape_since(last_date)  -> scrapes only events after last_date
                              last_date=None -> scrapes everything (first run)

The event list on ufcstats.com is ordered from most recent to oldest.
As soon as we encounter an event with a date <= last_date, we stop.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import random
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
RAW_DIR   = os.path.join(_BASE_DIR, "data", "raw")


# ─────────────────────────────────────────────
# HTTP Helpers
# ─────────────────────────────────────────────

def get_soup(url, retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/91.0.4472.124 Safari/537.36"
    }
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(1.0, 2.5))
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            return BeautifulSoup(r.content, "html.parser")
        except requests.exceptions.RequestException:
            print(f"    Attempt {attempt+1}/{retries} failed: {url}")
            time.sleep(5)
    print(f"Giving up after {retries} attempts: {url}")
    return None

def clean_text(text):
    return text.strip() if text else ""


# ─────────────────────────────────────────────
# Parsing the events list
# ─────────────────────────────────────────────

def parse_events_list(last_date=None):
    """
    Retrieves links and dates of events from the main page.
    If last_date is provided (datetime.date), stops as soon as an event is <= last_date.
    Returns a list of (event_url, event_date).
    """
    url = "http://ufcstats.com/statistics/events/completed?page=all"
    soup = get_soup(url)
    if not soup:
        return []

    events = []
    rows = soup.find_all("tr", class_="b-statistics__table-row")

    for row in rows:
        link = row.find("a", href=True)
        if not link or "event-details" not in link["href"]:
            continue

        event_url = link["href"]

        # The date is in the first td, inside a span
        tds = row.find_all("td")
        event_date = None
        if len(tds) >= 1:
            date_span = tds[0].find("span", class_="b-statistics__date")
            if date_span:
                date_str = clean_text(date_span.text)
                try:
                    event_date = datetime.strptime(date_str, "%B %d, %Y").date()
                except ValueError:
                    pass

        # Stop once we've passed last_date
        if last_date and event_date and event_date <= last_date:
            break

        events.append((event_url, event_date))

    return events


# ─────────────────────────────────────────────
# Parsing a single event
# ─────────────────────────────────────────────

def parse_event_metadata(soup_event):
    meta = {}
    for item in soup_event.find_all("li", class_="b-list__box-list-item"):
        text = clean_text(item.text)
        if text.startswith("Date:"):
            meta["event_date"] = text.replace("Date:", "").strip()
        elif text.startswith("Location:"):
            meta["event_location"] = text.replace("Location:", "").strip()
    return meta

def parse_two_values(col, default="0"):
    cells = col.find_all("p")
    r = clean_text(cells[0].text) if len(cells) > 0 else default
    b = clean_text(cells[1].text) if len(cells) > 1 else default
    return r, b

def parse_fight_details(fight_url, event_meta=None):
    """
    Scrapes all available stats for a fight.
    Tables:
      [0] totals    : KD, SIG_STR, TOTAL_STR, TD, SUB_ATT, REV, CTRL
      [1] by zone   : HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND
    Metadata: method, round, time, format, referee, fight_type, date, location
    """
    soup = get_soup(fight_url)
    if not soup:
        return None

    fighters = soup.find_all("h3", class_="b-fight-details__person-name")
    if len(fighters) < 2:
        return None

    status_divs = soup.find_all("i", class_="b-fight-details__person-status")
    data = {
        "R_Fighter": clean_text(fighters[0].text),
        "B_Fighter": clean_text(fighters[1].text),
        "R_Status":  clean_text(status_divs[0].text) if len(status_divs) > 0 else "",
        "B_Status":  clean_text(status_divs[1].text) if len(status_divs) > 1 else "",
        "URL": fight_url,
    }

    # Fight metadata
    label_map = {
        "Method": "method", "Round": "last_round",
        "Time": "last_round_time", "Time format": "format", "Referee": "referee",
    }
    for item in soup.find_all("i", class_="b-fight-details__text-item"):
        label_tag = item.find("i", class_="b-fight-details__label")
        if not label_tag:
            continue
        label = clean_text(label_tag.text).rstrip(":")
        value = clean_text(item.text.replace(label_tag.text, "").strip())
        if label in label_map:
            data[label_map[label]] = value

    ft = soup.find("i", class_="b-fight-details__fight-title")
    data["fight_type"] = clean_text(ft.text) if ft else ""

    if event_meta:
        data["date"]     = event_meta.get("event_date", "")
        data["location"] = event_meta.get("event_location", "")

    tables = soup.find_all("table", class_="b-fight-details__table")

    # Table 0: total stats
    if len(tables) > 0:
        rows = tables[0].find_all("tr", class_="b-fight-details__table-row")
        if len(rows) >= 2:
            cols = rows[1].find_all("td")
            if len(cols) >= 10:
                data["R_KD"],        data["B_KD"]        = parse_two_values(cols[1], "0")
                data["R_SIG_STR"],   data["B_SIG_STR"]   = parse_two_values(cols[2], "0 of 0")
                data["R_SIG_STR_pct"], data["B_SIG_STR_pct"] = parse_two_values(cols[3], "0%")
                data["R_TOTAL_STR"], data["B_TOTAL_STR"] = parse_two_values(cols[4], "0 of 0")
                data["R_TD"],        data["B_TD"]         = parse_two_values(cols[5], "0 of 0")
                data["R_TD_pct"],    data["B_TD_pct"]     = parse_two_values(cols[6], "0%")
                data["R_SUB_ATT"],   data["B_SUB_ATT"]   = parse_two_values(cols[7], "0")
                data["R_REV"],       data["B_REV"]        = parse_two_values(cols[8], "0")
                data["R_CTRL"],      data["B_CTRL"]       = parse_two_values(cols[9], "0:00")

    # Table 1: strikes by zone
    if len(tables) > 1:
        rows = tables[1].find_all("tr", class_="b-fight-details__table-row")
        if len(rows) >= 2:
            cols = rows[1].find_all("td")
            if len(cols) >= 9:
                data["R_HEAD"],     data["B_HEAD"]     = parse_two_values(cols[3], "0 of 0")
                data["R_BODY"],     data["B_BODY"]     = parse_two_values(cols[4], "0 of 0")
                data["R_LEG"],      data["B_LEG"]      = parse_two_values(cols[5], "0 of 0")
                data["R_DISTANCE"], data["B_DISTANCE"] = parse_two_values(cols[6], "0 of 0")
                data["R_CLINCH"],   data["B_CLINCH"]   = parse_two_values(cols[7], "0 of 0")
                data["R_GROUND"],   data["B_GROUND"]   = parse_two_values(cols[8], "0 of 0")

    return data


# ─────────────────────────────────────────────
# Main function: incremental scraping
# ─────────────────────────────────────────────

def scrape_since(last_date=None):
    """
    Scrapes UFC fights from ufcstats.com.

    - last_date=None  : first run, scrapes all events -> ufc_scraped_data.csv
    - last_date=date  : incremental run, scrapes only events after last_date
                        -> ufc_scraped_recent.csv

    Returns (DataFrame of scraped fights, fight count)
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    # Safety check: if first run (last_date=None) but the scraped file already
    # exists with data, reuse it instead of re-scraping (~10h)
    if last_date is None:
        existing_path = os.path.join(RAW_DIR, "ufc_scraped_data.csv")
        if os.path.exists(existing_path):
            try:
                df_existing = pd.read_csv(existing_path)
                if len(df_existing) > 0:
                    print(f"  Existing scraped file found: {existing_path} ({len(df_existing)} fights)")
                    print("  Reusing existing data (skipping re-scraping).")
                    return df_existing, len(df_existing)
            except Exception:
                pass  # corrupted file -> re-scrape

    mode = "incremental" if last_date else "full"
    print(f"UFC Scraper — mode {mode} (since: {last_date or 'beginning'})")

    events = parse_events_list(last_date)
    if not events:
        print("  No new events to scrape.")
        return pd.DataFrame(), 0

    print(f"  {len(events)} event(s) to scrape.")
    all_fights = []

    for i, (event_url, event_date) in enumerate(events):
        print(f"  [{i+1}/{len(events)}] {event_url.split('/')[-1]} ({event_date})")

        soup_event = get_soup(event_url)
        if soup_event is None:
            print("    Skip.")
            continue

        event_meta = parse_event_metadata(soup_event)

        fight_links = []
        for row in soup_event.find_all("tr", class_="b-fight-details__table-row"):
            if row.get("onclick"):
                a = row.find("a", href=True)
                if a and "fight-details" in a["href"]:
                    fight_links.append(a["href"])
        fight_links = list(set(fight_links))

        for fight_url in fight_links:
            fight_data = parse_fight_details(fight_url, event_meta=event_meta)
            if fight_data:
                all_fights.append(fight_data)

        print(f"    +{len(fight_links)} fights | Total: {len(all_fights)}")

    if not all_fights:
        return pd.DataFrame(), 0

    df = pd.DataFrame(all_fights)

    if last_date:
        # Incremental run: save recent fights separately ...
        recent_path = os.path.join(RAW_DIR, "ufc_scraped_recent.csv")
        df.to_csv(recent_path, index=False)
        print(f"  Saved: {recent_path} ({len(df)} new fights)")

        # ... and append them to the main scraped file
        main_path = os.path.join(RAW_DIR, "ufc_scraped_data.csv")
        if os.path.exists(main_path):
            df_main = pd.read_csv(main_path, low_memory=False)
            df_combined = pd.concat([df_main, df], ignore_index=True)
            # Deduplicate on (R_Fighter, B_Fighter, date) to be safe
            key_cols = ["R_Fighter", "B_Fighter", "date"]
            key_cols = [c for c in key_cols if c in df_combined.columns]
            if key_cols:
                df_combined = df_combined.drop_duplicates(subset=key_cols, keep="last")
            df_combined.to_csv(main_path, index=False)
            print(f"  Updated: {main_path} ({len(df_combined)} total fights)")
        else:
            df.to_csv(main_path, index=False)
            print(f"  Created: {main_path} ({len(df)} fights)")

        # Clear recent file now that it's merged into main
        open(recent_path, "w").close()
        print(f"  Cleared: {recent_path}")
    else:
        # Full run: save as main file
        main_path = os.path.join(RAW_DIR, "ufc_scraped_data.csv")
        df.to_csv(main_path, index=False)
        print(f"  Saved: {main_path} ({len(df)} fights)")

    return df, len(df)


if __name__ == "__main__":
    scrape_since(last_date=None)
