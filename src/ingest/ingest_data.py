"""
ingest_data.py — Scraper incrémental UFC Stats

Fonctions principales :
  scrape_since(last_date)  → scrape uniquement les events après last_date
                             last_date=None → scrape tout (premier run)

La liste des events sur ufcstats.com est ordonnée du plus récent au plus ancien.
Dès qu'on rencontre un event avec une date <= last_date, on s'arrête.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import random
from datetime import datetime

RAW_DIR = "data/raw"


# ─────────────────────────────────────────────
# Helpers HTTP
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
            print(f"    Tentative {attempt+1}/{retries} echouee : {url}")
            time.sleep(5)
    print(f"Abandon apres {retries} tentatives : {url}")
    return None

def clean_text(text):
    return text.strip() if text else ""


# ─────────────────────────────────────────────
# Parsing de la liste des events
# ─────────────────────────────────────────────

def parse_events_list(last_date=None):
    """
    Récupère les liens et dates des events depuis la page principale.
    Si last_date est fourni (datetime.date), s'arrête dès qu'un event est <= last_date.
    Retourne une liste de (event_url, event_date).
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

        # La date est dans le 2e td de la ligne (colonne "Date")
        tds = row.find_all("td")
        event_date = None
        if len(tds) >= 2:
            date_str = clean_text(tds[1].text)
            try:
                event_date = datetime.strptime(date_str, "%B %d, %Y").date()
            except ValueError:
                pass

        # Arrêt dès qu'on dépasse last_date
        if last_date and event_date and event_date <= last_date:
            break

        events.append((event_url, event_date))

    return events


# ─────────────────────────────────────────────
# Parsing d'un event
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
    Scrape toutes les stats disponibles d'un combat.
    Tables :
      [0] totaux     : KD, SIG_STR, TOTAL_STR, TD, SUB_ATT, REV, CTRL
      [1] par zone   : HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND
    Métadonnées : method, round, time, format, referee, fight_type, date, location
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

    # Métadonnées du combat
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

    # Table 0 : stats totales
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

    # Table 1 : frappes par zone
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
# Fonction principale : scrape incrémental
# ─────────────────────────────────────────────

def scrape_since(last_date=None):
    """
    Scrape les combats UFC depuis ufcstats.com.

    - last_date=None  : premier run, scrape tous les events → ufc_scraped_data.csv
    - last_date=date  : run incrémental, scrape seulement les events après last_date
                        → ufc_scraped_recent.csv

    Retourne (DataFrame des combats scrapés, nb combats)
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    # Garde-fou : si premier run (last_date=None) mais que le fichier scraped existe déjà
    # avec des données, on le réutilise au lieu de tout re-scraper (~10h)
    if last_date is None:
        existing_path = os.path.join(RAW_DIR, "ufc_scraped_data.csv")
        if os.path.exists(existing_path):
            try:
                df_existing = pd.read_csv(existing_path)
                if len(df_existing) > 0:
                    print(f"  Fichier scrape existant trouve : {existing_path} ({len(df_existing)} combats)")
                    print("  Reutilisation des donnees existantes (pas de re-scraping).")
                    return df_existing, len(df_existing)
            except Exception:
                pass  # fichier corrompu → on re-scrape

    mode = "incremental" if last_date else "complet"
    print(f"Scraper UFC — mode {mode} (depuis : {last_date or 'toujours'})")

    events = parse_events_list(last_date)
    if not events:
        print("  Aucun nouvel event a scraper.")
        return pd.DataFrame(), 0

    print(f"  {len(events)} event(s) a scraper.")
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

        print(f"    +{len(fight_links)} combats | Total : {len(all_fights)}")

    if not all_fights:
        return pd.DataFrame(), 0

    df = pd.DataFrame(all_fights)
    out_file = "ufc_scraped_recent.csv" if last_date else "ufc_scraped_data.csv"
    out_path = os.path.join(RAW_DIR, out_file)
    df.to_csv(out_path, index=False)
    print(f"  Sauvegarde : {out_path} ({len(df)} combats)")
    return df, len(df)


if __name__ == "__main__":
    scrape_since(last_date=None)
