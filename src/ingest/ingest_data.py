import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import random

# --- CONFIGURATION ---
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = "ufc_scraped_data.csv"
MAX_EVENTS = 500  # Tu peux y aller maintenant

def get_soup(url, retries=3):
    """Récupère le HTML avec gestion des erreurs et des retours à la ligne."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(retries):
        try:
            # Petite pause aléatoire pour faire "humain"
            time.sleep(random.uniform(1.0, 2.5)) 
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
            
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️ Tentative {attempt + 1}/{retries} échouée pour {url}")
            time.sleep(5) # On attend 5 secondes si on se fait bloquer avant de réessayer
            
    print(f"❌ Abandon de l'URL après {retries} tentatives : {url}")
    return None

def clean_text(text):
    return text.strip() if text else ""

def parse_fight_details(fight_url):
    # (Garde EXACTEMENT la même fonction parse_fight_details que je t'ai donnée avant)
    # Je ne la remets pas en entier ici pour que ce soit lisible, 
    # mais tu gardes tout le bloc qui extrait R_Fighter, R_CTRL, R_HEAD, etc.
    soup = get_soup(fight_url)
    if not soup: return None

    fighters = soup.find_all('h3', class_='b-fight-details__person-name')
    if len(fighters) < 2: return None
    r_fighter = clean_text(fighters[0].text)
    b_fighter = clean_text(fighters[1].text)
    
    status_divs = soup.find_all('i', class_='b-fight-details__person-status')
    r_status = clean_text(status_divs[0].text)
    b_status = clean_text(status_divs[1].text)

    tables = soup.find_all('table', class_='b-fight-details__table')
    
    data = {
        "R_Fighter": r_fighter, "B_Fighter": b_fighter,
        "R_Status": r_status, "B_Status": b_status,
        "URL": fight_url
    }

    if len(tables) > 0:
        rows = tables[0].find_all('tr', class_='b-fight-details__table-row')
        if len(rows) >= 2:
            cols = rows[1].find_all('td')
            ctrl_col = cols[9].find_all('p')
            data["R_CTRL"] = clean_text(ctrl_col[0].text) if len(ctrl_col) > 0 else "0:00"
            data["B_CTRL"] = clean_text(ctrl_col[1].text) if len(ctrl_col) > 1 else "0:00"
            td_col = cols[5].find_all('p')
            data["R_TD"] = clean_text(td_col[0].text) if len(td_col) > 0 else "0"
            data["B_TD"] = clean_text(td_col[1].text) if len(td_col) > 1 else "0"

    if len(tables) > 1:
        rows = tables[1].find_all('tr', class_='b-fight-details__table-row')
        if len(rows) >= 2:
            cols = rows[1].find_all('td')
            head = cols[3].find_all('p')
            data["R_HEAD"] = clean_text(head[0].text) if len(head) > 0 else "0"
            data["B_HEAD"] = clean_text(head[1].text) if len(head) > 1 else "0"
            body = cols[4].find_all('p')
            data["R_BODY"] = clean_text(body[0].text) if len(body) > 0 else "0"
            data["B_BODY"] = clean_text(body[1].text) if len(body) > 1 else "0"
            leg = cols[5].find_all('p')
            data["R_LEG"] = clean_text(leg[0].text) if len(leg) > 0 else "0"
            data["B_LEG"] = clean_text(leg[1].text) if len(leg) > 1 else "0"

    return data

def run_scraper():
    print("🚀 Démarrage du Scraper UFC (Version Anti-Crash)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # 1. Récupérer les événements
    url_events = "http://ufcstats.com/statistics/events/completed?page=all"
    soup = get_soup(url_events)
    if not soup:
        print("❌ Impossible de charger la page principale.")
        return

    event_links = []
    for link in soup.find_all('a', href=True):
        if "event-details" in link['href']:
            event_links.append(link['href'])
    
    event_links = list(set(event_links))
    print(f"📅 {len(event_links)} événements trouvés. Go !")
    
    # On va stocker les données au fur et à mesure
    all_fights_data = []
    
    for event_url in event_links[:MAX_EVENTS]:
        print(f"\n> Analyse Event: {event_url.split('/')[-1]} ...")
        
        soup_event = get_soup(event_url)
        
        # LE FIX EST LÀ : Si ça plante, on passe à la suite au lieu de crasher
        if soup_event is None:
            print(f"    ⏭️ Skip de l'événement (impossible de le charger).")
            continue
            
        fight_links = []
        for row in soup_event.find_all('tr', class_='b-fight-details__table-row'):
            if row.get('onclick'):
                a_link = row.find('a', href=True)
                if a_link and "fight-details" in a_link['href']:
                    fight_links.append(a_link['href'])
        
        fight_links = list(set(fight_links))
        
        for fight_url in fight_links:
            fight_data = parse_fight_details(fight_url)
            if fight_data:
                all_fights_data.append(fight_data)
        
        print(f"    ✅ +{len(fight_links)} combats. Total en mémoire : {len(all_fights_data)}")
        
        # SAUVEGARDE PROGRESSIVE : À chaque event terminé, on écrase le CSV avec les nouvelles données
        if all_fights_data:
            pd.DataFrame(all_fights_data).to_csv(output_path, index=False)

if __name__ == "__main__":
    run_scraper()