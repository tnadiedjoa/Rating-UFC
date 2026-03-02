"""
update_master.py -- Mise a jour incrementale du dataset master UFC

Fonction principale : update(last_date=None)
  - last_date=None  : premier run complet
  - last_date=date  : run incremental, seulement les combats apres last_date

Retourne (DataFrame master, nb_nouveaux_combats)
"""

import os
import re
import unicodedata
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Chemins (resolus par rapport a la position de ce fichier)
# ---------------------------------------------------------------------------

# src/processing/update_master.py  -> 3 dirname -> racine du repo
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.dirname(_THIS_DIR))

RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
KAGGLE1_DIR = os.path.join(RAW_DIR, "kaggle_1")
KAGGLE2_DIR = os.path.join(RAW_DIR, "kaggle_2")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MASTER_PATH = os.path.join(PROC_DIR, "ufc_master_enriched.csv")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Colonnes "X of Y" a splitter (prefixe R_ / B_ ajoute dynamiquement)
OF_COLS = [
    "SIG_STR", "TOTAL_STR", "TD",
    "HEAD", "BODY", "LEG",
    "DISTANCE", "CLINCH", "GROUND",
]

# Colonnes a garder de Kaggle 2
KAGGLE2_KEEP = [
    "RedFighter", "BlueFighter", "Date",
    "Gender", "WeightClass",
    "RedOdds", "BlueOdds", "RedExpectedValue", "BlueExpectedValue",
    "RedDecOdds", "BlueDecOdds", "RSubOdds", "BSubOdds", "RKOOdds", "BKOOdds",
    "Finish", "FinishDetails", "FinishRound", "FinishRoundTime", "TotalFightTimeSecs",
    "RMatchWCRank", "BMatchWCRank", "EmptyArena",
]

# Colonnes a garder de raw_fighter_details
FIGHTER_KEEP = [
    "fighter_name",
    "Height_cms", "Weight_lbs", "Reach_cms",
    "Stance", "DOB",
    "SLpM", "Str_Acc", "SApM", "Str_Def",
    "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
]


# ===========================================================================
# Helpers
# ===========================================================================

def normalize_name(name):
    """Lowercase, strip, normalise apostrophes et accents."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    # Apostrophes typographiques -> apostrophe simple
    name = name.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    # Suppression des diacritiques
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name


def parse_fight_key(r, b, date):
    """
    Cle unique d'un combat : paire de noms triee + date.
    Format retourne : "nom1|nom2|YYYY-MM-DD"
    """
    r_n = normalize_name(r)
    b_n = normalize_name(b)
    pair = "|".join(sorted([r_n, b_n]))
    date_str = ""
    if date is not None and not (isinstance(date, float) and pd.isna(date)):
        if isinstance(date, str) and date.strip():
            try:
                date_str = pd.to_datetime(date, dayfirst=False).strftime("%Y-%m-%d")
            except Exception:
                date_str = date.strip()[:10]
        elif hasattr(date, "strftime"):
            try:
                date_str = date.strftime("%Y-%m-%d")
            except Exception:
                pass
    return f"{pair}|{date_str}"


def parse_name_key(r, b):
    """Cle secondaire sans date (fallback pour combats sans date connue)."""
    r_n = normalize_name(r)
    b_n = normalize_name(b)
    return "|".join(sorted([r_n, b_n]))


def split_of(value):
    """
    Convertit "X of Y" en dict {"_landed": X, "_attempted": Y}.
    Retourne 0/0 en cas d'echec.
    """
    if not isinstance(value, str):
        return {"_landed": 0, "_attempted": 0}
    m = re.match(r"(\d+)\s+of\s+(\d+)", value.strip(), re.IGNORECASE)
    if m:
        return {"_landed": int(m.group(1)), "_attempted": int(m.group(2))}
    return {"_landed": 0, "_attempted": 0}


def time_to_seconds(t):
    """
    Convertit "M:SS" (ou "H:MM:SS") en secondes (int).
    Accepte egalement les valeurs numeriques.
    """
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return 0
    if isinstance(t, (int, float)):
        return int(t)
    parts = str(t).strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return 0


def pct_to_float(p):
    """
    Convertit "47%" en 0.47.
    Retourne NaN en cas d'echec.
    """
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return float("nan")
    if isinstance(p, (int, float)):
        return float(p) if float(p) <= 1.0 else float(p) / 100.0
    s = str(p).strip().replace("%", "")
    try:
        return float(s) / 100.0
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Helpers conversions physiques
# ---------------------------------------------------------------------------

def _height_to_cms(h):
    """Convertit "5' 11\"" en centimetres (float)."""
    if not isinstance(h, str) or not h.strip():
        return float("nan")
    m = re.match(r"(\d+)'\s*(\d*)", h.strip())
    if m:
        feet   = int(m.group(1))
        inches = int(m.group(2)) if m.group(2) else 0
        return round((feet * 12 + inches) * 2.54, 1)
    return float("nan")


def _weight_to_lbs(w):
    """Convertit "185 lbs." en float."""
    if not isinstance(w, str) or not w.strip():
        return float("nan")
    m = re.match(r"([\d.]+)\s*lbs", w.strip(), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return float("nan")


def _reach_to_cms(r):
    """Convertit '72"' en centimetres (float)."""
    if not isinstance(r, str) or not r.strip():
        return float("nan")
    m = re.match(r"([\d.]+)", r.strip())
    if m:
        return round(float(m.group(1)) * 2.54, 1)
    return float("nan")


# ===========================================================================
# Nettoyage commun d'un DataFrame de combats
# ===========================================================================

def clean_fight_df(df):
    """
    Applique sur un DataFrame de combats brut :
      - Renomme les colonnes avec point final (R_SIG_STR. -> R_SIG_STR)
      - Split "X of Y" pour chaque colonne OF_COLS cote R et B
      - Convertit CTRL -> CTRL_sec (secondes)
      - Convertit SIG_STR_pct et TD_pct en float [0..1]

    Modifie le DataFrame en place et le retourne.
    """
    df = df.copy()

    for side in ("R", "B"):
        # -- Split "X of Y" -------------------------------------------------
        for col in OF_COLS:
            raw_col = f"{side}_{col}"
            # Kaggle1 stocke certaines colonnes avec un point final
            alt_col = raw_col + "."
            if alt_col in df.columns and raw_col not in df.columns:
                df.rename(columns={alt_col: raw_col}, inplace=True)

            if raw_col in df.columns:
                parsed = df[raw_col].apply(split_of)
                df[f"{raw_col}_landed"]    = parsed.apply(lambda x: x["_landed"])
                df[f"{raw_col}_attempted"] = parsed.apply(lambda x: x["_attempted"])
                df.drop(columns=[raw_col], inplace=True)

        # -- CTRL -> CTRL_sec -----------------------------------------------
        ctrl_col = f"{side}_CTRL"
        if ctrl_col in df.columns:
            df[f"{side}_CTRL_sec"] = df[ctrl_col].apply(time_to_seconds)
            df.drop(columns=[ctrl_col], inplace=True)

        # -- SIG_STR_pct -> float -------------------------------------------
        for pct_col in (f"{side}_SIG_STR_pct", f"{side}_TD_pct"):
            if pct_col in df.columns:
                df[pct_col] = df[pct_col].apply(pct_to_float)

    return df


# ===========================================================================
# Chargement des sources
# ===========================================================================

def load_kaggle1(last_date=None):
    """
    Charge raw_total_fight_data.csv (separateur ';').
    Nettoie, renomme, filtre par last_date si fourni.
    Retourne un DataFrame vide si le fichier est absent.
    """
    path = os.path.join(KAGGLE1_DIR, "raw_total_fight_data.csv")
    if not os.path.exists(path):
        print(f"[update_master] Fichier introuvable : {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, sep=";", low_memory=False)

    # Normalisation de la casse des colonnes fighters
    rename_map = {}
    for old, new in [("R_fighter", "R_Fighter"), ("B_fighter", "B_Fighter")]:
        if old in df.columns:
            rename_map[old] = new
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Nettoyage combats
    df = clean_fight_df(df)

    # Colonnes Win
    if "Winner" in df.columns:
        df["R_Win"] = df["Winner"].str.strip().str.lower() == "red"
        df["B_Win"] = df["Winner"].str.strip().str.lower() == "blue"

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

    # Filtre incremental
    if last_date is not None and "date" in df.columns:
        last_dt = pd.to_datetime(last_date)
        df = df[df["date"] > last_dt].copy()

    df["source"] = "kaggle1"
    return df.reset_index(drop=True)


def load_kaggle2(last_date=None):
    """
    Charge ufc-master.csv depuis Kaggle 2.
    Garde seulement les colonnes utiles, renomme les noms de fighters.
    Retourne un DataFrame vide si le fichier est absent.
    """
    path = os.path.join(KAGGLE2_DIR, "ufc-master.csv")
    if not os.path.exists(path):
        print(f"[update_master] Fichier introuvable : {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Garder seulement les colonnes disponibles parmi KAGGLE2_KEEP
    keep = [c for c in KAGGLE2_KEEP if c in df.columns]
    df = df[keep].copy()

    # Renommage
    df.rename(columns={
        "RedFighter":  "R_Fighter",
        "BlueFighter": "B_Fighter",
        "Date":        "date",
    }, inplace=True)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

    # Filtre incremental
    if last_date is not None and "date" in df.columns:
        last_dt = pd.to_datetime(last_date)
        df = df[df["date"] > last_dt].copy()

    return df.reset_index(drop=True)


def load_fighter_details():
    """
    Charge raw_fighter_details.csv.
    Convertit hauteur, poids, portee et DOB.
    Retourne un DataFrame vide si le fichier est absent.
    """
    path = os.path.join(KAGGLE1_DIR, "raw_fighter_details.csv")
    if not os.path.exists(path):
        print(f"[update_master] Fichier introuvable : {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Conversions physiques
    if "Height" in df.columns:
        df["Height_cms"] = df["Height"].apply(_height_to_cms)
    if "Weight" in df.columns:
        df["Weight_lbs"] = df["Weight"].apply(_weight_to_lbs)
    if "Reach" in df.columns:
        df["Reach_cms"]  = df["Reach"].apply(_reach_to_cms)

    # DOB -> datetime
    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce", format="%b %d, %Y")

    # Garder seulement les colonnes utiles
    keep = [c for c in FIGHTER_KEEP if c in df.columns]
    df = df[keep].copy()

    return df.reset_index(drop=True)


def load_scraped(last_date=None):
    """
    Charge ufc_scraped_recent.csv (si last_date) ou ufc_scraped_data.csv.
    Applique le meme nettoyage que Kaggle 1.
    Retourne un DataFrame vide si le fichier est absent.
    """
    fname = "ufc_scraped_recent.csv" if last_date else "ufc_scraped_data.csv"
    path  = os.path.join(RAW_DIR, fname)

    if not os.path.exists(path):
        print(f"[update_master] Fichier introuvable : {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Nettoyage combats
    df = clean_fight_df(df)

    # Colonnes Win depuis Status
    if "R_Status" in df.columns:
        df["R_Win"] = df["R_Status"].str.strip().str.upper() == "W"
    if "B_Status" in df.columns:
        df["B_Win"] = df["B_Status"].str.strip().str.upper() == "W"

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

    # Filtre incremental
    if last_date is not None and "date" in df.columns:
        last_dt = pd.to_datetime(last_date)
        df = df[df["date"] > last_dt].copy()

    df["source"] = "scraped"
    return df.reset_index(drop=True)


# ===========================================================================
# Enrichissements
# ===========================================================================

def _add_keys(df):
    """Ajoute les colonnes _fight_key et _name_key au DataFrame."""
    df = df.copy()
    df["_fight_key"] = df.apply(
        lambda row: parse_fight_key(
            row.get("R_Fighter", ""),
            row.get("B_Fighter", ""),
            row.get("date", None),
        ),
        axis=1,
    )
    df["_name_key"] = df.apply(
        lambda row: parse_name_key(
            row.get("R_Fighter", ""),
            row.get("B_Fighter", ""),
        ),
        axis=1,
    )
    return df


def deduplicate(df):
    """
    Deduplique un DataFrame de combats sur la cle fight_key.
    La premiere occurrence (source plus fiable en tete) est conservee.
    """
    if df.empty:
        return df
    df = _add_keys(df)
    df = df.drop_duplicates(subset=["_fight_key"], keep="first")
    df = df.drop(columns=["_fight_key", "_name_key"])
    return df.reset_index(drop=True)


def enrich_with_kaggle2(df_fights, df_k2):
    """
    Joint le DataFrame principal avec Kaggle 2 sur la cle fight.
    Tentative 1 : fight_key (noms + date).
    Tentative 2 (fallback) : name_key (noms uniquement, sans date).
    Les colonnes de Kaggle 2 ne remplacent pas celles deja presentes.
    """
    if df_k2.empty or df_fights.empty:
        return df_fights

    df_k2 = df_k2.copy()
    df_k2 = _add_keys(df_k2)

    # Supprimer les doublons de cle dans k2
    df_k2_fk = df_k2.drop_duplicates(subset=["_fight_key"], keep="first")
    df_k2_nk = df_k2.drop_duplicates(subset=["_name_key"],  keep="first")

    # Colonnes a apporter (eviter doublons avec df_fights)
    shared_id_cols = {"R_Fighter", "B_Fighter", "date", "_fight_key", "_name_key"}
    new_cols = [
        c for c in df_k2.columns
        if c not in shared_id_cols and c not in df_fights.columns
    ]
    if not new_cols:
        return df_fights

    df_fights = _add_keys(df_fights)

    # -- Join principal par fight_key --------------------------------------
    merged = df_fights.merge(
        df_k2_fk[["_fight_key"] + new_cols],
        on="_fight_key",
        how="left",
    )

    # -- Fallback par name_key pour les lignes sans match -----------------
    sentinel_col = new_cols[0]
    missing_idx = merged.index[merged[sentinel_col].isna()]

    if len(missing_idx) > 0:
        fallback_left = merged.loc[missing_idx, ["_name_key"]].copy()
        fallback = fallback_left.merge(
            df_k2_nk[["_name_key"] + new_cols],
            on="_name_key",
            how="left",
        )
        for col in new_cols:
            merged.loc[missing_idx, col] = fallback[col].values

    merged.drop(columns=["_fight_key", "_name_key"], inplace=True)
    return merged.reset_index(drop=True)


def enrich_with_fighters(df_fights, df_fighters):
    """
    Joint les attributs physiques sur R_Fighter et B_Fighter.
    Les colonnes physiques recues prennent le prefixe R_ ou B_.
    """
    if df_fighters.empty or df_fights.empty:
        return df_fights

    phys_cols = [c for c in df_fighters.columns if c != "fighter_name"]
    df_fights = df_fights.copy()

    for side, col_name in [("R", "R_Fighter"), ("B", "B_Fighter")]:
        if col_name not in df_fights.columns:
            continue

        renamed = {c: f"{side}_{c}" for c in phys_cols}
        df_side = df_fighters.rename(columns={"fighter_name": col_name, **renamed})

        # Si des colonnes physiques existent deja, on les supprime pour eviter _x/_y
        existing = [c for c in renamed.values() if c in df_fights.columns]
        if existing:
            df_fights.drop(columns=existing, inplace=True)

        df_fights = df_fights.merge(df_side, on=col_name, how="left")

    return df_fights.reset_index(drop=True)


# ===========================================================================
# Telechargement Kaggle
# ===========================================================================

def download_kaggle_datasets():
    """
    Telechargement des deux datasets via l'API Kaggle.
    Ne leve pas d'exception si Kaggle est inaccessible ou non configure.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()

        os.makedirs(KAGGLE1_DIR, exist_ok=True)
        os.makedirs(KAGGLE2_DIR, exist_ok=True)

        print("[update_master] Telechargement Kaggle 1 (rajeevw/ufcdata)...")
        api.dataset_download_files(
            "rajeevw/ufcdata", path=KAGGLE1_DIR, unzip=True, quiet=False
        )

        print("[update_master] Telechargement Kaggle 2 (mdabbert/ultimate-ufc-dataset)...")
        api.dataset_download_files(
            "mdabbert/ultimate-ufc-dataset", path=KAGGLE2_DIR, unzip=True, quiet=False
        )

        print("[update_master] Telechargements termines.")

    except ImportError:
        print("[update_master] Module kaggle non installe -- telechargement ignore.")
    except Exception as exc:
        print(f"[update_master] Erreur Kaggle (mode offline ou cle invalide) : {exc}")


# ===========================================================================
# Fonction principale
# ===========================================================================

def update(last_date=None):
    """
    Met a jour le dataset master UFC de facon incrementale.

    Parametres
    ----------
    last_date : str | datetime.date | None
        Date de la derniere mise a jour connue (format ISO ou datetime.date).
        Si None, reconstruction complete depuis toutes les sources.

    Retourne
    --------
    tuple (df_master: pd.DataFrame, nb_nouveaux_combats: int)
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    # 1. Re-telechargement Kaggle -----------------------------------------
    download_kaggle_datasets()

    # 2. Kaggle 1 ----------------------------------------------------------
    print("[update_master] Chargement Kaggle 1...")
    df_k1 = load_kaggle1(last_date=last_date)
    print(f"  {len(df_k1)} combat(s) Kaggle 1 (apres filtre)")

    # 3. Kaggle 2 (metadata bookmaker / finish) ----------------------------
    print("[update_master] Chargement Kaggle 2...")
    df_k2 = load_kaggle2(last_date=last_date)
    print(f"  {len(df_k2)} combat(s) Kaggle 2 (apres filtre)")

    # 4. Attributs physiques -----------------------------------------------
    print("[update_master] Chargement attributs physiques...")
    df_fighters = load_fighter_details()
    print(f"  {len(df_fighters)} profil(s) de combattant")

    # 5. Donnees scrapees --------------------------------------------------
    print("[update_master] Chargement donnees scrapees...")
    df_scraped = load_scraped(last_date=last_date)
    print(f"  {len(df_scraped)} combat(s) scrapes (apres filtre)")

    # 6. Assemblage des nouvelles donnees ----------------------------------
    print("[update_master] Assemblage et enrichissement...")

    frames = [f for f in [df_k1, df_scraped] if not f.empty]
    if frames:
        df_new = pd.concat(frames, ignore_index=True, sort=False)
        df_new = deduplicate(df_new)
        print(f"  {len(df_new)} combat(s) uniques (Kaggle1 + scraped)")

        df_new = enrich_with_kaggle2(df_new, df_k2)
        df_new = enrich_with_fighters(df_new, df_fighters)
    else:
        print("[update_master] Aucune nouvelle donnee de combat.")
        df_new = pd.DataFrame()

    # 7. Fusion avec le master existant ------------------------------------
    master_exists = os.path.exists(MASTER_PATH)

    if master_exists:
        print("[update_master] Chargement du master existant...")
        try:
            df_master_old = pd.read_csv(MASTER_PATH, low_memory=False)
            if "date" in df_master_old.columns:
                df_master_old["date"] = pd.to_datetime(
                    df_master_old["date"], errors="coerce"
                )
            print(f"  {len(df_master_old)} combat(s) dans le master existant")
        except Exception as exc:
            print(f"[update_master] Impossible de lire le master existant : {exc}")
            df_master_old = pd.DataFrame()

        if df_new.empty and df_master_old.empty:
            df_master = pd.DataFrame()
            nb_nouveaux = 0
        elif df_new.empty:
            df_master = df_master_old
            nb_nouveaux = 0
        elif df_master_old.empty:
            df_master = df_new
            nb_nouveaux = len(df_master)
        else:
            n_avant = len(df_master_old)
            df_combined = pd.concat(
                [df_master_old, df_new], ignore_index=True, sort=False
            )
            df_master = deduplicate(df_combined)
            nb_nouveaux = len(df_master) - n_avant

    else:
        print("[update_master] Master inexistant -- creation depuis zero.")
        if not df_new.empty:
            df_master   = df_new
            nb_nouveaux = len(df_master)
        else:
            # Si last_date avait filtre toutes les lignes, on reconstruit tout
            print("[update_master] Reconstruction complete (last_date ignore)...")
            df_k1_full      = load_kaggle1(last_date=None)
            df_k2_full      = load_kaggle2(last_date=None)
            df_scraped_full = load_scraped(last_date=None)

            frames_full = [f for f in [df_k1_full, df_scraped_full] if not f.empty]
            if frames_full:
                df_all = pd.concat(frames_full, ignore_index=True, sort=False)
                df_all = deduplicate(df_all)
                df_all = enrich_with_kaggle2(df_all, df_k2_full)
                df_all = enrich_with_fighters(df_all, df_fighters)
                df_master = df_all
            else:
                df_master = pd.DataFrame()

            nb_nouveaux = len(df_master)

    # 8. Sauvegarde --------------------------------------------------------
    if not df_master.empty:
        df_master.to_csv(MASTER_PATH, index=False)
        print(
            f"[update_master] Master sauvegarde : {MASTER_PATH} "
            f"({len(df_master)} combats)"
        )
    else:
        print("[update_master] Rien a sauvegarder (DataFrame vide).")

    # 9. Retour ------------------------------------------------------------
    print(f"[update_master] Termine. Nouveaux combats : {nb_nouveaux}")
    return df_master, nb_nouveaux


# ---------------------------------------------------------------------------
# Point d'entree CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, n = update(last_date=None)
    print(f"Master shape : {df.shape}  |  Nouveaux : {n}")
