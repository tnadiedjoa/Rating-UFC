"""
feature_engineering.py — Vectorisation sans data leakage

Fonctions principales :
  build_features(df)          → features_df complet (vecteurs delta R-B)
  get_feature_cols(df, ...)   → liste des colonnes features

Technique anti-leakage : expanding().mean().shift(1) par fighter trié chronologiquement.
Pour un combat à la date T, les features = statistiques cumulées des combats AVANT T.
"""

import pandas as pd
import numpy as np
from typing import List

# ─────────────────────────────────────────────
# Colonnes de stats brutes (par fighter)
# ─────────────────────────────────────────────

STAT_COLS = [
    "KD",
    "SIG_STR_landed", "SIG_STR_attempted",
    "TOTAL_STR_landed", "TOTAL_STR_attempted",
    "TD_landed", "TD_attempted",
    "HEAD_landed", "HEAD_attempted",
    "BODY_landed", "BODY_attempted",
    "LEG_landed", "LEG_attempted",
    "DISTANCE_landed", "DISTANCE_attempted",
    "CLINCH_landed", "CLINCH_attempted",
    "GROUND_landed", "GROUND_attempted",
    "CTRL_sec", "SUB_ATT", "REV",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _parse_stat_of(val_str, which="landed"):
    """
    Parse '15 of 32' → (15, 32).
    val_str peut être une string ou un nombre.
    which: 'landed' ou 'attempted'
    """
    if pd.isna(val_str):
        return np.nan
    s = str(val_str).strip()
    if " of " in s:
        parts = s.split(" of ")
        try:
            landed = float(parts[0])
            attempted = float(parts[1])
            return landed if which == "landed" else attempted
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_ctrl(val_str):
    """Parse '2:35' → 155.0 secondes."""
    if pd.isna(val_str):
        return np.nan
    s = str(val_str).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_pct(val_str):
    """Parse '72%' → 0.72."""
    if pd.isna(val_str):
        return np.nan
    s = str(val_str).strip().replace("%", "")
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def _extract_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les colonnes numériques de stats depuis les colonnes R_*/B_*.
    Gère les formats '15 of 32', '2:35', '72%'.
    """
    out = df.copy()

    for prefix in ("R", "B"):
        # KD, SUB_ATT, REV → directs
        for col in ("KD", "SUB_ATT", "REV"):
            src = f"{prefix}_{col}"
            if src in out.columns:
                out[src] = pd.to_numeric(out[src], errors="coerce")

        # SIG_STR, TOTAL_STR, TD, HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND → "X of Y"
        for base in ("SIG_STR", "TOTAL_STR", "TD", "HEAD", "BODY", "LEG",
                     "DISTANCE", "CLINCH", "GROUND"):
            raw_col = f"{prefix}_{base}"
            if raw_col in out.columns:
                landed_col = f"{prefix}_{base}_landed"
                attempted_col = f"{prefix}_{base}_attempted"
                if landed_col not in out.columns:
                    out[landed_col] = out[raw_col].apply(lambda x: _parse_stat_of(x, "landed"))
                if attempted_col not in out.columns:
                    out[attempted_col] = out[raw_col].apply(lambda x: _parse_stat_of(x, "attempted"))

        # CTRL → secondes
        ctrl_col = f"{prefix}_CTRL"
        if ctrl_col in out.columns:
            out[f"{prefix}_CTRL_sec"] = out[ctrl_col].apply(_parse_ctrl)

        # SIG_STR_pct, TD_pct → ratio
        for pct_base in ("SIG_STR_pct", "TD_pct"):
            src = f"{prefix}_{pct_base}"
            if src in out.columns:
                out[src] = out[src].apply(_parse_pct)

    return out


# ─────────────────────────────────────────────
# Étape 1 — Long format : une ligne par (fighter, combat)
# ─────────────────────────────────────────────

def build_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit le DataFrame combat-centrique en format long :
    une ligne par (fighter, combat).
    Retourne un DataFrame avec colonnes : Fighter, Opponent, date, Win, + stats brutes.
    """
    df = _extract_numeric_stats(df)

    meta_cols = ["date", "WeightClass", "TotalFightTimeSecs", "method"]
    meta_cols = [c for c in meta_cols if c in df.columns]

    # Colonnes stats disponibles pour R et B
    actual_stat_cols = [c for c in STAT_COLS if f"R_{c}" in df.columns]
    r_cols = [f"R_{c}" for c in actual_stat_cols]
    b_cols = [f"B_{c}" for c in actual_stat_cols]

    # Coin rouge
    r_rows = df[["R_Fighter"] + meta_cols + ["R_Win"] + r_cols].copy()
    r_rows.columns = ["Fighter"] + meta_cols + ["Win"] + actual_stat_cols
    r_rows["Opponent"] = df["B_Fighter"].values

    # Coin bleu
    # B_Win = 1 - R_Win si disponible, sinon construire
    if "B_Win" in df.columns:
        b_win = df["B_Win"].values
    else:
        b_win = (1 - df["R_Win"].fillna(0)).values

    b_rows = df[["B_Fighter"] + meta_cols].copy()
    b_rows.columns = ["Fighter"] + meta_cols
    b_rows["Win"] = b_win
    for i, col in enumerate(actual_stat_cols):
        b_rows[col] = df[b_cols[i]].values
    b_rows["Opponent"] = df["R_Fighter"].values

    apps = pd.concat([r_rows, b_rows], ignore_index=True)
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    return apps.sort_values(["Fighter", "date"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# Étape 2 — Normalisation par minute
# ─────────────────────────────────────────────

def normalize_per_minute(apps: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les stats de volume par minute de combat.
    Ajoute des colonnes de précision (ratio).
    """
    apps = apps.copy()

    # Durée du combat en minutes
    duration_col = "TotalFightTimeSecs"
    if duration_col in apps.columns:
        minutes = apps[duration_col].clip(lower=1).fillna(900) / 60.0
    else:
        minutes = pd.Series(15.0, index=apps.index)

    vol_cols = [
        "KD", "SIG_STR_landed", "SIG_STR_attempted",
        "TOTAL_STR_landed", "TD_landed", "TD_attempted",
        "HEAD_landed", "BODY_landed", "LEG_landed",
        "CTRL_sec", "SUB_ATT",
    ]
    for col in vol_cols:
        if col in apps.columns:
            apps[f"{col}_pm"] = apps[col] / minutes

    # Précision (ratio, pas de normalisation temporelle)
    if "SIG_STR_landed" in apps.columns and "SIG_STR_attempted" in apps.columns:
        apps["SIG_STR_acc"] = (apps["SIG_STR_landed"]
                               / apps["SIG_STR_attempted"].clip(lower=1))
    if "TD_landed" in apps.columns and "TD_attempted" in apps.columns:
        apps["TD_acc"] = apps["TD_landed"] / apps["TD_attempted"].clip(lower=1)
    if "HEAD_landed" in apps.columns and "SIG_STR_landed" in apps.columns:
        apps["HEAD_rate"] = apps["HEAD_landed"] / apps["SIG_STR_landed"].clip(lower=1)
    if "BODY_landed" in apps.columns and "SIG_STR_landed" in apps.columns:
        apps["BODY_rate"] = apps["BODY_landed"] / apps["SIG_STR_landed"].clip(lower=1)
    if "LEG_landed" in apps.columns and "SIG_STR_landed" in apps.columns:
        apps["LEG_rate"] = apps["LEG_landed"] / apps["SIG_STR_landed"].clip(lower=1)

    # Méthode de victoire encodée
    if "method" in apps.columns:
        m = apps["method"].fillna("")
        apps["is_finish"] = m.str.contains("KO|TKO|Submission", case=False).astype(float)
        apps["is_KO"] = m.str.contains("KO|TKO", case=False).astype(float)
        apps["is_sub"] = m.str.contains("Submission", case=False).astype(float)

    return apps


# ─────────────────────────────────────────────
# Étape 3 — Expanding window + shift(1) anti-leakage
# ─────────────────────────────────────────────

MEAN_COLS = [
    "KD_pm", "SIG_STR_landed_pm", "SIG_STR_attempted_pm", "TOTAL_STR_landed_pm",
    "TD_landed_pm", "TD_attempted_pm", "HEAD_landed_pm", "BODY_landed_pm", "LEG_landed_pm",
    "CTRL_sec_pm", "SUB_ATT_pm",
    "SIG_STR_acc", "TD_acc", "HEAD_rate", "BODY_rate", "LEG_rate",
    "Win", "is_finish", "is_KO", "is_sub",
]


def compute_prelagged_stats(apps: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque fighter, calcule les statistiques cumulées AVANT chaque combat.
    Utilise expanding().mean().shift(1) pour éviter tout data leakage.
    Ajoute aussi : forme récente (3 derniers combats), n_fights_before.
    """
    available_mean_cols = [c for c in MEAN_COLS if c in apps.columns]
    results = []

    for fighter, grp in apps.groupby("Fighter"):
        g = grp.sort_values("date").copy()

        # Stats cumulées avg_* (expanding mean, puis shift pour exclure le combat actuel)
        for col in available_mean_cols:
            g[f"avg_{col}"] = g[col].expanding().mean().shift(1)

        # Forme récente : 3 derniers combats
        for col in ["Win", "KD_pm", "SIG_STR_landed_pm"]:
            if col in g.columns:
                g[f"recent3_{col}"] = g[col].rolling(3, min_periods=1).mean().shift(1)
        g["recent_wins_3"] = g["Win"].rolling(3, min_periods=1).sum().shift(1)

        # Nombre de combats avant ce combat (expérience UFC)
        g["n_fights_before"] = np.arange(len(g), dtype=float)

        results.append(g)

    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values(["date", "Fighter"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# Étape 4 — Vecteurs delta (R - B) + attributs physiques
# ─────────────────────────────────────────────

def build_delta_features(df: pd.DataFrame, apps_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruit un DataFrame combat-centrique avec des features delta (Rouge - Bleu).
    Chaque feature = différence entre les stats pré-combat de R et B.
    """
    pre_cols = [c for c in apps_stats.columns
                if c.startswith("avg_") or c.startswith("recent") or c == "n_fights_before"]

    # Indexer par (Fighter, date) pour la jointure
    red_stats = (apps_stats.set_index(["Fighter", "date"])[pre_cols]
                 .add_prefix("R_pre_"))
    blue_stats = (apps_stats.set_index(["Fighter", "date"])[pre_cols]
                  .add_prefix("B_pre_"))

    # Colonnes meta du combat
    meta_keep = ["R_Fighter", "B_Fighter", "date", "WeightClass", "R_Win",
                 "R_Height_cms", "R_Reach_cms", "R_Weight_lbs", "R_Stance", "R_DOB",
                 "B_Height_cms", "B_Reach_cms", "B_Weight_lbs", "B_Stance", "B_DOB",
                 "RMatchWCRank", "BMatchWCRank", "RedOdds", "BlueOdds", "EmptyArena"]
    meta_keep = [c for c in meta_keep if c in df.columns]
    meta = df[meta_keep].copy()
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")

    meta = meta.join(red_stats, on=["R_Fighter", "date"], how="left")
    meta = meta.join(blue_stats, on=["B_Fighter", "date"], how="left")

    feat = pd.DataFrame(index=meta.index)

    # Delta des stats cumulées
    r_pre_cols = [c for c in meta.columns if c.startswith("R_pre_")]
    for rc in r_pre_cols:
        bc = rc.replace("R_pre_", "B_pre_")
        name = rc.replace("R_pre_", "delta_")
        if bc in meta.columns:
            feat[name] = meta[rc] - meta[bc]

    # Attributs physiques delta
    if "R_Height_cms" in meta.columns and "B_Height_cms" in meta.columns:
        feat["delta_height"] = meta["R_Height_cms"] - meta["B_Height_cms"]
    if "R_Reach_cms" in meta.columns and "B_Reach_cms" in meta.columns:
        feat["delta_reach"] = meta["R_Reach_cms"] - meta["B_Reach_cms"]

    # Âge au moment du combat
    date_s = pd.to_datetime(meta["date"])
    if "R_DOB" in meta.columns:
        r_dob = pd.to_datetime(meta["R_DOB"], errors="coerce")
        feat["R_age"] = (date_s - r_dob).dt.days / 365.25
    if "B_DOB" in meta.columns:
        b_dob = pd.to_datetime(meta["B_DOB"], errors="coerce")
        feat["B_age"] = (date_s - b_dob).dt.days / 365.25
    if "R_age" in feat.columns and "B_age" in feat.columns:
        feat["delta_age"] = feat["R_age"] - feat["B_age"]

    # Stance
    if "R_Stance" in meta.columns:
        feat["R_is_southpaw"] = (meta["R_Stance"] == "Southpaw").astype(float)
    if "B_Stance" in meta.columns:
        feat["B_is_southpaw"] = (meta["B_Stance"] == "Southpaw").astype(float)
    if "R_Stance" in meta.columns and "B_Stance" in meta.columns:
        feat["same_stance"] = (meta["R_Stance"] == meta["B_Stance"]).astype(float)

    # Classement officiel
    if "RMatchWCRank" in meta.columns:
        rr = pd.to_numeric(meta["RMatchWCRank"], errors="coerce").fillna(99)
        br = pd.to_numeric(meta["BMatchWCRank"], errors="coerce").fillna(99)
        feat["delta_rank"] = rr - br
        feat["R_is_ranked"] = pd.to_numeric(meta["RMatchWCRank"], errors="coerce").notna().astype(float)
        feat["B_is_ranked"] = pd.to_numeric(meta["BMatchWCRank"], errors="coerce").notna().astype(float)

    # Cotes → probabilité implicite
    if "RedOdds" in meta.columns and "BlueOdds" in meta.columns:
        def ato_prob(odds):
            o = pd.to_numeric(odds, errors="coerce")
            return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))
        feat["R_implied_prob"] = ato_prob(meta["RedOdds"])
        feat["delta_implied_prob"] = (ato_prob(meta["RedOdds"])
                                      - ato_prob(meta["BlueOdds"]))

    # Contexte
    if "EmptyArena" in meta.columns:
        feat["empty_arena"] = pd.to_numeric(meta["EmptyArena"], errors="coerce").fillna(0)

    # Catégorie de poids (ordinal)
    wc_map = {
        "strawweight": 1, "flyweight": 2, "bantamweight": 3,
        "featherweight": 4, "lightweight": 5, "welterweight": 6,
        "middleweight": 7, "light heavyweight": 8, "heavyweight": 9,
    }
    if "WeightClass" in meta.columns:
        feat["weight_class_ord"] = meta["WeightClass"].apply(
            lambda x: next((v for k, v in wc_map.items()
                             if k in str(x).lower()), 5)
        )

    # Métadonnées (non utilisées comme features, utiles pour le notebook)
    feat["R_Fighter"] = meta["R_Fighter"].values
    feat["B_Fighter"] = meta["B_Fighter"].values
    feat["date"] = meta["date"].values
    feat["WeightClass"] = meta["WeightClass"].values if "WeightClass" in meta.columns else None
    feat["R_Win"] = meta["R_Win"].values

    if "R_pre_n_fights_before" in meta.columns:
        feat["n_fights_R"] = meta["R_pre_n_fights_before"].values
    if "B_pre_n_fights_before" in meta.columns:
        feat["n_fights_B"] = meta["B_pre_n_fights_before"].values

    return feat


# ─────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet : df master → features_df prêt pour le ML.
    Filtre les combats où chaque fighter a au moins 1 combat UFC précédent.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    apps = build_appearances(df)
    apps = normalize_per_minute(apps)
    apps_stats = compute_prelagged_stats(apps)
    fdf = build_delta_features(df, apps_stats)

    # Filtre : au moins 1 combat précédent pour R et B
    if "n_fights_R" in fdf.columns and "n_fights_B" in fdf.columns:
        fdf = fdf[(fdf["n_fights_R"] >= 1) & (fdf["n_fights_B"] >= 1)].copy()

    return fdf.reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame, include_odds: bool = True) -> List[str]:
    """
    Retourne la liste des colonnes features (delta_* + physiques + contexte).
    include_odds=False exclut les colonnes liées aux cotes de paris.
    """
    base_cols = [
        "R_age", "B_age",
        "R_is_southpaw", "B_is_southpaw", "same_stance",
        "R_is_ranked", "B_is_ranked", "delta_rank",
        "empty_arena", "weight_class_ord",
    ]
    if include_odds:
        base_cols += ["R_implied_prob", "delta_implied_prob"]

    feat_cols = [
        c for c in df.columns
        if c.startswith("delta_") or c in base_cols
    ]
    if not include_odds:
        feat_cols = [c for c in feat_cols if "implied_prob" not in c]

    return feat_cols
