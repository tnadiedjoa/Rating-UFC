"""
ranking.py — ML + Classement UFC

Fonctions principales :
  temporal_split(df)               → train / val / test chronologique
  train_models(X_tr, y_tr, ...)    → LR + SVM + Random Forest + XGBoost
  compare_models(models, X_te, y_te) → tableau de métriques
  rank_by_weights(fighter_stats)   → classement pondéré (WEIGHTS configurable)
  build_fighter_current_stats(df)  → profil actuel de chaque fighter
  rank_by_model(...)               → round-robin ML
  compute_elo(df)                  → rating Elo dynamique
  compare_with_official(ranking, df) → corrélation Spearman/Kendall vs UFC officiel
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
    roc_curve,
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from scipy.stats import spearmanr, kendalltau
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─────────────────────────────────────────────
# Poids par défaut pour le classement pondéré
# ─────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "win_rate":        0.20,
    "finish_rate":     0.15,
    "sig_str_per_min": 0.15,
    "sig_str_acc":     0.10,
    "td_per_min":      0.10,
    "td_acc":          0.10,
    "ctrl_per_min":    0.10,
    "kd_per_min":      0.05,
    "sub_att_per_min": 0.05,
}


# ─────────────────────────────────────────────
# Split temporel
# ─────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologique sans shuffle aléatoire.
    Retourne (train, val, test).
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ─────────────────────────────────────────────
# Preprocessing commun
# ─────────────────────────────────────────────

def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


# ─────────────────────────────────────────────
# Entraînement des modèles
# ─────────────────────────────────────────────

def train_models(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feat_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Entraîne LR, SVM, Random Forest et XGBoost.
    Retourne un dict {nom: modèle_ou_dict}.

    Format du dict retourné :
      "LogReg"       : sklearn Pipeline (fit)
      "SVM"          : sklearn Pipeline (fit)
      "RandomForest" : sklearn Pipeline (fit)
      "XGBoost"      : {"booster": xgb_model, "feat_cols": list, "type": "xgb"}
                       ou Pipeline si xgboost non disponible
    """
    tscv = TimeSeriesSplit(n_splits=5)
    models = {}

    # ── Logistic Regression ──
    print("  [1/4] Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]},
        cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    lr_grid.fit(X_tr, y_tr)
    models["LogReg"] = lr_grid.best_estimator_
    val_auc = roc_auc_score(y_val, models["LogReg"].predict_proba(X_val)[:, 1])
    print(f"    LR best params : {lr_grid.best_params_}  |  Val AUC : {val_auc:.3f}")

    # ── SVM ──
    print("  [2/4] SVM...")
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale"]},
        cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    svm_grid.fit(X_tr, y_tr)
    models["SVM"] = svm_grid.best_estimator_
    val_auc = roc_auc_score(y_val, models["SVM"].predict_proba(X_val)[:, 1])
    print(f"    SVM best params : {svm_grid.best_params_}  |  Val AUC : {val_auc:.3f}")

    # ── Random Forest ──
    print("  [3/4] Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 300], "max_depth": [4, 6, None],
         "min_samples_leaf": [5, 10]},
        cv=tscv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    rf_grid.fit(X_tr, y_tr)
    models["RandomForest"] = rf_grid.best_estimator_
    val_auc = roc_auc_score(y_val, models["RandomForest"].predict_proba(X_val)[:, 1])
    print(f"    RF best params : {rf_grid.best_params_}  |  Val AUC : {val_auc:.3f}")

    # ── XGBoost ──
    print("  [4/4] XGBoost...")
    if XGB_AVAILABLE and feat_cols:
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_tr_p = scl.fit_transform(imp.fit_transform(X_tr))
        X_val_p = scl.transform(imp.transform(X_val))

        dtrain = xgb.DMatrix(X_tr_p, label=y_tr, feature_names=feat_cols)
        dval = xgb.DMatrix(X_val_p, label=y_val, feature_names=feat_cols)
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "seed": 42,
            "verbosity": 0,
        }
        booster = xgb.train(
            xgb_params, dtrain, num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30, verbose_eval=100,
        )
        val_auc = roc_auc_score(y_val, booster.predict(dval))
        print(f"    XGB best round : {booster.best_iteration}  |  Val AUC : {val_auc:.3f}")
        models["XGBoost"] = {
            "booster": booster,
            "imputer": imp,
            "scaler": scl,
            "feat_cols": feat_cols,
            "type": "xgb",
        }
    else:
        # Fallback : gradient boosting sklearn
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=42)
        gb.fit(X_tr, y_tr)
        models["XGBoost"] = gb
        val_auc = roc_auc_score(y_val, gb.predict_proba(X_val)[:, 1])
        print(f"    GBT (fallback) — Val AUC : {val_auc:.3f}")

    return models


# ─────────────────────────────────────────────
# Prédiction unifiée
# ─────────────────────────────────────────────

def _predict_proba(model, X: np.ndarray, feat_cols: Optional[List[str]] = None) -> np.ndarray:
    """Retourne les probabilités P(R_Win=1) quelle que soit la structure du modèle."""
    if isinstance(model, dict) and model.get("type") == "xgb":
        X_p = model["scaler"].transform(model["imputer"].transform(X))
        dm = xgb.DMatrix(X_p, feature_names=model.get("feat_cols", feat_cols))
        return model["booster"].predict(dm)
    return model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────
# Comparaison des modèles
# ─────────────────────────────────────────────

def compare_models(
    models: Dict,
    X_te: np.ndarray,
    y_te: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compare accuracy, AUC, Brier score pour chaque modèle sur le test set.
    Retourne (results_df, {nom: probas_test}).
    """
    rows = []
    probas = {}

    for name, model in models.items():
        te_p = _predict_proba(model, X_te, feat_cols)
        probas[name] = te_p

        row = {
            "Model": name,
            "Test_Acc":   f"{accuracy_score(y_te, te_p > 0.5) * 100:.1f}%",
            "Test_AUC":   f"{roc_auc_score(y_te, te_p):.3f}",
            "Test_Brier": f"{brier_score_loss(y_te, te_p):.3f}",
        }
        if X_val is not None and y_val is not None:
            val_p = _predict_proba(model, X_val, feat_cols)
            row["Val_Acc"] = f"{accuracy_score(y_val, val_p > 0.5) * 100:.1f}%"
            row["Val_AUC"] = f"{roc_auc_score(y_val, val_p):.3f}"
        rows.append(row)

    results = pd.DataFrame(rows)
    cols = ["Model"]
    if "Val_Acc" in results.columns:
        cols += ["Val_Acc", "Val_AUC"]
    cols += ["Test_Acc", "Test_AUC", "Test_Brier"]
    return results[cols], probas


# ─────────────────────────────────────────────
# Classement pondéré
# ─────────────────────────────────────────────

def build_fighter_current_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques cumulées de carrière de chaque fighter
    à partir du DataFrame master (format combat-centrique).
    Retourne un DataFrame indexé par fighter avec les stats agrégées.
    """
    from src.processing.feature_engineering import (
        build_appearances, normalize_per_minute,
    )

    apps = build_appearances(df)
    apps = normalize_per_minute(apps)

    agg_cols = {
        "Win": "mean",
        "is_finish": "mean",
        "SIG_STR_landed_pm": "mean",
        "SIG_STR_acc": "mean",
        "TD_landed_pm": "mean",
        "TD_acc": "mean",
        "CTRL_sec_pm": "mean",
        "KD_pm": "mean",
        "SUB_ATT_pm": "mean",
        "date": "count",
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in apps.columns}

    stats = apps.groupby("Fighter").agg(agg_cols).reset_index()
    stats.rename(columns={"date": "n_fights"}, inplace=True)

    # Ajouter la division principale
    if "WeightClass" in apps.columns:
        main_div = (apps.groupby("Fighter")["WeightClass"]
                    .agg(lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) > 0 else "Unknown")
                    .reset_index())
        stats = stats.merge(main_div, on="Fighter", how="left")

    # Ajouter le dernier combat
    if "date" in apps.columns:
        last_date = (apps.sort_values("date").groupby("Fighter")["date"]
                     .last().reset_index().rename(columns={"date": "last_fight_date"}))
        stats = stats.merge(last_date, on="Fighter", how="left")

    return stats


def rank_by_weights(
    fighter_stats: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    weight_class: Optional[str] = None,
    min_fights: int = 3,
) -> pd.DataFrame:
    """
    Classement pondéré par les stats de carrière.
    Normalise chaque stat sur [0,1] puis calcule le score pondéré.

    Paramètres :
      fighter_stats : DataFrame issu de build_fighter_current_stats()
      weights       : dict {stat_key: poids} (DEFAULT_WEIGHTS si None)
      weight_class  : filtrer par division (None = toutes)
      min_fights    : nombre minimum de combats pour apparaître
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Mapping stat_key → colonne DataFrame
    col_map = {
        "win_rate":        "Win",
        "finish_rate":     "is_finish",
        "sig_str_per_min": "SIG_STR_landed_pm",
        "sig_str_acc":     "SIG_STR_acc",
        "td_per_min":      "TD_landed_pm",
        "td_acc":          "TD_acc",
        "ctrl_per_min":    "CTRL_sec_pm",
        "kd_per_min":      "KD_pm",
        "sub_att_per_min": "SUB_ATT_pm",
    }

    df = fighter_stats.copy()

    # Filtre
    if "n_fights" in df.columns:
        df = df[df["n_fights"] >= min_fights]
    if weight_class and "WeightClass" in df.columns:
        df = df[df["WeightClass"].str.lower().str.contains(weight_class.lower(), na=False)]

    if df.empty:
        return pd.DataFrame(columns=["Rank", "Fighter", "Score", "WeightClass"])

    # Normalisation min-max par colonne
    score = pd.Series(0.0, index=df.index)
    for key, w in weights.items():
        col = col_map.get(key)
        if col and col in df.columns:
            col_vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            mn, mx = col_vals.min(), col_vals.max()
            if mx > mn:
                normalized = (col_vals - mn) / (mx - mn)
            else:
                normalized = pd.Series(0.5, index=df.index)
            score += w * normalized

    df = df.copy()
    df["Score"] = score.values
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df["Score"] = df["Score"].round(4)

    cols = ["Rank", "Fighter", "Score"]
    if "n_fights" in df.columns:
        cols.append("n_fights")
    if "WeightClass" in df.columns:
        cols.append("WeightClass")
    return df[cols]


# ─────────────────────────────────────────────
# Classement ML (round-robin)
# ─────────────────────────────────────────────

def rank_by_model(
    fighter_stats: pd.DataFrame,
    model,
    preproc,
    feat_cols: List[str],
    weight_class: Optional[str] = None,
    min_fights: int = 3,
) -> pd.DataFrame:
    """
    Classement via tournoi round-robin :
    Score(A) = sum of P(A beats X) for all X in division.

    fighter_stats : DataFrame avec colonnes de stats par fighter.
    model         : modèle sklearn ou dict XGBoost.
    preproc       : sklearn Pipeline (imputer + scaler) FIT sur le train.
    feat_cols     : liste des colonnes features (doit correspondre à preproc).
    """
    df = fighter_stats.copy()
    if "n_fights" in df.columns:
        df = df[df["n_fights"] >= min_fights]
    if weight_class and "WeightClass" in df.columns:
        df = df[df["WeightClass"].str.lower().str.contains(weight_class.lower(), na=False)]

    if len(df) < 2:
        return pd.DataFrame(columns=["Rank", "Fighter", "Score"])

    fighters = df["Fighter"].tolist()
    scores = {f: 0.0 for f in fighters}

    for i, fa in enumerate(fighters):
        for fb in fighters[i + 1:]:
            pa = df[df["Fighter"] == fa].iloc[0]
            pb = df[df["Fighter"] == fb].iloc[0]

            # Construire le vecteur delta A-B
            row = {col: 0.0 for col in feat_cols}
            # Remplir avec les stats disponibles dans fighter_stats
            stat_map = {
                "delta_avg_Win": ("Win", "Win"),
                "delta_avg_SIG_STR_landed_pm": ("SIG_STR_landed_pm", "SIG_STR_landed_pm"),
                "delta_avg_SIG_STR_acc": ("SIG_STR_acc", "SIG_STR_acc"),
                "delta_avg_TD_landed_pm": ("TD_landed_pm", "TD_landed_pm"),
                "delta_avg_TD_acc": ("TD_acc", "TD_acc"),
                "delta_avg_CTRL_sec_pm": ("CTRL_sec_pm", "CTRL_sec_pm"),
                "delta_avg_KD_pm": ("KD_pm", "KD_pm"),
                "delta_avg_SUB_ATT_pm": ("SUB_ATT_pm", "SUB_ATT_pm"),
                "delta_avg_is_finish": ("is_finish", "is_finish"),
            }
            for delta_col, (ra_col, rb_col) in stat_map.items():
                if delta_col in feat_cols:
                    va = pa.get(ra_col, 0) if ra_col in pa.index else 0
                    vb = pb.get(rb_col, 0) if rb_col in pb.index else 0
                    row[delta_col] = (va or 0) - (vb or 0)

            X = pd.DataFrame([row])[feat_cols].fillna(0).values
            X_proc = preproc.transform(X)
            p_a_wins = float(_predict_proba(model, X_proc, feat_cols)[0])
            scores[fa] += p_a_wins
            scores[fb] += 1 - p_a_wins

    ranking = pd.DataFrame(
        [(f, s) for f, s in scores.items()], columns=["Fighter", "Score"]
    )
    ranking = ranking.sort_values("Score", ascending=False).reset_index(drop=True)
    ranking["Rank"] = ranking.index + 1
    ranking["Score"] = ranking["Score"].round(2)
    if "WeightClass" in df.columns:
        ranking = ranking.merge(
            df[["Fighter", "WeightClass"]].drop_duplicates("Fighter"),
            on="Fighter", how="left",
        )
    return ranking[["Rank", "Fighter", "Score"] +
                   (["WeightClass"] if "WeightClass" in ranking.columns else [])]


# ─────────────────────────────────────────────
# Elo dynamique
# ─────────────────────────────────────────────

def compute_elo(
    df_chronological: pd.DataFrame,
    K: float = 32,
    initial: float = 1500,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Calcule le rating Elo dynamique pour chaque fighter.
    df_chronological doit être trié par date (croissant).
    Retourne (elo_final_dict, elo_history_df).
    """
    df = df_chronological.sort_values("date").reset_index(drop=True)
    elo: Dict[str, float] = {}
    history = []

    for _, fight in df.iterrows():
        r = fight["R_Fighter"]
        b = fight["B_Fighter"]
        er = elo.get(r, initial)
        eb = elo.get(b, initial)

        expected_r = 1 / (1 + 10 ** ((eb - er) / 400))
        actual_r = fight["R_Win"]

        elo[r] = er + K * (actual_r - expected_r)
        elo[b] = eb + K * ((1 - actual_r) - (1 - expected_r))

        wc = fight.get("WeightClass", "") if "WeightClass" in fight.index else ""
        history.append({"date": fight["date"], "Fighter": r, "Elo": elo[r], "WeightClass": wc})
        history.append({"date": fight["date"], "Fighter": b, "Elo": elo[b], "WeightClass": wc})

    return elo, pd.DataFrame(history)


def elo_ranking(
    elo_final: Dict[str, float],
    weight_class: Optional[str] = None,
    elo_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construit le DataFrame de classement Elo.
    Si weight_class est fourni et elo_history disponible, filtre par division.
    """
    if weight_class and elo_history is not None:
        wc_fighters = (elo_history[
            elo_history["WeightClass"].str.lower().str.contains(weight_class.lower(), na=False)
        ]["Fighter"].unique())
        filtered = {f: e for f, e in elo_final.items() if f in wc_fighters}
    else:
        filtered = elo_final

    df = pd.DataFrame(
        [(f, e) for f, e in filtered.items()], columns=["Fighter", "Elo"]
    )
    df = df.sort_values("Elo", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df["Elo"] = df["Elo"].round(1)
    return df[["Rank", "Fighter", "Elo"]]


# ─────────────────────────────────────────────
# Comparaison avec le classement officiel UFC
# ─────────────────────────────────────────────

def compare_with_official(
    our_ranking: pd.DataFrame,
    df: pd.DataFrame,
    rank_col: str = "RMatchWCRank",
    fighter_col: str = "R_Fighter",
) -> Dict:
    """
    Compare notre classement avec les rangs officiels UFC.
    Retourne un dict avec Spearman rho, Kendall tau et p-values.
    """
    if not SCIPY_AVAILABLE:
        return {"error": "scipy non disponible"}

    # Fighters ayant un classement officiel
    ranked = (df[[fighter_col, rank_col]].dropna()
              .groupby(fighter_col)[rank_col]
              .mean()
              .reset_index())
    ranked.columns = ["Fighter", "official_rank"]

    # Jointure avec notre classement
    our = our_ranking[["Fighter", "Rank"]].copy()
    merged = ranked.merge(our, on="Fighter", how="inner")

    if len(merged) < 5:
        return {
            "n_fighters": len(merged),
            "message": f"Seulement {len(merged)} fighters comparables (minimum 5 requis)",
        }

    rho, p_spear = spearmanr(merged["official_rank"], merged["Rank"])
    tau, p_kend = kendalltau(merged["official_rank"], merged["Rank"])

    return {
        "n_fighters": len(merged),
        "spearman_rho": round(rho, 3),
        "p_spearman": round(p_spear, 4),
        "kendall_tau": round(tau, 3),
        "p_kendall": round(p_kend, 4),
        "fighters": merged,
    }
