"""Script de génération du notebook UFC_Pipeline.ipynb."""
import nbformat as nbf
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
c = []

# ─── Section 0 — Configuration ────────────────────────────────────────────────
c.append(nbf.v4.new_markdown_cell(
    "# UFC Pipeline — Scraping, ML & Classement\n\n"
    "**Un seul notebook pour tout :** scraping incrémental, mise à jour des données, "
    "feature engineering, modèles ML et classement par division.\n\n"
    "---\n"
    "## Section 0 — Configuration\n\n"
    "Modifiez les valeurs ci-dessous avant d'exécuter le notebook."
))

c.append(nbf.v4.new_code_cell(
    "# ─── PARAMÈTRES MODIFIABLES ──────────────────────────────────────────────\n"
    "\n"
    "# Poids pour le classement pondéré (doivent sommer à 1.0)\n"
    "WEIGHTS = {\n"
    "    'win_rate':        0.20,\n"
    "    'finish_rate':     0.15,\n"
    "    'sig_str_per_min': 0.15,\n"
    "    'sig_str_acc':     0.10,\n"
    "    'td_per_min':      0.10,\n"
    "    'td_acc':          0.10,\n"
    "    'ctrl_per_min':    0.10,\n"
    "    'kd_per_min':      0.05,\n"
    "    'sub_att_per_min': 0.05,\n"
    "}\n"
    "\n"
    "# Filtrer par division (None = toutes les divisions)\n"
    "# Exemples : 'Lightweight', 'Heavyweight', 'Middleweight'\n"
    "WEIGHT_CLASS_FILTER = None\n"
    "\n"
    "# Nombre minimum de combats UFC pour apparaître dans le classement\n"
    "MIN_FIGHTS_RANKING = 5\n"
    "\n"
    "# ─── SETUP ────────────────────────────────────────────────────────────────\n"
    "import sys, os\n"
    "# Ajouter la racine du projet au path pour les imports\n"
    "PROJECT_ROOT = os.path.abspath('')\n"
    "if PROJECT_ROOT not in sys.path:\n"
    "    sys.path.insert(0, PROJECT_ROOT)\n"
    "\n"
    "import json\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib\n"
    "import seaborn as sns\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "sns.set_theme(style='darkgrid', palette='muted')\n"
    "plt.rcParams['figure.figsize'] = (13, 5)\n"
    "pd.set_option('display.max_columns', 30)\n"
    "pd.set_option('display.width', 120)\n"
    "\n"
    "STATE_FILE = os.path.join(PROJECT_ROOT, 'data', 'state', 'last_run.json')\n"
    "MASTER_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ufc_master_enriched.csv')\n"
    "\n"
    "# Lire l'état du dernier run\n"
    "with open(STATE_FILE, 'r') as f:\n"
    "    state = json.load(f)\n"
    "\n"
    "print('Configuration chargée')\n"
    "print(f'Dernier run : {state[\"last_run\"] or \"jamais (premier run)\"}')\n"
    "print(f'Total combats connus : {state[\"n_fights_total\"]}')\n"
    "print(f'Filtre division : {WEIGHT_CLASS_FILTER or \"toutes\"}')\n"
    "print(f'Min combats pour ranking : {MIN_FIGHTS_RANKING}')"
))

# ─── Section 1 — Mise à jour des données ──────────────────────────────────────
c.append(nbf.v4.new_markdown_cell(
    "---\n"
    "## Section 1 — Mise à jour des données\n\n"
    "1. Scraping incrémental ufcstats.com\n"
    "2. Re-téléchargement des datasets Kaggle\n"
    "3. Fusion et déduplication → master CSV\n"
    "4. Mise à jour du state file"
))

c.append(nbf.v4.new_code_cell(
    "from datetime import date\n"
    "from src.ingest.ingest_data import scrape_since\n"
    "from src.processing.update_master import update\n"
    "\n"
    "# Convertir last_run en date si non null\n"
    "last_date = None\n"
    "if state['last_run']:\n"
    "    from datetime import datetime\n"
    "    last_date = datetime.strptime(state['last_run'], '%Y-%m-%d').date()\n"
    "\n"
    "print('=' * 60)\n"
    "if last_date:\n"
    "    print(f'Run INCREMENTAL — combats après le {last_date}')\n"
    "else:\n"
    "    print('PREMIER RUN — scraping de tous les combats')\n"
    "print('=' * 60)\n"
    "\n"
    "# 1. Scraping ufcstats\n"
    "print('\\n[1/2] Scraping ufcstats.com...')\n"
    "df_scraped, n_scraped = scrape_since(last_date=last_date)\n"
    "print(f'  Combats scrapés : {n_scraped}')\n"
    "\n"
    "# 2. Mise à jour du master\n"
    "print('\\n[2/2] Mise à jour du master CSV...')\n"
    "df_master, n_new = update(last_date=last_date)\n"
    "print(f'  Nouveaux combats ajoutés : {n_new}')\n"
    "print(f'  Total master : {len(df_master)} combats')"
))

c.append(nbf.v4.new_code_cell(
    "# Mise à jour du state file\n"
    "today_str = date.today().isoformat()\n"
    "state['last_run'] = today_str\n"
    "state['n_fights_total'] = len(df_master)\n"
    "os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)\n"
    "with open(STATE_FILE, 'w') as f:\n"
    "    json.dump(state, f, indent=2)\n"
    "\n"
    "print(f'State mis a jour : last_run = {today_str}')\n"
    "print(f'Total combats en base : {len(df_master)}')"
))

# ─── Section 2 — Aperçu du dataset ────────────────────────────────────────────
c.append(nbf.v4.new_markdown_cell(
    "---\n"
    "## Section 2 — Aperçu du dataset\n\n"
    "Structure du master CSV, statistiques globales, distributions."
))

c.append(nbf.v4.new_code_cell(
    "df = pd.read_csv(MASTER_CSV)\n"
    "df['date'] = pd.to_datetime(df['date'], errors='coerce')\n"
    "df = df.sort_values('date').reset_index(drop=True)\n"
    "\n"
    "print(f'Shape : {df.shape}')\n"
    "print(f'Periode : {df[\"date\"].min().date()} -> {df[\"date\"].max().date()}')\n"
    "print(f'Nb fighters uniques : {pd.concat([df[\"R_Fighter\"], df[\"B_Fighter\"]]).nunique()}')\n"
    "print(f'Taux victoire Rouge : {df[\"R_Win\"].mean()*100:.1f}%')\n"
    "\n"
    "# Colonnes disponibles\n"
    "print(f'\\nColonnes ({len(df.columns)}) :')\n"
    "print(', '.join(df.columns.tolist()))"
))

c.append(nbf.v4.new_code_cell(
    "# Aperçu des 5 combats les plus récents\n"
    "display_cols = [c for c in ['date','R_Fighter','B_Fighter','WeightClass',\n"
    "                             'method','last_round','R_Win'] if c in df.columns]\n"
    "print('5 combats les plus récents :')\n"
    "df[display_cols].tail(5)"
))

c.append(nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
    "\n"
    "# Distribution victoires\n"
    "win_counts = df['R_Win'].value_counts()\n"
    "axes[0].bar(['Blue wins', 'Red wins'],\n"
    "            [win_counts.get(0, 0), win_counts.get(1, 0)],\n"
    "            color=['steelblue', 'tomato'])\n"
    "axes[0].set_title('Distribution des victoires')\n"
    "for i, v in enumerate([win_counts.get(0, 0), win_counts.get(1, 0)]):\n"
    "    axes[0].text(i, v + 20, f'{v/len(df)*100:.1f}%', ha='center', fontweight='bold')\n"
    "\n"
    "# Combats par année\n"
    "df['year'] = df['date'].dt.year\n"
    "yearly = df.groupby('year').size()\n"
    "axes[1].bar(yearly.index, yearly.values, color='steelblue', edgecolor='white')\n"
    "axes[1].set_title('Combats par année')\n"
    "axes[1].tick_params(axis='x', rotation=45)\n"
    "\n"
    "# Méthodes de victoire\n"
    "if 'method' in df.columns:\n"
    "    methods = df['method'].value_counts().head(10)\n"
    "    axes[2].barh(methods.index[::-1], methods.values[::-1], color='steelblue')\n"
    "    axes[2].set_title('Top 10 méthodes de victoire')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ─── Section 3 — Feature Engineering ─────────────────────────────────────────
c.append(nbf.v4.new_markdown_cell(
    "---\n"
    "## Section 3 — Feature Engineering\n\n"
    "Vectorisation sans data leakage via `expanding().mean().shift(1)` par fighter.\n\n"
    "Pour un combat à la date T, chaque feature = statistique cumulée des combats **avant T**."
))

c.append(nbf.v4.new_code_cell(
    "from src.processing.feature_engineering import build_features, get_feature_cols\n"
    "\n"
    "print('Construction des features (peut prendre 30-60 secondes)...')\n"
    "features_df = build_features(df)\n"
    "\n"
    "feat_cols = get_feature_cols(features_df, include_odds=True)\n"
    "feat_cols_no_odds = get_feature_cols(features_df, include_odds=False)\n"
    "\n"
    "print(f'Shape features : {features_df.shape}')\n"
    "print(f'Nb features (avec cotes) : {len(feat_cols)}')\n"
    "print(f'Nb features (sans cotes) : {len(feat_cols_no_odds)}')\n"
    "print(f'Cible R_Win : {features_df[\"R_Win\"].mean()*100:.1f}% Rouge')"
))

c.append(nbf.v4.new_code_cell(
    "# Valeurs manquantes par feature\n"
    "null_pct = features_df[feat_cols].isnull().mean() * 100\n"
    "null_pct = null_pct.sort_values(ascending=False)\n"
    "\n"
    "print('Valeurs manquantes par feature :')\n"
    "for col, pct in null_pct.items():\n"
    "    if pct > 0:\n"
    "        print(f'  {col:45s}  {pct:.1f}%')\n"
    "print(f'\\n{(null_pct == 0).sum()} features sans valeurs manquantes')"
))

c.append(nbf.v4.new_code_cell(
    "# Top 20 features par corrélation avec R_Win\n"
    "num_feats = [c for c in feat_cols if features_df[c].dtype in ['float64', 'float32', 'int64']]\n"
    "corr = features_df[num_feats + ['R_Win']].corr()['R_Win'].drop('R_Win')\n"
    "top_corr = corr.abs().sort_values(ascending=False).head(20)\n"
    "vals = corr[top_corr.index].sort_values()\n"
    "\n"
    "plt.figure(figsize=(12, 6))\n"
    "colors = ['tomato' if v > 0 else 'steelblue' for v in vals]\n"
    "vals.plot(kind='barh', color=colors)\n"
    "plt.title('Top 20 features — corrélation avec R_Win')\n"
    "plt.axvline(0, color='black', linewidth=0.8)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print(f'Feature la plus prédictive : {top_corr.index[0]}  (|r| = {top_corr.iloc[0]:.3f})')"
))

# ─── Section 4 — Modèles ML ───────────────────────────────────────────────────
c.append(nbf.v4.new_markdown_cell(
    "---\n"
    "## Section 4 — Modèles ML\n\n"
    "Split temporel 70/15/15 + Logistic Regression, SVM, Random Forest, XGBoost.\n\n"
    "**Règle stricte :** preprocessing (imputation + scaling) fitté uniquement sur le train."
))

c.append(nbf.v4.new_code_cell(
    "from src.models.ranking import (\n"
    "    temporal_split, make_preprocessor, train_models,\n"
    "    compare_models, compute_elo, elo_ranking,\n"
    "    build_fighter_current_stats, rank_by_weights, rank_by_model,\n"
    "    compare_with_official,\n"
    ")\n"
    "\n"
    "train, val, test = temporal_split(features_df)\n"
    "\n"
    "X_tr  = train[feat_cols].values;  y_tr  = train['R_Win'].values\n"
    "X_val = val[feat_cols].values;    y_val = val['R_Win'].values\n"
    "X_te  = test[feat_cols].values;   y_te  = test['R_Win'].values\n"
    "\n"
    "# Preprocessing (fit UNIQUEMENT sur train)\n"
    "preproc = make_preprocessor()\n"
    "X_tr_p  = preproc.fit_transform(X_tr)\n"
    "X_val_p = preproc.transform(X_val)\n"
    "X_te_p  = preproc.transform(X_te)\n"
    "\n"
    "print(f'Train : {len(train):5d} combats  ({train[\"date\"].min().date()} -> {train[\"date\"].max().date()})')\n"
    "print(f'Val   : {len(val):5d} combats  ({val[\"date\"].min().date()} -> {val[\"date\"].max().date()})')\n"
    "print(f'Test  : {len(test):5d} combats  ({test[\"date\"].min().date()} -> {test[\"date\"].max().date()})')"
))

c.append(nbf.v4.new_code_cell(
    "# Baseline : cotes de paris\n"
    "if 'R_implied_prob' in features_df.columns:\n"
    "    test_odds = test[test['R_implied_prob'].notna()].copy()\n"
    "    if len(test_odds) > 10:\n"
    "        from sklearn.metrics import accuracy_score, roc_auc_score\n"
    "        pred_odds = (test_odds['R_implied_prob'] > 0.5).astype(int)\n"
    "        acc = accuracy_score(test_odds['R_Win'], pred_odds)\n"
    "        auc = roc_auc_score(test_odds['R_Win'], test_odds['R_implied_prob'])\n"
    "        print(f'Baseline cotes — Accuracy : {acc*100:.1f}%  |  AUC : {auc:.3f}')\n"
    "        print('Le modele ML doit depasser ces valeurs.')\n"
    "else:\n"
    "    print('Pas de cotes disponibles sur ce dataset.')"
))

c.append(nbf.v4.new_code_cell(
    "print('Entrainement des modeles...')\n"
    "models = train_models(X_tr_p, y_tr, X_val_p, y_val, feat_cols=feat_cols)"
))

c.append(nbf.v4.new_code_cell(
    "# Tableau de comparaison\n"
    "results_df, probas = compare_models(models, X_te_p, y_te, X_val_p, y_val, feat_cols=feat_cols)\n"
    "print('\\n=== Comparaison des modeles ===')\n"
    "print(results_df.to_string(index=False))"
))

c.append(nbf.v4.new_code_cell(
    "from sklearn.metrics import roc_curve, roc_auc_score\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "\n"
    "# Courbes ROC\n"
    "colors = ['tomato', 'steelblue', 'forestgreen', 'darkorange']\n"
    "for (name, proba), col in zip(probas.items(), colors):\n"
    "    fpr, tpr, _ = roc_curve(y_te, proba)\n"
    "    auc = roc_auc_score(y_te, proba)\n"
    "    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=col)\n"
    "axes[0].plot([0, 1], [0, 1], 'k--', label='Chance')\n"
    "axes[0].set_xlabel('False Positive Rate')\n"
    "axes[0].set_ylabel('True Positive Rate')\n"
    "axes[0].set_title('Courbes ROC — Test set')\n"
    "axes[0].legend()\n"
    "\n"
    "# Feature importance XGBoost\n"
    "if 'XGBoost' in models and isinstance(models['XGBoost'], dict):\n"
    "    import xgboost as xgb\n"
    "    booster = models['XGBoost']['booster']\n"
    "    imp = booster.get_score(importance_type='gain')\n"
    "    imp_df = pd.DataFrame(imp.items(), columns=['Feature', 'Importance'])\n"
    "    imp_df = imp_df.sort_values('Importance', ascending=True).tail(15)\n"
    "    axes[1].barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')\n"
    "    axes[1].set_title('XGBoost — Feature importance (gain)')\n"
    "elif 'RandomForest' in models:\n"
    "    rf = models['RandomForest']\n"
    "    imp_df = pd.DataFrame({'Feature': feat_cols, 'Importance': rf.feature_importances_})\n"
    "    imp_df = imp_df.sort_values('Importance', ascending=True).tail(15)\n"
    "    axes[1].barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')\n"
    "    axes[1].set_title('Random Forest — Feature importance')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ─── Section 5 — Classement ───────────────────────────────────────────────────
c.append(nbf.v4.new_markdown_cell(
    "---\n"
    "## Section 5 — Classement UFC\n\n"
    "Trois méthodes de ranking :\n"
    "1. **Classement pondéré** — basé sur les poids définis en Section 0\n"
    "2. **Classement ML round-robin** — `P(A bat B)` pour toutes les paires\n"
    "3. **Elo dynamique** — historique du niveau de chaque fighter\n\n"
    "Comparaison finale avec les classements officiels UFC (Spearman/Kendall)."
))

c.append(nbf.v4.new_code_cell(
    "# Profil actuel de chaque fighter\n"
    "fighter_stats = build_fighter_current_stats(df)\n"
    "print(f'Fighters avec stats : {len(fighter_stats)}')\n"
    "print(f'Fighters avec >= {MIN_FIGHTS_RANKING} combats : {(fighter_stats[\"n_fights\"] >= MIN_FIGHTS_RANKING).sum()}')"
))

c.append(nbf.v4.new_code_cell(
    "# ─── 1. Classement pondéré ─────────────────────────────────────────────\n"
    "print('=== CLASSEMENT PONDERE ===')\n"
    "if WEIGHT_CLASS_FILTER:\n"
    "    print(f'Division : {WEIGHT_CLASS_FILTER}')\n"
    "    w_rank = rank_by_weights(fighter_stats, WEIGHTS, WEIGHT_CLASS_FILTER, MIN_FIGHTS_RANKING)\n"
    "    print(w_rank.head(15).to_string(index=False))\n"
    "else:\n"
    "    divisions = ['Heavyweight', 'Light Heavyweight', 'Middleweight', 'Welterweight',\n"
    "                 'Lightweight', 'Featherweight', 'Bantamweight', 'Flyweight']\n"
    "    divisions = [d for d in divisions if d in fighter_stats.get('WeightClass', pd.Series()).values]\n"
    "    if not divisions:\n"
    "        # Toutes divisions confondues\n"
    "        w_rank = rank_by_weights(fighter_stats, WEIGHTS, None, MIN_FIGHTS_RANKING)\n"
    "        print('Top 20 global :')\n"
    "        print(w_rank.head(20).to_string(index=False))\n"
    "    else:\n"
    "        for div in divisions[:4]:  # Afficher les 4 premières\n"
    "            w_rank = rank_by_weights(fighter_stats, WEIGHTS, div, MIN_FIGHTS_RANKING)\n"
    "            if len(w_rank) > 0:\n"
    "                print(f'\\n{div.upper()} (Top 10) :')\n"
    "                print(w_rank.head(10).to_string(index=False))"
))

c.append(nbf.v4.new_code_cell(
    "# ─── 2. Classement ML round-robin ──────────────────────────────────────\n"
    "# Utiliser le meilleur modèle (sans cotes pour indépendance du marché)\n"
    "print('=== CLASSEMENT ML (ROUND-ROBIN) ===')\n"
    "\n"
    "# Preprocessing sans cotes\n"
    "X_tr_no = train[feat_cols_no_odds].values\n"
    "X_val_no = val[feat_cols_no_odds].values\n"
    "\n"
    "preproc_no = make_preprocessor()\n"
    "X_tr_no_p = preproc_no.fit_transform(X_tr_no)\n"
    "\n"
    "# Utiliser Logistic Regression (plus rapide pour le round-robin)\n"
    "best_model_name = results_df.sort_values('Test_AUC', ascending=False).iloc[0]['Model']\n"
    "print(f'Meilleur modele : {best_model_name}')\n"
    "\n"
    "# Pour le round-robin, utiliser LR sans cotes\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "lr_no = LogisticRegression(max_iter=1000, C=1, solver='liblinear', random_state=42)\n"
    "lr_no.fit(X_tr_no_p, y_tr)\n"
    "\n"
    "if WEIGHT_CLASS_FILTER:\n"
    "    ml_rank = rank_by_model(fighter_stats, lr_no, preproc_no, feat_cols_no_odds,\n"
    "                             WEIGHT_CLASS_FILTER, MIN_FIGHTS_RANKING)\n"
    "    print(f'Division : {WEIGHT_CLASS_FILTER}')\n"
    "    print(ml_rank.head(15).to_string(index=False))\n"
    "else:\n"
    "    for div in (['Lightweight', 'Heavyweight', 'Middleweight', 'Welterweight'][:3]):\n"
    "        ml_rank = rank_by_model(fighter_stats, lr_no, preproc_no, feat_cols_no_odds,\n"
    "                                 div, MIN_FIGHTS_RANKING)\n"
    "        if len(ml_rank) > 0:\n"
    "            print(f'\\n{div.upper()} (Top 10) :')\n"
    "            print(ml_rank.head(10).to_string(index=False))"
))

c.append(nbf.v4.new_code_cell(
    "# ─── 3. Elo dynamique ───────────────────────────────────────────────────\n"
    "print('=== ELO DYNAMIQUE ===')\n"
    "elo_final, elo_history = compute_elo(df)\n"
    "\n"
    "if WEIGHT_CLASS_FILTER:\n"
    "    elo_rank = elo_ranking(elo_final, WEIGHT_CLASS_FILTER, elo_history)\n"
    "    print(f'Division : {WEIGHT_CLASS_FILTER}')\n"
    "    print(elo_rank.head(15).to_string(index=False))\n"
    "else:\n"
    "    elo_rank_global = elo_ranking(elo_final)\n"
    "    print('Top 20 Elo global :')\n"
    "    print(elo_rank_global.head(20).to_string(index=False))"
))

c.append(nbf.v4.new_code_cell(
    "# Évolution Elo des légendes\n"
    "legends = ['Khabib Nurmagomedov', 'Jon Jones', 'Anderson Silva',\n"
    "           'Georges St-Pierre', 'Conor McGregor', 'Israel Adesanya']\n"
    "legends_in_data = [f for f in legends if f in elo_history['Fighter'].values]\n"
    "\n"
    "if legends_in_data:\n"
    "    plt.figure(figsize=(13, 6))\n"
    "    for fighter in legends_in_data:\n"
    "        fh = elo_history[elo_history['Fighter'] == fighter].sort_values('date')\n"
    "        plt.plot(fh['date'], fh['Elo'], marker='o', markersize=3, label=fighter)\n"
    "    plt.axhline(1500, linestyle='--', color='gray', label='Rating initial')\n"
    "    plt.title('Évolution du rating Elo — Légendes UFC')\n"
    "    plt.xlabel('Date')\n"
    "    plt.ylabel('Elo')\n"
    "    plt.legend()\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "else:\n"
    "    print('Fighters legendes non trouvés dans le dataset.')"
))

c.append(nbf.v4.new_code_cell(
    "# ─── 4. Comparaison avec les classements officiels UFC ──────────────────\n"
    "print('=== COMPARAISON VS CLASSEMENTS OFFICIELS UFC ===')\n"
    "\n"
    "# Elo ranking global — comparaison\n"
    "elo_rank_df = elo_ranking(elo_final)\n"
    "result = compare_with_official(elo_rank_df, df)\n"
    "\n"
    "if 'error' in result:\n"
    "    print(f'Erreur : {result[\"error\"]}')\n"
    "elif 'message' in result:\n"
    "    print(result['message'])\n"
    "else:\n"
    "    print(f'Fighters comparables : {result[\"n_fighters\"]}')\n"
    "    print(f'Spearman rho  : {result[\"spearman_rho\"]}  (p={result[\"p_spearman\"]})')\n"
    "    print(f'Kendall tau   : {result[\"kendall_tau\"]}  (p={result[\"p_kendall\"]})')\n"
    "    print()\n"
    "    if result['spearman_rho'] > 0:\n"
    "        print('Corrélation positive avec le classement officiel UFC.')\n"
    "    else:\n"
    "        print('Pas de corrélation significative avec le classement officiel.')\n"
    "    if 'fighters' in result and len(result['fighters']) > 0:\n"
    "        print('\\nDétail des fighters comparés :')\n"
    "        print(result['fighters'].to_string(index=False))"
))

c.append(nbf.v4.new_markdown_cell(
    "---\n"
    "## Fin du pipeline\n\n"
    "Pour relancer le pipeline sur de nouvelles données, exécutez à nouveau ce notebook. "
    "Seuls les combats postérieurs au dernier run seront scrapés et intégrés.\n\n"
    "Pour modifier les poids du classement, éditez le dict `WEIGHTS` en Section 0."
))

nb.cells = c
output_path = os.path.join(ROOT, "UFC_Pipeline.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook créé : {output_path}")
