"""Script de génération des 3 notebooks Jupyter du projet UFC ML."""
import nbformat as nbf
import os

os.makedirs("notebooks", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 01 — EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────

def make_nb01():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    c = []

    c.append(nbf.v4.new_markdown_cell(
        "# 01 — Exploration des données UFC\n\n"
        "Objectif : comprendre la structure, les distributions, les valeurs manquantes "
        "et identifier les features prometteuses pour le ML.\n\n"
        "**Source :** `data/processed/ufc_master_enriched.csv` — 7 727 combats × 111 colonnes"
    ))

    c.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "sns.set_theme(style='darkgrid', palette='muted')\n"
        "plt.rcParams['figure.figsize'] = (12, 5)\n\n"
        "df = pd.read_csv('../data/processed/ufc_master_enriched.csv')\n"
        "df['date'] = pd.to_datetime(df['date'], errors='coerce')\n"
        "df = df.sort_values('date').reset_index(drop=True)\n\n"
        "print(f'Shape : {df.shape}')\n"
        "print(f'Période : {df[\"date\"].min().date()} → {df[\"date\"].max().date()}')\n"
        "df.head(3)"
    ))

    c.append(nbf.v4.new_markdown_cell("## 1. Distribution de la cible (R_Win)\n\n"
        "Le coin rouge a un avantage historique connu (~57%). Ce biais doit être pris en compte."))

    c.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# Distribution globale\n"
        "win_counts = df['R_Win'].value_counts()\n"
        "axes[0].bar(['Blue wins', 'Red wins'], [win_counts.get(0,0), win_counts.get(1,0)],\n"
        "            color=['steelblue', 'tomato'])\n"
        "axes[0].set_title('Distribution des victoires globale')\n"
        "for i, v in enumerate([win_counts.get(0,0), win_counts.get(1,0)]):\n"
        "    axes[0].text(i, v + 20, f'{v/len(df)*100:.1f}%', ha='center', fontweight='bold')\n\n"
        "# Par année\n"
        "df['year'] = df['date'].dt.year\n"
        "yearly = df.groupby('year')['R_Win'].agg(['mean','count']).reset_index()\n"
        "yearly = yearly[yearly['count'] >= 10]\n"
        "axes[1].plot(yearly['year'], yearly['mean'] * 100, marker='o', color='tomato')\n"
        "axes[1].axhline(50, linestyle='--', color='gray', label='50%')\n"
        "axes[1].set_title('Taux de victoire Rouge par année')\n"
        "axes[1].set_ylabel('% victoires Rouge')\n"
        "axes[1].legend()\n\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'Taux de victoire Rouge global : {df[\"R_Win\"].mean()*100:.1f}%')"
    ))

    c.append(nbf.v4.new_markdown_cell("## 2. Valeurs manquantes"))

    c.append(nbf.v4.new_code_cell(
        "missing = (df.isnull().mean() * 100).sort_values(ascending=False)\n"
        "missing = missing[missing > 0]\n\n"
        "plt.figure(figsize=(14, 6))\n"
        "missing.head(40).plot(kind='bar', color='coral')\n"
        "plt.title('% de valeurs manquantes par colonne (top 40)')\n"
        "plt.ylabel('% manquant')\n"
        "plt.xticks(rotation=60, ha='right')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('Colonnes > 50% manquantes :')\n"
        "print(missing[missing > 50].to_string())"
    ))

    c.append(nbf.v4.new_markdown_cell("## 3. Distribution par catégorie de poids"))

    c.append(nbf.v4.new_code_cell(
        "weight_counts = df['WeightClass'].value_counts().dropna()\n\n"
        "plt.figure(figsize=(14, 5))\n"
        "weight_counts.plot(kind='bar', color='steelblue')\n"
        "plt.title('Nombre de combats par catégorie de poids')\n"
        "plt.ylabel('Nb combats')\n"
        "plt.xticks(rotation=45, ha='right')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "# Taux victoire rouge par division\n"
        "rd = df.groupby('WeightClass')['R_Win'].agg(['mean','count']).reset_index()\n"
        "rd = rd[rd['count'] >= 20].sort_values('mean', ascending=False)\n"
        "rd.columns = ['WeightClass','R_win_rate','n_fights']\n"
        "print(rd.to_string(index=False))"
    ))

    c.append(nbf.v4.new_markdown_cell("## 4. Attributs physiques (taille, allonge, poids)"))

    c.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n"
        "axes = axes.flatten()\n"
        "cols = [('R_Height_cms','Taille Red (cm)'),('R_Reach_cms','Allonge Red (cm)'),\n"
        "        ('R_Weight_lbs','Poids Red (lbs)'),('B_Height_cms','Taille Blue (cm)'),\n"
        "        ('B_Reach_cms','Allonge Blue (cm)'),('B_Weight_lbs','Poids Blue (lbs)')]\n"
        "for ax, (col, title) in zip(axes, cols):\n"
        "    data = df[col].dropna()\n"
        "    ax.hist(data, bins=40, color='steelblue', edgecolor='white')\n"
        "    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Moy={data.mean():.0f}')\n"
        "    ax.set_title(title); ax.legend(fontsize=8)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "# Allonge vs Taille\n"
        "plt.figure(figsize=(7, 5))\n"
        "plt.scatter(df['R_Height_cms'], df['R_Reach_cms'], alpha=0.2, s=10, color='steelblue')\n"
        "plt.xlabel('Taille (cm)'); plt.ylabel('Allonge (cm)')\n"
        "plt.title('Corrélation Taille / Allonge (coin Rouge)')\n"
        "r = df[['R_Height_cms','R_Reach_cms']].corr().iloc[0,1]\n"
        "plt.text(0.05, 0.93, f'r = {r:.2f}', transform=plt.gca().transAxes, fontsize=12)\n"
        "plt.tight_layout(); plt.show()"
    ))

    c.append(nbf.v4.new_markdown_cell("## 5. Corrélation des features numériques avec R_Win"))

    c.append(nbf.v4.new_code_cell(
        "num_cols = df.select_dtypes(include=np.number).columns.tolist()\n"
        "exclude = ['R_Win','B_Win','year','R_Status','B_Status']\n"
        "num_cols = [c for c in num_cols if c not in exclude]\n\n"
        "corr = df[num_cols + ['R_Win']].corr()['R_Win'].drop('R_Win')\n"
        "corr_top = corr.abs().sort_values(ascending=False).head(25)\n"
        "vals = corr[corr_top.index].sort_values()\n\n"
        "plt.figure(figsize=(12, 7))\n"
        "colors = ['tomato' if v > 0 else 'steelblue' for v in vals]\n"
        "vals.plot(kind='barh', color=colors)\n"
        "plt.title('Top 25 features — corrélation avec R_Win')\n"
        "plt.axvline(0, color='black', linewidth=0.8)\n"
        "plt.tight_layout(); plt.show()"
    ))

    c.append(nbf.v4.new_markdown_cell("## 6. Baseline : calibration des cotes de paris\n\n"
        "Le modèle ML doit dépasser l'accuracy des bookmakers pour apporter de la valeur."))

    c.append(nbf.v4.new_code_cell(
        "df_odds = df[['RedOdds','BlueOdds','R_Win']].dropna()\n\n"
        "def american_to_prob(odds):\n"
        "    odds = pd.to_numeric(odds, errors='coerce')\n"
        "    return np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100))\n\n"
        "p_r = american_to_prob(df_odds['RedOdds'])\n"
        "pred = (p_r > 0.5).astype(int)\n"
        "baseline_acc = (pred == df_odds['R_Win'].values).mean()\n\n"
        "print(f'Combats avec cotes : {len(df_odds)}')\n"
        "print(f'Accuracy baseline (cotes) : {baseline_acc*100:.1f}%')\n"
        "print('→ Le modèle ML doit dépasser ce seuil.')\n\n"
        "plt.figure(figsize=(10, 4))\n"
        "plt.hist(p_r, bins=50, color='tomato', edgecolor='white', alpha=0.7)\n"
        "plt.axvline(0.5, linestyle='--', color='black')\n"
        "plt.title('Distribution des probabilités implicites des cotes (coin Rouge)')\n"
        "plt.xlabel('Probabilité implicite')\n"
        "plt.tight_layout(); plt.show()"
    ))

    c.append(nbf.v4.new_markdown_cell("## 7. Volume de combats par année et justification du temporal split"))

    c.append(nbf.v4.new_code_cell(
        "yearly_counts = df.groupby('year').size()\n\n"
        "plt.figure(figsize=(12, 4))\n"
        "yearly_counts.plot(kind='bar', color='steelblue', edgecolor='white')\n"
        "plt.title('Nombre de combats par année')\n"
        "plt.xticks(rotation=45)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "n = len(df)\n"
        "print(f'Temporal split prévu (pas de random shuffle pour éviter le leakage) :')\n"
        "print(f'  Train (70%) : {int(n*0.70)} combats → jusqu\\'au {df.iloc[int(n*0.70)][\"date\"].date()}')\n"
        "print(f'  Val   (15%) : {int(n*0.15)} combats → jusqu\\'au {df.iloc[int(n*0.85)][\"date\"].date()}')\n"
        "print(f'  Test  (15%) : {int(n*0.15)} combats → jusqu\\'au {df[\"date\"].max().date()}')"
    ))

    c.append(nbf.v4.new_markdown_cell("## 8. Distribution des méthodes de victoire"))

    c.append(nbf.v4.new_code_cell(
        "methods = df['method'].value_counts().head(15)\n\n"
        "plt.figure(figsize=(12, 5))\n"
        "methods.plot(kind='bar', color='steelblue', edgecolor='white')\n"
        "plt.title('Top 15 méthodes de victoire')\n"
        "plt.xticks(rotation=45, ha='right')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "total = df['method'].notna().sum()\n"
        "for label, pattern in [('KO/TKO','KO|TKO'),('Submission','Submission'),('Décision','Decision')]:\n"
        "    count = df['method'].str.contains(pattern, case=False, na=False).sum()\n"
        "    print(f'{label:15s} : {count/total*100:.1f}%')"
    ))

    c.append(nbf.v4.new_markdown_cell("## 9. Top fighters — volume et win rate"))

    c.append(nbf.v4.new_code_cell(
        "r = df[['R_Fighter','R_Win']].rename(columns={'R_Fighter':'Fighter','R_Win':'Win'})\n"
        "b = df[['B_Fighter','B_Win']].rename(columns={'B_Fighter':'Fighter','B_Win':'Win'})\n"
        "apps = pd.concat([r, b], ignore_index=True)\n\n"
        "summary = apps.groupby('Fighter').agg(n_fights=('Win','count'), wins=('Win','sum')).reset_index()\n"
        "summary['win_rate'] = (summary['wins'] / summary['n_fights']).round(3)\n"
        "summary = summary[summary['n_fights'] >= 5]\n\n"
        "print('Top 15 fighters par nombre de combats UFC :')\n"
        "print(summary.sort_values('n_fights', ascending=False).head(15).to_string(index=False))\n\n"
        "print('\\nTop 15 fighters par win rate (min. 10 combats) :')\n"
        "print(summary[summary['n_fights'] >= 10].sort_values('win_rate', ascending=False).head(15).to_string(index=False))"
    ))

    nb.cells = c
    with open("notebooks/01_exploration.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("✓ notebooks/01_exploration.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 02 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def make_nb02():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    c = []

    c.append(nbf.v4.new_markdown_cell(
        "# 02 — Feature Engineering\n\n"
        "Objectif : construire le vecteur de features **sans data leakage** pour chaque combat.\n\n"
        "**Règle d'or :** pour un combat à la date T, les features = statistiques cumulées "
        "uniquement sur les combats **antérieurs** à T.\n\n"
        "**Technique :** `expanding().mean().shift(1)` par fighter trié chronologiquement.\n\n"
        "**Sortie :** `data/processed/features_dataset.csv`"
    ))

    c.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "df = pd.read_csv('../data/processed/ufc_master_enriched.csv')\n"
        "df['date'] = pd.to_datetime(df['date'], errors='coerce')\n"
        "df = df.sort_values('date').reset_index(drop=True)\n"
        "print(f'Shape source : {df.shape}')"
    ))

    c.append(nbf.v4.new_markdown_cell("## Étape 1 — Long format : une ligne par (fighter, combat)"))

    c.append(nbf.v4.new_code_cell(
        "STAT_COLS = [\n"
        "    'KD', 'SIG_STR_landed', 'SIG_STR_attempted',\n"
        "    'TOTAL_STR_landed', 'TOTAL_STR_attempted',\n"
        "    'TD_landed', 'TD_attempted',\n"
        "    'HEAD_landed', 'HEAD_attempted',\n"
        "    'BODY_landed', 'BODY_attempted',\n"
        "    'LEG_landed', 'LEG_attempted',\n"
        "    'DISTANCE_landed', 'DISTANCE_attempted',\n"
        "    'CLINCH_landed', 'CLINCH_attempted',\n"
        "    'GROUND_landed', 'GROUND_attempted',\n"
        "    'CTRL_sec', 'SUB_ATT', 'REV',\n"
        "]\n\n"
        "def create_appearances(df):\n"
        "    meta_cols = ['date', 'WeightClass', 'TotalFightTimeSecs', 'method']\n"
        "    r_cols = [f'R_{c}' for c in STAT_COLS if f'R_{c}' in df.columns]\n"
        "    b_cols = [f'B_{c}' for c in STAT_COLS if f'B_{c}' in df.columns]\n"
        "    actual_stat_cols = [c for c in STAT_COLS if f'R_{c}' in df.columns]\n\n"
        "    r_rows = df[['R_Fighter'] + meta_cols + ['R_Win'] + r_cols].copy()\n"
        "    r_rows.columns = ['Fighter'] + meta_cols + ['Win'] + actual_stat_cols\n"
        "    r_rows['Opponent'] = df['B_Fighter'].values\n\n"
        "    b_rows = df[['B_Fighter'] + meta_cols + ['B_Win'] + b_cols].copy()\n"
        "    b_rows.columns = ['Fighter'] + meta_cols + ['Win'] + actual_stat_cols\n"
        "    b_rows['Opponent'] = df['R_Fighter'].values\n\n"
        "    apps = pd.concat([r_rows, b_rows], ignore_index=True)\n"
        "    return apps.sort_values(['Fighter', 'date']).reset_index(drop=True)\n\n"
        "apps = create_appearances(df)\n"
        "print(f'Long format : {apps.shape}')\n"
        "apps.head(3)"
    ))

    c.append(nbf.v4.new_markdown_cell("## Étape 2 — Normalisation par minute + encodage des finishs"))

    c.append(nbf.v4.new_code_cell(
        "def normalize_stats(apps):\n"
        "    apps = apps.copy()\n"
        "    minutes = apps['TotalFightTimeSecs'].clip(lower=1).fillna(900) / 60.0\n\n"
        "    vol_cols = ['KD','SIG_STR_landed','SIG_STR_attempted','TOTAL_STR_landed',\n"
        "                'TD_landed','TD_attempted','HEAD_landed','BODY_landed',\n"
        "                'LEG_landed','CTRL_sec','SUB_ATT']\n"
        "    for col in vol_cols:\n"
        "        if col in apps.columns:\n"
        "            apps[f'{col}_pm'] = apps[col] / minutes\n\n"
        "    # Précision (ratio, pas de normalisation par temps)\n"
        "    apps['SIG_STR_acc'] = apps['SIG_STR_landed'] / apps['SIG_STR_attempted'].clip(lower=1)\n"
        "    apps['TD_acc']      = apps['TD_landed']      / apps['TD_attempted'].clip(lower=1)\n"
        "    apps['HEAD_rate']   = apps['HEAD_landed']    / apps['SIG_STR_landed'].clip(lower=1)\n"
        "    apps['BODY_rate']   = apps['BODY_landed']    / apps['SIG_STR_landed'].clip(lower=1)\n"
        "    apps['LEG_rate']    = apps['LEG_landed']     / apps['SIG_STR_landed'].clip(lower=1)\n\n"
        "    # Méthode de victoire\n"
        "    m = apps['method'].fillna('')\n"
        "    apps['is_finish'] = m.str.contains('KO|TKO|Submission', case=False).astype(float)\n"
        "    apps['is_KO']     = m.str.contains('KO|TKO',           case=False).astype(float)\n"
        "    apps['is_sub']    = m.str.contains('Submission',        case=False).astype(float)\n\n"
        "    return apps\n\n"
        "apps = normalize_stats(apps)\n"
        "print('Normalisation OK — nouvelles colonnes :', [c for c in apps.columns if '_pm' in c or 'acc' in c or 'rate' in c or 'is_' in c])"
    ))

    c.append(nbf.v4.new_markdown_cell(
        "## Étape 3 — Expanding window + shift(1) : stats cumulées AVANT chaque combat\n\n"
        "C'est le cœur de la prévention du data leakage.\n\n"
        "`shift(1)` garantit que pour le combat N, on utilise uniquement les N-1 combats précédents."
    ))

    c.append(nbf.v4.new_code_cell(
        "MEAN_COLS = [\n"
        "    'KD_pm','SIG_STR_landed_pm','SIG_STR_attempted_pm','TOTAL_STR_landed_pm',\n"
        "    'TD_landed_pm','TD_attempted_pm','HEAD_landed_pm','BODY_landed_pm','LEG_landed_pm',\n"
        "    'CTRL_sec_pm','SUB_ATT_pm',\n"
        "    'SIG_STR_acc','TD_acc','HEAD_rate','BODY_rate','LEG_rate',\n"
        "    'Win','is_finish','is_KO','is_sub',\n"
        "]\n"
        "MEAN_COLS = [c for c in MEAN_COLS if c in apps.columns]\n\n"
        "def compute_prelagged_stats(apps):\n"
        "    results = []\n"
        "    for fighter, grp in apps.groupby('Fighter'):\n"
        "        g = grp.sort_values('date').copy()\n"
        "        # Expanding mean + shift → stats cumulées AVANT ce combat\n"
        "        for col in MEAN_COLS:\n"
        "            g[f'avg_{col}'] = g[col].expanding().mean().shift(1)\n"
        "        # Forme récente : 3 derniers combats\n"
        "        for col in ['Win','KD_pm','SIG_STR_landed_pm']:\n"
        "            if col in g.columns:\n"
        "                g[f'recent3_{col}'] = g[col].rolling(3, min_periods=1).mean().shift(1)\n"
        "        g['recent_wins_3'] = g['Win'].rolling(3, min_periods=1).sum().shift(1)\n"
        "        # Expérience (nb combats précédents)\n"
        "        g['n_fights_before'] = np.arange(len(g), dtype=float)\n"
        "        results.append(g)\n"
        "    return pd.concat(results, ignore_index=True).sort_values(['date','Fighter'])\n\n"
        "print('Calcul des stats pré-combat (quelques secondes)...')\n"
        "apps_stats = compute_prelagged_stats(apps)\n"
        "print(f'Done. Shape : {apps_stats.shape}')\n"
        "apps_stats[['Fighter','date','n_fights_before','avg_Win','avg_SIG_STR_landed_pm']].head(8)"
    ))

    c.append(nbf.v4.new_markdown_cell("## Étape 4 — Reconstruction en format combat + vecteur delta (R - B)"))

    c.append(nbf.v4.new_code_cell(
        "# Séparer Red et Blue depuis la table long\n"
        "pre_cols = [c for c in apps_stats.columns\n"
        "            if c.startswith('avg_') or c.startswith('recent') or c == 'n_fights_before']\n\n"
        "red_stats  = apps_stats.set_index(['Fighter','date'])[pre_cols].add_prefix('R_pre_')\n"
        "blue_stats = apps_stats.set_index(['Fighter','date'])[pre_cols].add_prefix('B_pre_')\n\n"
        "# Joindre sur le dataframe des combats\n"
        "meta = df[['R_Fighter','B_Fighter','date','WeightClass','R_Win',\n"
        "           'R_Height_cms','R_Reach_cms','R_Weight_lbs','R_Stance','R_DOB',\n"
        "           'B_Height_cms','B_Reach_cms','B_Weight_lbs','B_Stance','B_DOB',\n"
        "           'RMatchWCRank','BMatchWCRank','RedOdds','BlueOdds','EmptyArena']].copy()\n\n"
        "meta = meta.join(red_stats,  on=['R_Fighter','date'], how='left')\n"
        "meta = meta.join(blue_stats, on=['B_Fighter','date'], how='left')\n\n"
        "print(f'Shape après jointure : {meta.shape}')\n"
        "meta.head(2)"
    ))

    c.append(nbf.v4.new_code_cell(
        "def build_delta_features(meta):\n"
        "    feat = pd.DataFrame(index=meta.index)\n\n"
        "    # Delta des stats cumulées (R - B)\n"
        "    r_pre = [c for c in meta.columns if c.startswith('R_pre_')]\n"
        "    for rc in r_pre:\n"
        "        bc = rc.replace('R_pre_', 'B_pre_')\n"
        "        name = rc.replace('R_pre_', 'delta_')\n"
        "        if bc in meta.columns:\n"
        "            feat[name] = meta[rc] - meta[bc]\n\n"
        "    # Attributs physiques delta\n"
        "    feat['delta_height'] = meta['R_Height_cms'] - meta['B_Height_cms']\n"
        "    feat['delta_reach']  = meta['R_Reach_cms']  - meta['B_Reach_cms']\n\n"
        "    # Age au moment du combat\n"
        "    date = pd.to_datetime(meta['date'])\n"
        "    r_dob = pd.to_datetime(meta['R_DOB'], errors='coerce')\n"
        "    b_dob = pd.to_datetime(meta['B_DOB'], errors='coerce')\n"
        "    feat['R_age']      = (date - r_dob).dt.days / 365.25\n"
        "    feat['B_age']      = (date - b_dob).dt.days / 365.25\n"
        "    feat['delta_age']  = feat['R_age'] - feat['B_age']\n\n"
        "    # Stance\n"
        "    feat['R_is_southpaw'] = (meta['R_Stance'] == 'Southpaw').astype(float)\n"
        "    feat['B_is_southpaw'] = (meta['B_Stance'] == 'Southpaw').astype(float)\n"
        "    feat['same_stance']   = (meta['R_Stance'] == meta['B_Stance']).astype(float)\n\n"
        "    # Classement officiel\n"
        "    rr = pd.to_numeric(meta['RMatchWCRank'], errors='coerce').fillna(99)\n"
        "    br = pd.to_numeric(meta['BMatchWCRank'], errors='coerce').fillna(99)\n"
        "    feat['delta_rank']   = rr - br\n"
        "    feat['R_is_ranked']  = (pd.to_numeric(meta['RMatchWCRank'], errors='coerce').notna()).astype(float)\n"
        "    feat['B_is_ranked']  = (pd.to_numeric(meta['BMatchWCRank'], errors='coerce').notna()).astype(float)\n\n"
        "    # Cotes → probabilité implicite\n"
        "    def ato_prob(odds):\n"
        "        o = pd.to_numeric(odds, errors='coerce')\n"
        "        return np.where(o > 0, 100/(o+100), -o/(-o+100))\n"
        "    feat['R_implied_prob']    = ato_prob(meta['RedOdds'])\n"
        "    feat['delta_implied_prob']= ato_prob(meta['RedOdds']) - ato_prob(meta['BlueOdds'])\n\n"
        "    # Contexte\n"
        "    feat['empty_arena'] = pd.to_numeric(meta['EmptyArena'], errors='coerce').fillna(0)\n\n"
        "    # Catégorie de poids (ordinal)\n"
        "    wc_map = {'Strawweight':1,'Flyweight':2,'Bantamweight':3,'Featherweight':4,\n"
        "              'Lightweight':5,'Welterweight':6,'Middleweight':7,'Light Heavyweight':8,'Heavyweight':9}\n"
        "    feat['weight_class_ord'] = meta['WeightClass'].map(\n"
        "        lambda x: next((v for k,v in wc_map.items() if k.lower() in str(x).lower()), 5)\n"
        "    )\n\n"
        "    # Métadonnées (non utilisées comme features)\n"
        "    feat['R_Fighter']    = meta['R_Fighter'].values\n"
        "    feat['B_Fighter']    = meta['B_Fighter'].values\n"
        "    feat['date']         = meta['date'].values\n"
        "    feat['WeightClass']  = meta['WeightClass'].values\n"
        "    feat['R_Win']        = meta['R_Win'].values\n"
        "    feat['n_fights_R']   = meta['R_pre_n_fights_before'].values\n"
        "    feat['n_fights_B']   = meta['B_pre_n_fights_before'].values\n\n"
        "    return feat\n\n"
        "features = build_delta_features(meta)\n"
        "print(f'Shape features brutes : {features.shape}')\n"
        "features.head(3)"
    ))

    c.append(nbf.v4.new_markdown_cell("## Étape 5 — Nettoyage final et sauvegarde"))

    c.append(nbf.v4.new_code_cell(
        "# Filtre : garder les combats où R et B ont chacun au moins 1 combat UFC précédent\n"
        "features_clean = features[\n"
        "    (features['n_fights_R'] >= 1) & (features['n_fights_B'] >= 1)\n"
        "].copy()\n\n"
        "print(f'Combats filtrés (>= 1 combat précédent chacun) : {len(features_clean)} / {len(features)}')\n"
        "print(f'Cible R_Win : {features_clean[\"R_Win\"].mean()*100:.1f}% rouge')\n\n"
        "# Colonnes features (delta + physique + contexte)\n"
        "FEAT_COLS = [c for c in features_clean.columns\n"
        "             if c.startswith('delta_') or\n"
        "                c in ['R_age','B_age','R_is_southpaw','B_is_southpaw','same_stance',\n"
        "                      'R_is_ranked','B_is_ranked','delta_rank',\n"
        "                      'empty_arena','weight_class_ord',\n"
        "                      'R_implied_prob','delta_implied_prob']]\n"
        "print(f'\\nNombre de features : {len(FEAT_COLS)}')\n"
        "print('Features :')\n"
        "for f in FEAT_COLS:\n"
        "    null_pct = features_clean[f].isnull().mean()*100\n"
        "    print(f'  {f:45s}  NaN: {null_pct:.1f}%')"
    ))

    c.append(nbf.v4.new_code_cell(
        "# Sauvegarde du dataset complet (avec cotes)\n"
        "out_path = '../data/processed/features_dataset.csv'\n"
        "features_clean.to_csv(out_path, index=False)\n"
        "print(f'Sauvegardé : {out_path}')\n"
        "print(f'Shape finale : {features_clean.shape}')\n\n"
        "# Version sans cotes (pour le ranking indépendant du marché)\n"
        "no_odds_cols = [c for c in features_clean.columns\n"
        "                if 'implied_prob' not in c and 'Odds' not in c]\n"
        "features_no_odds = features_clean[no_odds_cols]\n"
        "out_path2 = '../data/processed/features_dataset_no_odds.csv'\n"
        "features_no_odds.to_csv(out_path2, index=False)\n"
        "print(f'Sauvegardé (sans cotes) : {out_path2}')"
    ))

    c.append(nbf.v4.new_markdown_cell("## Visualisation des features les plus corrélées avec R_Win"))

    c.append(nbf.v4.new_code_cell(
        "corr = features_clean[FEAT_COLS + ['R_Win']].corr()['R_Win'].drop('R_Win')\n"
        "top = corr.abs().sort_values(ascending=False).head(20)\n"
        "vals = corr[top.index].sort_values()\n\n"
        "plt.figure(figsize=(11, 6))\n"
        "colors = ['tomato' if v > 0 else 'steelblue' for v in vals]\n"
        "vals.plot(kind='barh', color=colors)\n"
        "plt.title('Top 20 features delta — corrélation avec R_Win')\n"
        "plt.axvline(0, color='black', linewidth=0.8)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('Feature la plus prédictive :', top.index[0], f'({top.iloc[0]:.3f})')"
    ))

    nb.cells = c
    with open("notebooks/02_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("✓ notebooks/02_feature_engineering.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 03 — ML MODELS + RANKING
# ─────────────────────────────────────────────────────────────────────────────

def make_nb03():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    c = []

    c.append(nbf.v4.new_markdown_cell(
        "# 03 — Modèles ML & Ranking UFC\n\n"
        "**Pipeline :**\n"
        "1. Temporal split (train/val/test chronologique)\n"
        "2. Entraînement : Logistic Regression, SVM, XGBoost\n"
        "3. Comparaison + calibration des probabilités\n"
        "4. Génération du classement par division\n\n"
        "**Source :** `data/processed/features_dataset.csv`"
    ))

    c.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.svm import SVC\n"
        "from sklearn.model_selection import GridSearchCV, TimeSeriesSplit\n"
        "from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,\n"
        "                              log_loss, classification_report, roc_curve,\n"
        "                              ConfusionMatrixDisplay, calibration_curve)\n"
        "import xgboost as xgb\n\n"
        "sns.set_theme(style='darkgrid')\n"
        "plt.rcParams['figure.figsize'] = (12, 5)\n\n"
        "df = pd.read_csv('../data/processed/features_dataset.csv')\n"
        "df['date'] = pd.to_datetime(df['date'])\n"
        "df = df.sort_values('date').reset_index(drop=True)\n"
        "print(f'Shape : {df.shape}')\n"
        "print(f'Période : {df[\"date\"].min().date()} → {df[\"date\"].max().date()}')"
    ))

    c.append(nbf.v4.new_markdown_cell(
        "## 1. Temporal split\n\n"
        "**Règle :** jamais de `random_state` sur le split — on respecte l'ordre chronologique "
        "pour simuler une vraie mise en production (on prédit le futur depuis le passé)."
    ))

    c.append(nbf.v4.new_code_cell(
        "FEAT_COLS = [c for c in df.columns\n"
        "             if c.startswith('delta_') or\n"
        "                c in ['R_age','B_age','R_is_southpaw','B_is_southpaw','same_stance',\n"
        "                      'R_is_ranked','B_is_ranked','delta_rank',\n"
        "                      'empty_arena','weight_class_ord',\n"
        "                      'R_implied_prob','delta_implied_prob']]\n\n"
        "# Sans cotes (pour le ranking indépendant)\n"
        "FEAT_NO_ODDS = [c for c in FEAT_COLS if 'implied_prob' not in c]\n\n"
        "N = len(df)\n"
        "train_end = int(N * 0.70)\n"
        "val_end   = int(N * 0.85)\n\n"
        "train = df.iloc[:train_end]\n"
        "val   = df.iloc[train_end:val_end]\n"
        "test  = df.iloc[val_end:]\n\n"
        "print(f'Train : {len(train):5d} combats  ({train[\"date\"].min().date()} → {train[\"date\"].max().date()})')\n"
        "print(f'Val   : {len(val):5d} combats  ({val[\"date\"].min().date()} → {val[\"date\"].max().date()})')\n"
        "print(f'Test  : {len(test):5d} combats  ({test[\"date\"].min().date()} → {test[\"date\"].max().date()})')\n"
        "print(f'\\nNombre de features (avec cotes)   : {len(FEAT_COLS)}')\n"
        "print(f'Nombre de features (sans cotes)   : {len(FEAT_NO_ODDS)}')\n\n"
        "X_tr  = train[FEAT_COLS].values;  y_tr  = train['R_Win'].values\n"
        "X_val = val[FEAT_COLS].values;    y_val = val['R_Win'].values\n"
        "X_te  = test[FEAT_COLS].values;   y_te  = test['R_Win'].values"
    ))

    c.append(nbf.v4.new_markdown_cell("## 2. Preprocessing commun"))

    c.append(nbf.v4.new_code_cell(
        "preproc = Pipeline([\n"
        "    ('imp',   SimpleImputer(strategy='median')),\n"
        "    ('scale', StandardScaler()),\n"
        "])\n\n"
        "# Fit UNIQUEMENT sur le train\n"
        "X_tr_p  = preproc.fit_transform(X_tr)\n"
        "X_val_p = preproc.transform(X_val)\n"
        "X_te_p  = preproc.transform(X_te)\n\n"
        "print('Preprocessing OK')\n"
        "print(f'  Train  : {X_tr_p.shape}')\n"
        "print(f'  Val    : {X_val_p.shape}')\n"
        "print(f'  Test   : {X_te_p.shape}')"
    ))

    c.append(nbf.v4.new_markdown_cell("## 3. Baseline — Cotes de paris"))

    c.append(nbf.v4.new_code_cell(
        "# Baseline : prédire R_Win = 1 si la cote implicite de R > 50%\n"
        "df_odds = test[test['R_implied_prob'].notna()].copy()\n"
        "if len(df_odds) > 0:\n"
        "    pred_odds = (df_odds['R_implied_prob'] > 0.5).astype(int)\n"
        "    baseline_acc = accuracy_score(df_odds['R_Win'], pred_odds)\n"
        "    baseline_auc = roc_auc_score(df_odds['R_Win'], df_odds['R_implied_prob'])\n"
        "    print(f'Baseline cotes — Accuracy : {baseline_acc*100:.1f}%  |  AUC : {baseline_auc:.3f}')\n"
        "    print('→ Objectif : dépasser ces valeurs avec le ML')\n"
        "else:\n"
        "    print('Pas de cotes disponibles sur le test set (combats anciens)')"
    ))

    c.append(nbf.v4.new_markdown_cell("## 4. Logistic Regression"))

    c.append(nbf.v4.new_code_cell(
        "tscv = TimeSeriesSplit(n_splits=5)\n\n"
        "lr_grid = GridSearchCV(\n"
        "    LogisticRegression(max_iter=1000, random_state=42),\n"
        "    {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1','l2'],\n"
        "     'solver': ['liblinear']},\n"
        "    cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=0\n"
        ")\n"
        "lr_grid.fit(X_tr_p, y_tr)\n"
        "lr = lr_grid.best_estimator_\n"
        "print(f'Meilleurs params : {lr_grid.best_params_}')\n\n"
        "lr_val_proba = lr.predict_proba(X_val_p)[:,1]\n"
        "lr_te_proba  = lr.predict_proba(X_te_p)[:,1]\n\n"
        "print(f'Val  → Acc: {accuracy_score(y_val, lr_val_proba>0.5)*100:.1f}%  AUC: {roc_auc_score(y_val, lr_val_proba):.3f}')\n"
        "print(f'Test → Acc: {accuracy_score(y_te,  lr_te_proba >0.5)*100:.1f}%  AUC: {roc_auc_score(y_te,  lr_te_proba):.3f}')\n\n"
        "# Coefficients\n"
        "coef_df = pd.DataFrame({'feature': FEAT_COLS, 'coef': lr.coef_[0]})\n"
        "coef_df['abs'] = coef_df['coef'].abs()\n"
        "top_coef = coef_df.sort_values('abs', ascending=False).head(15)\n\n"
        "plt.figure(figsize=(11, 5))\n"
        "colors = ['tomato' if v > 0 else 'steelblue' for v in top_coef.sort_values('coef')['coef']]\n"
        "top_coef.sort_values('coef')['coef'].plot(kind='barh', color=colors,\n"
        "    index=top_coef.sort_values('coef')['feature'].values)\n"
        "plt.title('Logistic Regression — Top 15 coefficients')\n"
        "plt.axvline(0, color='black', linewidth=0.8)\n"
        "plt.tight_layout(); plt.show()"
    ))

    c.append(nbf.v4.new_markdown_cell("## 5. SVM"))

    c.append(nbf.v4.new_code_cell(
        "svm_grid = GridSearchCV(\n"
        "    SVC(probability=True, random_state=42),\n"
        "    {'C': [0.1, 1, 10], 'kernel': ['rbf','linear'], 'gamma': ['scale']},\n"
        "    cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=0\n"
        ")\n"
        "svm_grid.fit(X_tr_p, y_tr)\n"
        "svm = svm_grid.best_estimator_\n"
        "print(f'Meilleurs params : {svm_grid.best_params_}')\n\n"
        "svm_val_proba = svm.predict_proba(X_val_p)[:,1]\n"
        "svm_te_proba  = svm.predict_proba(X_te_p)[:,1]\n\n"
        "print(f'Val  → Acc: {accuracy_score(y_val, svm_val_proba>0.5)*100:.1f}%  AUC: {roc_auc_score(y_val, svm_val_proba):.3f}')\n"
        "print(f'Test → Acc: {accuracy_score(y_te,  svm_te_proba >0.5)*100:.1f}%  AUC: {roc_auc_score(y_te,  svm_te_proba):.3f}')"
    ))

    c.append(nbf.v4.new_markdown_cell("## 6. XGBoost"))

    c.append(nbf.v4.new_code_cell(
        "dtrain = xgb.DMatrix(X_tr_p,  label=y_tr,  feature_names=FEAT_COLS)\n"
        "dval   = xgb.DMatrix(X_val_p, label=y_val, feature_names=FEAT_COLS)\n"
        "dtest  = xgb.DMatrix(X_te_p,  label=y_te,  feature_names=FEAT_COLS)\n\n"
        "xgb_params = {\n"
        "    'objective': 'binary:logistic',\n"
        "    'eval_metric': ['auc','logloss'],\n"
        "    'max_depth': 4,\n"
        "    'learning_rate': 0.05,\n"
        "    'subsample': 0.8,\n"
        "    'colsample_bytree': 0.8,\n"
        "    'min_child_weight': 5,\n"
        "    'reg_lambda': 1.0,\n"
        "    'seed': 42,\n"
        "    'verbosity': 0,\n"
        "}\n"
        "model_xgb = xgb.train(\n"
        "    xgb_params, dtrain,\n"
        "    num_boost_round=500,\n"
        "    evals=[(dtrain,'train'),(dval,'val')],\n"
        "    early_stopping_rounds=30,\n"
        "    verbose_eval=100\n"
        ")\n\n"
        "xgb_val_proba = model_xgb.predict(dval)\n"
        "xgb_te_proba  = model_xgb.predict(dtest)\n\n"
        "print(f'Val  → Acc: {accuracy_score(y_val, xgb_val_proba>0.5)*100:.1f}%  AUC: {roc_auc_score(y_val, xgb_val_proba):.3f}')\n"
        "print(f'Test → Acc: {accuracy_score(y_te,  xgb_te_proba >0.5)*100:.1f}%  AUC: {roc_auc_score(y_te,  xgb_te_proba):.3f}')\n\n"
        "# Importance des features\n"
        "ax = xgb.plot_importance(model_xgb, max_num_features=20, importance_type='gain',\n"
        "                          height=0.5, figsize=(11,7))\n"
        "ax.set_title('XGBoost — Feature importance (gain)')\n"
        "plt.tight_layout(); plt.show()"
    ))

    c.append(nbf.v4.new_markdown_cell("## 7. Comparaison des modèles"))

    c.append(nbf.v4.new_code_cell(
        "models = {\n"
        "    'Logistic Regression': (lr_te_proba, lr_val_proba),\n"
        "    'SVM':                 (svm_te_proba, svm_val_proba),\n"
        "    'XGBoost':             (xgb_te_proba, xgb_val_proba),\n"
        "}\n\n"
        "rows = []\n"
        "for name, (te_p, val_p) in models.items():\n"
        "    rows.append({\n"
        "        'Model': name,\n"
        "        'Val_Acc':   f\"{accuracy_score(y_val, val_p>0.5)*100:.1f}%\",\n"
        "        'Val_AUC':   f\"{roc_auc_score(y_val, val_p):.3f}\",\n"
        "        'Test_Acc':  f\"{accuracy_score(y_te,  te_p >0.5)*100:.1f}%\",\n"
        "        'Test_AUC':  f\"{roc_auc_score(y_te,  te_p):.3f}\",\n"
        "        'Test_Brier':f\"{brier_score_loss(y_te, te_p):.3f}\",\n"
        "    })\n"
        "results = pd.DataFrame(rows)\n"
        "print(results.to_string(index=False))\n\n"
        "# Courbes ROC superposées\n"
        "plt.figure(figsize=(8, 6))\n"
        "colors = ['tomato','steelblue','forestgreen']\n"
        "for (name, (te_p, _)), col in zip(models.items(), colors):\n"
        "    fpr, tpr, _ = roc_curve(y_te, te_p)\n"
        "    auc = roc_auc_score(y_te, te_p)\n"
        "    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=col)\n"
        "plt.plot([0,1],[0,1],'k--', label='Chance')\n"
        "plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')\n"
        "plt.title('Courbes ROC — Test set'); plt.legend(); plt.tight_layout(); plt.show()"
    ))

    c.append(nbf.v4.new_markdown_cell("## 8. Calibration des probabilités"))

    c.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(8, 6))\n"
        "plt.plot([0,1],[0,1],'k--', label='Calibration parfaite')\n"
        "colors = ['tomato','steelblue','forestgreen']\n"
        "for (name, (te_p, _)), col in zip(models.items(), colors):\n"
        "    frac_pos, mean_pred = calibration_curve(y_te, te_p, n_bins=10)\n"
        "    plt.plot(mean_pred, frac_pos, marker='o', label=name, color=col)\n"
        "plt.xlabel('Probabilité prédite'); plt.ylabel('Fraction réelle positifs')\n"
        "plt.title('Courbes de calibration — Test set')\n"
        "plt.legend(); plt.tight_layout(); plt.show()\n\n"
        "print('Un modèle bien calibré suit la diagonale.')\n"
        "print('Critique pour le ranking : des probas mal calibrées faussent les scores.')"
    ))

    c.append(nbf.v4.new_markdown_cell(
        "## 9. Ranking UFC par division\n\n"
        "**Méthode :** chaque fighter actif combat (virtuellement) contre tous les autres "
        "de sa division. Son score = somme des `P(il gagne)` contre chaque adversaire.\n\n"
        "On utilise le **meilleur modèle sans cotes** pour un ranking indépendant du marché."
    ))

    c.append(nbf.v4.new_code_cell(
        "# Recharger le dataset sans cotes pour le ranking\n"
        "df_no = pd.read_csv('../data/processed/features_dataset_no_odds.csv')\n"
        "df_no['date'] = pd.to_datetime(df_no['date'])\n"
        "df_no = df_no.sort_values('date').reset_index(drop=True)\n\n"
        "FEAT_NO_ODDS = [c for c in df_no.columns\n"
        "                if c.startswith('delta_') or\n"
        "                   c in ['R_age','B_age','R_is_southpaw','B_is_southpaw','same_stance',\n"
        "                         'R_is_ranked','B_is_ranked','delta_rank',\n"
        "                         'empty_arena','weight_class_ord']]\n\n"
        "# Re-splitter et ré-entraîner XGBoost sans cotes\n"
        "N2 = len(df_no)\n"
        "tr2 = df_no.iloc[:int(N2*0.70)]\n"
        "va2 = df_no.iloc[int(N2*0.70):int(N2*0.85)]\n\n"
        "preproc2 = Pipeline([('imp', SimpleImputer(strategy='median')),\n"
        "                      ('scale', StandardScaler())])\n"
        "X_tr2 = preproc2.fit_transform(tr2[FEAT_NO_ODDS])\n"
        "X_va2 = preproc2.transform(va2[FEAT_NO_ODDS])\n\n"
        "d2train = xgb.DMatrix(X_tr2, label=tr2['R_Win'].values, feature_names=FEAT_NO_ODDS)\n"
        "d2val   = xgb.DMatrix(X_va2, label=va2['R_Win'].values, feature_names=FEAT_NO_ODDS)\n\n"
        "model_rank = xgb.train(\n"
        "    xgb_params, d2train, num_boost_round=500,\n"
        "    evals=[(d2train,'train'),(d2val,'val')],\n"
        "    early_stopping_rounds=30, verbose_eval=False\n"
        ")\n"
        "print('Modèle de ranking entraîné (sans cotes)')\n"
        "val_auc2 = roc_auc_score(va2['R_Win'].values, model_rank.predict(d2val))\n"
        "print(f'Val AUC (sans cotes) : {val_auc2:.3f}')"
    ))

    c.append(nbf.v4.new_code_cell(
        "# Profil actuel de chaque fighter = stats cumulées à leur dernier combat\n"
        "pre_r_cols = [c for c in df_no.columns if c.startswith('delta_') or\n"
        "              c in ['R_age','R_is_southpaw','R_is_ranked','n_fights_R']]\n\n"
        "# Pour le ranking, on reconstruit les stats de chaque fighter depuis les colonnes R_pre_\n"
        "# en lisant le CSV non-delta (features_dataset) et en prenant le dernier combat de chaque fighter\n"
        "df_full = pd.read_csv('../data/processed/features_dataset.csv')\n"
        "df_full['date'] = pd.to_datetime(df_full['date'])\n\n"
        "# Stats pre_R du dernier combat (= état le plus récent)\n"
        "r_profiles = []\n"
        "r_pre_cols = [c for c in df_full.columns if c.startswith('delta_') or\n"
        "              c in ['R_age','R_is_southpaw','R_is_ranked','n_fights_R']]\n\n"
        "# On utilise les dernières apparitions de chaque fighter en coin Rouge\n"
        "latest_r = df_full.sort_values('date').groupby('R_Fighter').last().reset_index()\n"
        "latest_b = df_full.sort_values('date').groupby('B_Fighter').last().reset_index()\n\n"
        "# Renommer B → R pour uniformiser\n"
        "b_to_r = {'B_Fighter':'Fighter','B_age':'age','B_is_southpaw':'is_southpaw',\n"
        "          'B_is_ranked':'is_ranked','n_fights_B':'n_fights','WeightClass':'WeightClass'}\n"
        "r_to_r = {'R_Fighter':'Fighter','R_age':'age','R_is_southpaw':'is_southpaw',\n"
        "          'R_is_ranked':'is_ranked','n_fights_R':'n_fights','WeightClass':'WeightClass'}\n\n"
        "prof_r = latest_r[list(r_to_r.keys())].rename(columns=r_to_r)\n"
        "prof_b = latest_b[list(b_to_r.keys())].rename(columns=b_to_r)\n"
        "profiles = pd.concat([prof_r, prof_b]).groupby('Fighter').last().reset_index()\n\n"
        "# Filtrer : min 3 combats UFC\n"
        "profiles = profiles[profiles['n_fights'] >= 3]\n"
        "print(f'Fighters actifs (min 3 combats) : {len(profiles)}')\n"
        "profiles.head(5)"
    ))

    c.append(nbf.v4.new_code_cell(
        "def rank_division(weight_class, profiles, df_full, model, preproc, feat_cols):\n"
        "    \"\"\"\n"
        "    Calcule le ranking des fighters d'une division via un tournament round-robin.\n"
        "    Pour chaque paire (A,B), prédit P(A bat B) et l'ajoute au score de A.\n"
        "    \"\"\"\n"
        "    # Fighters de cette division\n"
        "    wc_fighters = df_full[\n"
        "        df_full['WeightClass'].str.lower().str.contains(\n"
        "            weight_class.lower(), na=False)\n"
        "    ]['R_Fighter'].value_counts()\n"
        "    wc_fighters = wc_fighters[wc_fighters >= 3].index.tolist()\n"
        "    div_profiles = profiles[profiles['Fighter'].isin(wc_fighters)]\n\n"
        "    if len(div_profiles) < 2:\n"
        "        return pd.DataFrame()\n\n"
        "    fighters = div_profiles['Fighter'].tolist()\n"
        "    scores = {f: 0.0 for f in fighters}\n\n"
        "    for i, fa in enumerate(fighters):\n"
        "        for fb in fighters[i+1:]:\n"
        "            pa = div_profiles[div_profiles['Fighter'] == fa].iloc[0]\n"
        "            pb = div_profiles[div_profiles['Fighter'] == fb].iloc[0]\n\n"
        "            # Vecteur delta A-B (version simplifiée avec les colonnes disponibles)\n"
        "            row = {}\n"
        "            for col in feat_cols:\n"
        "                row[col] = 0.0  # défaut\n"
        "            row['delta_age']         = pa.get('age',25) - pb.get('age',25)\n"
        "            row['R_age']             = pa.get('age',25)\n"
        "            row['B_age']             = pb.get('age',25)\n"
        "            row['R_is_southpaw']     = pa.get('is_southpaw',0)\n"
        "            row['B_is_southpaw']     = pb.get('is_southpaw',0)\n"
        "            row['same_stance']       = float(pa.get('is_southpaw',0) == pb.get('is_southpaw',0))\n"
        "            row['R_is_ranked']       = pa.get('is_ranked',0)\n"
        "            row['B_is_ranked']       = pb.get('is_ranked',0)\n\n"
        "            X = pd.DataFrame([row])[feat_cols].fillna(0)\n"
        "            X_proc = preproc.transform(X)\n"
        "            d = xgb.DMatrix(X_proc, feature_names=feat_cols)\n"
        "            p_a_wins = float(model.predict(d)[0])\n"
        "            scores[fa] += p_a_wins\n"
        "            scores[fb] += (1 - p_a_wins)\n\n"
        "    ranking = pd.DataFrame([(f, s) for f,s in scores.items()],\n"
        "                           columns=['Fighter','Score'])\n"
        "    ranking = ranking.sort_values('Score', ascending=False).reset_index(drop=True)\n"
        "    ranking['Rank'] = ranking.index + 1\n"
        "    ranking['Score'] = ranking['Score'].round(2)\n"
        "    return ranking\n\n"
        "print('Fonction de ranking définie')"
    ))

    c.append(nbf.v4.new_code_cell(
        "# Ranking Lightweight (Khabib doit être en tête !)\n"
        "print('=== LIGHTWEIGHT ===')\n"
        "lw = rank_division('Lightweight', profiles, df_full, model_rank, preproc2, FEAT_NO_ODDS)\n"
        "print(lw.head(20).to_string(index=False))"
    ))

    c.append(nbf.v4.new_code_cell(
        "# Autres divisions\n"
        "divisions = ['Heavyweight', 'Middleweight', 'Welterweight',\n"
        "             'Featherweight', 'Bantamweight', 'Flyweight',\n"
        "             'Light Heavyweight']\n\n"
        "all_rankings = {}\n"
        "for div in divisions:\n"
        "    rank = rank_division(div, profiles, df_full, model_rank, preproc2, FEAT_NO_ODDS)\n"
        "    if len(rank) > 0:\n"
        "        all_rankings[div] = rank\n"
        "        print(f'\\n=== {div.upper()} (Top 10) ===')\n"
        "        print(rank.head(10).to_string(index=False))"
    ))

    c.append(nbf.v4.new_markdown_cell("## 10. Elo dynamique — historique du niveau de chaque fighter"))

    c.append(nbf.v4.new_code_cell(
        "def compute_elo(df_chrono, K=32, initial=1500):\n"
        "    elo = {}\n"
        "    history = []\n"
        "    for _, fight in df_chrono.iterrows():\n"
        "        r, b = fight['R_Fighter'], fight['B_Fighter']\n"
        "        er = elo.get(r, initial)\n"
        "        eb = elo.get(b, initial)\n"
        "        exp_r = 1 / (1 + 10**((eb - er)/400))\n"
        "        actual_r = fight['R_Win']\n"
        "        elo[r] = er + K * (actual_r - exp_r)\n"
        "        elo[b] = eb + K * ((1-actual_r) - (1-exp_r))\n"
        "        history.append({'date': fight['date'], 'Fighter': r, 'Elo': elo[r], 'WeightClass': fight.get('WeightClass','')})\n"
        "        history.append({'date': fight['date'], 'Fighter': b, 'Elo': elo[b], 'WeightClass': fight.get('WeightClass','')})\n"
        "    return elo, pd.DataFrame(history)\n\n"
        "elo_final, elo_history = compute_elo(df_full.sort_values('date'))\n\n"
        "# Ranking Elo global (top 20)\n"
        "elo_df = pd.DataFrame([(f,e) for f,e in elo_final.items()], columns=['Fighter','Elo'])\n"
        "elo_df = elo_df.sort_values('Elo', ascending=False).reset_index(drop=True)\n"
        "elo_df['Rank'] = elo_df.index + 1\n"
        "print('Top 20 Elo global :')\n"
        "print(elo_df.head(20).to_string(index=False))"
    ))

    c.append(nbf.v4.new_code_cell(
        "# Evolution du Elo pour quelques légendes\n"
        "legends = ['Khabib Nurmagomedov','Jon Jones','Anderson Silva','Georges St-Pierre','Conor McGregor']\n"
        "legends = [f for f in legends if f in elo_history['Fighter'].values]\n\n"
        "plt.figure(figsize=(13, 6))\n"
        "for fighter in legends:\n"
        "    fh = elo_history[elo_history['Fighter'] == fighter].sort_values('date')\n"
        "    plt.plot(fh['date'], fh['Elo'], marker='o', markersize=3, label=fighter)\n\n"
        "plt.axhline(1500, linestyle='--', color='gray', label='Rating initial')\n"
        "plt.title('Evolution du rating Elo — Légendes UFC')\n"
        "plt.xlabel('Date'); plt.ylabel('Elo')\n"
        "plt.legend(); plt.tight_layout(); plt.show()"
    ))

    c.append(nbf.v4.new_markdown_cell("## 11. Validation du ranking vs classements officiels UFC"))

    c.append(nbf.v4.new_code_cell(
        "from scipy.stats import spearmanr, kendalltau\n\n"
        "# Fighters avec classement officiel dans le dataset test\n"
        "ranked_test = test[test['RMatchWCRank'].notna()][['R_Fighter','RMatchWCRank','WeightClass']].copy()\n"
        "ranked_test = ranked_test.groupby('R_Fighter').agg(\n"
        "    official_rank=('RMatchWCRank','mean'),\n"
        "    weight_class=('WeightClass','first')\n"
        ").reset_index()\n\n"
        "# Comparer avec le ranking Elo\n"
        "elo_ranked = elo_df.set_index('Fighter')['Rank']\n"
        "common = ranked_test[ranked_test['R_Fighter'].isin(elo_ranked.index)]\n\n"
        "if len(common) >= 5:\n"
        "    off = common.set_index('R_Fighter')['official_rank']\n"
        "    our = elo_ranked[common['R_Fighter']]\n"
        "    rho, p1 = spearmanr(off.values, our.values)\n"
        "    tau, p2 = kendalltau(off.values, our.values)\n"
        "    print(f'Fighters comparables : {len(common)}')\n"
        "    print(f'Spearman rho : {rho:.3f}  (p={p1:.4f})')\n"
        "    print(f'Kendall tau  : {tau:.3f}  (p={p2:.4f})')\n"
        "    print('\\nUn rho positif = corrélation avec le classement officiel UFC')\n"
        "else:\n"
        "    print(f'Seulement {len(common)} fighters comparables — pas assez pour la corrélation.')"
    ))

    nb.cells = c
    with open("notebooks/03_ml_models.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("✓ notebooks/03_ml_models.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_nb01()
    make_nb02()
    make_nb03()
    print("\nTous les notebooks créés.")
