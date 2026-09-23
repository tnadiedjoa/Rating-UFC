"""
The published dataset: clean, documented tables in ``dataset/``, rebuilt by
the pipeline after every update.

    fights.csv    one row per UFC fight: result, method, judges' scores,
                  bonuses, fight totals of both fighters, odds, official ranks
    rounds.csv    one row per round of those fights, with the same statistics
    fighters.csv  one row per UFC fighter: profile, UFC and professional
                  records, current Elo rating
    rankings.csv  the official UFC rankings, week by week, since 2018
    records.csv   every professional fight of the fighters with a Wikipedia
                  record, inside and outside the UFC
    elo.csv       Elo rating of both fighters before and after every fight

The dataset card (``dataset/README.md``) is generated from the tables, so its
figures always match the files.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ufc_rating.config import DATASET_DIR, WIKI_RANKINGS_CSV, WIKI_RECORDS_CSV
from ufc_rating.processing.master import (
    ROUND_STAT_COLUMNS, STATS, read_official_rankings,
)

ROUND_STATS = list(ROUND_STAT_COLUMNS) + ["ctrl_sec"]
RESULT_COLUMNS = {"win": "wins", "loss": "losses", "draw": "draws", "nc": "no_contests"}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def fights_table(master: pd.DataFrame) -> pd.DataFrame:
    """One row per fight, descriptive columns first, then both fighters' totals."""
    winner = np.select([master["outcome"] == "r", master["outcome"] == "b"],
                       [master["r_id"], master["b_id"]], default=None)
    out = pd.DataFrame({
        "fight_id": master["fight_id"],
        "event_id": master["event_id"],
        "event": master["event_name"],
        "date": master["date"].dt.date,
        "location": master["location"],
        "division": master["division"],
        "weight_class": master["weight_class"],
        "title_fight": master["title_fight"],
        "scheduled_rounds": master["scheduled_rounds"],
        "time_format": master["time_format"],
        "r_fighter_id": master["r_id"],
        "r_fighter": master["r_name"],
        "b_fighter_id": master["b_id"],
        "b_fighter": master["b_name"],
        "outcome": master["outcome"],
        "winner_id": winner,
        "method": master["method"],
        "method_group": master["method_group"],
        "finish_round": master["finish_round"].astype("Int64"),
        "finish_time": master["finish_time"],
        "fight_seconds": master["fight_seconds"].astype("Int64"),
        "details": master["details"],
        "referee": master["referee"],
        "bonuses": master["bonuses"],
    })
    for j in (1, 2, 3):
        out[f"judge{j}"] = master[f"judge{j}_name"]
        out[f"judge{j}_r_score"] = master[f"judge{j}_r_score"].astype("Int64")
        out[f"judge{j}_b_score"] = master[f"judge{j}_b_score"].astype("Int64")
    for column in ("r_odds", "b_odds", "odds_source", "r_rank", "b_rank", "r_meta_rank", "b_meta_rank"):
        values = master[column] if column in master else pd.Series(np.nan, index=master.index)
        out[column] = values.astype("Int64") if column != "odds_source" else values
    stats = {f"{side}_{stat}": master[f"{side}_{stat}"].astype("Int64")
             for side in ("r", "b") for stat in STATS}
    return pd.concat([out, pd.DataFrame(stats)], axis=1)


def rounds_table(rounds: pd.DataFrame) -> pd.DataFrame:
    out = rounds[["fight_id", "date", "round", "seconds", "r_id", "b_id"]].rename(
        columns={"r_id": "r_fighter_id", "b_id": "b_fighter_id"})
    out["date"] = out["date"].dt.date
    out["seconds"] = out["seconds"].astype("Int64")
    stats = {f"{side}_{stat}": rounds[f"{side}_{stat}"].astype("Int64")
             for side in ("r", "b") for stat in ROUND_STATS}
    return pd.concat([out, pd.DataFrame(stats, index=rounds.index)], axis=1)


def fighters_table(master: pd.DataFrame, elo_history: pd.DataFrame,
                   records: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per fighter, from their most recent fight (profile) and all their fights (records)."""
    parts = []
    for side in ("r", "b"):
        result = np.select([master["outcome"] == side, master["outcome"].isin(["r", "b"]),
                            master["outcome"] == "draw"], ["win", "loss", "draw"], default="nc")
        parts.append(pd.DataFrame({
            "fighter_id": master[f"{side}_id"], "fighter": master[f"{side}_name"],
            "date": master["date"], "division": master["division"], "result": result,
            "dob": master[f"{side}_dob"], "height_cm": master[f"{side}_height_cm"],
            "reach_cm": master[f"{side}_reach_cm"], "stance": master[f"{side}_stance"],
        }))
    apps = pd.concat(parts, ignore_index=True).sort_values(["fighter_id", "date"])
    latest = apps.groupby("fighter_id").tail(1).set_index("fighter_id")
    counts = pd.crosstab(apps["fighter_id"], apps["result"]).reindex(
        columns=["win", "loss", "draw", "nc"], fill_value=0)

    out = pd.DataFrame({
        "fighter_id": latest.index,
        "fighter": latest["fighter"].values,
        "dob": latest["dob"].dt.date.values,
        "height_cm": latest["height_cm"].values,
        "reach_cm": latest["reach_cm"].values,
        "stance": latest["stance"].values,
        "division": apps.dropna(subset=["division"]).groupby("fighter_id")["division"].last()
                        .reindex(latest.index).values,
        "first_ufc_fight": apps.groupby("fighter_id")["date"].min().reindex(latest.index).dt.date.values,
        "last_ufc_fight": latest["date"].dt.date.values,
    })
    for result, column in RESULT_COLUMNS.items():
        out[f"ufc_{column}"] = counts[result].reindex(latest.index).values
    # the history is in processing order (date, then fight_id): keep it for tournament nights
    elo = elo_history.sort_values(["date", "fight_id"], kind="stable").groupby("fighter_id")["elo_after"].last()
    out["elo"] = elo.reindex(latest.index).round(1).values

    if records is not None and not records.empty:
        first = out.set_index("fighter_id")["first_ufc_fight"].map(pd.Timestamp)
        rec = records.assign(first_ufc=records["fighter_id"].map(first))
        totals = pd.crosstab(rec["fighter_id"], rec["result"]).reindex(
            columns=["win", "loss", "draw"], fill_value=0)
        before = rec[rec["date"] < rec["first_ufc"]].groupby("fighter_id").size()
        wiki = rec.groupby("fighter_id")["wiki_title"].first()
        idx = out["fighter_id"]
        out["wikipedia_page"] = idx.map(wiki).values
        for result in ("win", "loss", "draw"):
            out[f"pro_{RESULT_COLUMNS[result]}"] = idx.map(totals[result]).astype("Int64").values
        out["pro_fights_before_ufc"] = idx.map(before).fillna(0).astype("Int64").where(idx.isin(wiki.index)).values
    return out.sort_values("fighter").reset_index(drop=True)


def elo_table(elo_history: pd.DataFrame) -> pd.DataFrame:
    out = elo_history[["fight_id", "date", "fighter_id", "fighter_name", "elo_before", "elo_after"]].rename(
        columns={"fighter_name": "fighter"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out[["elo_before", "elo_after"]] = out[["elo_before", "elo_after"]].round(1)
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(master: pd.DataFrame, rounds: pd.DataFrame, elo_history: pd.DataFrame,
           out_dir: Path = DATASET_DIR, rankings_csv: Path = WIKI_RANKINGS_CSV,
           records_csv: Path = WIKI_RECORDS_CSV) -> dict:
    """Write every table and the dataset card to ``out_dir``. Returns the tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(records_csv, parse_dates=["date"]) if Path(records_csv).exists() else None
    tables = {
        "fights": fights_table(master),
        "rounds": rounds_table(rounds),
        "fighters": fighters_table(master, elo_history, records),
        "elo": elo_table(elo_history),
    }
    if Path(rankings_csv).exists():
        official = read_official_rankings(rankings_csv, master)
        official["date"] = official["date"].dt.date
        tables["rankings"] = official[["date", "system", "division", "rank", "note", "fighter", "fighter_id"]]
    if records is not None:
        rec = records.drop(columns=["fetched"], errors="ignore").copy()
        rec["date"] = rec["date"].dt.date
        tables["records"] = rec
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "README.md").write_text(dataset_card(tables), encoding="utf-8")
    return tables


def _coverage(series: pd.Series) -> str:
    return f"{series.notna().mean():.1%}"


def dataset_card(tables: dict) -> str:
    """The dataset README, with its figures computed from ``tables``."""
    f, r, fi = tables["fights"], tables["rounds"], tables["fighters"]
    dates = pd.to_datetime(f["date"])
    since_2010 = f[dates >= "2010-01-01"]
    decisions = f[f["method_group"] == "Decision"]
    size = {name: len(table) for name, table in tables.items()}
    lines = [
        "# UFC dataset",
        "",
        f"Every UFC fight from {dates.min():%d %B %Y} to {dates.max():%d %B %Y}: "
        f"{size['fights']:,} fights, {size['rounds']:,} rounds and {size['fighters']:,} fighters, "
        "with results, round-by-round statistics, judges' scores, bonuses, betting odds and "
        "official rankings. Built and updated weekly by the code of this repository.",
        "",
        "## Files",
        "",
        "| File | Rows | One row per |",
        "|---|---:|---|",
        f"| `fights.csv` | {size['fights']:,} | UFC fight |",
        f"| `rounds.csv` | {size['rounds']:,} | round of a fight with statistics |",
        f"| `fighters.csv` | {size['fighters']:,} | fighter with at least one UFC fight |",
        f"| `elo.csv` | {size['elo']:,} | fighter in a fight (Elo before and after) |",
    ]
    if "rankings" in tables:
        rk = tables["rankings"]
        lines.append(f"| `rankings.csv` | {size['rankings']:,} | fighter in a weekly official ranking "
                     f"({pd.to_datetime(rk['date']).min():%B %Y} onwards) |")
    if "records" in tables:
        lines.append(f"| `records.csv` | {size['records']:,} | professional fight of a fighter "
                     f"with a Wikipedia record ({tables['records']['fighter_id'].nunique():,} fighters) |")
    lines += [
        "",
        "All files are UTF-8 CSV. Fighters, fights and events keep the ids of ufcstats.com, so "
        "the tables join on `fighter_id`, `fight_id` and `event_id`.",
        "",
        "## Coverage",
        "",
        "| Field | Available for |",
        "|---|---|",
        f"| Fight totals (strikes, takedowns...) | {_coverage(f['r_sig_landed'])} of fights |",
        f"| Control time | {_coverage(f['r_ctrl_sec'])} of fights (not recorded before mid-1999) |",
        f"| Round-by-round statistics | {f['fight_id'].isin(r['fight_id']).mean():.1%} of fights "
        "(some 1990s tournament fights only have totals) |",
        f"| Judges' scores | {_coverage(decisions['judge1_r_score'])} of decisions "
        "(draws have scores but no orientation, see below) |",
        f"| Betting odds | {_coverage(since_2010['r_odds'])} of fights since 2010 |",
        f"| Official rank at fight time | fights since 2010 (ranked fighters only) |",
        f"| Reach | {_coverage(fi['reach_cm'])} of fighters |",
        f"| Date of birth | {_coverage(fi['dob'])} of fighters |",
    ]
    if "wikipedia_page" in fi:
        lines.append(f"| Professional record (Wikipedia) | {_coverage(fi['wikipedia_page'])} of fighters |")
    lines += [
        "",
        "## Things to know",
        "",
        "- **Corners.** `r_` is the fighter listed first on ufcstats.com. Since about 2010 this is "
        "the red corner; before, ufcstats lists the **winner first in every fight**, so the r/b "
        "sides must not be used as corners for those years (a model learning from them would learn "
        "the result).",
        "- **Outcome.** `outcome` is `r`, `b`, `draw` or `nc` (no contest); `winner_id` is empty "
        "for draws and no contests.",
        "- **Judges' scores** are oriented to the r/b sides using the winner (ufcstats writes them "
        "as loser - winner). They are left empty for draws, which cannot be oriented.",
        "- **Odds** are American odds, closing lines: the Ultimate UFC Dataset (to March 2026), "
        "completed by the median over the sportsbooks listed on bestfightodds.com for the fights it "
        "misses since 2023 and for every later event (`odds_source`).",
        "- **Official ranks** are 0 for the champion (and an interim champion), empty when unranked. "
        "`r_rank`/`b_rank` are the media-panel rankings; `r_meta_rank`/`b_meta_rank` the Meta UFC "
        "Rankings that replaced them from June 2026. In `rankings.csv`, `note` is `T` for a tied "
        "rank and `IC` for an interim champion.",
        "- **Bonuses** are listed per fight. Fight of the Night goes to both fighters; the other "
        "bonuses usually go to the winner.",
        "- **Elo** uses K = 80, with split and majority decisions counting half (tuned on fights "
        "before the models' test period).",
        "",
        "## Sources and licence",
        "",
        "| Data | Source | Licence |",
        "|---|---|---|",
        "| Fights, rounds, fighter profiles, judges, bonuses | [ufcstats.com](http://ufcstats.com), "
        "through the [UFC Datasets 1994-2025](https://www.kaggle.com/datasets/neelagiriaditya/"
        "ufc-datasets-1994-2025) mirror and this repository's scraper for the latest events | CC0 (mirror) |",
        "| Odds up to March 2026, ranks up to 2018 | [Ultimate UFC Dataset](https://www.kaggle.com/"
        "datasets/mdabbert/ultimate-ufc-dataset) | CC BY 4.0 |",
        "| Odds it misses since 2023, later odds | [bestfightodds.com](https://www.bestfightodds.com) (sportsbook consensus) | "
        "facts, source credited |",
        "| Official rankings, professional records | [English Wikipedia](https://en.wikipedia.org/"
        "wiki/UFC_rankings) | CC BY-SA 4.0 |",
        "",
        "Because it includes material from Wikipedia, the dataset as a whole is distributed under "
        "**CC BY-SA 4.0**: credit the sources above and share adaptations under the same licence.",
        "",
    ]
    return "\n".join(lines)
