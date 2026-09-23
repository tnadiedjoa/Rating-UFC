from pathlib import Path

import pandas as pd
import pytest

from ufc_rating.config import UFCSTATS_CSV
from ufc_rating.processing.master import STAT_COLUMNS, build_master, to_canonical

FIXTURES = Path(__file__).parent / "fixtures"


def raw_fight(fight_id, date, r, b, result="r", method="Decision - Unanimous",
              finish_round=3, finish_time="5:00", time_format="3 Rnd (5-5-5)",
              weight_class="Lightweight", event="UFC Test Night", stats=None):
    """One fight in the Kaggle-mirror layout. ``result``: 'r', 'b', 'draw' or 'nc'."""
    row = {
        "fight_id": fight_id, "event_id": f"ev-{date}", "event_name": event,
        "event_date": date, "weight_class": weight_class, "title_fight": 0,
        "method": method, "finish_round": finish_round, "finish_time": finish_time,
        "time_format": time_format,
        "result_status": {"draw": "draw", "nc": "no_contest"}.get(result, "win"),
        "winner_id": {"r": r, "b": b}.get(result),
        "r_fighter_id": r, "r_fighter_name": r.title(),
        "b_fighter_id": b, "b_fighter_name": b.title(),
    }
    for side in ("r", "b"):
        row.update({f"{side}_height": "5' 10\"", f"{side}_reach_inches": 70.0,
                    f"{side}_stance": "Orthodox", f"{side}_dob": "1990-01-01"})
        for suffix in STAT_COLUMNS.values():
            row[f"{side}_{suffix}"] = 10
    row.update(stats or {})
    return row


def make_master(rows) -> pd.DataFrame:
    master = to_canonical(pd.DataFrame(rows))
    return master.sort_values(["date", "fight_id"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def real_master():
    """Master table built from the versioned data snapshot (no odds, no file written)."""
    if not UFCSTATS_CSV.exists():
        pytest.skip("data snapshot not available")
    return build_master(scraped_csv=None, odds_csv=None, out_path=None)
