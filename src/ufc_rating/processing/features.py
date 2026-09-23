"""
Leakage-free fighter features and matchup vectors.

Three building blocks share one definition of every feature:

  fighter_states(master)      -> career state of each fighter AFTER each fight date
  pre_fight_features(master)  -> state BEFORE each fight date (the ML inputs)
  current_profiles(master)    -> state after the last fight (the ranking inputs)

Anti-leakage rule: a fight on date D only sees fights on dates strictly before
D. Career numbers are cumulative sums over past fights, aggregated per date and
shifted by one date, so a fighter's other bouts on the same night (1990s
tournaments) are excluded as well.

build_matchups() turns each decided fight into a vector of differences
between two fighters A and B, with A drawn at random from the two corners.
The r/b sides of the raw data are not usable as such: before 2010 ufcstats
lists the winner first (see processing.master).
"""

from typing import Optional

import numpy as np
import pandas as pd

from ufc_rating.config import SEED
from ufc_rating.processing.master import american_to_prob
from ufc_rating.ranking.elo import INITIAL_ELO, compute_elo

CAREER_FEATURES = [
    "n_fights",         # prior UFC fights (experience)
    "win_rate",         # wins / decided fights (draw = half a win)
    "finish_rate",      # wins by KO/TKO or submission / decided fights
    "ko_rate",          # wins by KO/TKO / decided fights
    "sub_rate",         # wins by submission / decided fights
    "finished_rate",    # losses by KO/TKO or submission / decided fights (durability)
    "slpm",             # significant strikes landed per minute
    "sapm",             # significant strikes absorbed per minute
    "sig_acc",          # significant strike accuracy
    "sig_def",          # share of opponent significant strikes avoided
    "td_per15",         # takedowns landed per 15 minutes
    "td_acc",           # takedown accuracy
    "td_def",           # share of opponent takedowns stopped
    "sub_per15",        # submission attempts per 15 minutes
    "kd_per15",         # knockdowns per 15 minutes
    "ctrl_pct",         # share of fight time spent in control
    "recent_win_rate",  # win rate over the last 3 decided fights
    "streak",           # current streak: +n wins in a row, -n losses in a row
    "days_since_last",  # layoff since the previous fight
]
PHYSICAL_FEATURES = ["age", "height_cm", "reach_cm", "southpaw"]
RATING_FEATURES = ["elo"]
FIGHTER_FEATURES = CAREER_FEATURES + PHYSICAL_FEATURES + RATING_FEATURES

STATS_FEATURES = [f"delta_{f}" for f in FIGHTER_FEATURES]
ODDS_FEATURES = STATS_FEATURES + ["odds_logit"]

_FINISH = ("KO/TKO", "Submission")

# UFC-wide averages over 1993-2026 (rounded), used as shrinkage priors.
PRIORS = {
    "win_rate": 0.50, "finish_rate": 0.26, "ko_rate": 0.17, "sub_rate": 0.10,
    "finished_rate": 0.26, "slpm": 3.5, "sapm": 3.5, "sig_acc": 0.45,
    "td_per15": 1.5, "td_acc": 0.37, "sub_per15": 0.5, "kd_per15": 0.3, "ctrl_pct": 0.21,
}
PRIOR_FIGHTS = 2.0         # decided fights, for win/finish rates
PRIOR_MINUTES = 15.0       # one full three-round fight, for per-minute rates
PRIOR_SIG_ATTEMPTS = 50.0  # about one fight of significant strike attempts
PRIOR_TD_ATTEMPTS = 3.0    # about one fight of takedown attempts


# ===========================================================================
# Long format: one row per (fight, fighter)
# ===========================================================================

def appearances(master: pd.DataFrame) -> pd.DataFrame:
    """Reshape the master table to one row per fighter and fight."""
    parts = []
    for side, opp in (("r", "b"), ("b", "r")):
        part = pd.DataFrame({
            "fight_id": master["fight_id"].values,
            "date": master["date"].values,
            "division": master["division"].values,
            "method_group": master["method_group"].values,
            "minutes": master["fight_seconds"].values / 60.0,
            "fighter_id": master[f"{side}_id"].values,
            "fighter_name": master[f"{side}_name"].values,
            "opponent_id": master[f"{opp}_id"].values,
            "result": np.select(
                [master["outcome"] == side, master["outcome"] == opp, master["outcome"] == "draw"],
                ["win", "loss", "draw"], default="nc"),
        })
        for stat in ("kd", "sig_landed", "sig_att", "td_landed", "td_att", "sub_att", "ctrl_sec"):
            part[stat] = master[f"{side}_{stat}"].values
        for stat in ("sig_landed", "sig_att", "td_landed", "td_att"):
            part[f"opp_{stat}"] = master[f"{opp}_{stat}"].values
        for attr in ("height_cm", "reach_cm", "stance", "dob"):
            part[attr] = master[f"{side}_{attr}"].values
        parts.append(part)
    apps = pd.concat(parts, ignore_index=True)
    return apps.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


# ===========================================================================
# Career state after each fight date
# ===========================================================================

def _streaks(results: pd.Series) -> np.ndarray:
    """Signed streak after each fight: wins count up, losses count down, draws reset, NC skip."""
    out = np.zeros(len(results))
    streak = 0
    for i, res in enumerate(results):
        if res == "win":
            streak = streak + 1 if streak > 0 else 1
        elif res == "loss":
            streak = streak - 1 if streak < 0 else -1
        elif res == "draw":
            streak = 0
        out[i] = streak
    return out


def _shrunk(num, den, prior_rate: float, prior_weight: float) -> np.ndarray:
    """
    Ratio shrunk toward a UFC-wide prior: (num + rate * w) / (den + w).

    Without it, small samples dominate: a single 7-second knockout reads as
    128 knockdowns per 15 minutes. The prior counts as ``prior_weight`` units
    of an average fight (minutes, attempts or decided fights).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return (num + prior_rate * prior_weight) / (den + prior_weight)


def fighter_states(master: pd.DataFrame, elo_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Career state of each fighter after each date on which they fought.
    Index: (fighter_id, date). Columns: the career features + elo.
    """
    apps = appearances(master)
    decided = apps["result"].isin(["win", "loss", "draw"])
    finish = apps["method_group"].isin(_FINISH)
    win_value = apps["result"].map({"win": 1.0, "loss": 0.0, "draw": 0.5})
    ctrl_known = apps["ctrl_sec"].notna()

    contrib = pd.DataFrame({
        "fighter_id": apps["fighter_id"],
        "fights": 1.0,
        "decided": decided.astype(float),
        "wins": win_value.fillna(0.0),
        "finish_wins": ((apps["result"] == "win") & finish).astype(float),
        "ko_wins": ((apps["result"] == "win") & (apps["method_group"] == "KO/TKO")).astype(float),
        "sub_wins": ((apps["result"] == "win") & (apps["method_group"] == "Submission")).astype(float),
        "finished": ((apps["result"] == "loss") & finish).astype(float),
        "minutes": apps["minutes"],
        "ctrl_minutes": apps["minutes"].where(ctrl_known, 0.0),
    })
    for stat in ("kd", "sig_landed", "sig_att", "td_landed", "td_att", "sub_att", "ctrl_sec",
                 "opp_sig_landed", "opp_sig_att", "opp_td_landed", "opp_td_att"):
        contrib[stat] = apps[stat].fillna(0.0)
    cum = contrib.groupby("fighter_id").cumsum()

    state = pd.DataFrame({"fighter_id": apps["fighter_id"], "date": apps["date"]})
    state["n_fights"] = cum["fights"]
    # UFC record (for display; not model inputs)
    for label, result in (("wins", "win"), ("losses", "loss"), ("draws", "draw")):
        state[f"record_{label}"] = (apps["result"] == result).groupby(apps["fighter_id"]).cumsum()
    P = PRIORS
    state["win_rate"] = _shrunk(cum["wins"], cum["decided"], P["win_rate"], PRIOR_FIGHTS)
    state["finish_rate"] = _shrunk(cum["finish_wins"], cum["decided"], P["finish_rate"], PRIOR_FIGHTS)
    state["ko_rate"] = _shrunk(cum["ko_wins"], cum["decided"], P["ko_rate"], PRIOR_FIGHTS)
    state["sub_rate"] = _shrunk(cum["sub_wins"], cum["decided"], P["sub_rate"], PRIOR_FIGHTS)
    state["finished_rate"] = _shrunk(cum["finished"], cum["decided"], P["finished_rate"], PRIOR_FIGHTS)
    state["slpm"] = _shrunk(cum["sig_landed"], cum["minutes"], P["slpm"], PRIOR_MINUTES)
    state["sapm"] = _shrunk(cum["opp_sig_landed"], cum["minutes"], P["sapm"], PRIOR_MINUTES)
    state["sig_acc"] = _shrunk(cum["sig_landed"], cum["sig_att"], P["sig_acc"], PRIOR_SIG_ATTEMPTS)
    state["sig_def"] = 1.0 - _shrunk(cum["opp_sig_landed"], cum["opp_sig_att"], P["sig_acc"], PRIOR_SIG_ATTEMPTS)
    state["td_per15"] = 15.0 * _shrunk(cum["td_landed"], cum["minutes"], P["td_per15"] / 15.0, PRIOR_MINUTES)
    state["td_acc"] = _shrunk(cum["td_landed"], cum["td_att"], P["td_acc"], PRIOR_TD_ATTEMPTS)
    state["td_def"] = 1.0 - _shrunk(cum["opp_td_landed"], cum["opp_td_att"], P["td_acc"], PRIOR_TD_ATTEMPTS)
    state["sub_per15"] = 15.0 * _shrunk(cum["sub_att"], cum["minutes"], P["sub_per15"] / 15.0, PRIOR_MINUTES)
    state["kd_per15"] = 15.0 * _shrunk(cum["kd"], cum["minutes"], P["kd_per15"] / 15.0, PRIOR_MINUTES)
    state["ctrl_pct"] = _shrunk(cum["ctrl_sec"] / 60.0, cum["ctrl_minutes"], P["ctrl_pct"], PRIOR_MINUTES)

    # Recent form: mean result of the last 3 decided fights (NC fights carry the value forward)
    recent = (win_value.where(decided)
              .groupby(apps["fighter_id"])
              .transform(lambda s: s.dropna().rolling(3, min_periods=1).mean().reindex(s.index)))
    state["recent_win_rate"] = recent.groupby(apps["fighter_id"]).ffill()
    state["streak"] = apps.groupby("fighter_id")["result"].transform(_streaks)

    if elo_history is None:
        elo_history = compute_elo(master)
    elo_after = elo_history.set_index(["fight_id", "fighter_id"])["elo_after"]
    state["elo"] = elo_after.reindex(pd.MultiIndex.from_arrays([apps["fight_id"], apps["fighter_id"]])).values

    # One row per (fighter, date): the state after the last fight of that date.
    # (groupby().last() would skip NaNs and pick up stale values.)
    return (state.drop_duplicates(["fighter_id", "date"], keep="last")
            .set_index(["fighter_id", "date"]))


def pre_fight_features(master: pd.DataFrame, elo_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Career state of each fighter just before each date on which they fought.
    Fighters with no earlier fight get n_fights = 0, Elo = 1500 and NaN elsewhere.
    """
    after = fighter_states(master, elo_history)
    before = after.groupby(level="fighter_id").shift(1)
    dates = after.index.get_level_values("date").to_series(index=after.index)
    before["days_since_last"] = (dates - dates.groupby(level="fighter_id").shift(1)).dt.days
    before["n_fights"] = before["n_fights"].fillna(0.0)
    before["streak"] = before["streak"].fillna(0.0)
    before["elo"] = before["elo"].fillna(INITIAL_ELO)
    return before


# ===========================================================================
# Physical attributes
# ===========================================================================

def southpaw_score(stance) -> float:
    """Southpaw = 1, switch = 0.5, orthodox and others = 0, unknown = NaN."""
    if not isinstance(stance, str):
        return np.nan
    return {"Southpaw": 1.0, "Switch": 0.5}.get(stance, 0.0)


def age_years(reference, dob):
    return (pd.to_datetime(reference) - pd.to_datetime(dob)).dt.days / 365.25


# ===========================================================================
# Matchups for the ML models
# ===========================================================================

def build_matchups(
    master: pd.DataFrame,
    seed: int = SEED,
    min_prior_fights: int = 1,
    elo_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    One row per decided fight (no draws, no NC), oriented as fighter A vs B
    with A chosen at random (fixed seed). Features are A minus B; the target
    ``a_wins`` is 1 when A won.

    Both fighters need ``min_prior_fights`` UFC fights before the bout: a
    debut carries no career statistics.
    """
    pre = pre_fight_features(master, elo_history)
    fights = master[master["outcome"].isin(["r", "b"])].reset_index(drop=True)

    sides = {}
    for side in ("r", "b"):
        key = pd.MultiIndex.from_arrays([fights[f"{side}_id"], fights["date"]])
        feats = pre.reindex(key).reset_index(drop=True)
        feats["age"] = age_years(fights["date"], fights[f"{side}_dob"])
        feats["height_cm"] = fights[f"{side}_height_cm"]
        feats["reach_cm"] = fights[f"{side}_reach_cm"]
        feats["southpaw"] = fights[f"{side}_stance"].map(southpaw_score)
        sides[side] = feats

    rng = np.random.default_rng(seed)
    a_is_r = rng.random(len(fights)) < 0.5

    def pick(r_values, b_values):
        return np.where(a_is_r, r_values, b_values), np.where(a_is_r, b_values, r_values)

    out = pd.DataFrame({
        "fight_id": fights["fight_id"],
        "date": fights["date"],
        "division": fights["division"],
        "title_fight": fights["title_fight"],
        "a_is_r": a_is_r,
    })
    out["a_id"], out["b_id"] = pick(fights["r_id"], fights["b_id"])
    out["a_name"], out["b_name"] = pick(fights["r_name"], fights["b_name"])
    out["a_wins"] = np.where(a_is_r, fights["outcome"] == "r", fights["outcome"] == "b").astype(int)

    for feat in FIGHTER_FEATURES:
        a_val, b_val = pick(sides["r"][feat].astype(float), sides["b"][feat].astype(float))
        out[f"a_{feat}"] = a_val
        out[f"b_{feat}"] = b_val
        out[f"delta_{feat}"] = a_val - b_val

    # Market view: implied probability of A with the bookmaker margin removed
    no_odds = pd.Series(np.nan, index=fights.index)
    r_imp = american_to_prob(fights.get("r_odds", no_odds))
    b_imp = american_to_prob(fights.get("b_odds", no_odds))
    a_imp, b_imp = pick(r_imp, b_imp)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["a_odds_prob"] = a_imp / (a_imp + b_imp)
        out["odds_logit"] = np.log(out["a_odds_prob"] / (1.0 - out["a_odds_prob"]))

    enough = (out["a_n_fights"] >= min_prior_fights) & (out["b_n_fights"] >= min_prior_fights)
    return out[enough].sort_values(["date", "fight_id"]).reset_index(drop=True)


# ===========================================================================
# Current profiles for the rankings
# ===========================================================================

def current_profiles(
    master: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    elo_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    One row per fighter with every FIGHTER_FEATURES column as of ``as_of``
    (default: the last fight date in the data), plus name, division (from
    the most recent fight with a known division), last fight date and bio.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else master["date"].max()
    master = master[master["date"] <= as_of]

    after = fighter_states(master, elo_history).reset_index()
    last = after.groupby("fighter_id").tail(1).set_index("fighter_id")

    apps = appearances(master)
    latest = apps.groupby("fighter_id").tail(1).set_index("fighter_id")
    known_div = apps.dropna(subset=["division"]).groupby("fighter_id").tail(1).set_index("fighter_id")

    profiles = last.drop(columns="date")
    profiles["fighter_name"] = latest["fighter_name"]
    profiles["division"] = known_div["division"].reindex(profiles.index)
    profiles["last_fight"] = latest["date"]
    profiles["days_since_last"] = (as_of - latest["date"]).dt.days
    profiles["age"] = age_years(pd.Series(as_of, index=latest.index), latest["dob"])
    profiles["height_cm"] = latest["height_cm"]
    profiles["reach_cm"] = latest["reach_cm"]
    profiles["southpaw"] = latest["stance"].map(southpaw_score)
    return profiles.reset_index()
