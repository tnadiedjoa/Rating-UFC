import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ufc_rating.models.training import model_grid, scores, temporal_split
from ufc_rating.ranking.rankings import eligible
from ufc_rating.ranking.round_robin import win_probability_matrix
from ufc_rating.ranking.weighted import DEFAULT_WEIGHTS, check_weights, weighted_scores


def toy_matchups(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 3000, n)), unit="D")
    X = rng.normal(size=(n, 3))
    y = (X @ [1.0, -0.5, 0.2] + rng.normal(scale=1.0, size=n) > 0).astype(int)
    df = pd.DataFrame(X, columns=["delta_x", "delta_y", "delta_z"])
    df["date"], df["fight_id"], df["a_wins"] = dates, [f"f{i}" for i in range(n)], y
    return df


def test_temporal_split_is_chronological_and_never_splits_a_date():
    train, val, test = temporal_split(toy_matchups())
    assert len(train) + len(val) + len(test) == 600
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert 0.6 < len(train) / 600 < 0.8


def test_logistic_regression_is_exactly_antisymmetric():
    df = toy_matchups()
    model, _ = model_grid()["LogReg"]
    X = df[["delta_x", "delta_y", "delta_z"]].to_numpy()
    X[::7, 0] = np.nan                                  # imputed as "no difference"
    model.fit(X, df["a_wins"])
    p = model.predict_proba(X)[:, 1]
    p_swapped = model.predict_proba(-X)[:, 1]
    np.testing.assert_allclose(p + p_swapped, 1.0, atol=1e-12)


def test_scores():
    s = scores([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4])
    assert s["accuracy"] == 1.0 and s["auc"] == 1.0 and s["n"] == 4


def profiles_fixture():
    return pd.DataFrame({
        "fighter_id": list("abcde"),
        "fighter_name": list("ABCDE"),
        "division": ["Heavyweight", "Light Heavyweight", "Heavyweight", "Heavyweight", "Heavyweight"],
        "last_fight": pd.to_datetime(["2026-01-01", "2026-01-01", "2020-01-01", "2025-06-01", "2026-03-01"]),
        "n_fights": [10, 10, 10, 2, 6],
    })


def test_eligibility_uses_exact_division_activity_and_experience():
    pool = eligible(profiles_fixture(), "Heavyweight", pd.Timestamp("2026-08-01"),
                    active_days=730, min_fights=5)
    # b: other division ('Heavyweight' is inside 'Light Heavyweight'), c: inactive, d: too few fights
    assert list(pool["fighter_id"]) == ["a", "e"]


def test_weights_are_validated():
    check_weights(DEFAULT_WEIGHTS, DEFAULT_WEIGHTS.keys())
    with pytest.raises(ValueError, match="sum to 1"):
        check_weights({"win_rate": 0.5}, ["win_rate"])
    with pytest.raises(ValueError, match="Unknown"):
        check_weights({"reach": 1.0}, ["win_rate"])


def test_weighted_scores_follow_the_statistics():
    pool = pd.DataFrame({k: [0.1, 0.5, 0.9] for k in DEFAULT_WEIGHTS})
    score = weighted_scores(pool)
    assert score.is_monotonic_increasing
    assert score.iloc[-1] == pytest.approx(1.0)


def test_round_robin_matrix_is_consistent_for_any_model():
    df = toy_matchups()
    features = ["delta_x", "delta_y", "delta_z"]
    model = RandomForestClassifier(n_estimators=50, random_state=0)   # not symmetric by design
    model.fit(df[features].to_numpy(), df["a_wins"])
    pool = pd.DataFrame(np.random.default_rng(1).normal(size=(6, 3)), columns=["x", "y", "z"])
    matrix = win_probability_matrix(pool, model, features)
    assert np.isnan(np.diag(matrix)).all()
    off = ~np.eye(6, dtype=bool)
    np.testing.assert_allclose((matrix + matrix.T)[off], 1.0)


def test_divisions_with_a_single_fighter_are_skipped():
    from ufc_rating.ranking.rankings import division_ranking
    profiles = profiles_fixture()
    table = division_ranking(profiles, "Light Heavyweight", model=None, features=[],
                             as_of=pd.Timestamp("2026-08-01"), min_fights=5)
    assert table.empty


def test_lower_is_better_statistics_are_inverted():
    pool = pd.DataFrame({"sapm": [1.0, 3.0, 6.0], "win_rate": [0.5, 0.5, 0.5]})
    score = weighted_scores(pool, {"sapm": 0.5, "win_rate": 0.5})
    assert score.is_monotonic_decreasing     # absorbing fewer strikes ranks higher
