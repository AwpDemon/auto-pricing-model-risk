import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance

from autopricing import metrics as M

rng = np.random.default_rng(0)


@pytest.fixture
def book():
    n = 200000
    w = rng.uniform(0.1, 1.0, n)
    true_rate = np.exp(rng.normal(-2.5, 0.6, n))
    counts = rng.poisson(true_rate * w)
    return counts / w, true_rate, w


def test_deviances_match_sklearn(book):
    y, mu, w = book
    assert M.poisson_deviance(y, mu, w) == pytest.approx(mean_poisson_deviance(y, mu, sample_weight=w), rel=1e-10)
    yy = rng.gamma(2.0, 500.0, 1000)
    mm = rng.gamma(2.0, 500.0, 1000)
    assert M.gamma_deviance(yy, mm) == pytest.approx(mean_gamma_deviance(yy, mm), rel=1e-10)
    assert M.tweedie_deviance(y, mu, 1.5, w) == pytest.approx(
        mean_tweedie_deviance(y, mu, sample_weight=w, power=1.5), rel=1e-10)


def test_deviance_minimised_by_truth(book):
    y, mu, w = book
    assert M.poisson_deviance(y, mu, w) < M.poisson_deviance(y, mu * 1.2, w)
    assert M.poisson_deviance(y, mu, w) < M.poisson_deviance(y, np.full_like(mu, np.average(y, weights=w)), w)


def test_gini_properties(book):
    y, mu, w = book
    g_true = M.gini(y, mu, w)
    g_rand = M.gini(y, rng.permutation(mu), w)
    g_rev = M.gini(y, -mu, w)
    assert g_true > 0.2
    assert abs(g_rand) < 0.05
    assert g_rev < -0.2
    # Gini is rank-based: any monotone transform of the score leaves it unchanged
    assert M.gini(y, np.log(mu), w) == pytest.approx(g_true)


def test_lorenz_endpoints(book):
    x, lc = M.lorenz_curve(*book)
    assert x[0] == 0 and lc[0] == 0
    assert x[-1] == pytest.approx(1) and lc[-1] == pytest.approx(1)
    assert np.all(np.diff(x) >= 0)


def test_exposure_bins_equal_exposure(book):
    _, mu, w = book
    b = M.exposure_bins(mu, w, 10)
    share = pd.Series(w).groupby(b).sum() / w.sum()
    assert set(b) == set(range(1, 11))
    assert share.between(0.095, 0.105).all()


def test_lift_table_calibrated_model(book):
    y, mu, w = book
    t = M.lift_table(y, mu, w, 10)
    assert t["predicted_rate"].is_monotonic_increasing
    assert (t["ae"] - 1).abs().max() < 0.1
    assert M.lift_ratio(t) > 3


def test_double_lift_identifies_better_model(book):
    y, mu, w = book
    noisy = mu * np.exp(rng.normal(0, 0.5, len(mu)))
    dl = M.double_lift(y, noisy, mu, w, 10)      # challenger = truth
    assert dl["challenger_abs_err"].mean() < dl["base_abs_err"].mean()


def test_calibration_slope_near_one(book):
    y, mu, w = book
    assert M.calibration_slope(y, mu, w)["slope"] == pytest.approx(1.0, abs=0.08)
    # Over-dispersed predictions (mu^1.5) give a slope below one
    assert M.calibration_slope(y, mu ** 1.5 * np.exp(1.2), w)["slope"] < 0.8


def test_psi_hand_computed():
    e = np.repeat([0, 1], [50, 50])
    a = np.repeat([0, 1], [70, 30])
    expected = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    assert M.psi(e, a, categorical=True) == pytest.approx(expected)


def test_psi_zero_for_same_distribution():
    x = rng.normal(size=50000)
    assert M.psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert M.psi(x, rng.normal(size=50000)) < 0.01
    assert M.psi(x, rng.normal(1.0, 1, size=50000)) > 0.25


def test_actual_vs_expected():
    df = pd.DataFrame({"f": ["a", "a", "b"], "w": [1.0, 1.0, 2.0], "loss": [1.0, 1.0, 1.0], "e": [1.0, 1.0, 2.0]})
    t = M.actual_vs_expected(df, "f", "loss", ["e"], "w").set_index("level")
    assert t.loc["a", "ae_e"] == pytest.approx(1.0)
    assert t.loc["b", "ae_e"] == pytest.approx(0.5)


def test_bootstrap_gini_detects_better_model(book):
    y, mu, w = book
    out = M.bootstrap_gini(y, rng.permutation(mu), mu, w, reps=30, seed=1)
    assert out["diff_ci"][0] > 0
    assert out["share_b_better"] == 1.0
