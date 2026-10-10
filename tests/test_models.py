"""GLM recovers known parameters; exposure offset behaves; GLM design conventions hold."""

import numpy as np
import pandas as pd
import pytest

from autopricing import glm as G
from autopricing.features import GLMDesign, add_bands


def simulate(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Region": rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1]),
        "LogDensity": rng.normal(6, 1.5, n),
        "Exposure": rng.uniform(0.05, 1.0, n),
    })
    beta = {"Intercept": -2.3, "Region[B]": np.log(1.5), "Region[C]": np.log(0.7), "LogDensity": 0.08}
    eta = beta["Intercept"] + beta["LogDensity"] * df["LogDensity"] + np.where(df["Region"] == "B", beta["Region[B]"], 0) \
        + np.where(df["Region"] == "C", beta["Region[C]"], 0)
    df["ClaimNb"] = rng.poisson(np.exp(eta) * df["Exposure"])
    sev_mu = 1500 * np.where(df["Region"] == "B", 1.3, 1.0)
    df["ClaimAmountCapped"] = np.where(df["ClaimNb"] > 0, rng.gamma(2.0, sev_mu / 2.0) * df["ClaimNb"], 0.0)
    return df, beta


def test_poisson_glm_recovers_relativities():
    df, beta = simulate()
    m = G.fit_frequency(df, ["Region", "LogDensity"])
    assert m.design.base["Region"] == "A"            # highest-exposure level is the base
    se = dict(zip(m.design.columns, m.result.bse, strict=True))
    for k, v in beta.items():
        assert abs(m.params[k] - v) < 3.5 * se[k], k          # within 3.5 standard errors
    rel = G.relativities(m, df).set_index("level")
    assert rel.loc["A", "relativity"] == 1.0
    assert rel.loc["B", "ci_low"] < 1.5 < rel.loc["B", "ci_high"]


def test_exposure_offset_scales_prediction():
    df, _ = simulate(5000)
    m = G.fit_frequency(df, ["Region", "LogDensity"])
    one = m.predict(df.head(10), exposure=np.ones(10))
    half = m.predict(df.head(10), exposure=np.full(10, 0.5))
    np.testing.assert_allclose(half, one * 0.5)
    np.testing.assert_allclose(m.predict(df.head(10)), one)


def test_poisson_glm_balance_property():
    """With a log link and an intercept, fitted claims equal observed claims on the training data."""
    df, _ = simulate(20000)
    m = G.fit_frequency(df, ["Region", "LogDensity"])
    assert m.predict(df, df["Exposure"]).sum() == pytest.approx(df["ClaimNb"].sum(), rel=1e-6)


def test_gamma_severity_recovers_relativity():
    df, _ = simulate()
    m = G.fit_severity(df, ["Region"])
    assert np.exp(m.params["Region[B]"]) == pytest.approx(1.3, abs=0.08)
    assert np.exp(m.params["Intercept"]) == pytest.approx(1500, rel=0.05)


def test_design_unseen_level_maps_to_base():
    df = pd.DataFrame({"Region": ["A", "A", "B"], "Exposure": [1.0, 1.0, 1.0]})
    d = GLMDesign(["Region"]).fit(df, df["Exposure"])
    X = d.transform(pd.DataFrame({"Region": ["Z"]}))
    assert X.loc[0, "Region[B]"] == 0 and X.loc[0, "Intercept"] == 1


def test_bands_cover_all_ages():
    df = pd.DataFrame({"DrivAge": [18, 20, 21, 25, 26, 70, 71, 100], "BonusMalus": [50] * 8, "VehAge": [0] * 8,
                       "VehPower": [4] * 8})
    b = add_bands(df)
    assert list(b["DrivAgeBand"]) == ["18-20", "18-20", "21-25", "21-25", "26-30", "61-70", "71+", "71+"]
    assert (b != "nan").all().all()


def test_design_matrix_equals_transform():
    df, _ = simulate(500)
    d = GLMDesign(["Region", "LogDensity"]).fit(df, df["Exposure"])
    np.testing.assert_array_equal(d.matrix(df), d.transform(df).to_numpy())
