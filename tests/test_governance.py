"""Fairness construction, monitoring thresholds, and doc/number consistency."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autopricing import fairness as FA
from autopricing import glm as G
from autopricing import monitoring as MO
from autopricing import report as R
from autopricing.config import ROOT, load_config

CFG = load_config()


def test_discrimination_free_price_does_not_depend_on_attribute():
    rng = np.random.default_rng(0)
    n = 20000
    d = rng.choice(["young", "old"], n, p=[0.2, 0.8])
    x = np.where(d == "young", rng.choice(["hi", "lo"], n, p=[0.8, 0.2]), rng.choice(["hi", "lo"], n, p=[0.2, 0.8]))
    df = pd.DataFrame({"Age": d, "X": x, "Exposure": 1.0})
    rate = 0.1 * np.where(d == "young", 2.0, 1.0) * np.where(x == "hi", 1.5, 1.0)
    df["ClaimNb"] = rng.poisson(rate)
    df["ClaimAmountCapped"] = df["ClaimNb"] * rng.gamma(2.0, 500.0, n)
    f = G.fit_frequency(df, ["Age", "X"])
    s = G.fit_severity(df, ["X"])
    p_d = pd.Series({"young": 0.2, "old": 0.8})
    h = FA.discrimination_free(f, s, df, "Age", p_d)
    # Two policies with the same X get the same price whatever their age
    t = pd.DataFrame({"x": x, "d": d, "h": h}).groupby(["x", "d"])["h"].mean().unstack()
    assert t.loc["hi", "young"] == pytest.approx(t.loc["hi", "old"])
    # ...but the *unaware* model, which drops Age, loads hi-X because it proxies youth
    un = G.fit_frequency(df, ["X"])
    hi_lo_unaware = np.exp(un.params["X[hi]"]) if "X[hi]" in un.params else 1 / np.exp(un.params["X[lo]"])
    assert hi_lo_unaware > 1.6        # more than the true 1.5: age leaks in through X


def test_rag_thresholds():
    thr = {"amber": 0.1, "red": 0.25}
    assert MO.rag(0.05, thr) == "GREEN"
    assert MO.rag(0.10, thr) == "AMBER"
    assert MO.rag(0.30, thr) == "RED"


def test_monitoring_flags_shifted_book():
    rng = np.random.default_rng(1)
    n = 20000

    def book(age_mu):
        return pd.DataFrame({
            "DrivAge": rng.normal(age_mu, 10, n).clip(18, 90), "BonusMalus": rng.integers(50, 100, n),
            "VehAge": rng.integers(0, 20, n), "VehPower": rng.integers(4, 12, n), "LogDensity": rng.normal(6, 1, n),
            "VehBrand": rng.choice(["B1", "B2"], n), "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(["R11", "R24"], n), "Area": rng.choice(["A", "B"], n), "Exposure": 1.0,
            "score": rng.gamma(2, 50, n),
        }).assign(loss=lambda d: rng.poisson(d["score"] / 1000) * 1000.0)

    ref = book(45)
    stable = MO.monitoring_report(ref, book(45), "score", "loss", 0.0, CFG)
    shifted = MO.monitoring_report(ref, book(30), "score", "loss", 0.0, CFG)
    psi_age = lambda r: r["table"].set_index("metric").loc["PSI DrivAge", "status"]  # noqa: E731
    assert psi_age(stable) == "GREEN"
    assert psi_age(shifted) == "RED"


def test_inject_replaces_only_marked_block():
    text = "a\n<!-- BEGIN:x -->\nold\n<!-- END:x -->\nb"
    assert R.inject(text, "x", "new") == "a\n<!-- BEGIN:x -->\nnew\n<!-- END:x -->\nb"


METRICS = ROOT / "reports" / "metrics.json"


@pytest.mark.skipif(not METRICS.exists(), reason="full-data run not present")
def test_readme_numbers_match_metrics_json():
    """The README results block must be exactly what the committed metrics.json renders to."""
    res = json.loads(METRICS.read_text())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, fn in R.BLOCKS.items():
        m = re.search(rf"<!-- BEGIN:{name} -->\n(.*?)\n<!-- END:{name} -->", readme, re.S)
        if m:
            assert m.group(1) == fn(res), f"README block '{name}' is stale; run `python -m autopricing report`"


@pytest.mark.skipif(not (ROOT / "models" / "glm_frequency.json").exists() or not METRICS.exists(),
                    reason="no saved model")
def test_saved_glm_scoring_matches_pipeline_prediction():
    from autopricing.scoring import score_frame
    pol = pd.DataFrame({"DrivAge": [19, 45], "BonusMalus": [100, 50], "VehAge": [0, 8], "VehPower": [6, 6],
                        "VehBrand": ["B12", "B1"], "VehGas": ["Regular", "Diesel"], "Region": ["R11", "R24"],
                        "Density": [5000.0, 50.0]})
    out = score_frame(pol)
    assert (out["risk_premium"] > 0).all()
    assert out.loc[0, "expected_frequency"] > out.loc[1, "expected_frequency"]   # young, malus driver
    assert Path(ROOT / "models" / "glm_severity.json").exists()
