"""End-to-end smoke test on the committed 20k-policy sample (what CI runs)."""

import json

import pytest

from autopricing.config import ROOT
from autopricing.pipeline import run


@pytest.mark.slow
def test_pipeline_runs_on_sample():
    res = run(sample=True)
    d = res["data"]
    assert d["profiles_crossing_splits"] == 0
    assert d["clean_policies"] < d["raw_policies"]
    pp = res["eval_test"]["pure_premium"]
    assert pp["GLM"]["gini"] > 0.0                       # better than random even on 20k rows
    assert abs(res["shap"]["additivity_max_abs_err"]) < 1e-6   # SHAP values sum to the raw score
    assert set(res["fairness"]["performance"]) == {"aware", "unaware", "disc_free"}
    out = ROOT / "out" / "sample"
    saved = json.loads((out / "reports" / "metrics.json").read_text())
    assert saved["data"]["clean_policies"] == d["clean_policies"]
    for fig in ["lorenz_pure_premium.png", "double_lift.png", "fairness_age_proxy.png", "monitoring_psi.png"]:
        assert (out / "docs" / "figures" / fig).stat().st_size > 10_000
