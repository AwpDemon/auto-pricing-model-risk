import numpy as np
import pandas as pd
import pytest

from autopricing import data as D
from autopricing.config import load_config

CFG = load_config()


def toy():
    # IDs 1-3 form a "claims-only block" (all have claims; most lack amounts), then a normal book.
    freq = pd.DataFrame({
        "IDpol": [1, 2, 3, 10, 11, 12, 13, 14],
        "ClaimNb": [1, 2, 1, 0, 1, 6, 0, 2],
        "Exposure": [0.5, 1.0, 0.2, 1.0, 1.5, 1.0, 0.3, 1.0],
        "Area": ["A"] * 8, "VehPower": [5] * 8, "VehAge": [1] * 8, "DrivAge": [40] * 8, "BonusMalus": [50] * 8,
        "VehBrand": ["B1"] * 8, "VehGas": ["Diesel"] * 8, "Density": [100.0] * 8, "Region": ["R11"] * 8,
    })
    sev = pd.DataFrame({"IDpol": [2, 11] + [12] * 6 + [14, 999], "ClaimAmount": [100.0, 200.0] + [50.0] * 6 + [300.0, 7.0]})
    return freq, sev


def test_claims_only_block_excluded():
    f, s = toy()
    r = D.clean(f, s, CFG)
    assert r.log["dq1_block_boundary_idpol"] == 10
    assert r.log["dq1_block_policies"] == 3
    assert r.log["dq1_block_claims_missing_amount_share"] == pytest.approx(2 / 3)
    assert r.policies["IDpol"].min() == 10


def test_counts_reconciled_capped_and_exposure_capped():
    f, s = toy()
    r = D.clean(f, s, CFG)
    p = r.policies.set_index("IDpol")
    assert p.loc[14, "ClaimNb"] == 1            # 2 reported, 1 record with an amount
    assert p.loc[12, "ClaimNb"] == 4            # 6 records, capped at 4
    assert p.loc[12, "ClaimAmount"] == 300.0    # amounts are NOT capped by the count cap
    assert p.loc[11, "Exposure"] == 1.0
    assert r.log["dq4_policies_exposure_capped"] == 1
    assert 999 not in set(r.claims["IDpol"])    # orphan claim record dropped


def test_split_keeps_rating_profiles_together():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({c: rng.integers(0, 3, n) for c in D.RATING_FEATURES})
    sp = D.assign_split(df, CFG)
    g = D.rating_group_key(df)
    assert pd.DataFrame({"g": g, "s": sp}).groupby("g")["s"].nunique().max() == 1
    assert (sp == D.assign_split(df, CFG)).all()                 # deterministic
    share = sp.value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.6, abs=0.1)


def test_large_loss_cap_and_load_preserve_total():
    claims = pd.DataFrame({"IDpol": [1, 1, 2, 3], "ClaimAmount": [100.0, 10000.0, 200.0, 300.0]})
    df = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimAmount": [10100.0, 200.0, 300.0]})
    train = pd.Series([True, True, True])
    out, load = D.apply_large_loss_cap(df, claims, 1000.0, train)
    assert out["ClaimAmountCapped"].tolist() == [1100.0, 200.0, 300.0]
    assert out["ClaimAmountCapped"].sum() * load == pytest.approx(df["ClaimAmount"].sum())
