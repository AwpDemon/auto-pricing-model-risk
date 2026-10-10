"""Proxy-discrimination analysis with driver age as the sensitive attribute.

freMTPL2 contains no protected class (race, sex, religion, national origin). Driver age is the
only attribute that some regimes treat as sensitive, so it is used as a *stand-in* to demonstrate
the method: (1) how well the other rating factors can reconstruct it (proxy strength), and
(2) how prices change between the "aware", "unaware" and "discrimination-free" constructions of
Lindholm, Richman, Tsanakas and Wuthrich (2022, ASTIN Bulletin 52(1)).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import glm as G
from . import metrics as M
from .features import gbm_frame

NON_AGE = ["BonusMalus", "VehAge", "VehPower", "VehBrand", "VehGas", "Region", "Area", "LogDensity"]
NON_AGE_CAT = ["VehBrand", "VehGas", "Region", "Area"]


def proxy_strength(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, test_target: pd.Series,
                   seed: int, n_jobs: int = 8) -> dict:
    """AUC of a classifier that predicts a sensitive flag from the *other* rating factors.

    AUC 0.5 = the other factors carry no information about the attribute; close to 1 = a strong proxy.
    Also reports the AUC from BonusMalus alone to show where the information sits.
    """
    Xtr, cats = gbm_frame(train, NON_AGE, NON_AGE_CAT)
    Xte, _ = gbm_frame(test, NON_AGE, NON_AGE_CAT, cats)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
              "seed": seed, "verbose": -1, "n_jobs": n_jobs, "deterministic": True, "force_row_wise": True}
    booster = lgb.train(params, lgb.Dataset(Xtr, target.astype(int)), num_boost_round=300)
    p_all = booster.predict(Xte)
    gain = pd.Series(booster.feature_importance("gain"), index=NON_AGE)
    return {
        "auc_all_non_age_factors": float(roc_auc_score(test_target, p_all)),
        # direction-free: a factor that is *lower* for the group is just as informative
        "auc_bonus_malus_only": float(max(a := roc_auc_score(test_target, test["BonusMalus"]), 1 - a)),
        "top_proxy_factor": str(gain.idxmax()),
        "gain_share": (gain / gain.sum()).round(4).to_dict(),
        "prevalence_test": float(test_target.mean()),
    }


def pure_premium_glm(train: pd.DataFrame, freq_feats: list[str], sev_feats: list[str], max_iter: int):
    f = G.fit_frequency(train, freq_feats, max_iter)
    s = G.fit_severity(train, sev_feats, max_iter=max_iter)
    return f, s


def discrimination_free(freq: G.FittedGLM, sev: G.FittedGLM, df: pd.DataFrame, attr: str,
                        p_d: pd.Series) -> np.ndarray:
    """h*(x) = sum_d mu(x, d) P(d): average the aware price over the *marginal* of D.

    Because P(d) does not depend on x, D cannot leak back in through correlated factors.
    """
    out = np.zeros(len(df))
    for d, p in p_d.items():
        tmp = df.assign(**{attr: d})
        out += p * freq.predict(tmp) * sev.predict(tmp)
    return out


def price_comparison(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    """Fit aware / unaware GLM pure-premium models and compare prices by age band on test."""
    gcfg, attr = cfg["glm"], cfg["fairness"]["sensitive_attribute"]
    ff, sf = gcfg["frequency_features"], gcfg["severity_features"]
    aware_f, aware_s = pure_premium_glm(train, ff, sf, gcfg["max_iter"])
    un_f, un_s = pure_premium_glm(train, [f for f in ff if f != attr], [f for f in sf if f != attr], gcfg["max_iter"])
    p_d = train.groupby(attr)["Exposure"].sum()
    p_d = p_d / p_d.sum()

    def preds(d: pd.DataFrame) -> dict[str, np.ndarray]:
        return {"aware": aware_f.predict(d) * aware_s.predict(d),
                "unaware": un_f.predict(d) * un_s.predict(d),
                "disc_free": discrimination_free(aware_f, aware_s, d, attr, p_d)}

    tr_p, te_p = preds(train), preds(test)
    # Rebalance so each construction collects the same total premium on train (revenue neutral).
    w_tr = train["Exposure"].to_numpy()
    target_total = np.sum(tr_p["aware"] * w_tr)
    factors = {k: target_total / np.sum(v * w_tr) for k, v in tr_p.items()}
    te_p = {k: v * factors[k] for k, v in te_p.items()}

    w = test["Exposure"].to_numpy()
    y = (test["ClaimAmountCapped"] / test["Exposure"]).to_numpy()
    frame = pd.DataFrame({"band": test[attr].astype(str).to_numpy(), "w": w, "loss": y * w,
                          **{k: v * w for k, v in te_p.items()}})
    t = frame.groupby("band").sum()
    port = frame[["loss", *te_p]].sum() / frame["w"].sum()
    table = pd.DataFrame({"exposure": t["w"]})
    table["actual_pp"] = t["loss"] / t["w"]
    for k in te_p:
        table[f"{k}_pp"] = t[k] / t["w"]
        table[f"{k}_index"] = table[f"{k}_pp"] / port[k]       # premium relative to portfolio average
        table[f"{k}_ae"] = t["loss"] / t[k]
    order = cfg_order(attr)
    table = table.reindex([o for o in order if o in table.index])

    perf = {k: {"gini": M.gini(y, v, w)} for k, v in te_p.items()}
    return {"table": table.reset_index().rename(columns={"index": "band", attr: "band"}), "performance": perf,
            "rebalance_factors": factors, "p_d": p_d.to_dict()}


def cfg_order(attr: str) -> list[str]:
    from .features import band_order
    return band_order(attr) or []
