"""Sensitivity analyses for the validation report: how much do results move when a judgement changes?"""

from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import data as D
from . import features as F
from . import glm as G
from . import metrics as M
from .features import gbm_frame

KEY_FACTORS = ["DrivAgeBand", "BonusMalusBand"]


def _freq_fit_eval(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    m = G.fit_frequency(train, cfg["glm"]["frequency_features"], cfg["glm"]["max_iter"])
    y, w = (test["ClaimNb"] / test["Exposure"]).to_numpy(), test["Exposure"].to_numpy()
    mu = m.predict(test)
    return {"model": m, "gini": M.gini(y, mu, w), "deviance": M.poisson_deviance(y, mu, w),
            "train_freq": float(train["ClaimNb"].sum() / train["Exposure"].sum())}


def _key_rel(m: G.FittedGLM) -> dict:
    p = m.params
    return {c: float(np.exp(p[c])) for c in p.index if any(c.startswith(k) for k in KEY_FACTORS)}


def data_decisions(freq_raw: pd.DataFrame, sev_raw: pd.DataFrame, cfg: dict, base_test_ids: np.ndarray) -> pd.DataFrame:
    """Re-run cleaning with one decision flipped at a time; evaluate on the *baseline* clean test set."""
    variants = {
        "baseline (all DQ decisions on)": {},
        "DQ-1 off: keep claims-only block with its reported counts": {"exclude_claims_only_block": False,
                                                                      "reconcile_claim_counts": False},
        "DQ-1 off: keep block, counts reconciled to amounts": {"exclude_claims_only_block": False},
        "DQ-2 off: reported counts, not reconciled": {"reconcile_claim_counts": False},
        "DQ-3 off: claim counts not capped": {"cap_claim_nb": 99},
        "DQ-4 off: exposure not capped": {"cap_exposure": 99.0},
    }
    rows, base_rel = [], None
    base_clean = F.add_bands(D.clean(freq_raw, sev_raw, cfg).policies)
    test = base_clean[base_clean["IDpol"].isin(base_test_ids)]
    y, w = (test["ClaimNb"] / test["Exposure"]).to_numpy(), test["Exposure"].to_numpy()
    for name, override in variants.items():
        c = {**cfg, "cleaning": {**cfg["cleaning"], **override}}
        df = F.add_bands(D.clean(freq_raw, sev_raw, c).policies)
        split = D.assign_split(df, cfg)     # hash-based: same profiles land in the same split in every variant
        train = df[(split == "train") & ~df["IDpol"].isin(base_test_ids)]
        r = _freq_fit_eval(train, test, cfg)
        rel = _key_rel(r["model"])
        if base_rel is None:
            base_rel = rel
        shift = max(abs(np.log(rel[k] / base_rel[k])) for k in base_rel)
        mu = r["model"].predict(test)
        rows.append({"variant": name, "train_policies": len(train), "train_claim_frequency": r["train_freq"],
                     "test_ae_frequency": float((y * w).sum() / (mu * w).sum()),
                     "test_gini": r["gini"], "test_poisson_deviance": r["deviance"],
                     "max_abs_log_relativity_shift_age_bm": shift,
                     "rel_DrivAge_18_20": rel.get("DrivAgeBand[18-20]"), "rel_BM_101plus": rel.get("BonusMalusBand[101+]")})
        del r, df, train, split
        gc.collect()
    return pd.DataFrame(rows)


def split_seeds(df: pd.DataFrame, cfg: dict, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        sp = D.assign_split(df, cfg, seed=s)
        r = _freq_fit_eval(df[sp == "train"], df[sp == "test"], cfg)
        rel = _key_rel(r["model"])
        rows.append({"seed": s, "test_gini": r["gini"], "test_poisson_deviance": r["deviance"],
                     "rel_DrivAge_18_20": rel.get("DrivAgeBand[18-20]"), "rel_BM_101plus": rel.get("BonusMalusBand[101+]")})
    return pd.DataFrame(rows)


def large_loss(df: pd.DataFrame, claims: pd.DataFrame, split: pd.Series, cfg: dict, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    tr_mask = split == "train"
    for q in quantiles:
        cap = D.large_loss_cap(claims, df.loc[tr_mask, "IDpol"], q)
        d2, load = D.apply_large_loss_cap(df, claims, cap, tr_mask)
        tr, te = d2[tr_mask], d2[split == "test"]
        f = G.fit_frequency(tr, cfg["glm"]["frequency_features"], cfg["glm"]["max_iter"])
        s = G.fit_severity(tr, cfg["glm"]["severity_features"], max_iter=cfg["glm"]["max_iter"])
        pp = f.predict(te) * s.predict(te) * load
        y, w = (te["ClaimAmount"] / te["Exposure"]).to_numpy(), te["Exposure"].to_numpy()
        rows.append({"cap_quantile": q, "cap_amount": cap if np.isfinite(cap) else None, "large_loss_load": load,
                     "test_gini_uncapped_losses": M.gini(y, pp, w),
                     "test_ae_uncapped_losses": float((y * w).sum() / (pp * w).sum()),
                     "sev_rel_DrivAge_18_20": float(np.exp(s.params.get("DrivAgeBand[18-20]", np.nan)))})
    return pd.DataFrame(rows)


def tweedie_power(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, cfg: dict, best_params: dict,
                  powers: list[float]) -> pd.DataFrame:
    g = cfg["gbm"]
    Xtr, cats = gbm_frame(train, g["features"], g["categorical"])
    Xva, _ = gbm_frame(valid, g["features"], g["categorical"], cats)
    Xte, _ = gbm_frame(test, g["features"], g["categorical"], cats)
    ytr = train["ClaimAmountCapped"] / train["Exposure"]
    yva = valid["ClaimAmountCapped"] / valid["Exposure"]
    yte, wte = (test["ClaimAmountCapped"] / test["Exposure"]).to_numpy(), test["Exposure"].to_numpy()
    rows = []
    for p in powers:
        params = {**best_params, "tweedie_variance_power": p}
        dtr = lgb.Dataset(Xtr, ytr, weight=train["Exposure"], categorical_feature=g["categorical"])
        dva = lgb.Dataset(Xva, yva, weight=valid["Exposure"], reference=dtr)
        b = lgb.train(params, dtr, g["num_boost_round"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(g["early_stopping_rounds"], verbose=False)])
        mu = b.predict(Xte, num_iteration=b.best_iteration)
        rows.append({"tweedie_power": p, "best_iteration": b.best_iteration, "test_gini": M.gini(yte, mu, wte),
                     "test_ae_before_balance_correction": float((yte * wte).sum() / (mu * wte).sum())})
    return pd.DataFrame(rows)


def split_leakage(df: pd.DataFrame, cfg: dict, best_params: dict, n_rounds: int) -> pd.DataFrame:
    """Does a naive row-level split flatter the GBM? Compare grouped vs ungrouped splits."""
    g = cfg["gbm"]
    rows = []
    for grouped in (True, False):
        c = {**cfg, "split": {**cfg["split"], "group_by_rating_features": grouped}}
        sp = D.assign_split(df, c)
        tr, te = df[sp == "train"], df[sp == "test"]
        Xtr, cats = gbm_frame(tr, g["features"], g["categorical"])
        Xte, _ = gbm_frame(te, g["features"], g["categorical"], cats)
        t0 = time.time()
        b = lgb.train(best_params, lgb.Dataset(Xtr, tr["ClaimNb"] / tr["Exposure"], weight=tr["Exposure"],
                                               categorical_feature=g["categorical"]), n_rounds)
        y, w = (te["ClaimNb"] / te["Exposure"]).to_numpy(), te["Exposure"].to_numpy()
        mu_te = b.predict(Xte)
        mu_tr = b.predict(Xtr)
        ytr, wtr = (tr["ClaimNb"] / tr["Exposure"]).to_numpy(), tr["Exposure"].to_numpy()
        rows.append({"split": "grouped by rating profile" if grouped else "naive row-level",
                     "train_gini": M.gini(ytr, mu_tr, wtr), "test_gini": M.gini(y, mu_te, w),
                     "test_poisson_deviance": M.poisson_deviance(y, mu_te, w), "seconds": round(time.time() - t0, 1)})
    return pd.DataFrame(rows)
