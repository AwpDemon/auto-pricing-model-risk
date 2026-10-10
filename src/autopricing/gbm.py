"""LightGBM challenger models: Poisson frequency and Tweedie pure premium."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import metrics as M
from .features import gbm_frame


@dataclass
class FittedGBM:
    booster: lgb.Booster
    features: list[str]
    categorical: list[str]
    categories: dict[str, list[str]]
    objective: str
    params: dict
    best_iteration: int
    grid_results: list[dict]
    balance: float = 1.0      # multiplicative balance correction fitted on train (see predict)

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return gbm_frame(df, self.features, self.categorical, self.categories)[0]

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Booster output before the balance correction."""
        return self.booster.predict(self.frame(df), num_iteration=self.best_iteration)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted annual rate (claims per year, or loss per year for Tweedie), balance-corrected.

        Gradient boosting with shrinkage, L2 leaf penalties and early stopping does not satisfy the
        GLM balance property (sum of predictions = sum of actuals on the training data). Finding V-03
        measured the raw Tweedie model under-predicting training losses; the standard fix is one
        multiplicative factor estimated on the training data, which leaves the ranking (Gini) unchanged.
        """
        return self.predict_raw(df) * self.balance


def _fit(train: pd.DataFrame, valid: pd.DataFrame, target_rate_col: str, objective: str, cfg: dict,
         seed: int, extra: dict | None = None) -> FittedGBM:
    g = cfg["gbm"]
    Xtr, cats = gbm_frame(train, g["features"], g["categorical"])
    Xva, _ = gbm_frame(valid, g["features"], g["categorical"], cats)
    dtr = lgb.Dataset(Xtr, train[target_rate_col], weight=train["Exposure"], categorical_feature=g["categorical"],
                      free_raw_data=False)
    dva = lgb.Dataset(Xva, valid[target_rate_col], weight=valid["Exposure"], reference=dtr)
    extra = extra or {}
    p_tw = extra.get("tweedie_variance_power")

    def val_dev(pred: np.ndarray) -> float:
        y, w = valid[target_rate_col], valid["Exposure"]
        return M.poisson_deviance(y, pred, w) if objective == "poisson" else M.tweedie_deviance(y, pred, p_tw, w)

    results, best = [], None
    for nl, mcs in itertools.product(g["grid"]["num_leaves"], g["grid"]["min_child_samples"]):
        params = {"objective": objective, "learning_rate": g["learning_rate"], "num_leaves": nl,
                  "min_child_samples": mcs, "n_jobs": g["n_jobs"], "seed": seed, "verbose": -1,
                  "deterministic": True, "force_row_wise": True, **g["common"], **extra}
        booster = lgb.train(params, dtr, num_boost_round=g["num_boost_round"], valid_sets=[dva],
                            callbacks=[lgb.early_stopping(g["early_stopping_rounds"], verbose=False)])
        dev = val_dev(booster.predict(Xva, num_iteration=booster.best_iteration))
        results.append({"num_leaves": nl, "min_child_samples": mcs, "best_iteration": booster.best_iteration,
                        "valid_deviance": dev})
        if best is None or dev < best[0]:
            best = (dev, booster, params)
    _, booster, params = best
    fitted = FittedGBM(booster, g["features"], g["categorical"], cats, objective, params,
                       booster.best_iteration, results)
    raw_tr = fitted.predict_raw(train)
    w_tr = train["Exposure"].to_numpy()
    fitted.balance = float(np.sum(train[target_rate_col].to_numpy() * w_tr) / np.sum(raw_tr * w_tr))
    return fitted


def fit_frequency(train: pd.DataFrame, valid: pd.DataFrame, cfg: dict, seed: int) -> FittedGBM:
    tr = train.assign(Freq=train["ClaimNb"] / train["Exposure"])
    va = valid.assign(Freq=valid["ClaimNb"] / valid["Exposure"])
    return _fit(tr, va, "Freq", "poisson", cfg, seed)


def fit_pure_premium(train: pd.DataFrame, valid: pd.DataFrame, cfg: dict, seed: int, power: float | None = None,
                     amount_col: str = "ClaimAmountCapped") -> FittedGBM:
    power = cfg["gbm"]["tweedie_power"] if power is None else power
    tr = train.assign(PP=train[amount_col] / train["Exposure"])
    va = valid.assign(PP=valid[amount_col] / valid["Exposure"])
    return _fit(tr, va, "PP", "tweedie", cfg, seed, {"tweedie_variance_power": power})
