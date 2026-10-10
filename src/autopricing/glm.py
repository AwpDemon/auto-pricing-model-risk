"""Frequency (Poisson, log link, log-exposure offset) and severity (Gamma, log link) GLMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import CONTINUOUS, GLMDesign


@dataclass
class FittedGLM:
    design: GLMDesign
    result: sm.genmod.generalized_linear_model.GLMResultsWrapper
    family: str

    @property
    def params(self) -> pd.Series:
        return pd.Series(self.result.params, index=self.design.columns)

    def predict_linear(self, df: pd.DataFrame) -> np.ndarray:
        return self.design.matrix(df) @ self.params.to_numpy()

    def predict(self, df: pd.DataFrame, exposure: np.ndarray | pd.Series | None = None) -> np.ndarray:
        """Expected value. For frequency without exposure this is the annual claim rate."""
        eta = self.predict_linear(df)
        if exposure is not None:
            eta = eta + np.log(np.asarray(exposure, dtype=float))
        return np.exp(eta)

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "link": "log",
            "features": self.design.features,
            "base_levels": self.design.base,
            "levels": self.design.levels,
            "coefficients": self.params.round(10).to_dict(),
            "deviance": float(self.result.deviance),
            "null_deviance": float(self.result.null_deviance),
            "n_obs": int(self.result.nobs),
            "dispersion_scale": float(self.result.scale),
        }


def fit_frequency(df: pd.DataFrame, features: list[str], max_iter: int = 100) -> FittedGLM:
    """Poisson GLM: log E[N] = log(exposure) + x'beta. The offset makes beta a log *rate*."""
    design = GLMDesign(features).fit(df, df["Exposure"])
    X = design.matrix(df)
    model = sm.GLM(df["ClaimNb"].to_numpy(dtype=float), X,
                   family=sm.families.Poisson(), offset=np.log(df["Exposure"].to_numpy()))
    # scale="X2": quasi-Poisson. Point estimates are unchanged; standard errors are inflated by the
    # Pearson dispersion so relativity CIs are not over-confident when counts are over-dispersed.
    res = model.fit(maxiter=max_iter, scale="X2")
    return FittedGLM(design, res, "poisson")


def severity_frame(df: pd.DataFrame, amount_col: str = "ClaimAmountCapped") -> pd.DataFrame:
    """Policies with at least one claim; response = average (capped) amount per claim."""
    s = df[df["ClaimNb"] > 0].copy()
    s["AvgSeverity"] = s[amount_col] / s["ClaimNb"]
    return s[s["AvgSeverity"] > 0]


def fit_severity(df: pd.DataFrame, features: list[str], amount_col: str = "ClaimAmountCapped",
                 max_iter: int = 100) -> FittedGLM:
    """Gamma GLM on average severity with var_weights = claim count.

    An average of n iid Gamma claims has variance phi*mu^2/n, so the claim count is the right
    precision weight (the frequency of the policy is irrelevant to the cost of a claim).
    """
    s = severity_frame(df, amount_col)
    design = GLMDesign(features).fit(s, s["ClaimNb"].astype(float))
    X = design.matrix(s)
    model = sm.GLM(s["AvgSeverity"].to_numpy(), X,
                   family=sm.families.Gamma(link=sm.families.links.Log()),
                   var_weights=s["ClaimNb"].to_numpy(dtype=float))
    res = model.fit(maxiter=max_iter)
    return FittedGLM(design, res, "gamma")


def relativities(fit: FittedGLM, df: pd.DataFrame, weight_col: str = "Exposure") -> pd.DataFrame:
    """Relativity table: exp(beta) with 95% Wald CI, p-value and the exposure in each level."""
    params = fit.params
    ci = pd.DataFrame(fit.result.conf_int(), index=fit.design.columns, columns=["lo", "hi"])
    pvals = pd.Series(fit.result.pvalues, index=fit.design.columns)
    rows = []
    for f in fit.design.features:
        if f in CONTINUOUS:
            rows.append({"factor": f, "level": "per unit", "relativity": np.exp(params[f]),
                         "ci_low": np.exp(ci.loc[f, "lo"]), "ci_high": np.exp(ci.loc[f, "hi"]),
                         "p_value": pvals[f], "weight": df[weight_col].sum(), "is_base": False})
            continue
        w = df.groupby(df[f].astype(str))[weight_col].sum()
        for lv in fit.design.levels[f]:
            col = f"{f}[{lv}]"
            base = lv == fit.design.base[f]
            rows.append({
                "factor": f, "level": lv,
                "relativity": 1.0 if base else np.exp(params[col]),
                "ci_low": 1.0 if base else np.exp(ci.loc[col, "lo"]),
                "ci_high": 1.0 if base else np.exp(ci.loc[col, "hi"]),
                "p_value": np.nan if base else pvals[col],
                "weight": float(w.get(lv, 0.0)), "is_base": base,
            })
    return pd.DataFrame(rows)
