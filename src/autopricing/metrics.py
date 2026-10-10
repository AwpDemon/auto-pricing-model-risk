"""Actuarial evaluation metrics, written out explicitly so each can be tested and explained.

Conventions: `y` is an observed *rate* (claims or loss per exposure year), `mu` a predicted rate,
`w` the exposure. Deviances are exposure-weighted means of the unit deviance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# --------------------------------------------------------------------------- deviances
def poisson_deviance(y, mu, w=None) -> float:
    y, mu = _arr(y), _arr(mu)
    w = np.ones_like(y) if w is None else _arr(w)
    unit = 2.0 * (xlogy(y, y / mu) - (y - mu))
    return float(np.sum(w * unit) / np.sum(w))


def gamma_deviance(y, mu, w=None) -> float:
    y, mu = _arr(y), _arr(mu)
    w = np.ones_like(y) if w is None else _arr(w)
    unit = 2.0 * (-np.log(y / mu) + (y - mu) / mu)
    return float(np.sum(w * unit) / np.sum(w))


def tweedie_deviance(y, mu, p: float, w=None) -> float:
    """Unit deviance for 1 < p < 2 (compound Poisson-Gamma)."""
    y, mu = _arr(y), _arr(mu)
    w = np.ones_like(y) if w is None else _arr(w)
    unit = 2.0 * (np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
                  - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))
    return float(np.sum(w * unit) / np.sum(w))


def deviance_explained(dev_model: float, dev_null: float) -> float:
    """D^2: share of null deviance removed by the model (the GLM analogue of R^2)."""
    return 1.0 - dev_model / dev_null


# --------------------------------------------------------------------------- ranking
def lorenz_curve(y, mu, w) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve: policies sorted by predicted rate (low -> high).

    x = cumulative share of exposure, y = cumulative share of actual losses/claims.
    """
    y, mu, w = _arr(y), _arr(mu), _arr(w)
    order = np.argsort(mu, kind="mergesort")
    loss = (y * w)[order]
    x = np.concatenate([[0.0], np.cumsum(w[order]) / w.sum()])
    lc = np.concatenate([[0.0], np.cumsum(loss) / loss.sum()])
    return x, lc


def gini(y, mu, w) -> float:
    """Gini = 1 - 2 * area under the ordered Lorenz curve (0 = random, higher = better ranking)."""
    x, lc = lorenz_curve(y, mu, w)
    return float(1.0 - 2.0 * np.trapezoid(lc, x))


def bootstrap_gini(y, mu_a, mu_b, w, reps: int, seed: int) -> dict:
    """Paired bootstrap of Gini(a), Gini(b) and the difference b - a on the same resamples."""
    y, mu_a, mu_b, w = _arr(y), _arr(mu_a), _arr(mu_b), _arr(w)
    rng = np.random.default_rng(seed)
    n = len(y)
    ga, gb = np.empty(reps), np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        ga[r], gb[r] = gini(y[i], mu_a[i], w[i]), gini(y[i], mu_b[i], w[i])
    d = gb - ga
    return {"a_ci": [float(np.percentile(ga, 2.5)), float(np.percentile(ga, 97.5))],
            "b_ci": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
            "diff_mean": float(d.mean()), "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_b_better": float((d > 0).mean()), "reps": reps}


# --------------------------------------------------------------------------- tables
def exposure_bins(score, w, n_bins: int) -> np.ndarray:
    """Assign equal-exposure bins (1..n) by ascending score."""
    score, w = _arr(score), _arr(w)
    order = np.argsort(score, kind="mergesort")
    cum = np.cumsum(w[order]) / w.sum()
    b = np.minimum((cum * n_bins - 1e-12).astype(int) + 1, n_bins)
    out = np.empty(len(score), dtype=int)
    out[order] = b
    return out


def lift_table(y, mu, w, n_bins: int = 10) -> pd.DataFrame:
    """Actual vs expected by equal-exposure decile of predicted rate (calibration + lift)."""
    y, mu, w = _arr(y), _arr(mu), _arr(w)
    b = exposure_bins(mu, w, n_bins)
    df = pd.DataFrame({"bin": b, "w": w, "actual": y * w, "expected": mu * w})
    t = df.groupby("bin").sum()
    t["actual_rate"] = t["actual"] / t["w"]
    t["predicted_rate"] = t["expected"] / t["w"]
    t["ae"] = t["actual"] / t["expected"]
    t = t.rename(columns={"w": "exposure"})
    return t[["exposure", "actual_rate", "predicted_rate", "ae"]].reset_index()


def lift_ratio(table: pd.DataFrame) -> float:
    """Top-decile actual rate / bottom-decile actual rate."""
    return float(table["actual_rate"].iloc[-1] / table["actual_rate"].iloc[0])


def double_lift(y, mu_base, mu_chal, w, n_bins: int = 10) -> pd.DataFrame:
    """Double-lift chart: sort by challenger/base ratio; within each bin compare both models to actual.

    Each series is divided by its own portfolio mean so only *relative* accuracy is compared.
    Where the challenger is higher than the base model, the challenger 'wins' if actual is too.
    """
    y, mu_base, mu_chal, w = _arr(y), _arr(mu_base), _arr(mu_chal), _arr(w)
    b = exposure_bins(mu_chal / mu_base, w, n_bins)
    df = pd.DataFrame({"bin": b, "w": w, "actual": y * w, "base": mu_base * w, "chal": mu_chal * w})
    t = df.groupby("bin").sum()
    tot = t[["actual", "base", "chal"]].sum() / t["w"].sum()
    out = pd.DataFrame({
        "bin": t.index, "exposure": t["w"].to_numpy(),
        "actual": (t["actual"] / t["w"] / tot["actual"]).to_numpy(),
        "base": (t["base"] / t["w"] / tot["base"]).to_numpy(),
        "challenger": (t["chal"] / t["w"] / tot["chal"]).to_numpy(),
    })
    out["base_abs_err"] = (out["base"] - out["actual"]).abs()
    out["challenger_abs_err"] = (out["challenger"] - out["actual"]).abs()
    return out


def calibration_slope(y, mu, w) -> dict:
    """Fit log E[y] = a + b * log(mu) by weighted Poisson quasi-likelihood. Ideal: a=0, b=1.

    b < 1 means predictions are too spread out (over-fitting); b > 1 means too compressed.
    """
    import statsmodels.api as sm

    y, mu, w = _arr(y), _arr(mu), _arr(w)
    X = sm.add_constant(np.log(mu))
    res = sm.GLM(y, X, family=sm.families.Poisson(), var_weights=w).fit()
    return {"intercept": float(res.params[0]), "slope": float(res.params[1]),
            "slope_ci": [float(v) for v in res.conf_int()[1]]}


def actual_vs_expected(df: pd.DataFrame, factor: str, actual_col: str, pred_cols: list[str], w_col: str) -> pd.DataFrame:
    """One-way A/E by rating-factor level. actual_col/pred_cols are *totals* (not rates)."""
    g = df.groupby(df[factor].astype(str))
    t = g[[w_col, actual_col, *pred_cols]].sum()
    for c in pred_cols:
        t[f"ae_{c}"] = t[actual_col] / t[c]
    t["actual_rate"] = t[actual_col] / t[w_col]
    return t.reset_index().rename(columns={factor: "level"})


# --------------------------------------------------------------------------- stability
def psi(expected, actual, bins: int = 10, categorical: bool = False, eps: float = 1e-6) -> float:
    """Population Stability Index: sum (a - e) * ln(a / e) over bins defined on `expected`.

    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
    """
    if categorical:
        e = pd.Series(expected).astype(str).value_counts(normalize=True)
        a = pd.Series(actual).astype(str).value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=0).to_numpy(), a.reindex(idx, fill_value=0).to_numpy()
    else:
        expected, actual = _arr(expected), _arr(actual)
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
