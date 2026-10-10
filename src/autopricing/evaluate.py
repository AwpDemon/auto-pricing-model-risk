"""Model comparison on a hold-out set: frequency, severity and pure premium."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M


def frequency_metrics(df: pd.DataFrame, preds: dict[str, np.ndarray], n_bins: int) -> dict:
    """preds are annual claim rates. Deviance is exposure-weighted Poisson deviance of the rate."""
    y, w = (df["ClaimNb"] / df["Exposure"]).to_numpy(), df["Exposure"].to_numpy()
    out = {}
    null = np.full(len(df), preds["Null"][0]) if "Null" in preds else None
    d_null = M.poisson_deviance(y, null, w) if null is not None else np.nan
    for name, mu in preds.items():
        dev = M.poisson_deviance(y, mu, w)
        lt = M.lift_table(y, mu, w, n_bins)
        out[name] = {
            "poisson_deviance": dev,
            "deviance_explained": M.deviance_explained(dev, d_null) if name != "Null" else 0.0,
            "gini": M.gini(y, mu, w) if name != "Null" else 0.0,
            "ae_total": float((y * w).sum() / (mu * w).sum()),
            "lift_top_bottom": M.lift_ratio(lt) if name != "Null" else 1.0,
            "calibration": M.calibration_slope(y, mu, w) if name != "Null" else None,
        }
    return out


def pure_premium_metrics(df: pd.DataFrame, preds: dict[str, np.ndarray], power: float, n_bins: int,
                         loss_col: str = "ClaimAmountCapped") -> dict:
    y, w = (df[loss_col] / df["Exposure"]).to_numpy(), df["Exposure"].to_numpy()
    d_null = M.tweedie_deviance(y, preds["Null"], power, w)
    out = {}
    for name, mu in preds.items():
        dev = M.tweedie_deviance(y, mu, power, w)
        lt = M.lift_table(y, mu, w, n_bins)
        out[name] = {
            f"tweedie_deviance_p{power}": dev,
            "deviance_explained": M.deviance_explained(dev, d_null) if name != "Null" else 0.0,
            "gini": M.gini(y, mu, w) if name != "Null" else 0.0,
            "ae_total": float((y * w).sum() / (mu * w).sum()),
            "lift_top_bottom": M.lift_ratio(lt) if name != "Null" else 1.0,
            "calibration": M.calibration_slope(y, mu, w) if name != "Null" else None,
            "max_decile_abs_ae_error": float((lt["ae"] - 1).abs().max()) if name != "Null" else None,
        }
    return out


def severity_metrics(sev_df: pd.DataFrame, preds: dict[str, np.ndarray]) -> dict:
    y, w = sev_df["AvgSeverity"].to_numpy(), sev_df["ClaimNb"].to_numpy(dtype=float)
    d_null = M.gamma_deviance(y, preds["Null"], w)
    out = {}
    for name, mu in preds.items():
        dev = M.gamma_deviance(y, mu, w)
        out[name] = {"gamma_deviance": dev, "deviance_explained": M.deviance_explained(dev, d_null),
                     "gini": M.gini(y, mu, w) if name != "Null" else 0.0,
                     "ae_total": float((y * w).sum() / (mu * w).sum())}
    return out


def shap_relativities(contrib: np.ndarray, frame: pd.DataFrame, feature_names: list[str], df: pd.DataFrame,
                      feature: str, band_col: str, order: list[str]) -> pd.DataFrame:
    """Mean SHAP (log scale) by band, re-based so the exposure-weighted geometric mean relativity is 1."""
    j = feature_names.index(feature)
    s = pd.DataFrame({"band": df[band_col].astype(str).to_numpy(), "phi": contrib[:, j],
                      "w": df["Exposure"].to_numpy()})
    g = s.groupby("band").apply(lambda t: np.average(t["phi"], weights=t["w"]), include_groups=False)
    wt = s.groupby("band")["w"].sum()
    rel = np.exp(g - np.average(g, weights=wt.reindex(g.index)))
    return rel.reindex([o for o in order if o in rel.index])


def glm_band_relativities(rel_freq: pd.DataFrame, rel_sev: pd.DataFrame, df: pd.DataFrame, factor: str,
                          order: list[str]) -> pd.Series:
    """Freq x sev relativity per band, re-based to an exposure-weighted geometric mean of 1 on `df`."""
    rf = rel_freq[rel_freq["factor"] == factor].set_index("level")["relativity"]
    rs = rel_sev[rel_sev["factor"] == factor].set_index("level")["relativity"] if factor in set(rel_sev["factor"]) \
        else pd.Series(1.0, index=rf.index)
    log_r = np.log(rf * rs.reindex(rf.index).fillna(1.0))
    wt = df.groupby(df[factor].astype(str))["Exposure"].sum().reindex(log_r.index).fillna(0)
    # Same geometric re-basing as shap_relativities so the two curves are directly comparable.
    r = np.exp(log_r - np.average(log_r, weights=wt))
    return r.reindex([o for o in order if o in r.index])
