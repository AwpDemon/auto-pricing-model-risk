"""Ongoing-monitoring metrics with Red/Amber/Green thresholds.

The same function is meant to run on every new batch of policies once their losses have
developed: input stability (PSI per rating factor and on the score), outcome accuracy
(actual / expected) and discrimination power (Gini drop vs the validation baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M

NUMERIC = ["DrivAge", "BonusMalus", "VehAge", "VehPower", "LogDensity"]
CATEGORICAL = ["VehBrand", "VehGas", "Region", "Area"]


def rag(value: float, thr: dict) -> str:
    if np.isnan(value):
        return "n/a"
    return "RED" if value >= thr["red"] else "AMBER" if value >= thr["amber"] else "GREEN"


def monitoring_report(reference: pd.DataFrame, current: pd.DataFrame, score_col: str, loss_col: str,
                      baseline_gini: float, cfg: dict) -> dict:
    mc = cfg["monitoring"]
    thr = mc["thresholds"]
    rows = []
    for f in NUMERIC:
        v = M.psi(reference[f], current[f], bins=mc["psi_bins"])
        rows.append({"metric": f"PSI {f}", "value": v, "status": rag(v, thr["psi"])})
    for f in CATEGORICAL:
        v = M.psi(reference[f], current[f], categorical=True)
        rows.append({"metric": f"PSI {f}", "value": v, "status": rag(v, thr["psi"])})
    v = M.psi(reference[score_col], current[score_col], bins=mc["psi_bins"])
    rows.append({"metric": "PSI score", "value": v, "status": rag(v, thr["psi"])})

    w = current["Exposure"]
    ae = float(current[loss_col].sum() / (current[score_col] * w).sum())
    rows.append({"metric": "A/E (capped losses)", "value": ae, "status": rag(abs(ae - 1), thr["ae_ratio"])})
    g = M.gini(current[loss_col] / w, current[score_col], w)
    rows.append({"metric": "Gini", "value": g, "status": rag(baseline_gini - g, thr["gini_drop"])})
    table = pd.DataFrame(rows)
    worst = "RED" if (table["status"] == "RED").any() else "AMBER" if (table["status"] == "AMBER").any() else "GREEN"
    return {"table": table, "overall": worst, "n_policies": len(current), "exposure": float(w.sum())}
