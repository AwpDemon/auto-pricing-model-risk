"""Batch scoring with the saved GLM artefacts (no refit, no statsmodels needed at score time)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .features import CONTINUOUS, add_bands

REQUIRED = ["DrivAge", "BonusMalus", "VehAge", "VehPower", "VehBrand", "VehGas", "Region", "Density"]


class SavedGLM:
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.coef = spec["coefficients"]

    def linear(self, df: pd.DataFrame) -> np.ndarray:
        eta = np.full(len(df), self.coef["Intercept"])
        for f in self.spec["features"]:
            if f in CONTINUOUS:
                eta += self.coef[f] * df[f].to_numpy(dtype=float)
                continue
            v = df[f].astype(str).to_numpy()
            for lv in self.spec["levels"][f]:
                key = f"{f}[{lv}]"
                if key in self.coef:
                    eta += self.coef[key] * (v == lv)
        return eta


def score_frame(df: pd.DataFrame, model_dir: Path = ROOT / "models", metrics_path: Path = ROOT / "reports" / "metrics.json") -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    freq = SavedGLM(json.loads((model_dir / "glm_frequency.json").read_text()))
    sev = SavedGLM(json.loads((model_dir / "glm_severity.json").read_text()))
    load = json.loads(metrics_path.read_text())["data"]["large_loss_load"]
    x = add_bands(df.assign(LogDensity=np.log(df["Density"].clip(lower=1.0))))
    out = df.copy()
    out["expected_frequency"] = np.exp(freq.linear(x))
    out["expected_severity_capped"] = np.exp(sev.linear(x))
    out["risk_premium"] = out["expected_frequency"] * out["expected_severity_capped"] * load
    return out


def score_csv(inp: Path, outp: Path) -> None:
    score_frame(pd.read_csv(inp)).to_csv(outp, index=False)
