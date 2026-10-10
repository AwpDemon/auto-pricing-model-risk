"""Rating-factor engineering for the GLM and feature frames for the GBM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Bands are chosen so each level carries enough exposure for a credible relativity and so that
# the shape seen in the one-way EDA (e.g. the steep young-driver gradient) can be expressed.
BANDS: dict[str, tuple[str, list[float], list[str]]] = {
    "DrivAgeBand": ("DrivAge", [17, 20, 25, 30, 40, 50, 60, 70, np.inf],
                    ["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]),
    "BonusMalusBand": ("BonusMalus", [0, 50, 57, 65, 75, 90, 100, np.inf],
                       ["50", "51-57", "58-65", "66-75", "76-90", "91-100", "101+"]),
    "VehAgeBand": ("VehAge", [-1, 0, 2, 5, 10, 15, np.inf], ["0", "1-2", "3-5", "6-10", "11-15", "16+"]),
    "VehPowerBand": ("VehPower", [0, 4, 5, 6, 7, 8, np.inf], ["4", "5", "6", "7", "8", "9+"]),
}
CONTINUOUS = {"LogDensity"}


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for band, (src, edges, labels) in BANDS.items():
        out[band] = pd.cut(out[src], bins=edges, labels=labels, right=True).astype(str)
    # Interaction candidate (finding V-04): bonus-malus means something different for a 22-year-old
    # (new driver, starts at 100) than for a 50-year-old (100 = recent claims). Coarse bands keep cells credible.
    age = pd.cut(out["DrivAge"], [0, 25, 30, 45, np.inf], labels=["<=25", "26-30", "31-45", "46+"]).astype(str)
    bm = pd.cut(out["BonusMalus"], [0, 50, 75, 100, np.inf], labels=["50", "51-75", "76-100", "101+"]).astype(str)
    # Treatment-coded product terms only: cells involving age 46+ or BM 50 are "other" (the base), so
    # with the finer main effects in the model the 9 interaction coefficients stay identifiable.
    cell = age + "|" + bm
    out["AgeXBM"] = np.where((age == "46+") | (bm == "50"), "other", cell)
    return out


def band_order(name: str) -> list[str] | None:
    return BANDS[name][2] if name in BANDS else None


@dataclass
class GLMDesign:
    """Treatment-coded design with base level = highest-exposure level (actuarial convention).

    Choosing the largest level as base makes every relativity read as "relative to the most
    common risk", and gives the intercept the smallest standard error.
    """

    features: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)
    base: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, weight: pd.Series) -> GLMDesign:
        self.columns = ["Intercept"]
        for f in self.features:
            if f in CONTINUOUS:
                self.columns.append(f)
                continue
            expo = weight.groupby(df[f].astype(str)).sum()
            order = band_order(f) or sorted(expo.index)
            order = [lv for lv in order if lv in expo.index]
            self.levels[f] = order
            self.base[f] = str(expo.idxmax())
            self.columns += [f"{f}[{lv}]" for lv in order if lv != self.base[f]]
        return self

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Design matrix written straight into one float64 array (no intermediate DataFrame copy)."""
        X = np.zeros((len(df), len(self.columns)))
        X[:, 0] = 1.0
        pos = {c: i for i, c in enumerate(self.columns)}
        for f in self.features:
            if f in CONTINUOUS:
                X[:, pos[f]] = df[f].to_numpy(dtype=float)
                continue
            v = df[f].astype(str).to_numpy()
            for lv in self.levels[f]:
                if lv != self.base[f]:
                    X[:, pos[f"{f}[{lv}]"]] = v == lv
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, np.ndarray] = {"Intercept": np.ones(len(df))}
        for f in self.features:
            if f in CONTINUOUS:
                cols[f] = df[f].to_numpy(dtype=float)
                continue
            v = df[f].astype(str).to_numpy()
            # Unseen levels fall back to the base level (documented limitation).
            for lv in self.levels[f]:
                if lv != self.base[f]:
                    cols[f"{f}[{lv}]"] = (v == lv).astype(float)
        return pd.DataFrame(cols, index=df.index)[self.columns]


def gbm_frame(df: pd.DataFrame, features: list[str], categorical: list[str],
              categories: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Feature frame for LightGBM with a fixed category vocabulary (fit on train)."""
    X = df[features].copy()
    cats = categories or {c: sorted(df[c].astype(str).unique()) for c in categorical}
    for c in categorical:
        X[c] = pd.Categorical(df[c].astype(str), categories=cats[c])
    return X, cats
