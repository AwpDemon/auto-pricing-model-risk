"""Static figures (PNG) for the README and the validation report.

One categorical order everywhere: GLM = blue, GBM = orange, third series = aqua, actual = ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
GLM_C, GBM_C, THIRD_C, FOURTH_C = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
STATUS = {"GREEN": "#1a7f37", "AMBER": "#c98500", "RED": "#d1242f"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "lines.linewidth": 2,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eda_oneway(df: pd.DataFrame, factors: list[str], orders: dict[str, list[str]], path: Path) -> None:
    """Observed claim frequency by level; exposure share printed under each bar."""
    n = len(factors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.4 * rows))
    overall = df["ClaimNb"].sum() / df["Exposure"].sum()
    for ax, f in zip(np.ravel(axes), factors, strict=False):
        t = df.groupby(df[f].astype(str)).agg(expo=("Exposure", "sum"), n=("ClaimNb", "sum"))
        t = t.reindex([o for o in orders.get(f, sorted(t.index)) if o in t.index])
        t["freq"] = t["n"] / t["expo"]
        share = t["expo"] / t["expo"].sum()
        x = np.arange(len(t))
        ax.bar(x, t["freq"], color=GLM_C, width=0.7)
        ax.axhline(overall, color=INK2, lw=1, ls="--")
        ax.set_xticks(x, [f"{lv}\n{s:.0%}" for lv, s in zip(t.index, share, strict=True)], fontsize=7)
        ax.set_title(f)
        ax.set_ylabel("claims per exposure-year")
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    fig.suptitle("Observed claim frequency by rating factor (train). Under each bar: share of exposure. "
                 "Dashed: portfolio mean", fontsize=10, color=INK2)
    _save(fig, path)


def severity_distribution(claims: pd.Series, cap: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(np.log10(claims.clip(lower=1)), bins=80, color=GLM_C)
    ax.axvline(np.log10(cap), color=INK, lw=1.5, ls="--")
    ax.text(np.log10(cap), ax.get_ylim()[1] * 0.9, f"  large-loss cap {cap:,.0f}", color=INK, fontsize=8)
    ax.set_xlabel("claim amount (log10 scale, currency units as published)")
    ax.set_ylabel("claims")
    ax.set_title("Individual claim severity (train)")
    _save(fig, path)


def lorenz(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    colors = [GLM_C, GBM_C, THIRD_C, FOURTH_C]
    for (name, (x, y, g)), c in zip(curves.items(), colors, strict=False):
        ax.plot(x, y, color=c, label=f"{name} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="random (Gini 0)")
    ax.set_xlabel("cumulative share of exposure (sorted by predicted risk, low to high)")
    ax.set_ylabel("cumulative share of actual losses")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def lift_chart(tables: dict[str, pd.DataFrame], path: Path, ylabel: str) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, t), c in zip(axes, tables.items(), [GLM_C, GBM_C], strict=False):
        ax.bar(t["bin"], t["actual_rate"], color=GRID, width=0.7, label="actual")
        ax.plot(t["bin"], t["predicted_rate"], color=c, marker="o", ms=5, label="predicted")
        ax.set_xticks(t["bin"])
        ax.set_xlabel("decile of predicted risk (equal exposure)")
        ax.set_title(f"{name}: lift = {t['actual_rate'].iloc[-1] / t['actual_rate'].iloc[0]:.1f}x")
        ax.legend(loc="upper left")
    axes[0].set_ylabel(ylabel)
    _save(fig, path)


def double_lift_chart(t: pd.DataFrame, path: Path, base: str, chal: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t["bin"], t["actual"], color=INK, marker="s", ms=5, label="actual")
    ax.plot(t["bin"], t["base"], color=GLM_C, marker="o", ms=5, label=base)
    ax.plot(t["bin"], t["challenger"], color=GBM_C, marker="o", ms=5, label=chal)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xticks(t["bin"])
    ax.set_xlabel(f"decile of {chal} / {base} prediction ratio")
    ax.set_ylabel("loss cost relative to portfolio mean")
    ax.set_title("Double lift: where the models disagree, which one is right?")
    ax.legend(loc="upper left")
    _save(fig, path)


def calibration_chart(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (name, t), c, off in zip(tables.items(), [GLM_C, GBM_C], [-0.15, 0.15], strict=False):
        ax.bar(t["bin"] + off, t["ae"] - 1, width=0.3, bottom=1, color=c, label=name)
    ax.axhline(1, color=INK, lw=1)
    ax.axhspan(0.9, 1.1, color=GRID, alpha=0.5, lw=0)
    ax.set_xticks(range(1, len(next(iter(tables.values()))) + 1))
    ax.set_xlabel("decile of predicted risk")
    ax.set_ylabel("actual / expected")
    ax.set_title("Calibration by decile (test). Shaded band: +/-10%")
    ax.legend(loc="upper right")
    _save(fig, path)


def ae_by_factor(tables: dict[str, pd.DataFrame], orders: dict[str, list[str]], path: Path) -> None:
    n = len(tables)
    fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(12, 3.4 * int(np.ceil(n / 2))))
    for ax, (f, t) in zip(np.ravel(axes), tables.items(), strict=False):
        t = t.set_index("level")
        t = t.reindex([o for o in orders.get(f, sorted(t.index)) if o in t.index])
        x = np.arange(len(t))
        ax.plot(x, t["ae_glm"], color=GLM_C, marker="o", ms=4, label="GLM")
        ax.plot(x, t["ae_gbm"], color=GBM_C, marker="o", ms=4, label="LightGBM")
        ax.axhline(1, color=INK, lw=1)
        ax.set_xticks(x, t.index, rotation=90 if len(t) > 9 else 0, fontsize=7)
        ax.set_title(f"A/E by {f} (test, capped pure premium)")
        ax.set_ylabel("actual / expected")
        ax.legend(loc="upper right")
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    _save(fig, path)


def relativity_chart(rel: pd.DataFrame, factors: list[str], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(factors), figsize=(4.2 * len(factors), 3.6))
    for ax, f in zip(np.atleast_1d(axes), factors, strict=False):
        t = rel[rel["factor"] == f]
        x = np.arange(len(t))
        ax.errorbar(x, t["relativity"], yerr=[t["relativity"] - t["ci_low"], t["ci_high"] - t["relativity"]],
                    fmt="o", color=GLM_C, ms=5, capsize=3, lw=1.5)
        base = t.reset_index(drop=True).index[t["is_base"].to_numpy()]
        ax.plot(base, [1.0] * len(base), "s", color=INK, ms=6, label="base level")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_xticks(x, t["level"], rotation=45, fontsize=7)
        ax.set_title(f)
        ax.legend(loc="upper center")
    np.atleast_1d(axes)[0].set_ylabel("relativity (95% CI)")
    fig.suptitle(title, fontsize=10, color=INK2)
    _save(fig, path)


def glm_vs_shap(comp: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(comp), figsize=(5.2 * len(comp), 3.8))
    for ax, (f, t) in zip(np.atleast_1d(axes), comp.items(), strict=False):
        x = np.arange(len(t))
        ax.plot(x, t["glm"], color=GLM_C, marker="o", ms=5, label="GLM relativity (freq x sev)")
        ax.plot(x, t["gbm_shap"], color=GBM_C, marker="o", ms=5, label="LightGBM SHAP-implied")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_xticks(x, t["level"], rotation=45, fontsize=7)
        ax.set_title(f)
        ax.set_ylabel("relativity (exposure-weighted geometric mean = 1)")
        ax.legend(loc="upper right")
    _save(fig, path)


def shap_importance(imp: pd.Series, path: Path) -> None:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(imp.index, imp.to_numpy(), color=GBM_C, height=0.6)
    for i, v in enumerate(imp.to_numpy()):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("mean |SHAP| on log pure premium (test sample)")
    ax.set_title("LightGBM Tweedie: what drives the prediction")
    _save(fig, path)


def fairness_chart(t: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(t))
    actual_idx = t["actual_pp"] / (t["actual_pp"] * t["exposure"]).sum() * t["exposure"].sum()
    ax.bar(x, actual_idx, color=GRID, width=0.7, label="actual loss cost")
    ax.plot(x, t["aware_index"], color=GLM_C, marker="o", ms=5, label="aware (uses age)")
    ax.plot(x, t["unaware_index"], color=GBM_C, marker="o", ms=5, label="unaware (age dropped)")
    ax.plot(x, t["disc_free_index"], color=THIRD_C, marker="o", ms=5, label="discrimination-free")
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xticks(x, t["band"])
    ax.set_xlabel("driver age band")
    ax.set_ylabel("price relative to portfolio average")
    ax.set_title("Price by driver age: aware vs unaware vs discrimination-free (test)")
    ax.legend(loc="upper right")
    _save(fig, path)


def psi_chart(reports: dict[str, pd.DataFrame], thr: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(reports)
    metrics = [m for m in reports[names[0]]["metric"] if m.startswith("PSI")]
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    for i, (name, c) in enumerate(zip(names, [GLM_C, GBM_C, THIRD_C], strict=False)):
        t = reports[name].set_index("metric").loc[metrics]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, t["value"], width=width * 0.9, color=c, label=name)
    ax.axhline(thr["amber"], color=STATUS["AMBER"], lw=1, ls="--")
    ax.axhline(thr["red"], color=STATUS["RED"], lw=1, ls="--")
    ax.text(-0.6, thr["amber"], "amber 0.10", color=INK2, fontsize=7, va="bottom", ha="left")
    ax.text(-0.6, thr["red"], "red 0.25", color=INK2, fontsize=7, va="bottom", ha="left")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(x, [m.replace("PSI ", "") for m in metrics], rotation=30, fontsize=8)
    ax.set_ylabel("PSI vs training book (symlog)")
    ax.set_title("Monitoring: population stability by scenario")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.32))
    _save(fig, path)
