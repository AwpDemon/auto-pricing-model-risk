"""End-to-end pipeline: data -> EDA -> GLM + GBM -> validation -> fairness -> monitoring -> reports.

    python -m autopricing run            # full data (downloads ~7.7 MB from OpenML on first run)
    python -m autopricing run --sample   # committed 20k-policy sample, used by CI
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import data as D
from . import evaluate as E
from . import fairness as FA
from . import features as F
from . import gbm as B
from . import glm as G
from . import metrics as M
from . import monitoring as MO
from . import plots as P
from . import report as R
from . import sensitivity as S
from .config import ROOT, load_config

log = logging.getLogger("autopricing")


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.steps: dict[str, float] = {}
        self._last = self.t0

    def step(self, name: str) -> None:
        now = time.time()
        self.steps[name] = round(now - self._last, 1)
        log.info("%-28s %6.1fs", name, now - self._last)
        self._last = now


def run(config_path: str | None = None, sample: bool = False, skip_sensitivity: bool = False) -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    out = ROOT / "out" / "sample" if sample else ROOT
    fig_dir, rep_dir, tab_dir, model_dir = out / "docs" / "figures", out / "reports", out / "reports" / "tables", out / "models"
    for d in (fig_dir, rep_dir, tab_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    T = Timer()
    res: dict = {"config": {"seed": seed, "sample": sample}, "timings_s": T.steps}

    # ------------------------------------------------------------------ 1. data
    data_dir = ROOT / (cfg["data"]["sample_dir"] if sample else cfg["data"]["raw_dir"])
    if not sample and not (data_dir / "freMTPL2freq.pq").exists():
        D.download(cfg, ROOT)
    freq_raw, sev_raw = D.load_raw(data_dir)
    cr = D.clean(freq_raw, sev_raw, cfg)
    df = F.add_bands(cr.policies)
    split = D.assign_split(df, cfg)
    tr_mask = split == "train"
    cap = D.large_loss_cap(cr.claims, df.loc[tr_mask, "IDpol"], cfg["cleaning"]["large_loss_quantile"])
    df, load = D.apply_large_loss_cap(df, cr.claims, cap, tr_mask)
    df["split"] = split
    train, valid, test = (df[split == s].copy() for s in ("train", "validation", "test"))
    groups = D.rating_group_key(df)
    res["data"] = {
        **cr.log,
        "split_policies": {s: int((split == s).sum()) for s in ("train", "validation", "test")},
        "split_exposure": {s: float(df.loc[split == s, "Exposure"].sum()) for s in ("train", "validation", "test")},
        "split_claims": {s: int(df.loc[split == s, "ClaimNb"].sum()) for s in ("train", "validation", "test")},
        "rating_profiles": int(groups.nunique()),
        "rows_sharing_profile_share": float(groups.duplicated(keep=False).mean()),
        "profiles_crossing_splits": int(df.assign(g=groups).groupby("g")["split"].nunique().gt(1).sum()),
        "large_loss_cap": cap, "large_loss_load": load,
        "large_loss_share_of_train_losses": 1 - 1 / load,
        "claims_above_cap_train": int((cr.claims.loc[cr.claims["IDpol"].isin(train["IDpol"]), "ClaimAmount"] > cap).sum()),
        "train_claim_frequency": float(train["ClaimNb"].sum() / train["Exposure"].sum()),
        "train_mean_severity": float(cr.claims.loc[cr.claims["IDpol"].isin(train["IDpol"]), "ClaimAmount"].mean()),
        "train_median_severity": float(cr.claims.loc[cr.claims["IDpol"].isin(train["IDpol"]), "ClaimAmount"].median()),
        "train_pure_premium": float(train["ClaimAmount"].sum() / train["Exposure"].sum()),
        "area_logdensity_spearman": float(df["Area"].rank().corr(df["LogDensity"].rank())),
        # Large-loss volatility: how much of each split's cost sits above the cap, and its largest claim.
        "uncapped_to_capped_ratio_by_split": {
            s: float(df.loc[split == s, "ClaimAmount"].sum() / df.loc[split == s, "ClaimAmountCapped"].sum())
            for s in ("train", "validation", "test")},
        "largest_claim_by_split": {
            s: float(cr.claims.loc[cr.claims["IDpol"].isin(df.loc[split == s, "IDpol"]), "ClaimAmount"].max())
            for s in ("train", "validation", "test")},
    }
    T.step("data")

    # ------------------------------------------------------------------ 2. EDA
    orders = {k: v[2] for k, v in F.BANDS.items()}
    P.eda_oneway(train, ["DrivAgeBand", "BonusMalusBand", "VehAgeBand", "VehPowerBand", "Area", "VehGas"], orders,
                 fig_dir / "eda_oneway_frequency.png")
    tr_claims = cr.claims[cr.claims["IDpol"].isin(train["IDpol"])]
    P.severity_distribution(tr_claims["ClaimAmount"], cap, fig_dir / "eda_severity_distribution.png")
    eda = {}
    for f in ["DrivAgeBand", "BonusMalusBand", "VehAgeBand", "VehPowerBand", "Area", "VehGas", "Region", "VehBrand"]:
        t = train.groupby(train[f].astype(str)).agg(exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"),
                                                    loss=("ClaimAmountCapped", "sum"))
        t = t.reindex([o for o in orders.get(f, sorted(t.index)) if o in t.index])
        t["frequency"] = t["claims"] / t["exposure"]
        t["avg_severity_capped"] = t["loss"] / t["claims"].replace(0, np.nan)
        t["exposure_share"] = t["exposure"] / t["exposure"].sum()
        eda[f] = t.reset_index().rename(columns={f: "level"})
    R.write_table(pd.concat({k: v for k, v in eda.items()}, names=["factor"]).reset_index(level=0),
                  tab_dir / "eda_oneway.md", floatfmt=".4f")
    T.step("eda")

    # ------------------------------------------------------------------ 3. GLMs
    gcfg = cfg["glm"]
    glm_f = G.fit_frequency(train, gcfg["frequency_features"], gcfg["max_iter"])
    glm_s = G.fit_severity(train, gcfg["severity_features"], max_iter=gcfg["max_iter"])
    rel_f = G.relativities(glm_f, train)
    rel_s = G.relativities(glm_s, G.severity_frame(train), "ClaimNb")
    R.write_table(rel_f, tab_dir / "glm_frequency_relativities.md", floatfmt=".3f")
    R.write_table(rel_s, tab_dir / "glm_severity_relativities.md", floatfmt=".3f")
    rel_f.to_csv(tab_dir / "glm_frequency_relativities.csv", index=False)
    rel_s.to_csv(tab_dir / "glm_severity_relativities.csv", index=False)
    P.relativity_chart(rel_f, ["DrivAgeBand", "BonusMalusBand", "VehAgeBand"], fig_dir / "glm_frequency_relativities.png",
                       "Poisson GLM frequency relativities (quasi-Poisson 95% CI, base = highest-exposure level)")
    (model_dir / "glm_frequency.json").write_text(json.dumps(glm_f.to_dict(), indent=2))
    (model_dir / "glm_severity.json").write_text(json.dumps(glm_s.to_dict(), indent=2))
    res["glm"] = {
        "frequency": {"n_params": len(glm_f.design.columns), "pearson_dispersion": float(glm_f.result.pearson_chi2 / glm_f.result.df_resid),
                      "deviance_dispersion": float(glm_f.result.deviance / glm_f.result.df_resid),
                      "converged": bool(glm_f.result.converged), "train_deviance": float(glm_f.result.deviance),
                      "params_p_gt_0_05": int((pd.Series(glm_f.result.pvalues) > 0.05).sum())},
        "severity": {"n_params": len(glm_s.design.columns), "gamma_dispersion": float(glm_s.result.scale),
                     "converged": bool(glm_s.result.converged), "n_claim_policies": int(glm_s.result.nobs),
                     "params_p_gt_0_05": int((pd.Series(glm_s.result.pvalues) > 0.05).sum())},
    }
    T.step("glm")

    # ------------------------------------------------------------------ 4. GBM challengers
    gbm_f = B.fit_frequency(train, valid, cfg, seed)
    gbm_pp = B.fit_pure_premium(train, valid, cfg, seed)
    gbm_f.booster.save_model(str(model_dir / "lgbm_frequency.txt"), num_iteration=gbm_f.best_iteration)
    gbm_pp.booster.save_model(str(model_dir / "lgbm_tweedie_pure_premium.txt"), num_iteration=gbm_pp.best_iteration)
    res["gbm"] = {
        "frequency": {"best_params": {k: gbm_f.params[k] for k in ("num_leaves", "min_child_samples")},
                      "best_iteration": gbm_f.best_iteration, "balance_factor": gbm_f.balance,
                      "grid": gbm_f.grid_results},
        "pure_premium": {"best_params": {k: gbm_pp.params[k] for k in ("num_leaves", "min_child_samples")},
                         "best_iteration": gbm_pp.best_iteration, "tweedie_power": cfg["gbm"]["tweedie_power"],
                         "balance_factor": gbm_pp.balance,
                         "grid": gbm_pp.grid_results},
    }
    T.step("gbm")

    # ------------------------------------------------------------------ 5. Evaluation (validation + test)
    p = cfg["gbm"]["tweedie_power"]
    nb = cfg["evaluation"]["n_bins"]
    base_freq = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    base_pp = float(train["ClaimAmountCapped"].sum() / train["Exposure"].sum())
    sev_tr = G.severity_frame(train)
    base_sev = float(np.average(sev_tr["AvgSeverity"], weights=sev_tr["ClaimNb"]))
    bm_only = G.fit_frequency(train, ["BonusMalusBand"], gcfg["max_iter"])
    glm_int = G.fit_frequency(train, gcfg["frequency_features"] + gcfg["interaction_candidate"], gcfg["max_iter"])
    res["glm"]["frequency_interaction_candidate"] = {"terms": gcfg["interaction_candidate"],
                                                     "n_params": len(glm_int.design.columns),
                                                     "train_deviance": float(glm_int.result.deviance)}

    def predict_all(d: pd.DataFrame) -> dict:
        return {
            "freq": {"Null": np.full(len(d), base_freq), "GLM (BonusMalus only)": bm_only.predict(d),
                     "GLM": glm_f.predict(d), "GLM + age x BM interaction": glm_int.predict(d),
                     "LightGBM": gbm_f.predict(d)},
            "pp": {"Null": np.full(len(d), base_pp), "GLM": glm_f.predict(d) * glm_s.predict(d),
                   "GLM + age x BM interaction": glm_int.predict(d) * glm_s.predict(d),
                   "LightGBM": gbm_pp.predict(d), "LightGBM freq x GLM sev": gbm_f.predict(d) * glm_s.predict(d),
                   "LightGBM (before balance correction)": gbm_pp.predict_raw(d)},
        }

    for name, d in (("validation", valid), ("test", test)):
        pr = predict_all(d)
        sev_d = G.severity_frame(d)
        res[f"eval_{name}"] = {
            "frequency": E.frequency_metrics(d, pr["freq"], nb),
            "pure_premium": E.pure_premium_metrics(d, pr["pp"], p, nb),
            "severity": E.severity_metrics(sev_d, {"Null": np.full(len(sev_d), base_sev), "GLM": glm_s.predict(sev_d)}),
        }
    res["eval_train"] = {"pure_premium": E.pure_premium_metrics(train, predict_all(train)["pp"], p, nb)}

    pr = predict_all(test)
    test["pp_glm"], test["pp_gbm"] = pr["pp"]["GLM"], pr["pp"]["LightGBM"]
    test["freq_glm"], test["freq_gbm"] = pr["freq"]["GLM"], pr["freq"]["LightGBM"]
    y_pp, w = (test["ClaimAmountCapped"] / test["Exposure"]).to_numpy(), test["Exposure"].to_numpy()
    y_f = (test["ClaimNb"] / test["Exposure"]).to_numpy()
    lifts = {"GLM": M.lift_table(y_pp, test["pp_glm"], w, nb), "LightGBM": M.lift_table(y_pp, test["pp_gbm"], w, nb)}
    dl = M.double_lift(y_pp, test["pp_glm"], test["pp_gbm"], w, nb)
    for k, t in lifts.items():
        R.write_table(t, tab_dir / f"lift_pure_premium_{k.lower()}.md", floatfmt=".4f")
    R.write_table(dl, tab_dir / "double_lift_pure_premium.md", floatfmt=".4f")
    res["double_lift"] = {"glm_mean_abs_err": float(dl["base_abs_err"].mean()),
                          "gbm_mean_abs_err": float(dl["challenger_abs_err"].mean()),
                          "bins_gbm_closer": int((dl["challenger_abs_err"] < dl["base_abs_err"]).sum())}
    reps = cfg["evaluation"]["bootstrap_reps"] if not sample else 30
    res["bootstrap_gini_test"] = {
        "pure_premium": M.bootstrap_gini(y_pp, test["pp_glm"], test["pp_gbm"], w, reps, seed),
        "frequency": M.bootstrap_gini(y_f, test["freq_glm"], test["freq_gbm"], w, reps, seed),
    }
    P.lorenz({k: (*M.lorenz_curve(y_pp, v, w), M.gini(y_pp, v, w)) for k, v in
              {"GLM": test["pp_glm"], "LightGBM": test["pp_gbm"]}.items()},
             fig_dir / "lorenz_pure_premium.png", "Ordered Lorenz curve, pure premium (test)")
    P.lift_chart(lifts, fig_dir / "lift_pure_premium.png", "capped loss cost per exposure-year")
    P.double_lift_chart(dl, fig_dir / "double_lift.png", "GLM", "LightGBM")
    P.calibration_chart(lifts, fig_dir / "calibration_by_decile.png")
    ae_tabs = {}
    test["loss"] = test["ClaimAmountCapped"]
    test["e_glm"], test["e_gbm"] = test["pp_glm"] * test["Exposure"], test["pp_gbm"] * test["Exposure"]
    for f in ["DrivAgeBand", "BonusMalusBand", "Region", "VehBrand"]:
        t = M.actual_vs_expected(test, f, "loss", ["e_glm", "e_gbm"], "Exposure")
        t = t.rename(columns={"ae_e_glm": "ae_glm", "ae_e_gbm": "ae_gbm"})
        ae_tabs[f] = t
    R.write_table(pd.concat(ae_tabs, names=["factor"]).reset_index(level=0), tab_dir / "ae_by_factor_test.md", floatfmt=".3f")
    res["ae_by_factor_worst"] = {
        f: {"glm_max_abs_dev": float((t["ae_glm"] - 1).abs().where(t["Exposure"] > 1000).max()),
            "gbm_max_abs_dev": float((t["ae_gbm"] - 1).abs().where(t["Exposure"] > 1000).max())}
        for f, t in ae_tabs.items()}
    P.ae_by_factor(ae_tabs, orders, fig_dir / "ae_by_factor.png")
    T.step("evaluation")

    # ------------------------------------------------------------------ 6. Interpretability
    rng = np.random.default_rng(seed)
    n_shap = min(cfg["evaluation"]["shap_sample"], len(test))
    sh = test.iloc[np.sort(rng.choice(len(test), n_shap, replace=False))]
    contrib = gbm_pp.booster.predict(gbm_pp.frame(sh), num_iteration=gbm_pp.best_iteration, pred_contrib=True)
    raw = gbm_pp.booster.predict(gbm_pp.frame(sh), num_iteration=gbm_pp.best_iteration, raw_score=True)
    res["shap"] = {"n": n_shap, "additivity_max_abs_err": float(np.abs(contrib.sum(axis=1) - raw).max())}
    imp = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=gbm_pp.features)
    res["shap"]["mean_abs"] = imp.sort_values(ascending=False).round(4).to_dict()
    P.shap_importance(imp, fig_dir / "shap_importance.png")
    comp = {}
    for feat, band in (("DrivAge", "DrivAgeBand"), ("BonusMalus", "BonusMalusBand"), ("VehAge", "VehAgeBand")):
        s_rel = E.shap_relativities(contrib, gbm_pp.frame(sh), gbm_pp.features, sh, feat, band, orders[band])
        g_rel = E.glm_band_relativities(rel_f, rel_s, sh, band, orders[band])
        comp[band] = pd.DataFrame({"level": s_rel.index, "glm": g_rel.reindex(s_rel.index).to_numpy(),
                                   "gbm_shap": s_rel.to_numpy()})
    P.glm_vs_shap(comp, fig_dir / "glm_vs_shap.png")
    R.write_table(pd.concat(comp, names=["factor"]).reset_index(level=0), tab_dir / "glm_vs_shap_relativities.md",
                  floatfmt=".3f")
    T.step("interpretability")

    # ------------------------------------------------------------------ 7. Fairness / proxy
    young = cfg["fairness"]["young_driver_max_age"]
    proxy = {
        f"young_le_{young}": FA.proxy_strength(train, test, train["DrivAge"] <= young, test["DrivAge"] <= young, seed,
                                               cfg["gbm"]["n_jobs"]),
        "senior_ge_71": FA.proxy_strength(train, test, train["DrivAge"] >= 71, test["DrivAge"] >= 71, seed,
                                          cfg["gbm"]["n_jobs"]),
    }
    pc = FA.price_comparison(train, test, cfg)
    R.write_table(pc["table"], tab_dir / "fairness_price_by_age.md", floatfmt=".3f")
    P.fairness_chart(pc["table"], fig_dir / "fairness_age_proxy.png")
    bm_by_age = train.groupby("DrivAgeBand").apply(lambda t: np.average(t["BonusMalus"], weights=t["Exposure"]),
                                                   include_groups=False)
    res["fairness"] = {"proxy": proxy, "performance": pc["performance"], "rebalance_factors": pc["rebalance_factors"],
                       "mean_bonus_malus_by_age": bm_by_age.reindex(orders["DrivAgeBand"]).round(2).to_dict(),
                       "price_index_young_18_20": {k: float(pc["table"].set_index("band").loc["18-20", f"{k}_index"])
                                                   for k in ("aware", "unaware", "disc_free")},
                       "price_index_range": {k: [float(pc["table"][f"{k}_index"].min()), float(pc["table"][f"{k}_index"].max())]
                                             for k in ("aware", "unaware", "disc_free")}}
    T.step("fairness")

    # ------------------------------------------------------------------ 8. Monitoring scenarios
    train["pp_glm"] = glm_f.predict(train) * glm_s.predict(train)
    valid_pp = glm_f.predict(valid) * glm_s.predict(valid)
    base_gini = M.gini(valid["ClaimAmountCapped"] / valid["Exposure"], valid_pp, valid["Exposure"])
    mon, mon_tables = {}, {}
    for sc in cfg["monitoring"]["scenarios"]:
        cur = test if sc["filter"] is None else test.query(sc["filter"])
        if len(cur) < 50:
            continue
        r = MO.monitoring_report(train, cur, "pp_glm", "ClaimAmountCapped", base_gini, cfg)
        mon[sc["name"]] = {"overall": r["overall"], "n_policies": r["n_policies"], "exposure": r["exposure"],
                           "metrics": r["table"].round(4).to_dict(orient="records")}
        mon_tables[sc["name"]] = r["table"]
    res["monitoring"] = {"baseline_validation_gini": base_gini, "scenarios": mon}
    P.psi_chart(mon_tables, cfg["monitoring"]["thresholds"]["psi"], fig_dir / "monitoring_psi.png")
    R.write_table(pd.concat(mon_tables, names=["scenario"]).reset_index(level=0), tab_dir / "monitoring_scenarios.md",
                  floatfmt=".4f")
    T.step("monitoring")

    # ------------------------------------------------------------------ 9. Sensitivity
    if not skip_sensitivity:
        sc = cfg["sensitivity"]
        sens = {
            "data_decisions": S.data_decisions(freq_raw, sev_raw, cfg, test["IDpol"].to_numpy()),
            "split_seeds": S.split_seeds(df, cfg, sc["split_seeds"]),
            "large_loss": S.large_loss(df.drop(columns=["ClaimAmountCapped"]), cr.claims, split, cfg,
                                       sc["large_loss_quantiles"]),
            "tweedie_power": S.tweedie_power(train, valid, test, cfg, gbm_pp.params, sc["tweedie_powers"]),
            "split_leakage": S.split_leakage(df, cfg, gbm_f.params, max(gbm_f.best_iteration, 50)),
        }
        for k, t in sens.items():
            R.write_table(t, tab_dir / f"sensitivity_{k}.md", floatfmt=".4f")
        res["sensitivity"] = {k: t.to_dict(orient="records") for k, t in sens.items()}
        T.step("sensitivity")

    # ------------------------------------------------------------------ 10. Write
    res["timings_s"]["total"] = round(time.time() - T.t0, 1)
    (rep_dir / "metrics.json").write_text(json.dumps(R.jsonable(res), indent=2))
    if not sample:
        R.inject_all(ROOT, res)
    log.info("done in %.1fs -> %s", time.time() - T.t0, rep_dir / "metrics.json")
    return res
