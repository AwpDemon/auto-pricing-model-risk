"""Markdown rendering and injection of generated numbers into README and the validation report.

Documents carry marker pairs (<!-- BEGIN:name --> / <!-- END:name -->). Everything between a pair
is regenerated from reports/metrics.json on every full run, so prose and numbers cannot drift.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def write_table(df: pd.DataFrame, path: Path, floatfmt: str = ".4f") -> None:
    path.write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n", encoding="utf-8")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating | float):
        v = float(o)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _f(v, fmt=".4f"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)


def render_results(res: dict) -> str:
    d, ev = res["data"], res["eval_test"]
    fq, pp = ev["frequency"], ev["pure_premium"]
    pkey = next(k for k in pp["GLM"] if k.startswith("tweedie_deviance"))
    lines = [
        f"Hold-out **test** set: {d['split_policies']['test']:,} policies, {d['split_exposure']['test']:,.0f} exposure-years, "
        f"{d['split_claims']['test']:,} claims (never used for fitting or tuning).",
        "",
        "**Pure premium (frequency x severity), capped losses**",
        "",
        "| Model | Tweedie deviance (p=1.5) | Deviance explained | Gini | Top/bottom decile lift | Calibration slope | A/E |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["Null", "GLM", "GLM + age x BM interaction", "LightGBM", "LightGBM (before balance correction)",
                 "LightGBM freq x GLM sev"]:
        m = pp[name]
        cal = m["calibration"]["slope"] if m["calibration"] else None
        lines.append(f"| {name} | {_f(m[pkey], '.2f')} | {_f(m['deviance_explained'], '.2%')} | {_f(m['gini'], '.3f')} | "
                     f"{_f(m['lift_top_bottom'], '.1f')}x | {_f(cal, '.3f')} | {_f(m['ae_total'], '.3f')} |")
    lines += ["", "**Claim frequency (Poisson, exposure offset)**", "",
              "| Model | Poisson deviance | Deviance explained | Gini | Calibration slope | A/E |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in ["Null", "GLM (BonusMalus only)", "GLM", "GLM + age x BM interaction", "LightGBM"]:
        m = fq[name]
        cal = m["calibration"]["slope"] if m["calibration"] else None
        lines.append(f"| {name} | {_f(m['poisson_deviance'], '.4f')} | {_f(m['deviance_explained'], '.2%')} | "
                     f"{_f(m['gini'], '.3f')} | {_f(cal, '.3f')} | {_f(m['ae_total'], '.3f')} |")
    b = res["bootstrap_gini_test"]
    lines += ["",
              f"Paired bootstrap ({b['pure_premium']['reps']} resamples of the test set): LightGBM minus GLM Gini = "
              f"{b['pure_premium']['diff_mean']:+.3f} on pure premium (95% CI {b['pure_premium']['diff_ci'][0]:+.3f} to "
              f"{b['pure_premium']['diff_ci'][1]:+.3f}) and {b['frequency']['diff_mean']:+.3f} on frequency "
              f"(95% CI {b['frequency']['diff_ci'][0]:+.3f} to {b['frequency']['diff_ci'][1]:+.3f}).",
              ]
    sv = ev["severity"]
    lines += ["", f"Severity GLM (Gamma) on test claims: deviance explained {sv['GLM']['deviance_explained']:.2%}, "
                  f"Gini {sv['GLM']['gini']:.3f}. Severity is close to unpredictable from these rating factors; "
                  "almost all of the pricing signal is in frequency."]
    return "\n".join(lines)


def render_data(res: dict) -> str:
    d = res["data"]
    return "\n".join([
        "| Step | Rows affected |", "|---|---:|",
        f"| Raw policies (freMTPL2freq) | {d['raw_policies']:,} |",
        f"| Raw claim records (freMTPL2sev) | {d['raw_claim_records']:,} |",
        f"| DQ-1: claims-only block removed (IDpol < {d['dq1_block_boundary_idpol']:,}; {d['dq1_block_share_with_claims']:.0%} have claims, "
        f"{d['dq1_block_claims_missing_amount_share']:.1%} of those have no amount) | {d['dq1_block_policies']:,} |",
        f"| DQ-2: policies whose ClaimNb disagreed with claim records (reconciled) | {d['dq2_policies_count_mismatch']:,} |",
        f"| DQ-3: policies with ClaimNb capped at 4 | {d['dq3_policies_claimnb_capped']:,} |",
        f"| DQ-4: policies with exposure capped at 1 year | {d['dq4_policies_exposure_capped']:,} |",
        f"| Clean policies / exposure-years / claims | {d['clean_policies']:,} / {d['clean_exposure_years']:,.0f} / {d['clean_claims']:,} |",
        f"| Large-loss cap (99.5th pct of train claims) / claims above it / load factor | {d['large_loss_cap']:,.0f} / "
        f"{d['claims_above_cap_train']:,} / {d['large_loss_load']:.3f} |",
        f"| Split train / validation / test (policies) | {d['split_policies']['train']:,} / {d['split_policies']['validation']:,} / "
        f"{d['split_policies']['test']:,} |",
    ])


def render_fairness(res: dict) -> str:
    fa = res["fairness"]
    young = next(k for k in fa["proxy"] if k.startswith("young"))
    px = fa["proxy"][young]
    pi = fa["price_index_young_18_20"]
    perf = fa["performance"]
    return "\n".join([
        f"- Other rating factors predict a young driver (age <= {young.split('_')[-1]}) with AUC **{px['auc_all_non_age_factors']:.3f}** "
        f"(BonusMalus alone: {px['auc_bonus_malus_only']:.3f}; top proxy by gain: {px['top_proxy_factor']}).",
        f"- Price index for drivers aged 18-20 (1.00 = portfolio average): aware **{pi['aware']:.2f}**, "
        f"unaware (age removed) **{pi['unaware']:.2f}**, discrimination-free **{pi['disc_free']:.2f}**.",
        f"- Test Gini: aware {perf['aware']['gini']:.3f}, unaware {perf['unaware']['gini']:.3f}, "
        f"discrimination-free {perf['disc_free']['gini']:.3f}.",
    ])


def render_monitoring(res: dict) -> str:
    lines = ["| Scenario | Policies | Overall | Worst PSI (factor) | A/E |", "|---|---:|---|---|---:|"]
    for name, sc in res["monitoring"]["scenarios"].items():
        psis = [m for m in sc["metrics"] if m["metric"].startswith("PSI")]
        worst = max(psis, key=lambda m: m["value"])
        ae = next(m for m in sc["metrics"] if m["metric"].startswith("A/E"))
        lines.append(f"| {name} | {sc['n_policies']:,} | {sc['overall']} | {worst['value']:.3f} ({worst['metric'][4:]}) | "
                     f"{ae['value']:.3f} |")
    return "\n".join(lines)


BLOCKS = {"results": render_results, "data": render_data, "fairness": render_fairness, "monitoring": render_monitoring}


def inject(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def inject_all(root: Path, res: dict) -> None:
    targets = [root / "README.md", *(root / "docs" / "mrm").glob("*.md")]
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new = text
        for name, fn in BLOCKS.items():
            if f"<!-- BEGIN:{name} -->" in new:
                new = inject(new, name, fn(res))
        if new != text:
            path.write_text(new, encoding="utf-8")
