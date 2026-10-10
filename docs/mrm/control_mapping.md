# Control mapping (crosswalk)

Maps what the frameworks ask for to the evidence in this repository, and states the gap where the
repository cannot provide it. This is a portfolio exercise: it shows how a pricing model's documentation
would be organised to answer these frameworks. It is **not legal advice** and not a compliance opinion.

**Frameworks referenced** (retrieved and checked 2026-10-10):

| Short name | Document | Status / scope |
|---|---|---|
| NAIC | NAIC Model Bulletin: Use of Artificial Intelligence Systems by Insurers, adopted 4 Dec 2023 | Bulletin interpreting existing law; adopted by about 25 states + DC by mid/late 2026 (secondary sources, check NAIC map). Insurers. |
| NYDFS | NY DFS Insurance Circular Letter No. 7 (2024), 11 Jul 2024 | AI systems and external consumer data in underwriting and pricing. NY-authorised insurers. |
| CO | Colorado SB21-169; 3 CCR 702-10 Regulation 10-1-1 as amended Aug 2025 (effective 15 Oct 2025) | Governance and risk-management framework for ECDIS, algorithms and predictive models; extended to private passenger auto and health benefit plans; full compliance for those lines due 1 Jul 2026. |
| NIST | NIST AI Risk Management Framework 1.0 (Jan 2023) | Voluntary; functions Govern, Map, Measure, Manage. |
| SR 26-2 | Fed/OCC/FDIC Revised Guidance on Model Risk Management, 17 Apr 2026 | **Supersedes SR 11-7 (2011)** and SR 21-8. Bank guidance; used here as the reference vocabulary for model risk management, not because it binds insurers. |
| EU AI Act | Regulation (EU) 2024/1689, Annex III 5(c) | High-risk: risk assessment and pricing for natural persons in **life and health** insurance. **Motor TPL pricing is not listed**; the Act's high-risk obligations do not apply to this model as such. |

## Crosswalk

Status: **Met** = evidence in repo; **Partial** = demonstrated with a stated limitation; **Gap** = would need
organisational evidence a single-author repo cannot provide.

| # | Requirement (paraphrased) | Source | Evidence in this repo | Status |
|---|---|---|---|---|
| C1 | Maintain an inventory of models/AI systems with owner, purpose, risk rating | NAIC (AIS program: governance); CO; SR 26-2 (inventory); NIST GOVERN 1.6 | `docs/mrm/model_inventory.yaml` | Met |
| C2 | Risk-based tiering: depth of validation proportional to materiality and complexity | SR 26-2; NAIC (risk-based approach); NIST MAP 5.1 | Inventory tier + rationale; challenger deliberately tier-raised | Met |
| C3 | Documented model purpose, intended use, out-of-scope uses, limitations | NAIC; NIST MAP 1.1, MAP 2.2; SR 26-2 (development documentation) | `model_card.md`, `model_development.md` | Met |
| C4 | Data governance: provenance, licensing, quality, suitability, lineage | NAIC (data practices); NYDFS (ECDIS data quality); CO (ECDIS inventory); NIST MAP 2.3 / MEASURE 2.1 | `data/raw/MANIFEST.json` (URL, sha256, licence), DQ-1..DQ-4 counts in `reports/metrics.json`, `issues_log.md` | Met |
| C5 | Conceptual soundness of the method for its purpose | SR 26-2 (validation); NAIC (validation, testing) | `validation_report.md` section 2 | Met (self-validated, see C14) |
| C6 | Outcomes analysis: back-testing, calibration, discrimination power on hold-out data | SR 26-2; NAIC; NIST MEASURE 2.3, 2.5 | `validation_report.md` section 4; lift/calibration/A/E tables in `reports/tables/` | Partial: no out-of-time data |
| C7 | Benchmarking against an alternative model | SR 26-2; NIST MEASURE 2.5 | LightGBM challenger, BonusMalus-only GLM, null model; paired bootstrap; double lift | Met |
| C8 | Sensitivity and stability analysis of key assumptions | SR 26-2; NIST MEASURE 2.5 | `reports/tables/sensitivity_*.md` (data decisions, split seed, large-loss cap, Tweedie power, split leakage) | Met |
| C9 | Testing for unfair discrimination / proxy discrimination against protected classes | NAIC (bias testing); NYDFS (quantitative assessment, proxy analysis, at least annually); CO (quantitative testing w.r.t. race); NIST MEASURE 2.11 | `fairness_price_by_age.md`, proxy AUC, discrimination-free price | **Partial**: no protected class in data; method shown with age as stand-in. Real compliance needs BIFSG-style inference or collected data. |
| C10 | Explainability: ability to explain factors driving a decision to regulators and consumers | NYDFS (adverse decision reasons); CO (consumer information); NIST MEASURE 2.9 | GLM relativity tables with CIs; SHAP for challenger; GLM-vs-SHAP comparison | Met for GLM; Partial for GBM |
| C11 | Ongoing monitoring for drift and performance degradation, with thresholds and escalation | NAIC; CO (monitor model drift); SR 26-2; NIST MANAGE 4.1, MEASURE 2.4 | `monitoring_plan.md`, `monitoring.py`, scenario results | Met (design + demonstration; no live feed) |
| C12 | Issue tracking and remediation | SR 26-2; NIST MANAGE 1.3 | `issues_log.md` | Met |
| C13 | Version control, reproducibility, change management | NAIC (internal controls); NIST GOVERN 1.4 | git history, pinned `requirements.txt`, seeded config, CI, one-command pipeline, generated numbers injected into docs | Met |
| C14 | Independent validation / effective challenge by competent, independent, influential reviewers | SR 26-2; NAIC (validation) | Validation written in an independent style | **Gap**: developer and validator are the same person |
| C15 | Board / senior-management oversight, written AIS program, accountable owner | NAIC; NYDFS; CO | Owner field in inventory | **Gap**: organisational |
| C16 | Third-party / vendor model and data oversight | NAIC; NYDFS | Not applicable: no vendor data or models (open-source libraries pinned) | n/a |
| C17 | Consumer notice and specific reasons for adverse decisions | NYDFS; CO | Relativity tables would support reason codes | Gap: no consumer process |
| C18 | Record retention of testing and documentation | NAIC; CO | Committed reports/tables/figures, metrics.json | Partial |

## NIST AI RMF function view

| Function | Where it shows up here |
|---|---|
| **Govern** | Inventory and tiering (C1, C2), issues log ownership (C12), change control (C13); organisational parts are gaps (C14, C15) |
| **Map** | Model card context, intended and out-of-scope use, stakeholders, impact on policyholders' premiums (C3), data provenance (C4) |
| **Measure** | Deviance, Gini, calibration, A/E, bootstrap (C6, C7), sensitivity (C8), proxy and discrimination-free analysis (C9), explainability (C10) |
| **Manage** | Monitoring thresholds and escalation (C11), remediation plan for each finding (C12), decision on champion vs challenger |
