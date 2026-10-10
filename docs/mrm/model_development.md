# Model development document: MTPL-RP-GLM-001

Developer's record of how the model was built and why. The validation report challenges it.
All counts and metrics are in `reports/metrics.json` and `reports/tables/`; data counts are injected below.

## 1. Purpose and scope

Estimate the **expected annual claim cost (risk premium)** of a French motor third-party-liability policy
from its rating factors, as the loss-cost component of a rate indication. Out of scope: expenses, profit
and contingency loads, underwriting accept/decline, reserving, any other product or jurisdiction.

## 2. Data

| Item | Value |
|---|---|
| Source | OpenML 41214 `freMTPL2freq` (policies) and 41215 `freMTPL2sev` (claims); parquet mirrors at data.openml.org |
| Licence | CC0 per OpenML; upstream R package CASdatasets (Dutang and Charpentier) is GPL (>= 2) |
| Retrieval | `python -m autopricing download` writes `data/raw/MANIFEST.json` with URL, size and sha256 |
| Unit | Policy-period: exposure (years), claim count, rating factors; claims: one row per claim with amount |
| Period | Not documented by the source |

<!-- BEGIN:data -->
| Step | Rows affected |
|---|---:|
| Raw policies (freMTPL2freq) | 678,013 |
| Raw claim records (freMTPL2sev) | 26,639 |
| DQ-1: claims-only block removed (IDpol < 24,952; 100% have claims, 96.9% of those have no amount) | 9,387 |
| DQ-2: policies whose ClaimNb disagreed with claim records (reconciled) | 23 |
| DQ-3: policies with ClaimNb capped at 4 | 8 |
| DQ-4: policies with exposure capped at 1 year | 1,224 |
| Clean policies / exposure-years / claims | 668,626 / 353,560 / 26,098 |
| Large-loss cap (99.5th pct of train claims) / claims above it / load factor | 39,196 / 79 / 1.258 |
| Split train / validation / test (policies) | 401,585 / 133,414 / 133,627 |
<!-- END:data -->

Cleaning decisions DQ-1 to DQ-4 and the large-loss cap are explained, with their rationale, in
`docs/INTERVIEW_GUIDE.md` section 2 and logged in `issues_log.md`. Each is re-tested in the validation
sensitivity analysis.

**Split.** 60/20/20 train/validation/test. Assignment is by a hash of the full rating profile (all nine
rating features) and the seed, so identical profiles always land in the same split (38% of rows share a
profile with another row) and a profile's assignment does not depend on which other rows are present.

## 3. Methodology

### 3.1 Frequency

Poisson GLM, log link, offset log(exposure):
`log E[N] = log(exposure) + b0 + sum_j b_j x_j`.

Rating factors: driver age (8 bands), bonus-malus (7 bands), vehicle age (6 bands), vehicle power (6 bands),
vehicle brand (11 levels), fuel (2), region (21 levels), log(population density) linear. Base level of each
categorical factor = level with the highest training exposure. Standard errors: quasi-Poisson (Pearson
dispersion) because the data are over-dispersed relative to Poisson.

Excluded: `Area` (Spearman 0.98 with log density: a coarsening of the same information).

### 3.2 Severity

Gamma GLM, log link, on the average capped claim amount per claim-bearing policy, `var_weights` = claim
count. Factors: driver age band, bonus-malus band, vehicle age band, fuel, log density. Brand, power and
region were not used: severity signal is weak and these add many parameters for little evidence.

### 3.3 Large losses

Individual claims are capped at the 99.5th percentile of training claim amounts. The excess is charged as a
flat multiplicative **large-loss load** = total training losses / total capped training losses.

### 3.4 Risk premium

`risk premium = frequency GLM x severity GLM x large-loss load`. Saved coefficients (`models/*.json`) and
the load (`reports/metrics.json`) are all that is needed to score; `python -m autopricing score` does this
without refitting.

### 3.5 Challengers and benchmarks

| Model | Role |
|---|---|
| Null (one rate for everyone) | Floor for deviance explained |
| GLM with bonus-malus only | Shows how much of the signal one factor carries |
| GLM + age x bonus-malus interaction (9 product cells) | v1.1 candidate raised by validation finding V-04 |
| LightGBM Poisson (frequency) and LightGBM Tweedie p = 1.5 (pure premium) | ML challengers; small grid on validation deviance, early stopping, then a multiplicative balance correction fitted on training data |
| LightGBM frequency x GLM severity | Isolates whether the GBM's advantage is in frequency |

## 4. Development testing

- Unit tests (`tests/`): deviance formulas against scikit-learn; Gini, Lorenz, PSI, double-lift properties;
  GLM recovers known coefficients on simulated data within 3.5 standard errors; exposure offset scaling;
  Poisson balance property; Gamma severity recovery; data-cleaning rules on toy data; split integrity;
  discrimination-free price invariance; monitoring RAG logic; README numbers equal `metrics.json`.
- End-to-end run on a committed 20k-policy sample in CI.

## 5. Assumptions and limitations (developer view)

1. Frequency and severity independent given rating factors.
2. Claims-only block (DQ-1) is a non-representative extract; inferred from the data, not documented by the source.
3. Recorded claim amounts are treated as final (no development / IBNR).
4. Large-loss excess is not rating-factor dependent (flat load).
5. Random grouped split; no out-of-time sample exists in this dataset.
6. Banding is judgemental (chosen from one-way exposure and frequency); finer bands or splines could fit better.
