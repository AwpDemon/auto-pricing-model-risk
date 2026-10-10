# Interview guide: auto-pricing-model-risk

This guide is for Ali. It teaches every method in the repo from scratch, explains each design decision
and why the alternative was rejected, and gives practice answers for the questions an actuarial,
pricing-analytics or model-risk interviewer is likely to ask. All numbers quoted here come from
`reports/metrics.json` (full-data run, `python -m autopricing run`). If you re-run with a different
config, re-read the numbers from that file instead of from this guide.

Read it in order once. Before an interview, re-read section 9 (questions) and section 10 (limits).

---

## 0. The 30-second version (memorise this)

> "I priced French motor third-party liability insurance on 669 thousand real policies. I built the
> classic actuarial model, a Poisson frequency GLM with an exposure offset times a Gamma severity GLM,
> and a LightGBM Tweedie challenger, and then I validated both the way a model-risk team would:
> deviance, Gini and Lorenz curves, lift and double-lift charts, calibration by decile, actual-versus-expected
> by rating factor, bootstrap confidence intervals, and sensitivity tests on every data decision.
> I found and documented real data problems (a 9,387-row block of claims-only records with 97% of
> amounts missing), showed that dropping driver age from the model does not remove age from the price
> because the bonus-malus score is a proxy, and packaged it with a model card, validation report,
> monitoring plan with PSI thresholds, and a control mapping to the NAIC AI bulletin, NIST AI RMF and
> the bank model-risk guidance."

Then stop and let them pick a thread. Every thread is covered below.

---

## 1. The business problem

An insurer must charge each policyholder, **before** the year starts, a premium that covers the claims
that policyholder is expected to cause, plus expenses and profit. The part that covers expected claims
is the **risk premium** or **pure premium**:

```
pure premium = E[total claim cost per year of cover]
             = E[number of claims per year] x E[cost per claim]
             = frequency x severity
```

The split into frequency times severity is valid when the number of claims and the size of each claim
are independent given the rating factors (the standard "collective risk model"). It is useful because
the two parts behave very differently: frequency is driven by who drives and how (age, bonus-malus,
density), severity is noisy and dominated by a few huge bodily-injury claims.

**Third-party liability (TPL)** covers damage the insured driver causes to *other people*. It is
compulsory in France and across the EU. Bodily-injury liability is why a few claims run into the millions.

**Rating factors** are what the insurer is allowed to and chooses to price on: driver age, vehicle age,
power, brand, fuel, region, population density, and the **bonus-malus** coefficient.

**Bonus-malus (BM)**: the French system ("coefficient de réduction-majoration") starts a new driver at
100, multiplies by 0.95 for each claim-free year (down to a floor of 50) and by 1.25 for each at-fault
claim. So BM = 50 means a long claim-free history; BM above 100 means recent claims. It is the single
strongest predictor, and it is mechanically related to age (you cannot reach 50 without about 13
claim-free years), which is the heart of the fairness section.

---

## 2. The data and every cleaning decision

**Source.** `freMTPL2freq` (one row per policy: exposure, claim count, rating factors) and `freMTPL2sev`
(one row per claim: amount), OpenML datasets 41214 and 41215. OpenML lists the licence as **CC0**; the
original R package they come from, CASdatasets (Dutang and Charpentier), is licensed GPL (>= 2). Say
both if asked; the repo commits only a 20k-row sample and a download script.

**Exposure** is the fraction of a year the policy was in force (0.5 = six months). A policy observed
for half a year has half the chance of a claim, so exposure must enter the model (section 3).

Every cleaning step is counted by the code (`data.clean`), and each one is a finding in the issues log:

| ID | Problem found | Decision | Why |
|---|---|---|---|
| DQ-1 | Policies with IDpol below 24,952 (9,387 rows) **all** have a claim and 97% of those claims have no amount in the severity table. Claim frequency in that block is about 2 per year versus about 0.07 elsewhere. | Exclude the block | It is an extract of claim records, not a sample of the portfolio. Leaving it in inflates the base frequency and tilts relativities toward whatever profile those rows have. Sensitivity test shows the effect. |
| DQ-2 | 23 remaining policies have a claim count that disagrees with the number of claim records. | Set ClaimNb = number of claim records with an amount | Frequency and severity must count the same claims, otherwise frequency x severity double-counts or misses cost. |
| DQ-3 | 8 policies have more than 4 claims (up to 16). | Cap at 4 for the frequency target | Probably data errors or fleet-like records; with a Poisson likelihood they get large leverage. Amounts are not capped by this. |
| DQ-4 | 1,224 policies have exposure above 1 year. | Cap exposure at 1 | Annual contracts; > 1 is a data artefact. |
| Large losses | The top 0.5% of training claims carry about 21% of all training claim cost (cap about 39k vs a mean claim of about 2.2k and a median of 1.2k; the largest claim in the data is 4.08M). | Cap each claim at the 99.5th percentile of **training** claims and spread the excess back as a flat load factor (1.258) | One 4-million claim in the training data would otherwise drive the severity coefficients of whatever cell it fell in. Large bodily-injury claims are close to random with respect to rating factors, so charging them as a flat load is the standard actuarial treatment. The cap is computed on training data only so the test set is untouched. |

**Split: train 60 / validation 20 / test 20, grouped by rating profile.** About 38% of rows share their
full rating profile with another row. Some are probably the same car and driver over two contract
periods. If one copy is in training and the other in test, the test score is flattered (leakage), and
flexible models (GBM) benefit more than rigid ones (GLM). So every row with an identical profile goes to
the same split. Validation is used for GBM early stopping and hyper-parameter choice; test is used once,
for the final comparison. The pipeline asserts zero profiles cross splits.

There is **no time variable** in this dataset, so the split is not out-of-time. That is a real
limitation (section 10): a real validation would hold out the most recent year.

---

## 3. GLMs from scratch

### 3.1 What a GLM is

Ordinary regression says `y = x'b + noise` with normal noise. A **generalized linear model** keeps the
linear predictor `eta = x'b` but changes two things:

1. **Distribution** of y, chosen from the exponential family to match the data: Poisson for counts,
   Gamma for positive skewed amounts, Tweedie for "zero or a positive amount".
2. **Link function** g with `g(E[y]) = eta`. Pricing uses the **log link**: `log E[y] = x'b`, so
   `E[y] = exp(b0) * exp(b1 x1) * ...` - a **multiplicative** tariff. Each `exp(b)` is a **relativity**.

Why multiplicative? Because a young driver in a dense city is riskier by roughly "young factor times city
factor", and because that is how rating tables are filed and explained to regulators.

The parameters are estimated by **maximum likelihood**, numerically via iteratively reweighted least
squares (IRLS). statsmodels does this.

### 3.2 Frequency: Poisson with an exposure offset

```
N_i ~ Poisson(lambda_i * e_i)
log E[N_i] = log(e_i) + x_i'b
```

`log(e_i)` is an **offset**: a term in the linear predictor with its coefficient fixed at 1. It encodes
"expected claims are proportional to time at risk". Then `exp(x'b)` is the **annual claim rate**, and the
relativities are rate relativities. The unit test `test_exposure_offset_scales_prediction` checks that
halving exposure halves the prediction.

Why not model `N / e` with weights? With a Poisson GLM the two are mathematically equivalent (same
estimates). The offset form is what actuaries write down, so it is used here; LightGBM uses the
rate-with-weights form because it has no offset argument in the Python API.

**Balance property**: a Poisson GLM with log link and an intercept reproduces total observed claims
exactly on the training data (tested in `test_poisson_glm_balance_property`). This is why GLMs are "self
calibrated" on the training book at the total level.

**Over-dispersion**: Poisson assumes variance = mean. The Pearson dispersion on the training data is
about 1.73, i.e. variance is larger than Poisson. Point estimates stay consistent, but standard errors
would be too small. The code fits **quasi-Poisson** standard errors (`scale="X2"`): same coefficients,
standard errors multiplied by sqrt(dispersion). A negative binomial model is the alternative; noted as
a future improvement.

### 3.3 Severity: Gamma GLM with claim-count weights

Response: the average capped claim amount of each policy that had claims. Distribution: Gamma, log link.
Gamma has variance proportional to mean squared (constant coefficient of variation), which matches
claim amounts: a 10,000 claim varies more in absolute terms than a 1,000 claim.

Weight: `var_weights = number of claims`, because the average of n claims has variance `phi * mu^2 / n`.
A policy with three claims tells you three times as much about average severity as a policy with one.

Result to remember: severity explains almost nothing out of sample (deviance explained on test 0.02%,
Gini 0.02). Only a few levels are significant (driver age 71+ at +21%, BM 76-100 at +14-22%, vehicle
age 16+ at -14%); 16 of 21 parameters are not. Frequency carries almost all the pricing signal, which is
typical for motor TPL, and validation finding V-11 recommends simplifying the severity model.

### 3.4 Encoding and base levels

Continuous factors (age, BM, vehicle age, power) are **banded**, because the effects are not linear
(age is U-shaped, BM jumps at 100). Bands were chosen from the one-way EDA so each band has enough
exposure. Density enters as `log(Density)`, linear: the one-way plot is close to linear on the log scale,
and it saves parameters. `Area` is dropped from the GLM because it is a coarsened version of density
(Spearman correlation printed in `metrics.json`, `area_logdensity_spearman`), so keeping both would be
collinear.

**Base level** = the level with the most exposure (e.g. driver age 41-50, BM 50, Region R24). Every
relativity then reads "versus the most common risk", and the intercept has the smallest standard error.

### 3.5 Reading the relativity table

`reports/tables/glm_frequency_relativities.md` has, for each level: relativity, 95% CI, p-value and
exposure. Example readings:

- BM 101+ has a frequency relativity of about 8.4 versus BM 50: drivers with recent claims claim much more.
- Driver age 18-20 is only about 1.11 versus age 41-50 **after controlling for BM** (and not significant:
  95% CI 0.95 to 1.29), while the raw one-way frequency for 18-20 is about 3x the portfolio average.
  Ages 21-40 even come out *below* the base (0.57 to 0.75). The difference is BM: young drivers start at
  BM 100, so BM already "explains" a large part of their risk. This is multicollinearity in action and
  it is exactly why the fairness section matters.
- Small regions (R21, R42, R43) have wide confidence intervals: not credible. A production tariff would
  group them or credibility-weight them toward the national average.

---

## 4. Tweedie and the GBM challenger

### 4.1 Tweedie in one paragraph

The **Tweedie** family with power 1 < p < 2 is a compound Poisson-Gamma distribution: a Poisson number
of Gamma-sized claims. It has a point mass at zero (no claims) and a continuous positive part. That is
exactly the shape of "loss cost per policy", so you can model pure premium directly in one model instead
of frequency times severity. Variance = phi * mu^p: p = 1 would be Poisson, p = 2 Gamma, and p = 1.5 is
the usual middle. The repo uses p = 1.5 and tests 1.3 and 1.7 (sensitivity table).

### 4.2 Gradient boosting

LightGBM builds hundreds of shallow decision trees, each correcting the previous ones, on the log scale
(same log link idea). It finds interactions and non-linearities automatically. Settings that matter:

- **Objective**: `poisson` for frequency (target claims per year, weight exposure), `tweedie` for pure
  premium (target capped loss per year, weight exposure).
- **Early stopping** on the validation set's deviance: stop adding trees when validation stops
  improving, so the test set is never used for tuning.
- **Small grid** (num_leaves x min_child_samples), choosing on validation deviance. `min_child_samples`
  is in the hundreds because claims are rare; a leaf with 20 policies would fit noise.

### 4.3 Why the GLM is still the production candidate

Model risk is not just accuracy. The GBM is a reasonable challenger, but:

- A GLM produces a **rating table** a regulator can review; most US states require rate filings with
  justification of each factor. A GBM produces 500 trees.
- GLM relativities are **monotone and stable** where you want them; a GBM can produce local wiggles
  (e.g. a price that dips at age 37) that are hard to justify to a customer.
- The GBM's edge is real here and I say so: +0.027 Gini on pure premium (95% CI +0.011 to +0.050) and
  it wins the double lift in 8 of 10 deciles. The hybrid "GBM frequency x GLM severity" is best of all
  (Gini 0.356), which shows the gap is in **frequency**. An age x BM interaction in the GLM recovers about
  a fifth of the gap. The honest recommendation, which is what most carriers do: keep the GLM as the filed
  tariff, use the GBM to find the missing structure, and add it to the GLM (finding V-04).
- The raw GBM was also **not balanced**: it under-predicted training losses by 11.6% because boosting
  with shrinkage and early stopping does not have the GLM's balance property. I fixed it with one
  multiplicative factor fitted on training data (finding V-03). Knowing that a GBM can rank well and
  still be mis-priced in total is a good model-risk talking point.

---

## 5. Evaluation metrics, explained

### 5.1 Deviance

The **deviance** is twice the log-likelihood gap between your model and a "saturated" model that fits
every observation perfectly. It is the GLM version of the residual sum of squares, with the right
penalty for each distribution (Poisson deviance for counts, Gamma for amounts, Tweedie for loss cost).
Lower is better. On its own the number is not interpretable, so we compare to the **null model** (one
constant rate for everyone):

```
deviance explained D^2 = 1 - D(model) / D(null)
```

Expect small numbers. Insurance claims are mostly luck: even a perfect model cannot know who will crash
this year. A frequency D^2 of a few percent is normal on this dataset. Do not apologise for it; explain it.

Why not accuracy, RMSE or AUC? Accuracy is meaningless for rates; RMSE is dominated by large claims and
does not respect the count nature; AUC treats "1 claim in 2 weeks" and "1 claim in 1 year" the same
because it ignores exposure. Deviance respects the distribution, exposure and scale.

### 5.2 Gini and the ordered Lorenz curve

Sort policies from lowest to highest **predicted** rate. Walk along them, plotting cumulative share of
exposure (x) against cumulative share of **actual** losses (y). If the model were useless, losses would
accumulate evenly: the diagonal. A good model puts the low-risk policies first, so the curve sags below
the diagonal. **Gini = 1 - 2 x (area under the curve)**: 0 = random, higher = better ranking.

Key properties:
- It measures **ranking (discrimination)**, not level. Multiplying all predictions by 2 does not change
  it (tested: `test_gini_properties`). So it must always be paired with a calibration check.
- It is exposure-weighted, so it handles part-year policies correctly.
- It is noisy on rare events, hence the paired bootstrap confidence interval.

### 5.3 Lift chart (decile chart) and calibration

Sort by prediction, cut into 10 bins of equal exposure, and compare average predicted and average actual
in each bin. Two things to read:
- **Lift**: actual in the top decile divided by actual in the bottom decile. How much the model separates.
- **Calibration**: in each decile, A/E (actual / expected) should be close to 1. A model can rank well
  and still be mis-priced in a segment.

**Calibration slope**: regress log actual on log predicted. Slope 1 = perfect spread. Slope below 1 =
predictions too extreme (overfitting, typical of GBMs that are not regularised enough); above 1 =
predictions too timid.

### 5.4 Double-lift chart

The tool for comparing two models on the **same** business question: "where they disagree, who is
right?" Sort policies by the ratio GBM prediction / GLM prediction. In the bottom deciles the GBM thinks
the GLM overcharges; in the top deciles it thinks the GLM undercharges. Plot actual loss cost (each series
relative to its own mean) against both models. If actual follows the GBM line, the GBM is capturing real
risk the GLM misses. This is the standard chart pricing teams use to decide whether to move to a new model.

### 5.5 Actual versus expected by rating factor

For each level of each factor (e.g. each region), compare actual losses to predicted. A GLM with a factor
in it is close to 1 on that factor in-sample by construction (score equations); on **test** data it shows
whether the relativity generalises. Factors not in the model (or small levels) are where you see problems.

### 5.6 Bootstrap

Resample test policies with replacement 200 times, recompute both Ginis **on the same resample** (paired),
take the 2.5th and 97.5th percentiles of the difference. If the interval excludes zero, the difference is
not a fluke of this particular test set.

---

## 6. Interpretability

- **GLM**: the relativity table *is* the explanation. Each factor's effect is one number per level with a CI.
- **GBM**: SHAP values. For each prediction, SHAP splits the log-prediction into a contribution per
  feature that adds up exactly to the model output (the pipeline checks this additivity to 1e-6). They
  come from cooperative game theory (Shapley values): each feature gets its average marginal
  contribution over all orderings. LightGBM's built-in TreeSHAP (`pred_contrib=True`) computes them
  exactly for trees; it is the same algorithm as `shap.TreeExplainer`.
- **GLM vs SHAP chart** (`docs/figures/glm_vs_shap.png`): average SHAP by age band and by BM band, turned
  into relativities on the same scale as the GLM's. Where the curves agree, the GBM learned the same story;
  where they differ, that is either an interaction the GLM misses or GBM noise. This is how you make a
  black box reviewable.

---

## 7. Fairness and proxy discrimination

### 7.1 Concepts

- **Protected attribute**: race, colour, religion, national origin, sex (and others depending on law).
  Using them in pricing is prohibited in most US states and, for sex, across the EU since 2012
  (Test-Achats ruling).
- **Proxy discrimination**: a model that does not use the protected attribute still charges by it, because
  other inputs are correlated with it. Removing the attribute ("fairness through unawareness") does not fix this.
- **Unfair discrimination** in US insurance law means rate differences **not justified by expected cost**.
  Pricing differences that reflect real risk are generally allowed for non-protected factors. So the legal
  question is not just "does the price differ by group" but "does it differ beyond what risk justifies,
  or through a protected class".

### 7.2 What the repo actually does

The dataset has **no protected attributes**. That is a limitation to state, not hide. So the repo uses
**driver age** as a stand-in sensitive attribute (age is protected in some contexts, and some
jurisdictions restrict its use in pricing) to demonstrate the method:

1. **Proxy strength.** Train a classifier to predict "young driver (age 25 or under)" from all the *other*
   rating factors. Its AUC says how much age information leaks through them. Most of it comes from
   bonus-malus, as expected from how BM works.
2. **Three prices** (Lindholm, Richman, Tsanakas and Wüthrich, 2022, "Discrimination-free insurance pricing"):
   - **Aware** price mu(x, d): uses age.
   - **Unaware** price mu(x): age dropped and the model refit. Age effects partly re-enter through BM.
   - **Discrimination-free** price h(x) = sum over d of mu(x, d) P(d): use the aware model but average
     over the *population* age distribution P(d), not the policyholder's own. Because P(d) does not depend
     on x, age cannot sneak back in through correlated factors. Two people with identical non-age factors
     pay the same, and the non-age relativities are the "clean" ones (not contaminated by age proxying).
   All three are rebalanced to collect the same total premium on the training book.
3. Compare price by age band, A/E by age band, and Gini for the three constructions.

What came out (test set, `reports/tables/fairness_price_by_age.md`):

| Driver age 18-20 | Price index (1 = portfolio average) | A/E |
|---|---:|---:|
| Actual loss cost | about 5.0 | |
| Aware (uses age) | 4.08 | 1.22 |
| Unaware (age dropped, refit) | 2.50 | 1.99 |
| Discrimination-free | 2.64 | 1.89 |

How to read it, carefully, because this is the part that impresses or sinks you:

1. **Bonus-malus is a near-perfect proxy for youth**: the other factors identify drivers aged 25 or under
   with AUC 0.94, and BM alone gets 0.91. That is mechanical: everybody starts at 100.
2. **Dropping age halves the young-driver loading but does not remove it** (4.08 to 2.50). The rest comes
   back through BM. So "we do not use age" would be a misleading statement about this price.
3. **The discrimination-free price is close to the unaware price** (2.64 vs 2.50). Interpretation: in this
   data, most of what the unaware model charges young drivers is explained by BM's *own* risk effect
   (as estimated with age controlled), not by BM standing in for age. The pure proxy effect is modest:
   unaware and discrimination-free differ by at most about 8% in any band. Do not overclaim either way.
   Why is discrimination-free slightly *higher* for the young? Holding age fixed, a high BM is more
   alarming for a 45-year-old (recent claims) than for a 20-year-old (just started), so the aware model's
   BM effect is steeper; the discrimination-free price applies that steeper BM effect to young drivers,
   most of whom have high BM.
4. **The cost**: remember A/E above 1 means the price is too *low*. Both age-blind prices under-charge
   18-20 year-olds by about half (A/E 1.9 to 2.0) and drivers over 40 (A/E 1.07 to 1.30), and
   over-charge drivers aged 21-40 (A/E 0.73 to 0.81), whose BM looks worse than their risk because they
   are new drivers. That is a cross-subsidy created by removing age. Test Gini drops from 0.306 (aware) to 0.288
   (unaware) and 0.286 (discrimination-free).
5. **Conclusion you can defend**: removing a variable is not the same as removing its effect; the method
   lets a company see and quantify the trade-off, and the decision belongs to the business, the actuaries
   and legal, not to the model.

### 7.3 What to say about the limits

- No protected attribute in the data, so no real disparate-impact test was possible. With US data you
  would use BIFSG (Bayesian Improved First name Surname Geocoding) to impute race probabilistically, which
  is what Colorado's draft testing methodology for life insurers and some NAIC discussions point toward,
  and run the three-step test NY DFS CL 2024-7 describes.
- Region and density could proxy for socio-economic or ethnic composition. Testing that needs external
  census data joined at a geography level the dataset does not give (regions are anonymised-ish codes).
- A four-fifths-style ratio is an employment-law heuristic, not an insurance standard; it is mentioned in
  the docs only as context, and the repo does not claim any legal conclusion.
- Age is a *legitimate* rating factor in French motor insurance. The analysis demonstrates a governance
  method; it does not say using age is wrong.

---

## 8. Model risk management

### 8.1 Vocabulary

- **Model**: a quantitative method that turns inputs into estimates used for decisions. A pricing GLM is a model.
- **Model risk**: losses from a model that is wrong (bad design, bad data, bugs) or used wrongly (outside
  its intended scope).
- **Three lines**: developers build (1st line), independent validation challenges (2nd line), internal
  audit checks the process (3rd line).
- **Effective challenge**: critical review by people with competence, independence and influence to make
  changes happen.
- **Validation pillars**: conceptual soundness (is the design right for the purpose?), ongoing monitoring
  (is it still working?), outcomes analysis (do predictions match reality, back-testing, benchmarking).
- **Model tiering**: more material and complex models get more validation. A pricing model that sets
  premiums for the whole motor book is high materiality.

### 8.2 The frameworks (be precise)

- **SR 11-7 / OCC 2011-12** (Federal Reserve and OCC, April 2011): the bank model-risk guidance where the
  vocabulary above comes from. **It was rescinded and replaced on 17 April 2026 by SR 26-2, "Revised
  Guidance on Model Risk Management"** (Fed, OCC, FDIC), which keeps the disciplines (inventory, tiering,
  validation, monitoring) but is more principles-based and risk-based. It applies to banks, not insurers.
  Saying this correctly is a strong signal: most candidates still quote SR 11-7 as current.
- **NAIC Model Bulletin on the Use of AI Systems by Insurers** (adopted 4 December 2023): asks insurers for
  a written AI Systems (AIS) program with governance, risk management, internal controls, validation and
  testing for unfair discrimination and errors, and third-party oversight. It is a bulletin, not a law:
  it explains how existing unfair-trade-practice and market-conduct laws apply to AI. Adopted by roughly
  25 states plus DC by mid/late 2026 (secondary sources; check the NAIC map for the current count).
- **NY DFS Insurance Circular Letter 2024-7** (11 July 2024): AI and external consumer data in underwriting
  and pricing; insurers must show their models and data do not proxy for protected classes, with a
  documented quantitative assessment, board/senior-management oversight, and responsibility for vendors.
- **Colorado SB21-169 and Regulation 10-1-1**: originally life insurance; amended in August 2025 (effective
  15 October 2025) to cover **private passenger auto** and health benefit plans, with a governance and
  risk-management framework and quantitative testing for unfair discrimination with respect to race.
  Full compliance for auto insurers was due 1 July 2026 (per the Colorado DOI amendment as reported by the
  CAS Actuarial Review).
- **NIST AI RMF 1.0** (January 2023): voluntary; four functions Govern, Map, Measure, Manage.
- **EU AI Act** (Regulation 2024/1689): Annex III point 5(c) makes "risk assessment and pricing in relation
  to natural persons in the case of life and health insurance" high-risk. **Motor insurance pricing is not
  in that list.** Say this precisely; it is a common mistake.

### 8.3 What is in `docs/mrm/`

| Document | What it answers |
|---|---|
| `model_card.md` | What the model is, intended use, out-of-scope uses, data, metrics, limits, owner. |
| `model_development.md` | How it was built and why: the developer's document. |
| `validation_report.md` | Independent-style review: conceptual soundness, data, outcomes, benchmarking, sensitivity, findings, rating. |
| `issues_log.md` | Each finding: severity, evidence, owner, action, status. |
| `monitoring_plan.md` | What is measured after go-live, thresholds (RAG), frequency, escalation. |
| `control_mapping.md` | Each requirement in NAIC / NY DFS / Colorado / NIST / SR 26-2 mapped to the artefact in this repo that evidences it, and gaps. |
| `model_inventory.yaml` | The inventory record with risk tier and rationale. |

### 8.4 PSI (population stability index)

For a variable, bin the reference (training) distribution into 10 quantile bins; compute the share of
the new book in each bin; `PSI = sum (new% - ref%) x ln(new% / ref%)`. Rules of thumb: below 0.10 stable,
0.10 to 0.25 investigate, above 0.25 significant shift. It detects **input drift** before losses arrive,
which matters because motor claims take months to report and bodily-injury claims years to settle. The
repo runs the monitor on the hold-out book (expected green) and on two **real** sub-books (Paris region,
under-30 drivers) to show it raising flags. Those are subsets of the real data used as stress scenarios,
not new data.

---

## 8.5 The numbers to know cold (test set unless stated)

| What | Value |
|---|---|
| Clean policies / exposure-years / claims | 668,626 / 353,560 / 26,098 |
| Train / validation / test policies | 401,585 / 133,414 / 133,627 |
| Training claim frequency, mean and median claim | 0.074 per year; 2,189; 1,172 |
| GLM pure-premium Gini / deviance explained | 0.306 / 4.8% |
| LightGBM pure-premium Gini / deviance explained | 0.334 / 6.0% |
| GBM minus GLM Gini, pure premium (95% CI) | +0.027 (+0.011 to +0.050) |
| Frequency Gini GLM vs GBM | 0.291 vs 0.322 |
| Top/bottom decile lift, GLM | 5.7x |
| Calibration slope GLM (pure premium / frequency) | 0.956 / 0.958 |
| A/E GLM on test (capped) | 0.992 |
| Large-loss cap / load | about 39.2k / 1.258 |
| Claims-only block (DQ-1) | 9,387 rows; keeping it gives test A/E 0.74 and Gini 0.247 |
| Proxy AUC (under-26 from other factors / BM only) | 0.940 / 0.914 |
| Raw GBM balance factor | 1.116 |

## 9. Likely interview questions with strong answers

**Q: Why an offset and not exposure as a feature?**
A: Claims scale with time at risk by definition, so exposure's coefficient should be exactly 1 on the
log scale. An offset imposes that. As a feature, the model would estimate a coefficient that absorbs
selection effects (e.g. short-tenure policies cancelled after a claim) and you would be pricing on them.

**Q: Why Poisson for counts? Your data is over-dispersed.**
A: Poisson gives consistent coefficient estimates even under over-dispersion as long as the mean is
right. What breaks is the standard errors, so I used quasi-Poisson standard errors (Pearson dispersion
about 1.7). Negative binomial is the next step and I list it as an open item.

**Q: Why Gamma for severity, not lognormal?**
A: Gamma GLM models the mean directly on the log link, so predictions are unbiased for the mean cost,
which is what the premium needs. A lognormal regression models the mean of log amount; converting back
needs a variance correction and is biased if that is wrong. Gamma also has constant coefficient of variation,
which fits claims reasonably below the cap.

**Q: How did you treat large losses?**
A: Capped individual claims at the 99.5th percentile of training claims (about 39k) and loaded the excess
back as a flat factor of 1.258. Large BI claims are close to random with respect to rating factors, so
modelling them per cell adds noise: without the cap the severity relativity for age 18-20 jumps from 1.20
to 2.71 on a handful of claims. But I also flag it as the weakest assumption (finding V-06): the single
4.08M claim landed in my test set, where uncapped losses are 1.74x capped versus 1.26x in training.

**Q: What is the Gini and why is yours "low"?**
A: Ranking quality on the ordered Lorenz curve. It is bounded well below 1 for insurance because claims
are mostly random; what matters is the comparison to the alternative model and its confidence interval.

**Q: The GBM has the better Gini. Why not deploy it?**
A: Look at the confidence interval and the double-lift. Then weigh: rate-filing explainability, stability
of relativities, monotonicity, and the governance cost of a black box. My recommendation is to keep the GLM
as the filed tariff and use the GBM as a challenger to discover interactions to add, which is common practice.

**Q: How do you know the model is not overfit?**
A: Separate validation and test sets; early stopping on validation; calibration slope near 1 on test;
train-versus-test deviance comparison; and a seed-stability check with three different splits.

**Q: What would you validate first if you were the second line?**
A: Data: the claims-only block (DQ-1) is the kind of issue that silently corrupts a model. Then whether
frequency and severity count the same claims, then the large-loss treatment, then out-of-time performance,
which this dataset cannot test.

**Q: What is proxy discrimination and how did you test it?**
A: "I used driver age as a stand-in sensitive attribute, since the data has no protected class. The other
factors reconstruct under-26 drivers with AUC 0.94, mostly through bonus-malus. Dropping age cut the
18-20 price index from 4.1 to 2.5 times average, so age effects partly survive through the proxy. I then
implemented the Lindholm-Wüthrich discrimination-free price, which came out close to the unaware one,
meaning most of the residual is BM's own risk signal. Either age-blind price under-charges the youngest
drivers by about half, over-charges 21-40 year-olds, and costs about 0.02 of Gini. The tool quantifies the trade-off; the choice is a
business and legal one."

**Q: Is SR 11-7 relevant to an insurer?**
A: Not directly; it was bank guidance, and it was replaced by SR 26-2 in April 2026. Its vocabulary
(conceptual soundness, outcomes analysis, effective challenge) is the lingua franca of model risk, and
insurers' AI governance (NAIC bulletin, NY DFS, Colorado) asks for the same disciplines plus
unfair-discrimination testing.

**Q: What is the difference between developer testing and validation?**
A: Independence and purpose. The developer shows the model works; the validator tries to show where it
does not, challenges assumptions, re-performs key calculations and rates residual risk. In this project
I played both roles, which is itself a limitation I state: it is "independent-style", not independent.

**Q: What would you do with real company data?**
A: Out-of-time validation on the latest accident year; development of open claims to ultimate (IBNR) so
the losses are complete; credibility weighting of small cells; telematics or claim-history variables;
proper protected-class testing with BIFSG; and a rate-impact analysis (how many customers' premiums move
by more than 10%) before any change.

---

## 10. Limitations to volunteer (before they find them)

1. **No time dimension**: random (grouped) split, not out-of-time. Real validation needs a later period.
2. **Losses are as recorded**, not developed to ultimate. Open claims and IBNR are ignored.
3. **No protected attributes**: the fairness work demonstrates the method with age as a stand-in.
4. **Self-validated**: I built and validated it, so "independent-style", not independent.
5. **No expenses, profit, or regulatory constraints**: this is a risk premium, not a filed rate.
6. **Data provenance**: the source does not document the claims-only block; my treatment is an inference
   from the data, stated as such.
7. **Single country, undocumented period**: French policies; the source does not state the observation
   years. Not transferable to US pricing without re-fitting on US data.

---

## 11. How to demo it in 3 minutes

1. Open the README: results table, Lorenz curve, double lift. ("Here is what the models do.")
2. Open `docs/mrm/issues_log.md`: DQ-1. ("Here is what I found that would have broken it.")
3. Open `docs/figures/fairness_age_proxy.png`. ("Here is why dropping a variable is not fairness.")
4. Open `docs/mrm/control_mapping.md`. ("Here is how this maps to what a regulator asks for.")
5. Run `pytest -q`. ("And it is reproducible and tested.")
