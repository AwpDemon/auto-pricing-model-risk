# Monitoring plan: Motor TPL risk-premium GLM

Implementation: `src/autopricing/monitoring.py` (`monitoring_report`), thresholds in `config/default.yaml`.
The pipeline demonstrates it on three books carved from the real hold-out data (results below). There is
no live data feed; this plan describes how the same code would run in production.

## What is monitored

| # | Metric | Population | Frequency | GREEN | AMBER | RED | Why |
|---|---|---|---|---|---|---|---|
| M1 | PSI of each rating factor (DrivAge, BonusMalus, VehAge, VehPower, LogDensity, VehBrand, VehGas, Region, Area) vs training book | New business + renewals written in the period | Monthly | < 0.10 | 0.10 - 0.25 | >= 0.25 | Input drift is visible immediately, months before losses develop. |
| M2 | PSI of the predicted risk premium | Same | Monthly | < 0.10 | 0.10 - 0.25 | >= 0.25 | Catches mix shift that individual factors miss (combinations). |
| M3 | Actual / expected (capped losses) | Accident period with at least 12 months of development | Quarterly | abs(A/E - 1) < 5% | 5 - 10% | >= 10% | Level adequacy. Must use developed losses or IBNR-adjusted ultimates; raw early losses are biased low. |
| M4 | A/E by rating factor level (levels with > 1,000 exposure-years) | Same | Quarterly | all within 10% | any 10 - 20% | any > 20% | Relativity drift; tells you *which* factor to refit. |
| M5 | Gini on capped losses vs validation baseline | Same | Semi-annual | drop < 0.03 | 0.03 - 0.06 | >= 0.06 | Ranking degradation. Only compare whole-book batches: Gini on a sub-segment is lower by construction (narrower range of risk). |
| M6 | Large-loss load: actual excess-of-cap losses / capped losses | Rolling 3 accident years | Annual | within 15% of 0.336 | 15 - 30% | > 30% | The flat load is the most assumption-driven parameter. |
| M7 | Price by age band: unaware vs discrimination-free index for the youngest band | Same | Annual | Reviewed by governance; no automatic threshold | | | Fairness review (proxy effect); NY DFS expects at least annual testing. |
| M8 | Data quality: share of claims without amounts, exposure > 1, ClaimNb > 4 | Each data load | Each load | 0 new DQ-1-type blocks | any increase | new block | DQ-1 showed how a feed can silently contain a non-representative extract. |

## Escalation

- **GREEN**: logged in the monitoring pack, no action.
- **AMBER**: model owner investigates within one month and records the cause in the issues log
  (mix shift that the model handles correctly is an acceptable explanation if M3/M4 stay green).
- **RED**: model owner and validation are notified within 5 business days; options are (a) recalibrate the
  intercept (level only), (b) refit relativities, (c) restrict use (e.g. exclude a new channel) until refit.
  Any RED on M3 or M5 triggers a re-validation.
- Two consecutive AMBERs on the same metric are treated as RED.

## Demonstration on real sub-books

Reference = training book. "Current" = three books from the hold-out set. The second and third are
**real subsets used as stress scenarios** (what the monitor would show if the insurer's new business
came mostly from the Paris region or from young drivers); they are not new data.

<!-- BEGIN:monitoring -->
| Scenario | Policies | Overall | Worst PSI (factor) | A/E |
|---|---:|---|---|---:|
| Hold-out test book (expected: stable) | 133,627 | AMBER | 0.000 (Region) | 0.992 |
| Paris region only (R11), real subset | 13,810 | RED | 12.186 (Region) | 1.032 |
| Young drivers (age < 30), real subset | 17,748 | RED | 10.986 (DrivAge) | 1.040 |
<!-- END:monitoring -->

Reading: the hold-out book is stable, as it should be for a random split. The concentrated books trip the
PSI thresholds on the factor that defines them (Region, DrivAge) and on correlated factors, and the score
PSI shows whether the shift moves predicted premiums. Full per-metric results are in
`reports/tables/monitoring_scenarios.md`.

## Known limitations of the monitoring design

- PSI depends on the binning; deciles of the training distribution are used throughout and should not be
  changed between periods.
- A/E needs developed losses. In this dataset losses are as recorded; in production use ultimates.
- Small books make PSI noisy: below about 1,000 policies treat AMBER as informational.
