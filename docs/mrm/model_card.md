# Model card: Motor TPL risk-premium GLM (MTPL-RP-GLM-001 v1.0.0)

**Portfolio project.** Built on public data to demonstrate pricing and model-risk practice. Not used to
price any real policy.

## Model details
- **Type**: Poisson GLM (log link, log-exposure offset) for claim frequency x Gamma GLM (log link) for capped
  claim severity x flat large-loss load (1.258).
- **Owner / developer**: Ali Askari. **Validation**: self, independent-style (see validation report).
- **Code**: `src/autopricing/`; parameters `models/glm_frequency.json`, `models/glm_severity.json`.
- **Licence**: code MIT; data CC0 (OpenML) / GPL >= 2 (CASdatasets upstream).

## Intended use
- Expected annual claim cost (risk premium) per policy for French motor third-party liability, as the
  loss-cost input to a rate indication reviewed by a pricing actuary.

## Out-of-scope uses
- Final customer premium (no expenses, profit, taxes, competitive or regulatory adjustments).
- Underwriting accept/decline, claims handling, reserving.
- Other countries or products; US use without refitting and protected-class testing.
- Any use of the GLM output without the large-loss load (it predicts capped losses).

## Inputs
Driver age, bonus-malus coefficient, vehicle age, vehicle power, vehicle brand, fuel, region, population
density (log). No external consumer data, no credit, no telematics.

## Training data
freMTPL2freq/sev, 668,626 policies after cleaning (353,560 exposure-years, 26,098 claims); 60/20/20
train/validation/test split grouped by rating profile. Four data-quality decisions documented (DQ-1..4),
the most important being exclusion of a 9,387-row claims-only block.

## Performance (hold-out test, 133,627 policies)

| Metric | GLM | LightGBM challenger |
|---|---:|---:|
| Pure-premium Gini | 0.306 | 0.334 |
| Pure-premium deviance explained (Tweedie p=1.5) | 4.84% | 5.97% |
| Frequency Gini | 0.291 | 0.322 |
| Calibration slope (pure premium) | 0.956 | 1.018 |
| Actual / expected (capped losses) | 0.992 | 0.993 (after balance correction) |
| Top/bottom decile lift | 5.7x | 7.8x |

## Fairness considerations
No protected attributes are available. With driver age as a stand-in sensitive attribute, the other rating
factors identify drivers aged 25 or under with AUC 0.94 (mostly bonus-malus). Removing age halves the
youngest drivers' loading but does not remove it; age-blind prices create cross-subsidies. Details:
validation report section 8, finding V-08.

## Known limitations
No out-of-time validation; losses as recorded (not developed); flat and volatile large-loss load; GLM leaves
measurable lift unpriced versus the GBM; sparse regions and brands have wide confidence intervals; severity
model has negligible predictive power. Full list: `issues_log.md`.

## Monitoring
PSI per factor and on the score (0.10 / 0.25), A/E (5% / 10%), Gini drop, large-loss load, fairness review;
see `monitoring_plan.md`.
