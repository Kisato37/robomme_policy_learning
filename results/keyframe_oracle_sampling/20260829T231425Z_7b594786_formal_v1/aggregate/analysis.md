# Formal keyframe selector analysis

## Validity and provenance

The fail-closed audit passed for exactly 3,200 scientific cells (16 tasks x 50 episodes x 4 arms). Every selector call was replayed against the frozen arm definition and preregistered RandomSamp seed table.

The run contains 57 recorded infrastructure failures; each was superseded through its explicitly authorized fresh-reset retry chain. No protocol deviations or unresolved risks were detected.

## Primary comparison

| Comparison | Estimate (pp) | 95% hierarchical bootstrap CI (pp) | Paired randomization p |
|---|---:|---:|---:|
| OC - U | -6.250 | [-12.625, -0.500] | 9.9999e-06 |

The estimate is the equal-task-weighted paired success-rate difference.

## Prespecified secondary comparisons

| Comparison | Estimate (pp) | 95% CI (pp) | Raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| O - U | -29.250 | [-40.875, -18.250] | 9.9999e-06 | 3.99996e-05 |
| OC - O | 23.000 | [13.500, 33.625] | 9.9999e-06 | 3.99996e-05 |
| R - U | -4.000 | [-9.125, 1.000] | 0.00786992 | 0.0157398 |
| OC - R | -2.250 | [-8.125, 3.125] | 0.180888 | 0.180888 |

Holm adjustment is applied across exactly the four preregistered secondary comparisons.

## Preregistered decision

Classification: `decisive_test_time_no_go`.

Satisfied conditions: `decisive_test_time_no_go`.

The practical-effect threshold is 3.0 percentage points. The decision uses confidence-interval rules exactly as preregistered; p-values are not substituted into the GO/NO-GO rule.

## Frozen prior-exposure sensitivity

Excluded 1 frozen task/episode blocks.

OC - U sensitivity estimate: -6.273 pp; 95% CI [-12.656, -0.500] pp.

## Reproducibility

- Formal evaluation commit: `7b59478619db222eefcd30575e268e1384f75953`
- Aggregation commit: `ac77b3ca3983875cbbc588ab6156107f9c284443`
- Analysis seed: `2026082502`
- Bootstrap replicates: `100000`
- Randomization replicates: `100000`
- Result-set SHA-256: `7eee5bf81e1aef5e95589c7bea9672fc78018887f14c6e2d2fda2a280d06cedd`
- Selector-trace-set SHA-256: `bd4ed32f3744d4e4182669c93ef98aaa89b853ae054003df7865f4a32898f1aa`
