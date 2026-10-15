# Frozen meta-reader writer ablation

**Measured result: no replacement passes the full predeclared three-seed rule.** Removing RMS normalization greatly reduces repeated-write disturbances; calibrated raw steps improve the first-order family at long horizons, while higher-order readers remain fragile.

All six upstream checkpoints were frozen: seeds 41/97/127 for the first-order and higher-order readers. Each checkpoint/writer/schedule used 128 held-out streams, 64 post-initialization observations, actual 48-byte state and a byte-identical four-write RMS prefix. Six candidate step sizes were selected using only development seeds 11/23/37, 128 streams each and horizon 64. There are 120 held-out runs: 6 checkpoints × 5 policies × 4 schedules.

These values use the published task generators and seeds; they are not a replication of the upstream task receipt. The original weights are unchanged. The same-step raw arm isolates RMS removal; calibrated and accepted arms additionally change step selection/acceptance.

## First-order reader checkpoints

Mean across three training seeds. Recall MSE uses the independent-rewrite schedule. Repeat drift and changed bytes measure the **actual 16th unchanged observation** in the repeat schedule, not the additional diagnostic sham write.

| Writer | MSE at 1 rewrite | MSE at 64 | Repeat-16 drift | Repeat-16 bytes changed | Full-rule seed passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| rms_reference | 0.091795 | 0.203014 | 0.0080028639 | 16.667% | reference/control |
| raw_same_step | 0.115380 | 0.200834 | 2.6585546e-06 | 0.982% | 0/3 |
| raw_calibrated | 0.078756 | 0.177400 | 1.0145873e-06 | 0.244% | 2/3 |
| raw_backtracked | 0.078756 | 0.177400 | 1.0145873e-06 | 0.244% | 2/3 |
| no_write | 0.189296 | 0.527110 | 0 | 0.000% | reference/control |

## Higher-order reader checkpoints

Mean across three training seeds. Recall MSE uses the independent-rewrite schedule. Repeat drift and changed bytes measure the **actual 16th unchanged observation** in the repeat schedule, not the additional diagnostic sham write.

| Writer | MSE at 1 rewrite | MSE at 64 | Repeat-16 drift | Repeat-16 bytes changed | Full-rule seed passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| rms_reference | 0.507364 | 0.194678 | 0.017354972 | 16.667% | reference/control |
| raw_same_step | 2.994465 | 0.265060 | 0.00011097392 | 0.244% | 0/3 |
| raw_calibrated | 1.688802 | 0.221642 | 2.1509919e-06 | 0.347% | 0/3 |
| raw_backtracked | 0.432657 | 0.195700 | 1.8644743e-06 | 0.277% | 1/3 |
| no_write | 0.588779 | 0.777788 | 0 | 0.000% | reference/control |

## Delayed correction and preserved history

After the initial key-0 correction A→A′, 63 distractor observations rewrite keys 1 and 2. At observation 64, score the corrected key 0 and never-rewritten key 3 separately. These are means across the three training seeds in the correction schedule. They expose information hidden by the full-memory aggregate.

| Reader family | Writer | Delayed A′ MSE (key 0) | Preserved binding MSE (key 3) |
| --- | --- | ---: | ---: |
| First-order | rms_reference | 0.513920 | 0.310062 |
| First-order | raw_same_step | 0.512065 | 0.112923 |
| First-order | raw_calibrated | 0.324488 | 0.242888 |
| First-order | raw_backtracked | 0.324488 | 0.242888 |
| First-order | no_write | 1.016454 | 0.077964 |
| Higher-order | rms_reference | 0.438721 | 0.874504 |
| Higher-order | raw_same_step | 0.210948 | 0.679012 |
| Higher-order | raw_calibrated | 0.360448 | 0.784021 |
| Higher-order | raw_backtracked | 0.371802 | 0.773312 |
| Higher-order | no_write | 0.790476 | 1.510553 |

## Interpretation

- In the first-order family, calibration improves mean 64-step MSE from 0.203014 to 0.177400 (12.6% lower). Repeat-16 collateral drift falls from 0.008003 to 0.000001015, about 7,900-fold. However, seed 41 misses the useful-learning noninferiority band at the short horizon, so the family-level gate fails.
- In the higher-order family, same-step normalization removal worsens mean one-rewrite MSE from 0.507 to 2.994. Accepted updates prevent much of that damage, but only one of three seeds passes the combined rule.
- No-write has zero churn and poor correction recall. Raw policies can likewise stop moving before reaching the target tolerance. Reduced byte changes are not sufficient evidence of successful memory.
- The exact float delta control scores zero MSE for every schedule, using supplied one-hot addresses. This is a strong control for this explicit-key task. The indexed uint8 table uses 12 bytes and incurs only value-rounding error.
- Backtracking guarantees only current-key loss decrease after quantization. Other keys can still be harmed.
- Both reader families were trained around the original RMS writer. A writer swap is a distribution shift; this frozen result does not decide how retrained raw/signed/vector readers will compare.
- No controller, rotational bank, natural-image task, general phase advantage or biological mechanism is established.

## Uncertainty and decision rule

Receipts include per-stream paired arm-minus-reference differences and their standard errors at each reported horizon. The paired-history schedule clusters each pair of related histories before estimating uncertainty: 128 streams represent 64 independent history pairs. All arms use the same underlying streams; treating them as independent would discard that covariance. The figure's recall error bars instead show SEM across the three training seeds.

A checkpoint passes only if actual repeat-16 collateral drift is at most half the reference, updated-key MSE at horizons 1 and 64 stays within `max(0.005, 5% of reference)`, and full-memory MSE at 64 stays within 0.005 of reference. A family requires all three training seeds to pass. These are predeclared effect bands, not a statistical significance claim. Near-zero reference drift is classified floor-limited.

## Calibration

All first-order calibrated and accepted arms selected eta=1/16. Higher-order raw calibration selected 1/64 for all three seeds; accepted updates selected 1/16 for seeds 41/127 and 1/64 for seed 97. Full candidate scores are in the receipt. No held-out stream selected eta.

## Receipts and reproduction

- [Compact summary and every checkpoint verdict](summary.json)
- [Scalar metrics](metrics.csv)
- [Full per-stream errors and per-write traces, gzip JSON](full_receipt.json.gz)
- [Figure](comparison.png)
- [Checkpoint manifest](../../checkpoints/meta/PROVENANCE.json)
- [Review corrections](../../docs/IMPLEMENTATION_REVIEW.md)

Run `python -m sillisalaatti.benchmark --checkpoint-dir checkpoints/meta --output runs/frozen` from the repository root. The CLI writes a complete uncompressed summary; the checked-in full receipt is compressed for size. Source hashes and environment are included. Values may vary slightly with PyTorch/device numerical behavior.

All inference persistent state is passed explicitly. Writer-call counts are batched API calls, not matched FLOPs. Timing excludes the common four-write initialization and is descriptive CPU timing. Calibration time is additional. Shared weights, buffers, evaluator targets, candidate reads and autograd temporaries are not included in the 48-byte data state.
