# First writer benchmark: review record

The first increment implements the approved writer ablation, causal streams, exact-key controls, matched phase/signed/vector training interfaces, checkpoint resume and reporting. The temporal controller and rotational bank remain separate stages. The 600-episode retrained comparison has not been measured; six 16-episode runs verify the training path.

An independent whole-branch review identified these measurement issues. All were corrected before regenerating the published evaluation receipts:

| Finding | Correction and verification |
| --- | --- |
| Primary drift rule used a diagnostic 17th repeat | Score the actual 16th unchanged observation; retain the extra sham only as a secondary metric. Regression distinguishes the two. |
| Missing paired uncertainty | Retain per-stream errors and calculate arm-minus-reference paired differences. Cluster related history pairs; duplicate-history regression checks that uncertainty does not shrink artificially. |
| Delayed correction was hidden by aggregate recall | Report corrected key 0 and preserved key 3 separately, including paired uncertainty. Regression exposes loss despite improving aggregate recall. |
| New schedule-specific metrics broke CSV field collection | Build the CSV schema from the union of scalar fields across schedules. CLI regression covers mixed rows. |
| Training plots were labeled as frozen-reader experiments | Derive the figure title and embedded PNG provenance from the experiment protocol. Regression checks the training title. |

Review ruling: the plot label was treated as an important provenance correction, because it could make a reader attribute results to the wrong experiment. It was implemented rather than deferred. No minor findings remain deferred.

Validation: 39 focused tests pass on Python 3.12 / PyTorch 2.5.1 CPU. These include reference-writer byte parity, quantized no-op and stall cases, accepted-update behavior, safe checkpoint loading, exact delta recall, deterministic causal tasks, exact episode-boundary CPU resume, CLI receipts and figure provenance. CUDA execution and CUDA resume equivalence remain unmeasured.

The final frozen evaluation contains 120 runs over six verified upstream checkpoints, each with 128 streams and 64 post-initialization observations. Full receipts retain per-stream horizon errors and scalar per-write traces. Source SHA-256 hashes, original checkpoint hashes, development selections, environment and call/timing costs accompany the measurements.
