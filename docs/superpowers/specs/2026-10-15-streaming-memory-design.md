# SilliSalaatti: controlled streaming-memory design

Date: 15 October 2026. Status: **proposed; awaiting user review**.

## Intent and scope

The user supplied SilliSalaatti for a continuation of the Tupsu, temporal-lens, and writable-graph work, with Gemini's proposed RMS-normalization explanation as an input. The intended outcome is a runnable experiment that identifies which update mechanisms improve useful retained information.

Design assumption: begin with the normalization ablation and a streaming benchmark, then introduce the temporal controller and rotational states as separately reported stages. This preserves a causal comparison before combining mechanisms.

Success is an informative measured result, including a negative result. A lower byte-change count without successful new learning is not success. No claim of a Transformer replacement, new biological mechanism, or general phase-memory superiority follows from the proposed tasks.

The [source audit](../../AUDIT.md) corrects the model attribution, state-size conflation, existing no-op threshold, and automatic-orthogonality claim.

## Stage 1: establish the writer comparison

### Reuse and provenance

Start from PhaseGraphFastWeightCell commit `f33c35cc3d0237ecf27ec851454047b2e35e0d4d`. Vendor only the required reader/quantizer components under `sillisalaatti/_vendor/phasegraph`, retaining MIT provenance and a record of imported file hashes. Keep the new writer and benchmark outside the vendor directory. Do not modify the upstream repositories.

Use the 4x4/two-channel configuration: 48 actual uint8 memory bytes. Keep reader architecture, initialization, keys, data, quantization, top-k limit, and error tolerance identical across writer ablations. Include a no-write control to expose apparently stable but uninformative memory.

### Separate two comparisons

1. **Frozen reader:** load an explicitly supplied upstream trained checkpoint, record its SHA-256, and apply alternative numerical writers without changing reader weights. For the pure correction/repetition ablation, initialize bindings once with the reference writer and fork byte-identical states into all arms. Report the separate end-to-end variant, where each writer initializes its own state, under a different label. If no checkpoint is available, mark this comparison `NOT_RUN`; do not replace it silently with a random reader. No checkpoint was evaluated during the design audit.
2. **Retrained reader:** train each writer/reader combination from the same initialization and streams, using the same optimization schedule. This tests whether the reader can adapt to its update law. Keep frozen-reader and retrained results in distinct reports.

### Writer arms

| Arm | Update policy | Purpose |
| --- | --- | --- |
| `rms_reference` | Upstream per-observation RMS normalization, top-k, step and tolerance | Reproduce the original numerical rule |
| `raw_same_step` | Remove RMS division only | One-variable arithmetic ablation |
| `raw_calibrated` | Raw gradient; learning rate selected on development data | Detect whether a negative raw result was a scale mismatch |
| `raw_backtracked` | Calibrated raw proposal; accept only a quantized current-key loss decrease | Test finite-step overshoot separately |
| `no_write` | Preserve initialized memory through the correction/repetition interval | Stability-without-learning control |

All gradient arms retain the upstream tolerance 0.002 and top-k=8 in the primary run. `raw_same_step` retains `step_bins=8` exactly. Calibrated arms choose `eta` from `{1/256, 1/64, 1/16, 1/4, 1, 4}` using development streams only; `eta` directly multiplies the unnormalized gradient in normalized-state coordinates. In the frozen-reader comparison, minimize full-memory development MSE after 64 rewrites. For retrained arms, use a fixed 64-episode development training allocation for every candidate, rank by the same development metric, and restart canonical training from the original initialization with the chosen value. Report calibration compute separately. Break ties in favor of smaller `eta`. Freeze the chosen values before held-out evaluation.

The no-write arm uses a copy of the reference-initialized byte state and suppresses all subsequent updates. It does not receive a fresh initialization after a correction.

Backtracking tests multipliers `{1, 1/2, 1/4, 1/8}` of its calibrated proposal, always starting from the unchanged current state. Quantize each candidate before evaluation. Commit the first candidate with finite current-key MSE at least `1e-12` below the current error; otherwise preserve the state. Count every attempted and accepted read/gradient operation. Current-key monotonicity is not a guarantee of protection for other keys.

Do not introduce a persistent floating shadow state, error accumulator, or momentum in this stage. Small raw steps that round away must be recorded as such.

## Streaming task and information access

Four explicitly supplied one-hot keys each map to three continuous values in `[-1,1]`. Each write receives only the current key and its new value. The read receives memory and a query key. Model weights are frozen during held-out evaluation.

Each episode initializes all four bindings, then runs one of these frozen schedules:

- Independent random rewrites, evaluated after 1, 4, 16 and 64 rewrites.
- Unchanged repetitions of one binding, evaluated after 1, 4 and 16 repetitions.
- A targeted correction A to A-prime, followed by bursts of valid updates to other keys, then delayed queries for A-prime and untouched bindings.
- Paired histories with identical final input/query but a different earlier untouched binding.

Other-key updates are the distractors; they remain legitimate facts the memory must learn. No oracle importance flag is supplied. Repetition is identified only through the observation and current memory, not a hidden task label.

The harness may retain ground-truth bindings for offline scoring and training losses. Those targets, source histories, probe masks, and future queries must never enter an inference writer. A write rule cannot inspect unrelated target values to choose its step. Report current-key loss, full-memory accuracy, and untouched-answer drift separately.

Use training seeds `{41,97,127}`. Use development-only seeds `{11,23,37}`, with 128 development streams each. Use 128 independently generated held-out streams per training seed and schedule; held-out generator seeds are `1_000_000_000 + training_seed + 10_000 * schedule_index`, with schedule indices in the order above. Publish the generators and complete configuration before canonical training.

The primary retrained comparison uses 600 episodes, batch size 8, horizons `{1,4,16,64}`, segment length 8, Adam learning rate 0.002, and gradient clipping 1.0. Per-seed schedules and initialization are shared. Report optimizer updates and token exposure; episode counts alone do not establish equal training effort. Follow the upstream first-order stop-Jacobian-gradient training path for the primary writer comparison. Second-order meta-learning is a separately named follow-up, not another simultaneous change.

### Measurements and decision bands

Record per stream and aggregate by training seed:

- Full and newest-binding MSE at every horizon.
- Untouched-binding accuracy and actual output drift after each write.
- Changed-byte fraction, circular phase distance, and proposal magnitude before rounding.
- Residual before/after each repeated input; distinguish learning repeats from repeats already below tolerance.
- Rejected proposals, quantization-stalled proposals, and accepted writes.
- Wall time, reader calls, state-gradient evaluations, peak temporary memory when measurable, shared parameter bytes, and physical resident-state bytes.

Primary question: can a raw/accepted arm reduce repeat-induced untouched-answer drift by at least 50% relative to the RMS reference while meeting both useful-learning requirements below?

1. Updated-binding MSE must not exceed reference MSE by more than `max(0.005, 0.05 * reference_MSE)` at 1 and 64 rewrites.
2. Full-memory MSE after 64 rewrites must not exceed reference by more than 0.005.

Require the direction of the repeat-drift improvement and both noninferiority conditions in all three training seeds. If reference drift is below `1e-10`, label that comparison floor-limited instead of declaring a ratio win. Report results even when every decision band fails. These are practical predeclared bands for this experiment, not universal memory thresholds.

Use paired stream-level uncertainty within each seed and show all three seed summaries. Do not treat all pooled streams as independent training runs.

## Stage 2: determine whether graph transport earns its cost

Once Stage 1 is reported, compare separately trained phase, signed-real transport, and direct-vector readers under a writer selected from development results, not the Stage-1 held-out scores. Each gets 48 physical data bytes. Use matched shared parameter counts where the architectures permit, identical training streams, and identical optimizer-update exposure. Explain any mismatch explicitly. Stage-2 held-out generator seeds use the Stage-1 formula with a `2_000_000_000` base, keeping this evaluation independent of the earlier selection process.

The strong task-specific control is an ordinary float32 delta matrix for the 4x3 known-key task: also 48 bytes. With one-hot keys and full-strength correction, it can retain exact latest values; it is not a learned language-model baseline. Include a uint8 indexed table as a second storage-format control, reporting its smaller payload and quantization floor honestly.

Storage matching does not imply compute matching. Report quality against measured time and reader/gradient calls, including the extra candidate reads of backtracking. Do not claim matched FLOPs from matched step counts, and do not waste dummy operations to manufacture equality.

A phase-specific result requires improved retained-answer quality against the signed-real reader at equal storage and a documented cost tradeoff across all three seeds. Beating positive-only gains is insufficient. If signed-real matches phase, retain that result and attribute any gain more narrowly to the writer or transport architecture.

## Stage 3: train the temporal controller

After Stage 1/2, introduce a differentiable, data-dependent facilitating trace. It receives present observation features, current-key error, and its previous value. It sets correction strength and a separately parameterized forgetting decision. Offline delayed recall trains the controller; held-out writes never see future labels.

Define forgetting explicitly: move toward the reader's declared blank state, using shortest circular displacement for phase bytes and linear displacement for signed/vector bytes, then quantize. Apply the corrective proposal from that resulting state. A control fixes forgetting strength to zero. This reset-like operation is a candidate forgetting mechanism, not proof that a fractional phase change erases a proportional amount of information. Do not multiply raw phase angles toward zero and treat that as coordinate-independent decay. Report the two operations and their collateral effects separately.

The first controller has one float32 trace: four additional resident bytes, making the 48-byte memory plus controller **52 bytes**. Every controller control carries and reports the same trace allocation. Any extra persistent controller statistic increases the budget and requires an explicit protocol amendment. No gradient, optimizer state, or candidate memory may persist uncounted between observations.

Controls are adaptive trace, matched tonic correction strength, periodic admission at matched write rate, and shuffled controller outputs. A shuffled baseline receives an online causal permutation/delay construction or a separately labeled offline replay diagnostic; future controller outputs cannot be inserted into an online baseline. Compare both correction success and retained old answers at matched accepted-write count, plus actual computational cost.

The proposed trace is inspired by Tupsu and Titans. It does not identify Martinotti inhibition in cortex. The initial gate changes plastic writes; it does not claim to stabilize a recurrent excitatory population. HRM-style multi-step reasoning remains outside this stage.

## Stage 4: test rotational temporal coordinates independently

Compare a leaky bank with damped rotational states at the same real-coordinate budget. Count each complex scalar as two real scalars. Match or disclose input-projection and readout parameter counts. Include a generic concat reader rather than testing only softmax lens routing.

Before training, inspect lag-response rank, singular values, normalized coherence, and conditioning for lags `{2,20,200}`. Normalizing diagnostic columns must not hide vanishing absolute signal under noise. Measure prediction both without noise and under a frozen development-selected perturbation.

Use same-present/different-history pairs and a repaired later-context task with explicit candidate-role tags or distinct positions. Enumerate the finite task to verify that distinct labels are never assigned to identical full observations; an exact explicit-history oracle must score 100% before any compact model is judged.

The question is whether rotations improve useful temporal separation. They are not assumed to create orthogonal addresses, recover discarded information, or inherit biological support from EEG recurrence.

## Intended implementation and delivery

The subsequent implementation plan should provide a small PyTorch package, a streaming benchmark command, JSON/CSV receipts, checkpoint save/resume, a compact plotting command, Windows-friendly run instructions, focused correctness tests, and CPU smoke CI. CUDA must be optional; the full run should fit the user's RTX 3060 with a configurable batch size. No GUI or large model/data download is required for the initial gate.

Tests should verify true uint8 round trips, no hidden sequence cache, correct tolerance no-op behavior, accepted-update loss decrease after quantization, correction learning versus no-write, correct data/target separation, deterministic schedule replay, checkpoint resume, and valid role-preserving tasks. They should not encode a presumed phase advantage.

Publish the source revision, environment, configuration, data seeds, state/weight byte accounting, elapsed costs, full per-seed metrics, and explicit verdict. Separate smoke checks, constructed arithmetic diagnostics, frozen-checkpoint comparisons, and canonical trained results.

## Review boundary

This commit contains the audit and proposed design only. After written-spec review, produce the implementation plan and select execution method before product implementation. Changes requested during review should update this document before the experiment is frozen.
