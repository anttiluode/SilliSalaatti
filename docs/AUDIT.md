# Source audit: scaling, quantization, and temporal roles

Reviewed 15 October 2026. This is a source audit with a constructed numerical diagnostic, not a trained-network reproduction.

Follow-up: the approved writer comparison is now implemented and measured on six real upstream checkpoints. See the [frozen-reader report](../results/frozen_meta/REPORT.md). The distinctions below describe the original audit and remain relevant to interpreting that experiment.

## 1. Locate the actual writer

Pinned source: [PhaseGraphFastWeightCell at f33c35c](https://github.com/anttiluode/PhaseGraphFastWeightCell/tree/f33c35cc3d0237ecf27ec851454047b2e35e0d4d).

[`phasegraph/meta_ttt.py`](https://github.com/anttiluode/PhaseGraphFastWeightCell/blob/f33c35cc3d0237ecf27ec851454047b2e35e0d4d/phasegraph/meta_ttt.py), `MetaDeltaReader._gradient_step`, computes:

1. Current-key prediction error and its state gradient.
2. Gradient divided by detached per-example RMS, clamped below at `1e-9`.
3. Optional top-k selection; the default permits eight state coordinates.
4. A proposal scaled by `step_bins / 256`, followed by byte quantization.
5. A no-op when per-example mean squared error is at most `tolerance**2`.

Defaults are `grid_size=4`, `channels=2`, `step_bins=8`, and `tolerance=0.002`: **48 persistent bytes**. A separately configured 8x8/eight-channel graph has 896 bytes, but that was not the reported 48-byte sequential meta-training result.

[GraphNN at 0e3d4ca](https://github.com/anttiluode/GraphNN/tree/0e3d4ca441aa6a549c5f596dc113eff79a8d21d8) used a learned CNN/MLP writer and context-dependent blending gate for its image rewrite experiment. Its 85.46% unchanged-A byte-change statistic cannot be attributed to an RMS gradient operation in another model.

## 2. What normalization implies

For a fixed local reader Jacobian, uniformly scaling a nonzero residual scales its gradient by the same factor. Dividing by gradient RMS cancels that factor, until the RMS clamp or error tolerance becomes active.

This is a **scale-insensitive proposal above the deadband**, not proof that every microscopic error produces a write. Top-k selection, state sensitivity, quantization, and the existing tolerance all affect the actual result.

Removing normalization retains residual magnitude. It also changes the effective learning rate, and small proposals can round back to the same byte state. Consequently:

- Less byte churn alone does not establish better memory.
- Exact same-step removal tests the arithmetic change, but is not a fair comparison of well-calibrated algorithms.
- A separate development-only step-size calibration is required.
- Quantized current-key error must be checked after a proposed accepted update.

## 3. Constructed numerical check

The diagnostic used NumPy, float64 arithmetic, a 48-byte state initialized to code 128, the scalar reader `prediction = state[0] / 256`, squared error, the source rule's eight-bin scale, and the source tolerance 0.002. Only the first gradient coordinate was nonzero. Top-k selection therefore leaves it unchanged. Circular nearest-byte rounding matches the source quantizer away from ties and seams in these cases.

| Initial residual | Rule | Changed bytes | First-coordinate change | Squared error before | Squared error after |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.001 | RMS | 0 | 0 bins | 0.000001 | 0.000001 |
| 0.001 | Raw | 0 | 0 bins | 0.000001 | 0.000001 |
| 0.004 | RMS | 1 | -55 bins | 0.000016 | 0.044455086914 |
| 0.004 | Raw | 0 | 0 bins | 0.000016 | 0.000016 |
| 0.020 | RMS | 1 | -55 bins | 0.000400 | 0.037964086914 |
| 0.020 | Raw | 0 | 0 bins | 0.000400 | 0.000400 |
| 0.100 | RMS | 1 | -55 bins | 0.010000 | 0.013189086914 |
| 0.100 | Raw | 1 | -2 bins | 0.010000 | 0.008498535156 |

With one active coordinate, `gradient / RMS(gradient)` has magnitude `sqrt(48)`, so an eight-bin base scale can propose about 55 bins in that coordinate. Eight bins is not an eight-bin per-coordinate bound.

The 0.004 residual is correctable to below tolerance with a one-bin change, yet the raw proposal at the unchanged learning rate does not cross the rounding boundary. This demonstrates how apparent no-op stability can also be failure to learn.

These results establish possibilities for the update arithmetic. They do **not** establish how much normalization contributed to the trained nonlinear graph's errors. A frozen-checkpoint ablation and separately trained comparison remain necessary. No trained checkpoint was run in this audit.

The diagnostic is reproducible with the following core calculation, applied to each residual in the table:

```python
state = np.full(48, 128, dtype=np.uint8)
work = state.astype(np.float64) / 256
target = work[0] - residual
gradient = np.zeros(48)
gradient[0] = 2 * residual
direction = gradient / max(np.sqrt(np.mean(gradient**2)), 1e-9)  # RMS arm
# direction = gradient                                        # raw arm
proposal = np.mod(np.round((work - (8 / 256) * direction) * 256), 256)
candidate = proposal.astype(np.uint8)
if residual**2 <= 0.002**2:
    candidate = state.copy()
after_squared_error = (candidate[0] / 256 - target)**2
```

## 4. Tupsu's controller has indirect training credit

Pinned source: [`Tupsu.forward`](https://github.com/anttiluode/Tupsu/blob/599df774de0d0c4bb59a8a80b058182d96f58cba/tupsu/model.py) calls its Martinotti-inspired controller under `torch.no_grad()` with detached mean bursts. Burst parameters can learn through attention/pooling, but episode boundaries do not receive direct gradients through that controller.

This is a design property, not automatically a bug. A new differentiable write controller needs its own comparison with fixed and shuffled timing; useful control cannot be inferred from a biological name or cleaner-looking boundaries.

## 5. The old reinterpretation task discarded its answer

Pinned source: [`make_later_context_batch`](https://github.com/anttiluode/MultipleTemporalLenses/blob/de74308099ffc4d0e20370cea9dfdf930a911ffb/src/multiple_temporal_lenses/tasks.py).

The generator writes first and second candidates at the same position with weights 0.5 and 0.5. It then uses a later context bit to request the first or second candidate. Swapping the candidate order leaves the observable mixture and context unchanged while reversing the requested label.

For six candidates and two context values, 60 equally weighted ordered-pair/context cases collapse to 30 observations, each with two equally likely labels. Enumerating these cases yields expected Bayes accuracy **0.5**. A replacement must retain candidate roles through distinct positions or explicit tags and first pass an exact explicit-history oracle.

The prior [BrainLoops interpretation audit](https://github.com/anttiluode/BrainLoops/blob/main/docs/interpretation/2026-10-05-resonance-valves.md#related-memory-test-correction) already records this correction. Other temporal-lens gates retain their verdicts.

## 6. Rotations supply candidate coordinates, not guaranteed addresses

Damped rotational modes have lag responses involving `r**L * exp(i * omega * L)`. Frequencies, damping, sampled lags, input projections, and readout jointly determine their separation. Arbitrary frequencies do not guarantee orthogonal lags; real leaky banks can also retain separable finite-lag distinctions.

Measure lag-response rank, singular values, coherence, and prediction at matched real-state budgets. A complex scalar consumes two real coordinates. EEG recurrence in BrainLoops does not establish that a particular oscillatory controller or memory architecture is present in cortex.
