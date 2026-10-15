# SilliSalaatti

A controlled research project connecting Tupsu's temporal controller, MultipleTemporalLenses' compact history, and GraphNN's writable response geometry to error-correcting recurrent memory.

**Status: source audit and experimental design prepared for review. No streaming model has been implemented or benchmarked here yet.**

The first question is deliberately narrow:

> Does removing per-observation gradient RMS normalization improve useful repeated-write stability, or merely make quantized memory stop learning?

Read the [source audit](docs/AUDIT.md) and [proposed experimental design](docs/superpowers/specs/2026-10-15-streaming-memory-design.md).

## What the audit establishes

- The RMS-normalized error-gradient writer belongs to **PhaseGraphFastWeightCell**, not GraphNN's CNN/MLP writer. Its published sequential experiment used **48 state bytes**; GraphNN's phase-only image experiment used **896**.
- That writer already has an error tolerance. Above it, normalization largely removes the gradient's magnitude; below it, writes are suppressed.
- A constructed scalar-reader diagnostic reproduces normalized overshoot **and** raw-gradient updates that vanish under byte rounding. This identifies possible failure modes, not the cause of the trained graph's results.
- Damped rotations can produce richer temporal response directions. They do not automatically make different lags orthogonal.
- The old temporal-lens reinterpretation task erased candidate roles. Its expected accuracy ceiling was 50%; the replacement task must preserve roles and pass an exact oracle first.

## Proposed order

1. **Writer ablation:** normalized, raw, development-calibrated raw, and accepted backtracking updates on identical frozen readers and streaming data.
2. **Learned memory comparison:** phase graph, signed-real transport, and direct vector, with an exact keyed delta matrix as a strong task-specific control.
3. **Temporal controller:** add a differentiable surprise trace only after the writer comparison; compare adaptive, tonic, periodic, and shuffled controls.
4. **Temporal basis:** independently compare leaky states with equal-size damped rotations on role-preserving history tasks.

Long-horizon recall, successful correction, untouched-binding drift, byte changes, and compute are evaluated together. A memory that stops changing but fails to learn a correction does not pass.

The plan retains local attention as an optional later hybrid component. Full language modeling, an HRM-style reasoning loop, and a GUI are outside the first experiment.

## Provenance

Original projects: [Tupsu](https://github.com/anttiluode/Tupsu), [MultipleTemporalLenses](https://github.com/anttiluode/MultipleTemporalLenses), [BrainLoops](https://github.com/anttiluode/BrainLoops), [GraphNN](https://github.com/anttiluode/GraphNN), and [PhaseGraphFastWeightCell](https://github.com/anttiluode/PhaseGraphFastWeightCell).

Relevant primary papers: [Gated DeltaNet](https://arxiv.org/abs/2412.06464), [Titans](https://arxiv.org/abs/2501.00663), and [Mamba-3](https://arxiv.org/abs/2603.15569).

MIT license. Upstream experiments keep their original results and interpretation boundaries.
