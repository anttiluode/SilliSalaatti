# Writer Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Ship the first usable writer-ablation benchmark and measure the supplied upstream meta-trained checkpoints.

**Architecture:** Preserve pinned upstream readers and place numerical writers outside them. A causal task generator supplies only the current key/value; an independent evaluator retains scoring truth. Training, checkpoint loading, and reports share these interfaces.

**Tech Stack:** Python 3.10+, PyTorch 2.5+, pytest, optional matplotlib; CPU and optional CUDA.

**Spec:** ../specs/2026-10-15-streaming-memory-design.md

## Global Constraints

- First increment implements Stage 1 and its Stage-2 comparison controls; temporal-controller and rotational-basis experiments remain separately staged.
- Four one-hot keys, three values, 48-byte uint8 graph/vector state; no inference floating shadow state or target/history cache.
- Reference tolerance 0.002, top-k 8, normalized step 8/256; raw calibration grid `{1/256,1/64,1/16,1/4,1,4}`; backtracking factors `{1,1/2,1/4,1/8}`.
- Training seeds 41/97/127; development 11/23/37; canonical budget 600 episodes, batch 8, segment 8, Adam 0.002, clip 1.0.
- Frozen-reader and retrained receipts are distinct. Smaller runs are labeled pilots. Exact known-key delta is a strong task-specific control.
- Stage-1 code is checked on real meta-trained artifacts; global-writer checkpoints are rejected explicitly.

## Review Focus

- Wrong checkpoint family: reject with an actionable error, never silently reinterpret weights.
- Quantization stalls: report them alongside correction error, never count all frozen states as successful no-ops.
- Backtracking with mixed batches: each case independently accepts its first improving quantized candidate.
- Resume under changed configuration: reject incompatible configuration and reproduce uninterrupted optimizer/RNG continuation.
- Arbitrary supplied sizes/NaNs: validate key/value/state shapes, finite numbers and arguments before running.

## Task 1: byte readers, checkpoint provenance, and writer policies

Files: `sillisalaatti/_vendor/phasegraph/{cell,quantization,meta_ttt}.py`, `sillisalaatti/checkpoints.py`, `sillisalaatti/writers.py`, `pyproject.toml`, `tests/test_writers.py`.

Interfaces: `load_checkpoint(path, device='cpu') -> (MetaDeltaReader, metadata)`; `WriteConfig(rule, eta, tolerance, topk)`; `update(model, state, observation, config, training=False) -> (state, stats)`.

- [x] Write tests for reference byte equality, tolerance no-op, raw rounding stall, accepted quantized loss decrease, mixed-batch acceptance, bad inputs, and wrong checkpoint family.
- [x] Run `python -m pytest tests/test_writers.py -q`; expected missing-package/interface failures.
- [x] Vendor the pinned MIT sources; implement validated pure-state writers and safe weights-only checkpoint loading with SHA-256 provenance.
- [x] Run the task tests and full available suite; expected all green. Commit.

## Task 2: causal streams, exact controls, and benchmark receipts

Files: `sillisalaatti/tasks.py`, `sillisalaatti/benchmark.py`, `tests/test_benchmark.py`.

Interfaces: `make_stream(seed, count, horizon, schedule, device) -> Stream(inputs, targets, queries)`; `evaluate(model, config, stream, init_config=None) -> receipt`; `python -m sillisalaatti.benchmark --checkpoint PATH --output DIR`.

- [x] Test deterministic independent/repeat/correction/paired streams, byte-identical initialization forks, no-write failure to learn corrections, exact-delta recall, and separate changed-byte/answer-error measures.
- [x] Run the new tests; expected missing task/evaluator interfaces.
- [x] Implement generators, per-write metrics, exact float delta and indexed-uint8 controls, development-only calibration, JSON/CSV receipts and explicit thresholds from the spec.
- [x] Run task tests and full suite; expected all green. Commit.

## Task 3: first-order retraining and exact resume

Files: `sillisalaatti/train.py`, `tests/test_training.py`.

Interfaces: `train(config, output, resume=None) -> checkpoint`; CLI exposes spec budget, kind, rule, seed, device, checkpoint interval, and segment size.

- [x] Test successful finite learning steps, byte-grid training forwards, safe checkpoint round trip, interrupted/resumed equivalence, and changed-config rejection.
- [x] Run the new tests; expected missing trainer.
- [x] Implement first-order segmented training, recorded optimizer-update counts, RNG/optimizer checkpoint state, development-only calibration support, and frozen evaluation after training.
- [x] Run task tests and full suite; expected all green. Commit.

## Task 4: runnable delivery, real checkpoint experiment, and review

Files: `sillisalaatti/plot.py`, `.github/workflows/tests.yml`, `.github/workflows/benchmark.yml`, `README.md`, `results/frozen_meta/`, `tests/test_cli.py`.

- [x] Test CLI argument failures and a small actual checkpoint-to-JSON/CSV workflow; run to observe the missing CLI/report behavior.
- [x] Add concise run instructions, optional plotting, CPU smoke CI and a manual full-budget workflow. Bundle the small verified meta checkpoints with artifact/hash provenance so the first run needs no account access.
- [x] Run frozen-reader ablations over all three checkpoint seeds; keep development calibration separate and publish every arm. Run a small retraining pilot, explicitly labeled.
- [x] Run `python -m pytest -q`, compile checks and CLI examples. Obtain a fresh whole-branch review, address important findings with regression tests, and update PR #1 with the runnable result.

## Execution

The user approved the spec and said go. Execute directly in this session. Preserve source-data provenance, report any unavailable measurements, and do not merge without an explicit merge instruction.
