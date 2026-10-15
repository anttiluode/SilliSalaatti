"""Frozen-checkpoint writer ablations and honest exact-key controls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from ._vendor.phasegraph.meta_ttt import read_all
from .checkpoints import load_checkpoint
from .tasks import make_stream, SCHEDULES
from .writers import WriteConfig, update, RULES, ETA_GRID


def _horizons(stream):
    return set((1, 4, 16, 64, stream.horizon))


def paired_difference(reference, arm, *, clustered=False):
    """Within-stream arm minus reference uncertainty, clustered by base history."""
    a = torch.as_tensor(reference, dtype=torch.float64)
    b = torch.as_tensor(arm, dtype=torch.float64)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('paired values must be nonempty, equally sized finite vectors')
    differences = b - a
    if clustered:
        if len(a) % 2:
            raise ValueError('history-pair clustering requires even stream count')
        differences = differences.reshape(-1, 2).mean(1)
    return {
        'mean_difference': float(differences.mean()),
        'standard_error': float(differences.std(unbiased=True) / len(differences)**.5) if len(differences) > 1 else 0.,
        'effective_histories': len(differences), 'stream_count': len(a),
    }


def _errors(predicted, truth, *, clustered=False):
    per_stream = (predicted - truth).square().mean((1, 2))
    uncertainty = paired_difference(torch.zeros_like(per_stream), per_stream, clustered=clustered)
    return {
        'mse': float(per_stream.mean()),
        'mse_standard_error': uncertainty['standard_error'],
        'effective_histories': uncertainty['effective_histories'],
        'per_stream_mse': per_stream.cpu().tolist(),
    }


def _answer_metrics(predicted, truth, previous, keys, schedule):
    count = len(predicted)
    untouched = torch.arange(4, device=predicted.device)[None] != keys[:, None]
    squared = (predicted - truth).square()
    updated = squared[torch.arange(count, device=predicted.device), keys].mean(-1)
    unrelated = squared[untouched].reshape(count, 3, 3).mean((1, 2))
    drift = (predicted - previous).square()[untouched].reshape(count, 3, 3).mean((1, 2))
    metrics = {
        'updated_mse': float(updated.mean()), 'untouched_mse': float(unrelated.mean()),
        'untouched_drift_mse': float(drift.mean()),
        'per_stream_updated_mse': updated.cpu().tolist(),
        'per_stream_untouched_mse': unrelated.cpu().tolist(),
        'per_stream_untouched_drift_mse': drift.cpu().tolist(),
    }
    if schedule in ('correction', 'paired'):
        for key, name in ((0, 'corrected_key0_mse'), (3, 'preserved_key3_mse')):
            per_stream = squared[:, key].mean(-1)
            metrics[name] = float(per_stream.mean())
            metrics['per_stream_' + name] = per_stream.cpu().tolist()
    return metrics


def evaluate(model, config: WriteConfig, stream, init_config: WriteConfig | None = None) -> dict:
    if model.slots != 4 or model.value_size != 3:
        raise ValueError('This benchmark requires four keys and three values')
    model.eval()
    initializer = init_config or WriteConfig(eta=model.step_bins / 256,
                                            tolerance=model.tolerance, topk=model.topk)
    state = None
    for obs in stream.inputs[:, :4].unbind(1):
        state, _ = update(model, state, obs, initializer)
    initial_hash = hashlib.sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    with torch.no_grad():
        old = read_all(model, state)
    rows, trace = {}, []
    costs = {'reader_calls': 0, 'gradient_calls': 0, 'metric_reader_calls': 1, 'sham_reader_calls': 0, 'sham_gradient_calls': 0}
    began = time.perf_counter()
    for step, obs in enumerate(stream.inputs[:, 4:].unbind(1), start=1):
        state, stats = update(model, state, obs, config)
        costs['reader_calls'] += stats['reader_calls']
        costs['gradient_calls'] += stats['gradient_calls']
        with torch.no_grad():
            predicted = read_all(model, state)
            keys = obs[:, :4].argmax(-1)
            untouched = torch.arange(4, device=state.device)[None] != keys[:, None]
            row = {
                'step': step, **stats,
                **_answer_metrics(predicted, stream.targets[:, step + 3], old, keys, stream.schedule),
            }
        costs['metric_reader_calls'] += 1
        trace.append({k: v for k, v in row.items() if not k.startswith('per_stream_')})
        if step in _horizons(stream):
            sham, sham_stats = update(model, state, obs, config)
            with torch.no_grad():
                sham_prediction = read_all(model, sham)
            costs['sham_reader_calls'] += sham_stats['reader_calls'] + 1
            costs['sham_gradient_calls'] += sham_stats['gradient_calls']
            rows[str(step)] = {
                **row, **_errors(predicted, stream.targets[:, step + 3], clustered=stream.schedule == 'paired'),
                'sham_changed_byte_fraction': sham_stats['changed_byte_fraction'],
                'sham_untouched_drift_mse': float((sham_prediction - predicted).square()[untouched].mean()),
                'sham_before_mse': sham_stats['before_mse'], 'sham_after_mse': sham_stats['after_mse'],
            }
        old = predicted
    return {
        'writer': asdict(config), 'schedule': stream.schedule, 'data_seed': stream.seed,
        'count': len(stream.inputs), 'state_bytes': model.payload_bytes,
        'initial_state_sha256': initial_hash,
        'initialization': 'shared RMS-reference four-write prefix' if init_config is None else 'explicit writer four-write prefix',
        'initialization_writer': asdict(initializer),
        'seconds_excluding_common_initialization': time.perf_counter() - began,
        'costs': costs, 'horizons': rows, 'trace': trace,
    }


def exact_control(stream, kind='delta') -> dict:
    if kind not in ('delta', 'indexed_uint8'):
        raise ValueError('unknown exact control')
    count = len(stream.inputs)
    codes = torch.zeros(count, 4, 3, dtype=torch.uint8, device=stream.inputs.device) if kind == 'indexed_uint8' else None
    state = torch.zeros(count, 4, 3, device=stream.inputs.device)
    rows = {}
    began = time.perf_counter()
    for t, obs in enumerate(stream.inputs.unbind(1)):
        keys = obs[:, :4].argmax(1)
        if codes is None:
            # Orthogonal one-hot keys make full-strength delta correction exact.
            state[torch.arange(count, device=state.device), keys] = obs[:, 4:]
        else:
            codes[torch.arange(count, device=state.device), keys] = ((obs[:, 4:] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
            state = codes.float() / 127.5 - 1
        horizon = t - 3
        if horizon in _horizons(stream):
            rows[str(horizon)] = _errors(state, stream.targets[:, t], clustered=stream.schedule == 'paired')
    return {'control': kind, 'state_bytes': 48 if codes is None else 12,
            'seconds': time.perf_counter() - began, 'horizons': rows,
            'access': 'given exact one-hot keys; no learned parameters'}


def calibrate(model, rule: str, count: int = 128, horizon: int = 64, grid=ETA_GRID) -> dict:
    candidates = []
    device = str(next(model.parameters()).device)
    for eta in sorted(grid):
        scores = []
        for seed in (11, 23, 37):
            receipt = evaluate(model, WriteConfig(rule=rule, eta=eta),
                               make_stream(seed, count, horizon, 'independent', device))
            scores.append(receipt['horizons'][str(horizon)]['mse'])
        candidates.append({'eta': eta, 'mse': sum(scores) / len(scores), 'seed_mse': scores})
    best = min(candidates, key=lambda row: (row['mse'], row['eta']))
    return {'eta': best['eta'], 'seeds': [11, 23, 37], 'count': count,
            'horizon': horizon, 'candidates': candidates, 'selection': 'development only'}


def write_receipts(output: Path, result: dict):
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output / 'summary.json')
    scalar_rows = []
    for run in result['runs']:
        for horizon, row in run['horizons'].items():
            scalar_rows.append({
                'checkpoint': run['checkpoint']['file'], 'seed': run['checkpoint']['seed'],
                'rule': run['writer']['rule'], 'eta': run['writer']['eta'],
                'schedule': run['schedule'], 'horizon': int(horizon),
                **{k: v for k, v in row.items() if isinstance(v, (int, float))},
            })
    if scalar_rows:
        with (output / 'metrics.csv').open('w', newline='') as handle:
            fields = list(dict.fromkeys(key for row in scalar_rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(scalar_rows)


def decision_bands(result: dict) -> dict:
    verdicts, comparisons = [], []
    for name in sorted({r['checkpoint']['file'] for r in result['runs']}):
        runs = [r for r in result['runs'] if r['checkpoint']['file'] == name]
        indexed = {(r['writer']['rule'], r['schedule']): r for r in runs}
        for rule in ('raw_same_step', 'raw_calibrated', 'raw_backtracked'):
            for schedule in SCHEDULES:
                if ('rms_reference', schedule) not in indexed or (rule, schedule) not in indexed:
                    continue
                reference = indexed[('rms_reference', schedule)]
                arm = indexed[(rule, schedule)]
                for h in reference['horizons'].keys() & arm['horizons'].keys():
                    for metric in ('mse', 'updated_mse', 'untouched_drift_mse', 'corrected_key0_mse', 'preserved_key3_mse'):
                        field = 'per_stream_' + metric
                        a, b = reference['horizons'][h], arm['horizons'][h]
                        if field in a and field in b:
                            comparisons.append({'checkpoint': name, 'rule': rule, 'schedule': schedule,
                                                'horizon': int(h), 'metric': metric,
                                                **paired_difference(a[field], b[field], clustered=schedule == 'paired')})
            needed = [(a, s) for a in ('rms_reference', rule) for s in ('repeat', 'independent')]
            if any(key not in indexed for key in needed):
                continue
            ref_repeat, arm_repeat = (indexed[(a, 'repeat')]['horizons'] for a in ('rms_reference', rule))
            if '16' not in ref_repeat or any('64' not in indexed[(a, 'independent')]['horizons'] for a in ('rms_reference', rule)):
                continue
            ref_drift, arm_drift = (r['16']['untouched_drift_mse'] for r in (ref_repeat, arm_repeat))
            learned = True
            for h in ('1', '64'):
                reference = indexed[('rms_reference', 'independent')]['horizons'][h]
                arm = indexed[(rule, 'independent')]['horizons'][h]
                learned &= arm['updated_mse'] <= reference['updated_mse'] + max(.005, .05 * reference['updated_mse'])
                if h == '64':
                    learned &= arm['mse'] <= reference['mse'] + .005
            verdict = 'FLOOR_LIMITED' if ref_drift < 1e-10 else ('PASS_THIS_CHECKPOINT' if arm_drift <= .5 * ref_drift and learned else 'FAIL_THIS_CHECKPOINT')
            verdicts.append({'checkpoint': name, 'rule': rule, 'verdict': verdict, 'useful_learning': bool(learned),
                             'reference_repeat_drift': ref_drift, 'arm_repeat_drift': arm_drift})
    return {'checkpoint_verdicts': verdicts, 'paired_comparisons': comparisons,
            'primary_repeat_measure': 'actual untouched-binding drift at repeat 16; extra sham write is secondary',
            'claim': 'A family-level pass requires all three training seeds; checkpoint families must remain separate.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, nargs='+')
    p.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--output', type=Path, default=Path('runs/frozen'))
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--horizon', type=int, default=64)
    p.add_argument('--dev-count', type=int, default=128)
    p.add_argument('--dev-horizon', type=int, default=64)
    p.add_argument('--eta', type=float, help='Manually override calibrated eta; receipt is labeled diagnostic')
    p.add_argument('--rules', nargs='+', choices=RULES, default=list(RULES))
    p.add_argument('--schedules', nargs='+', choices=SCHEDULES, default=list(SCHEDULES))
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args(argv)
    if min(args.count, args.horizon, args.dev_count, args.dev_horizon, args.threads) < 1:
        p.error('counts, horizons and threads must be positive')
    if 'paired' in args.schedules and args.count % 2:
        p.error('paired schedule requires even --count')
    files = args.checkpoint or (sorted(args.checkpoint_dir.glob('*.pt')) if args.checkpoint_dir else [])
    if not files:
        p.error('supply --checkpoint PATH or --checkpoint-dir DIR containing .pt files')
    if args.eta is not None:
        WriteConfig(eta=args.eta)
    torch.set_num_threads(args.threads)
    result = {
        'protocol': 'SilliSalaatti frozen-reader writer ablation v1',
        'status': 'canonical_frozen_settings' if (args.count, args.horizon, args.dev_count, args.dev_horizon, args.eta) == (128, 64, 128, 64, None) else 'diagnostic_pilot',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': args.device},
        'arguments': {k: str(v) if isinstance(v, Path) else ([str(x) for x in v] if k == 'checkpoint' and v else v) for k, v in vars(args).items()},
        'calibration': [], 'runs': [], 'controls': [],
        'limits': ['Frozen reader; changing its writer is a test-time distribution shift.',
                   'Equal persistent bytes do not imply equal compute.',
                   'Known one-hot keys make the exact delta control unusually strong.',
                   'Controller and rotational temporal states are not implemented in this increment.'],
    }
    for path in files:
        model, metadata = load_checkpoint(path, args.device)
        etas = {}
        for rule in ('raw_calibrated', 'raw_backtracked'):
            if rule in args.rules:
                calibration = ({'eta': args.eta, 'selection': 'manual diagnostic override'} if args.eta is not None else
                               calibrate(model, rule, args.dev_count, args.dev_horizon))
                result['calibration'].append({'checkpoint': metadata['file'], 'rule': rule, **calibration})
                etas[rule] = calibration['eta']
                print(json.dumps({'calibrated': metadata['file'], 'rule': rule, 'eta': calibration['eta']}), flush=True)
        for schedule in args.schedules:
            index = SCHEDULES.index(schedule)
            seed = 1_000_000_000 + (metadata['seed'] or 0) + 10_000 * index
            stream = make_stream(seed, args.count, args.horizon, schedule, args.device)
            result['controls'].append({'checkpoint': metadata['file'], 'schedule': schedule,
                                       'data_seed': seed, 'delta': exact_control(stream),
                                       'indexed_uint8': exact_control(stream, 'indexed_uint8')})
            for rule in args.rules:
                config = WriteConfig(rule=rule, eta=etas.get(rule, model.step_bins / 256),
                                     tolerance=model.tolerance, topk=model.topk)
                receipt = evaluate(model, config, stream)
                receipt['checkpoint'] = metadata
                result['runs'].append(receipt)
                result['decision_bands'] = decision_bands(result)
                write_receipts(args.output, result)
                print(json.dumps({'checkpoint': metadata['file'], 'rule': rule, 'schedule': schedule,
                                  'mse': receipt['horizons'][str(args.horizon)]['mse']}), flush=True)
    result['decision_bands'] = decision_bands(result)
    write_receipts(args.output, result)


if __name__ == '__main__':
    main()
