"""First-order segmented reader training, with exact episode-boundary resume."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random

import torch

from ._vendor.phasegraph.meta_ttt import MetaDeltaReader, read_all
from .benchmark import evaluate, write_receipts
from .checkpoints import load_checkpoint
from .tasks import make_stream, SCHEDULES
from .writers import WriteConfig, update, ETA_GRID, RULES


@dataclass(frozen=True)
class TrainConfig:
    kind: str = 'phase'
    rule: str = 'rms_reference'
    eta: float = 8/256
    seed: int = 41
    episodes: int = 600
    batch_size: int = 8
    segment: int = 8
    learning_rate: float = 0.002
    train_horizons: tuple[int, ...] = (1, 4, 16, 64)
    grid_size: int = 4
    channels: int = 2
    width: int = 64
    transport_steps: int = 4
    threads: int = 2
    device: str = 'cpu'
    checkpoint_every: int = 25

    def __post_init__(self):
        WriteConfig(rule=self.rule, eta=self.eta)
        if self.rule == 'no_write' or self.kind not in ('phase', 'signed', 'vector'):
            raise ValueError('train a phase/signed/vector reader with a writing rule')
        fields = (self.episodes, self.batch_size, self.segment, self.channels, self.width,
                  self.transport_steps, self.threads, self.checkpoint_every)
        if any(type(x) is not int or x < 1 for x in fields) or type(self.grid_size) is not int or self.grid_size < 2:
            raise ValueError('training sizes must be positive integers; grid_size >= 2')
        if not self.train_horizons or any(type(x) is not int or x < 1 for x in self.train_horizons):
            raise ValueError('training horizons must be positive')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')


def train(config: TrainConfig, output: str | Path, resume: str | Path | None = None,
          stop_after: int | None = None) -> Path:
    """Save and return last.pt. stop_after simulates a planned interruption."""
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop_after must be a positive episode number')
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    model = MetaDeltaReader(kind=config.kind, grid_size=config.grid_size,
                            channels=config.channels, width=config.width,
                            transport_steps=config.transport_steps, second_order=False).to(config.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    schedule = random.Random(config.seed + 1717)
    start, updates, trace = 0, 0, []
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved.get('train_config') != asdict(config):
            raise ValueError('Resume configuration differs from the saved training configuration')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['training_state']['optimizer'])
        state = saved['training_state']
        start, updates, trace = state['next_episode'], state['optimizer_updates'], state['trace']
        schedule.setstate(state['schedule_rng'])
        torch.set_rng_state(state['torch_rng'])
        if config.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'last.pt'
    write_config = WriteConfig(rule=config.rule, eta=config.eta)

    def save(next_episode):
        payload = {
            'protocol': 'sillisalaatti_v1', 'config': model.config(),
            'model': model.state_dict(), 'seed': config.seed, 'train_config': asdict(config),
            'training_state': {
                'next_episode': next_episode, 'optimizer_updates': updates,
                'optimizer': optimizer.state_dict(), 'trace': trace,
                'schedule_rng': schedule.getstate(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if config.device == 'cuda' else [],
            },
            'environment': {'torch': str(torch.__version__), 'device': config.device},
        }
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    end = min(config.episodes, stop_after) if stop_after is not None else config.episodes
    for episode in range(start, end):
        horizon = schedule.choice(config.train_horizons)
        stream = make_stream(config.seed * 100000 + episode + 6000000,
                             config.batch_size, horizon, 'independent', config.device)
        state = None
        for begin in range(0, stream.inputs.shape[1], config.segment):
            if state is not None:
                state = state.detach()
            last = min(begin + config.segment, stream.inputs.shape[1])
            for t in range(begin, last):
                state, _ = update(model, state, stream.inputs[:, t], write_config, training=True)
            if last < 4:
                continue
            prediction = read_all(model, state)
            loss = (prediction - stream.targets[:, last - 1]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'non-finite training loss at episode {episode + 1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            updates += 1
        trace.append({'episode': episode + 1, 'horizon': horizon,
                      'loss': float(loss.detach()), 'optimizer_updates': updates})
        if (episode + 1) % config.checkpoint_every == 0 or episode + 1 == end:
            save(episode + 1)
        if episode == 0 or (episode + 1) % max(1, config.episodes // 4) == 0:
            print(json.dumps({'train': config.kind, 'rule': config.rule, 'seed': config.seed, **trace[-1]}), flush=True)
    if start >= end:
        save(start)
    return path


def calibrate_training(config: TrainConfig, output: Path, count: int = 128) -> dict:
    """Equal 64-episode development training for every candidate and dev seed."""
    rows = []
    for eta in ETA_GRID:
        scores, updates = [], 0
        for seed in (11, 23, 37):
            candidate = replace(config, seed=seed, eta=eta, episodes=64)
            path = train(candidate, output / f'eta{eta}-seed{seed}')
            model, _ = load_checkpoint(path, config.device)
            receipt = evaluate(model, WriteConfig(rule=config.rule, eta=eta),
                               make_stream(seed, count, 64, 'independent', config.device),
                               init_config=WriteConfig(rule=config.rule, eta=eta))
            scores.append(receipt['horizons']['64']['mse'])
            updates += torch.load(path, weights_only=True)['training_state']['optimizer_updates']
        rows.append({'eta': eta, 'mse': sum(scores) / 3, 'seed_mse': scores,
                     'optimizer_updates': updates})
    best = min(rows, key=lambda r: (r['mse'], r['eta']))
    result = {'eta': best['eta'], 'seeds': [11, 23, 37], 'episodes_per_candidate_seed': 64,
              'count': count, 'candidates': rows, 'selection': 'development training only'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('phase', 'signed', 'vector'), default='phase')
    p.add_argument('--rule', choices=RULES[:-1], default='rms_reference')
    p.add_argument('--eta', type=float, default=None)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--episodes', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--segment', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=.002)
    p.add_argument('--train-horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--output', type=Path, default=Path('runs/trained'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--eval-count', type=int, default=128)
    p.add_argument('--eval-horizon', type=int, default=64)
    args = p.parse_args(argv)
    if args.eval_count < 2 or args.eval_count % 2 or args.eval_horizon < 1:
        p.error('eval-count must be even and >=2; eval-horizon must be positive')
    eta = args.eta
    if eta is None and args.resume:
        eta = torch.load(args.resume, weights_only=True)['train_config']['eta']
    config = TrainConfig(kind=args.kind, rule=args.rule, eta=8/256 if eta is None else eta,
                         seed=args.seed, episodes=args.episodes, batch_size=args.batch_size,
                         segment=args.segment, learning_rate=args.learning_rate,
                         train_horizons=tuple(args.train_horizons), threads=args.threads, device=args.device)
    calibration = None
    if eta is None and args.rule in ('raw_calibrated', 'raw_backtracked'):
        calibration = calibrate_training(config, args.output / 'development')
        config = replace(config, eta=calibration['eta'])
    path = train(config, args.output, args.resume, args.stop_after)
    saved = torch.load(path, weights_only=True)
    model, metadata = load_checkpoint(path, args.device)
    result = {
        'protocol': 'SilliSalaatti retrained reader v1',
        'status': '600_episode_budget_complete' if saved['training_state']['next_episode'] == 600 and config.episodes == 600 else 'diagnostic_pilot_or_partial',
        'train_config': asdict(config), 'optimizer_updates': saved['training_state']['optimizer_updates'],
        'calibration': calibration or {'selection': 'manual eta or uncalibrated reference rule'},
        'runs': [],
    }
    for index, schedule in enumerate(SCHEDULES):
        stream = make_stream(2_000_000_000 + config.seed + index * 10_000,
                             args.eval_count, args.eval_horizon, schedule, args.device)
        receipt = evaluate(model, WriteConfig(rule=config.rule, eta=config.eta), stream,
                           init_config=WriteConfig(rule=config.rule, eta=config.eta))
        receipt['checkpoint'] = metadata
        result['runs'].append(receipt)
    write_receipts(args.output, result)
    print(f'Saved {path} and {args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
