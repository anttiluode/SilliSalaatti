"""Causal, explicitly keyed streams. Truth is for the evaluator, never the writer."""
from dataclasses import dataclass

import torch
from torch import Tensor

SCHEDULES = ('independent', 'repeat', 'correction', 'paired')


@dataclass(frozen=True)
class Stream:
    inputs: Tensor
    targets: Tensor
    schedule: str
    seed: int
    slots: int = 4
    value_size: int = 3

    @property
    def horizon(self):
        return self.inputs.shape[1] - self.slots


def make_stream(seed: int, count: int, horizon: int, schedule: str,
                device: str = 'cpu') -> Stream:
    if any(type(x) is not int or x < 1 for x in (count, horizon)) or schedule not in SCHEDULES:
        raise ValueError('count/horizon must be positive integers and schedule must be known')
    if schedule == 'paired':
        if count % 2:
            raise ValueError('paired streams require an even count')
        base = make_stream(seed, count // 2, horizon, 'correction')
        inputs = base.inputs.repeat_interleave(2, dim=0)
        # Keep role/key 3 explicit and change only its earlier binding.
        inputs[1::2, 3, 4:] *= -1
    else:
        g = torch.Generator().manual_seed(seed)
        inputs = torch.zeros(count, 4 + horizon, 7)
        initial = torch.rand(count, 4, 3, generator=g) * 2 - 1
        for key in range(4):
            inputs[:, key, key] = 1
            inputs[:, key, 4:] = initial[:, key]
        for step in range(horizon):
            if schedule == 'repeat':
                keys = torch.zeros(count, dtype=torch.long)
                values = initial[:, 0]
            elif schedule == 'correction':
                keys = torch.zeros(count, dtype=torch.long) if step == 0 else torch.full((count,), 1 + step % 2)
                values = -initial[:, 0] if step == 0 else torch.rand(count, 3, generator=g) * 2 - 1
            else:
                keys = torch.randint(4, (count,), generator=g)
                values = torch.rand(count, 3, generator=g) * 2 - 1
            inputs[torch.arange(count), step + 4, keys] = 1
            inputs[:, step + 4, 4:] = values
    truth = torch.zeros(count, 4, 3)
    targets = []
    for obs in inputs.unbind(1):
        truth[torch.arange(count), obs[:, :4].argmax(-1)] = obs[:, 4:]
        targets.append(truth.clone())
    return Stream(inputs.to(device), torch.stack(targets, 1).to(device), schedule, seed)
