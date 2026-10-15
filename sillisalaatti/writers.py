"""Numerical writer ablations; no state or observation history is kept here."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ._vendor.phasegraph.quantization import quantize_phase
from ._vendor.phasegraph.meta_ttt import interval_ste

RULES = ('rms_reference', 'raw_same_step', 'raw_calibrated', 'raw_backtracked', 'no_write')
ETA_GRID = (1/256, 1/64, 1/16, 1/4, 1., 4.)


@dataclass(frozen=True)
class WriteConfig:
    rule: str = 'rms_reference'
    eta: float = 8/256
    tolerance: float = 0.002
    topk: int | None = 8

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f'Unknown writer rule: {self.rule}')
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError('eta must be finite and positive')
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError('tolerance must be finite and nonnegative')
        if self.topk is not None and (type(self.topk) is not int or self.topk < 1):
            raise ValueError('topk must be a positive integer or None')


def update(model, state: Tensor | None, observation: Tensor, config: WriteConfig,
           *, training: bool = False) -> tuple[Tensor, dict]:
    """Return the new explicit state and transient diagnostics.

    Inference requires/returns uint8. Training returns byte-rounded floats
    with a straight-through derivative, using first-order state gradients.
    Reported call counts are whole-batch calls, not per-case FLOP estimates.
    """
    model.reader._input(observation, model.slots + model.value_size, 'observation')
    if config.topk is not None and config.topk > model.payload_bytes:
        raise ValueError('topk exceeds state bytes')
    if state is None:
        state = model.initial_state(len(observation), packed=not training)
    if not training and state.dtype != torch.uint8:
        raise ValueError('inference state must be uint8')
    with torch.inference_mode(False), torch.enable_grad():
        obs = observation if training else observation.detach().clone()
        normal = model.reader._state(state, len(obs))
        work = normal if training and normal.requires_grad else normal.detach().clone().requires_grad_(True)
        q, target = obs[:, :model.slots], obs[:, model.slots:]
        predicted = model.read(work, q)
        before = (predicted - target).square().mean(-1)
        if not bool(torch.isfinite(before).all()):
            raise ValueError('non-finite current-key loss')
        active = before.detach() > config.tolerance**2
        reader_calls, gradient_calls = 1, 0
        stalled = torch.zeros_like(active)
        accepted = torch.zeros_like(active)
        proposal_magnitude = 0.0
        if config.rule == 'no_write':
            chosen = work
        else:
            gradient = torch.autograd.grad(before.sum(), work, create_graph=False, retain_graph=training)[0]
            gradient_calls = 1
            flat = gradient.flatten(1)
            if not bool(torch.isfinite(flat).all()):
                raise ValueError('non-finite state gradient')
            direction = flat
            if config.rule == 'rms_reference':
                rms = flat.square().mean(-1).sqrt().detach().clamp_min(1e-9)
                direction = flat / rms[:, None]
            if config.topk is not None:
                indices = flat.detach().abs().topk(config.topk, dim=-1).indices
                direction = direction * torch.zeros_like(flat).scatter_(1, indices, 1.)
            direction = direction.reshape_as(work)
            proposal_magnitude = float((config.eta * direction).detach().abs().mean())
            chosen = work
            pending = active.clone()
            factors = (1., .5, .25, .125) if config.rule == 'raw_backtracked' else (1.,)
            for index, factor in enumerate(factors):
                proposal = work - config.eta * factor * direction
                candidate = quantize_phase(proposal) if model.kind == 'phase' else interval_ste(proposal)
                unchanged = (candidate.detach() == work.detach()).flatten(1).all(1)
                if index == 0:
                    stalled = active & unchanged & (direction.detach().flatten(1).abs().sum(1) > 0)
                if config.rule == 'raw_backtracked':
                    with torch.no_grad():
                        error = (model.read(candidate.detach(), q) - target).square().mean(-1)
                    reader_calls += 1
                    take = pending & torch.isfinite(error) & (error < before.detach() - 1e-12)
                else:
                    take = pending
                chosen = torch.where(take[:, None, None], candidate, chosen)
                accepted |= take & ~unchanged
                # torch.where saves take for backward; it can alias pending.
                pending = pending & ~take
                if not bool(pending.any()):
                    break
        packed = model.reader.pack_state(chosen)
        old_packed = model.reader.pack_state(work.detach())
        with torch.no_grad():
            after = (model.read(packed, q) - target).square().mean(-1)
        reader_calls += 1
        differences = (packed.to(torch.int16) - old_packed.to(torch.int16)).abs()
        if model.kind == 'phase':
            differences = torch.minimum(differences, 256 - differences)
        stats = {
            'before_mse': float(before.detach().mean()), 'after_mse': float(after.mean()),
            'changed_byte_fraction': float((packed != old_packed).float().mean()),
            'mean_distance_bins': float(differences.float().mean()),
            'quantization_stall_fraction': float(stalled.float().mean()),
            'accepted_fraction': float(accepted.float().mean()),
            'rejected_fraction': float((active & ~accepted).float().mean()) if config.rule == 'raw_backtracked' else 0.,
            'already_correct_fraction': float((~active).float().mean()),
            'proposal_mean_abs': proposal_magnitude,
            'reader_calls': reader_calls, 'gradient_calls': gradient_calls,
        }
        return (chosen if training else packed), stats
