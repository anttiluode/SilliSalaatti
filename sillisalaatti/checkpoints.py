"""Strict, weights-only checkpoint loading with immutable file provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ._vendor.phasegraph.meta_ttt import MetaDeltaReader


def load_checkpoint(path: str | Path, device: str = 'cpu') -> tuple[MetaDeltaReader, dict]:
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('protocol') not in ('meta_ttt_v1', 'sillisalaatti_v1'):
        raise ValueError('Expected a meta_ttt_v1 or sillisalaatti_v1 checkpoint; global MLP-writer checkpoints are a different experiment.')
    if not isinstance(payload.get('config'), dict) or not isinstance(payload.get('model'), dict):
        raise ValueError('Checkpoint needs model and config dictionaries.')
    model = MetaDeltaReader(**payload['config']).to(device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    metadata = {
        'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'protocol': payload['protocol'], 'seed': payload.get('seed'),
        'config': payload['config'], 'state_bytes': model.payload_bytes,
        'parameter_count': model.param_count(),
        'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
        'buffer_bytes': sum(p.numel() * p.element_size() for p in model.buffers()),
    }
    return model, metadata
