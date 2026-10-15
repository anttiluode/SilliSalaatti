from dataclasses import replace
import torch
import pytest

from sillisalaatti.train import TrainConfig, train
from sillisalaatti.checkpoints import load_checkpoint


def tiny():
    return TrainConfig(episodes=4, batch_size=2, segment=8, width=8,
                       train_horizons=(1, 4), threads=1)


def test_training_checkpoint_loads_and_records_finite_updates(tmp_path):
    path = train(tiny(), tmp_path / 'run')
    model, metadata = load_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    assert metadata['protocol'] == 'sillisalaatti_v1'
    assert model.payload_bytes == 48
    assert checkpoint['training_state']['next_episode'] == 4
    assert checkpoint['training_state']['optimizer_updates'] >= 4
    assert all(torch.isfinite(torch.tensor(row['loss'])) for row in checkpoint['training_state']['trace'])


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    config = tiny()
    full = train(config, tmp_path / 'full')
    partial = train(config, tmp_path / 'resumed', stop_after=2)
    resumed = train(config, tmp_path / 'resumed', resume=partial)
    a = torch.load(full, weights_only=True)
    b = torch.load(resumed, weights_only=True)
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key]), key
    assert a['training_state']['trace'] == b['training_state']['trace']
    assert a['training_state']['optimizer_updates'] == b['training_state']['optimizer_updates']


def test_resume_rejects_changed_training_configuration(tmp_path):
    config = tiny()
    checkpoint = train(config, tmp_path / 'run', stop_after=1)
    with pytest.raises(ValueError, match='configuration'):
        train(replace(config, eta=0.5), tmp_path / 'run', resume=checkpoint)


@pytest.mark.parametrize('kwargs', [{'episodes': 0}, {'batch_size': 0}, {'learning_rate': float('nan')}, {'kind': 'bad'}, {'rule': 'no_write'}])
def test_invalid_training_configuration(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
