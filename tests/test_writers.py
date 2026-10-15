from pathlib import Path

import pytest
import torch

from sillisalaatti.checkpoints import load_checkpoint
from sillisalaatti.writers import WriteConfig, update
from sillisalaatti._vendor.phasegraph.meta_ttt import MetaDeltaReader


def model():
    torch.manual_seed(41)
    return MetaDeltaReader(second_order=False).eval()


def observation(m, batch=2):
    return torch.cat((torch.eye(m.slots)[:batch], torch.full((batch, m.value_size), 0.3)), 1)


def test_reference_matches_upstream_byte_for_byte():
    m = model()
    x = observation(m)
    state = m.initial_state(2)
    for _ in range(3):
        expected = m.write(x, state)
        actual, stats = update(m, state, x, WriteConfig())
        assert torch.equal(actual, expected)
        assert actual.dtype == torch.uint8
        assert actual.numel() // len(x) == 48
        assert stats['gradient_calls'] == 1
        state = actual


def test_already_correct_observation_is_exact_noop():
    m = model()
    state = m.initial_state(2)
    x = observation(m)
    with torch.no_grad():
        x[:, m.slots:] = m.read(state, x[:, :m.slots])
    for rule in ('rms_reference', 'raw_same_step', 'raw_calibrated', 'raw_backtracked'):
        after, stats = update(m, state, x, WriteConfig(rule=rule))
        assert torch.equal(after, state)
        assert stats['changed_byte_fraction'] == 0


def test_raw_tiny_step_is_reported_as_rounding_stall():
    m = model()
    state = m.initial_state(2)
    after, stats = update(m, state, observation(m), WriteConfig(rule='raw_calibrated', eta=1e-12))
    assert torch.equal(after, state)
    assert stats['quantization_stall_fraction'] == 1
    assert stats['after_mse'] > 0.002**2


def test_backtracking_never_increases_quantized_current_key_loss():
    m = model()
    state = m.initial_state(2)
    x = observation(m)
    after, stats = update(m, state, x, WriteConfig(rule='raw_backtracked', eta=4))
    with torch.no_grad():
        before = (m.read(state, x[:, :4]) - x[:, 4:]).square().mean(1)
        final = (m.read(after, x[:, :4]) - x[:, 4:]).square().mean(1)
    assert torch.all(final <= before + 1e-12)
    assert stats['reader_calls'] >= 2


def test_mixed_batch_keeps_already_correct_case_unchanged():
    m = model()
    state = m.initial_state(2)
    x = observation(m)
    with torch.no_grad():
        x[0, 4:] = m.read(state[:1], x[:1, :4])[0]
    after, _ = update(m, state, x, WriteConfig(rule='raw_backtracked', eta=0.25))
    assert torch.equal(after[0], state[0])


@pytest.mark.parametrize('kwargs', [{'eta': 0}, {'eta': float('nan')}, {'topk': 0}, {'tolerance': -1}, {'rule': 'typo'}])
def test_invalid_writer_configuration(kwargs):
    with pytest.raises(ValueError):
        WriteConfig(**kwargs)


def test_rejects_bad_observation_without_changing_input_state():
    m = model()
    state = m.initial_state(2)
    before = state.clone()
    x = observation(m)
    x[0, -1] = float('nan')
    with pytest.raises(ValueError):
        update(m, state, x, WriteConfig())
    assert torch.equal(state, before)


def test_checkpoint_is_safe_strict_and_records_hash(tmp_path):
    m = model()
    p = tmp_path / 'meta.pt'
    torch.save({'config': m.config(), 'model': m.state_dict(), 'protocol': 'meta_ttt_v1', 'seed': 41}, p)
    restored, metadata = load_checkpoint(p)
    assert len(metadata['sha256']) == 64
    assert metadata['seed'] == 41
    assert restored.payload_bytes == 48
    for key, value in m.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    torch.save({'config': {}, 'model': {}, 'protocol': 'global_writer'}, p)
    with pytest.raises(ValueError, match='meta_ttt_v1'):
        load_checkpoint(p)
