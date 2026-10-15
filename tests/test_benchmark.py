import pytest
import torch

from sillisalaatti.tasks import make_stream
from sillisalaatti.benchmark import evaluate, exact_control, calibrate
from sillisalaatti.writers import WriteConfig
from sillisalaatti._vendor.phasegraph.meta_ttt import MetaDeltaReader


def test_stream_determinism_and_latest_value_truth():
    a = make_stream(41, 4, 16, 'independent')
    b = make_stream(41, 4, 16, 'independent')
    assert torch.equal(a.inputs, b.inputs)
    truth = torch.zeros(4, 4, 3)
    for step, obs in enumerate(a.inputs.unbind(1)):
        keys = obs[:, :4].argmax(1)
        truth[torch.arange(4), keys] = obs[:, 4:]
        assert torch.equal(truth, a.targets[:, step])


def test_repeats_and_paired_histories_preserve_roles():
    repeated = make_stream(7, 4, 16, 'repeat')
    assert torch.equal(repeated.inputs[:, 4:], repeated.inputs[:, :1].expand(-1, 16, -1))
    paired = make_stream(7, 4, 16, 'paired')
    assert torch.equal(paired.inputs[0, 4:], paired.inputs[1, 4:])
    assert not torch.equal(paired.targets[0, -1, 3], paired.targets[1, -1, 3])
    assert torch.equal(paired.targets[0, -1, 0], paired.targets[1, -1, 0])


def test_exact_delta_solves_every_schedule_and_reports_bytes():
    for schedule in ('independent', 'repeat', 'correction', 'paired'):
        stream = make_stream(41, 4, 16, schedule)
        result = exact_control(stream, 'delta')
        assert result['state_bytes'] == 48
        assert result['horizons']['16']['mse'] == 0
        indexed = exact_control(stream, 'indexed_uint8')
        assert indexed['state_bytes'] == 12
        assert indexed['horizons']['16']['mse'] < 1e-4


def test_no_write_is_stable_but_does_not_represent_correction():
    torch.manual_seed(41)
    model = MetaDeltaReader(second_order=False).eval()
    stream = make_stream(41, 4, 4, 'correction')
    reference = evaluate(model, WriteConfig(), stream)
    frozen = evaluate(model, WriteConfig(rule='no_write'), stream)
    assert frozen['initial_state_sha256'] == reference['initial_state_sha256']
    assert frozen['horizons']['4']['changed_byte_fraction'] == 0
    assert frozen['horizons']['4']['mse'] > 0
    assert len(frozen['horizons']['4']['per_stream_mse']) == 4


def test_calibration_uses_only_declared_development_seeds():
    torch.manual_seed(41)
    model = MetaDeltaReader(second_order=False).eval()
    result = calibrate(model, 'raw_calibrated', count=2, horizon=1, grid=(0.01, 0.1))
    assert result['seeds'] == [11, 23, 37]
    assert result['eta'] in (0.01, 0.1)
    assert len(result['candidates']) == 2


@pytest.mark.parametrize('args', [(1, 0, 2, 'repeat'), (1, 3, 2, 'paired'), (1, 2, 0, 'repeat'), (1, 2, 2, 'bad')])
def test_invalid_task_configuration_is_rejected(args):
    with pytest.raises(ValueError):
        make_stream(*args)
