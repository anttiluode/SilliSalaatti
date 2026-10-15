import json
from copy import deepcopy

from PIL import Image
import pytest
import torch

from sillisalaatti.benchmark import _answer_metrics, _errors, paired_difference, decision_bands, main
from sillisalaatti.plot import plot


def test_primary_repeat_verdict_uses_actual_repeat16_not_extra_sham():
    def receipt(rule, schedule):
        h = {'mse': 0.1, 'updated_mse': 0.1}
        horizons = {'1': deepcopy(h), '64': deepcopy(h)} if schedule == 'independent' else {
            '16': {'untouched_drift_mse': 1.0, 'sham_untouched_drift_mse': 1.0 if rule == 'rms_reference' else 0.0}}
        return {'checkpoint': {'file': 'test.pt'}, 'writer': {'rule': rule}, 'schedule': schedule, 'horizons': horizons}
    runs = [receipt(r, s) for r in ('rms_reference', 'raw_same_step') for s in ('repeat', 'independent')]
    result = decision_bands({'runs': runs})['checkpoint_verdicts'][0]
    assert result['verdict'] == 'FAIL_THIS_CHECKPOINT'
    runs[0]['horizons']['16']['untouched_drift_mse'] = 0.0
    assert decision_bands({'runs': runs})['checkpoint_verdicts'][0]['verdict'] == 'FLOOR_LIMITED'


def test_paired_arm_difference_preserves_correlated_uncertainty():
    result = paired_difference([0., 100., 200.], [1., 101., 201.])
    assert result['mean_difference'] == 1
    assert result['standard_error'] == 0
    clustered = paired_difference([0., 0., 0., 0.], [1., 1., 3., 3.], clustered=True)
    assert clustered['effective_histories'] == 2
    assert clustered['standard_error'] == 1
    with pytest.raises(ValueError):
        paired_difference([1., 2.], [1.])


def test_history_pair_duplicates_do_not_artificially_shrink_sem():
    truth = torch.zeros(2, 4, 3)
    predicted = torch.stack((torch.ones(4, 3), torch.ones(4, 3) * 2))
    independent = _errors(predicted, truth)
    duplicated = _errors(predicted.repeat_interleave(2, 0), truth.repeat_interleave(2, 0), clustered=True)
    assert duplicated['mse_standard_error'] == pytest.approx(independent['mse_standard_error'])
    assert duplicated['effective_histories'] == 2


def test_delayed_corrected_binding_is_visible_despite_aggregate_improvement():
    truth = torch.zeros(2, 4, 3)
    old = torch.ones_like(truth)
    prediction = torch.zeros_like(truth)
    prediction[:, 0] = 1.5
    keys = torch.ones(2, dtype=torch.long)
    metrics = _answer_metrics(prediction, truth, old, keys, 'correction')
    assert prediction.square().mean() < old.square().mean()
    assert metrics['corrected_key0_mse'] == 2.25
    assert metrics['preserved_key3_mse'] == 0
    assert metrics['per_stream_corrected_key0_mse'] == [2.25, 2.25]
    assert len(metrics['per_stream_updated_mse']) == 2
    assert len(metrics['per_stream_untouched_drift_mse']) == 2


def test_training_plot_metadata_identifies_retrained_evaluation(tmp_path):
    data = {'protocol': 'SilliSalaatti retrained reader v1', 'status': 'diagnostic_pilot', 'runs': [{
        'checkpoint': {'config': {'second_order': False}}, 'writer': {'rule': 'rms_reference'},
        'schedule': 'independent', 'horizons': {'1': {'mse': 0.2}},
    }]}
    source = tmp_path / 'summary.json'
    source.write_text(json.dumps(data))
    plot(source, tmp_path / 'plot.png')
    assert 'Retrained-reader' in Image.open(tmp_path / 'plot.png').info['Title']


def test_csv_accepts_schedule_specific_delayed_binding_metrics(tmp_path):
    checkpoint = __import__('pathlib').Path(__file__).resolve().parents[1] / 'checkpoints/meta/seed41-phase_first_order.pt'
    main(['--checkpoint', str(checkpoint), '--output', str(tmp_path), '--count', '2',
          '--horizon', '1', '--rules', 'no_write', '--schedules', 'independent', 'correction'])
    assert 'corrected_key0_mse' in (tmp_path / 'metrics.csv').read_text().splitlines()[0]
