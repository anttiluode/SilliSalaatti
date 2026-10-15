import hashlib
import json
from pathlib import Path

import pytest
import torch

from sillisalaatti.benchmark import main, evaluate
from sillisalaatti.tasks import make_stream
from sillisalaatti.checkpoints import load_checkpoint
from sillisalaatti.writers import WriteConfig
from sillisalaatti.plot import plot


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_checkpoints_match_their_provenance():
    folder = ROOT / 'checkpoints/meta'
    manifest = json.loads((folder / 'PROVENANCE.json').read_text())
    assert len(manifest['files']) == 6
    for record in manifest['files']:
        path = folder / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        model, meta = load_checkpoint(path)
        assert model.payload_bytes == 48
        assert meta['seed'] == record['seed']


def test_real_checkpoint_cli_writes_json_csv_and_plot(tmp_path):
    main(['--checkpoint', str(ROOT / 'checkpoints/meta/seed41-phase_first_order.pt'),
          '--output', str(tmp_path), '--count', '2', '--horizon', '1', '--eta', '0.0625',
          '--rules', 'rms_reference', 'raw_calibrated', '--schedules', 'correction'])
    data = json.loads((tmp_path / 'summary.json').read_text())
    assert data['status'] == 'diagnostic_pilot'
    assert len(data['runs']) == 2
    assert len((tmp_path / 'metrics.csv').read_text().splitlines()) == 3
    plot(tmp_path / 'summary.json', tmp_path / 'comparison.png')
    assert (tmp_path / 'comparison.png').stat().st_size > 1000


def test_explicit_initialization_is_labeled_and_changes_the_fork():
    model, _ = load_checkpoint(ROOT / 'checkpoints/meta/seed41-phase_first_order.pt')
    stream = make_stream(41, 2, 1, 'correction')
    raw = WriteConfig(rule='raw_calibrated', eta=1e-12)
    own = evaluate(model, raw, stream, init_config=raw)
    shared = evaluate(model, raw, stream)
    assert own['initialization'] == 'explicit writer four-write prefix'
    assert own['initial_state_sha256'] != shared['initial_state_sha256']


def test_bad_cli_arguments_exit_cleanly():
    with pytest.raises(SystemExit):
        main(['--count', '0'])
