# Retraining plumbing pilot

Six 16-episode CPU runs, seed 41: phase/signed/vector × RMS-reference/raw-same-step. Every model used batch 8, horizons 1/4/16/64, segment 8 and 76 optimizer updates. Each was evaluated on four schedules, 16 streams each and horizon 64, with its own writer used for initialization.

These runs verify finite training, quantized state, controls and the reporting path. They do not establish a phase advantage or the 600-episode trained comparison. Checkpoint/resume equivalence is separately tested in pytest.

[Compact metrics](summary.json) · [CSV](metrics.csv) · [Complete trace receipt](full_receipt.json.gz). Pilot weights are reproducible from the trainer and are not bundled. The six original upstream frozen checkpoints are bundled separately.

The compact and full receipts include all six training configurations, checkpoint hashes, optimizer-update counts and episode loss traces. To reproduce an individual pilot, use `python -m sillisalaatti.train --kind phase --rule rms_reference --seed 41 --episodes 16 --eval-count 16 --eval-horizon 64 --output runs/pilot`, then repeat for signed/vector and raw_same_step.
