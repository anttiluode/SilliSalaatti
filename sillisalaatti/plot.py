"""Plot actual answer error and repeat-write drift from a JSON receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(summary: str | Path, output: str | Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(Path(summary).read_text())
    runs = data['runs']
    families = sorted({bool(r['checkpoint']['config']['second_order']) for r in runs})
    rules = ['rms_reference', 'raw_same_step', 'raw_calibrated', 'raw_backtracked', 'no_write']
    labels = ['RMS', 'Raw same step', 'Raw calibrated', 'Raw accepted', 'No write']
    fig, axes = plt.subplots(len(families), 2, figsize=(11, 3.5 * len(families)), squeeze=False)
    for row, family in enumerate(families):
        chosen = [r for r in runs if bool(r['checkpoint']['config']['second_order']) == family]
        schedule = 'independent' if any(r['schedule'] == 'independent' for r in chosen) else chosen[0]['schedule']
        for column, (selected_schedule, metric, title) in enumerate([
            (schedule, 'mse', f'{schedule}: final full-memory MSE'),
            ('repeat', 'untouched_drift_mse', 'Repeat-16 collateral drift (log scale)'),
        ]):
            means, errors, ticks = [], [], []
            for rule, label in zip(rules, labels):
                values = []
                for run in chosen:
                    if run['writer']['rule'] != rule or run['schedule'] != selected_schedule:
                        continue
                    horizons = run['horizons']
                    h = '16' if column == 1 and '16' in horizons else max(horizons, key=int)
                    values.append(horizons[h][metric])
                if values:
                    means.append(float(np.mean(values)))
                    errors.append(float(np.std(values, ddof=1) / len(values)**.5) if len(values) > 1 else 0.)
                    ticks.append(label)
            ax = axes[row, column]
            if means:
                heights = np.maximum(means, 1e-12) if column else means
                ax.bar(range(len(means)), heights, yerr=errors if not column else None,
                       color=['#376b83', '#c4834d', '#4b8e70', '#8273a7', '#a2a2a2'][:len(means)], capsize=3)
                ax.set_xticks(range(len(means)), ticks, rotation=15, ha='right')
                if column:
                    ax.set_yscale('log')
                ax.grid(axis='y', alpha=.2)
                ax.set_axisbelow(True)
            else:
                ax.text(.5, .5, 'Schedule not evaluated', transform=ax.transAxes, ha='center')
            ax.set_title(('Higher-order' if family else 'First-order') + ' reader — ' + title, fontsize=10)
    prefix = 'Frozen-reader writer ablation' if 'frozen' in data.get('protocol', '').lower() else 'Retrained-reader evaluation'
    title = prefix + ': retained answers and collateral changes'
    fig.suptitle(title, fontsize=13)
    fig.text(.5, .01, 'Bars average available training seeds; MSE error bars show SEM across training seeds. Zero drift is plotted at 1e-12.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .96))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, metadata={'Title': title, 'Description': data.get('status', '')})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', type=Path)
    p.add_argument('--output', type=Path, default=Path('comparison.png'))
    args = p.parse_args()
    plot(args.summary, args.output)


if __name__ == '__main__':
    main()
