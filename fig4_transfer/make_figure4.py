#!/usr/bin/env python3
"""Build Figure 4E-H from the two benchmark runs.

The published panels combine two runs of `benchmark_transfer.py`, because the
G x Y condition reported in the paper is the *frozen-gate* one:

  results/known_frozen_gy   `--methods bilinear-rw --task-information known --g-lr 0`
                            the descriptor is mapped once through the Linear(2,8)
                            encoder and the gate is then frozen: no critic
                            evaluation, no replay, no gradient update
  results/known             `--methods sf-gpi sf-actor --task-information known`
                            the successor-feature baselines, same descriptor

The successor-feature rows are identical in both runs (they never read the gate);
this script takes the G x Y rows from the frozen run and the baseline rows from
the other, writes the merged table to results/figure4/, renders the four panels
with the same plotting code as a single run, and prints the episode-5 summary.

Usage
-----
    python make_figure4.py                       # from the shipped result CSVs
    python make_figure4.py --out results/figure4
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark_transfer as bt  # noqa: E402

GY_METHOD = 'bilinear-rw'
SF_METHODS = ('sf-gpi', 'sf-actor')
LABELS = {'bilinear-rw': 'G×Y', 'sf-gpi': 'SF-GPI', 'sf-actor': 'SF-actor'}


def _read(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def merge(gy_dir, sf_dir, out_dir):
    """Write the merged results CSV + a metadata file recording provenance."""
    gy_rows = [r for r in _read(Path(gy_dir) / 'adaptation_results.csv')
               if r['method'] == GY_METHOD]
    sf_rows = [r for r in _read(Path(sf_dir) / 'adaptation_results.csv')
               if r['method'] in SF_METHODS]
    if not gy_rows:
        sys.exit(f'no {GY_METHOD} rows in {gy_dir}')
    missing = set(SF_METHODS) - {r['method'] for r in sf_rows}
    if missing:
        sys.exit(f'missing {sorted(missing)} rows in {sf_dir}')

    gy_meta = json.loads((Path(gy_dir) / 'metadata.json').read_text())
    sf_meta = json.loads((Path(sf_dir) / 'metadata.json').read_text())
    if float(gy_meta.get('g_lr', 1.0)) != 0.0:
        sys.exit(f'{gy_dir} is not the frozen-gate run (g_lr={gy_meta.get("g_lr")}); '
                 'rerun it with --g-lr 0')
    for name, meta in (('G×Y', gy_meta), ('baselines', sf_meta)):
        if meta.get('task_information') != 'known':
            sys.exit(f'{name} run is not the known-task condition '
                     f'(task_information={meta.get("task_information")})')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = gy_rows + sf_rows
    with (out_dir / 'adaptation_results.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # plot_results reads these two keys to draw G×Y as a frozen (zero-shot)
    # reference line rather than an adaptation curve
    merged_meta = {
        'figure': 'Figure 4E-H',
        'task_information': 'known',
        'g_lr': 0.0,
        'gy_source': {'dir': str(gy_dir), 'metadata': gy_meta},
        'baseline_source': {'dir': str(sf_dir), 'metadata': sf_meta},
    }
    (out_dir / 'metadata.json').write_text(json.dumps(merged_meta, indent=2, default=str))
    return out_dir, rows


def summarise(rows):
    """Episode-5 mean ± SEM across seeds, after averaging directions per seed."""
    last = max(int(r['episode']) for r in rows)
    out = {}
    for method in (GY_METHOD, *SF_METHODS):
        sub = [r for r in rows if r['method'] == method and int(r['episode']) == last]
        seeds = sorted({r['seed'] for r in sub})

        def stat(key, scale=1.0):
            per_seed = [st.mean(float(r[key]) * scale for r in sub if r['seed'] == s)
                        for s in seeds]
            sem = st.stdev(per_seed) / len(per_seed) ** 0.5 if len(per_seed) > 1 else 0.0
            return st.mean(per_seed), sem

        out[method] = {
            'reward_per_step': stat('reward_per_step'),
            'gmac': stat('cumulative_macs', 1e-9),
            'model_seconds': stat('model_seconds'),
            'gradient_steps': stat('gradient_steps'),
        }
    return last, out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gy-dir', default=str(HERE / 'results' / 'known_frozen_gy'),
                   help='run supplying the frozen-gate G×Y rows')
    p.add_argument('--sf-dir', default=str(HERE / 'results' / 'known'),
                   help='run supplying the successor-feature baseline rows')
    p.add_argument('--out', default=str(HERE / 'results' / 'figure4'),
                   help='merged results + figure go here')
    args = p.parse_args()

    out_dir, rows = merge(args.gy_dir, args.sf_dir, args.out)
    bt.plot_results(argparse.Namespace(output_dir=str(out_dir)))

    last, s = summarise(rows)
    gy = s[GY_METHOD]
    print(f'\nepisode {last}, mean ± SEM over 5 seeds (directions averaged within seed)')
    print(f'{"":10s} {"reward/step":>16s} {"cumulative GMAC":>20s} {"model time (s)":>18s} {"grad steps":>12s}')
    for m in (GY_METHOD, *SF_METHODS):
        v = s[m]
        print(f'{LABELS[m]:10s} '
              f'{v["reward_per_step"][0]:9.3f} ± {v["reward_per_step"][1]:<4.3f} '
              f'{v["gmac"][0]:13.2f} ± {v["gmac"][1]:<5.2f} '
              f'{v["model_seconds"][0]:11.2f} ± {v["model_seconds"][1]:<4.2f} '
              f'{v["gradient_steps"][0]:8.0f}')
    for m in SF_METHODS:
        print(f'  {LABELS[m]} vs G×Y: {s[m]["gmac"][0] / gy["gmac"][0]:.1f}x cumulative compute, '
              f'{s[m]["model_seconds"][0] / gy["model_seconds"][0]:.1f}x model time')
    print(f'\nmerged table + panels → {out_dir}')


if __name__ == '__main__':
    main()
