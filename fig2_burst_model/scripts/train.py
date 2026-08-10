#!/usr/bin/env python3
"""Train the burst-neuron Reacher agent.

Examples
--------
    python scripts/train.py --episodes 2000 --outdir runs/fresh
    python scripts/train.py --resume --ckpt checkpoints/burst_vprob_reacher_rep0.pickle
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reacher_burst import config as C          # noqa: E402
from reacher_burst.train import train          # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episodes', type=int, default=50000)
    p.add_argument('--outdir', default='runs')
    p.add_argument('--ckpt', default=None,
                   help='checkpoint to resume FROM (default: '
                        '<outdir>/reacher_checkpoint.pickle). '
                        'New checkpoints are always written into --outdir.')
    p.add_argument('--resume', action='store_true', help='load --ckpt and continue')
    p.add_argument('--start-episode', type=int, default=None,
                   help='episode offset for the noise/torque schedules '
                        '(default when resuming: inferred from the learning curve)')
    p.add_argument('--plot-every', type=int, default=100,
                   help='episodes between figure + checkpoint dumps (0 = never)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--lr', type=float, default=C.LR_ADAM)
    p.add_argument('--no-progress', action='store_true')
    args = p.parse_args()

    train(n_episodes=args.episodes, outdir=args.outdir,
          ckpt_path=args.ckpt, resume=args.resume, plot_every=args.plot_every,
          seed=args.seed, lr=args.lr, progress=not args.no_progress,
          start_episode=args.start_episode)


if __name__ == '__main__':
    main()
