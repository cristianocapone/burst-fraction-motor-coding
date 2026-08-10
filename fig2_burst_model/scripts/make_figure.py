#!/usr/bin/env python3
"""Reproduce the burst-neuron Reacher figure from a trained checkpoint.

Loads the checkpoint, runs one 400-step evaluation episode (4 goal blocks, no
learning) and renders the 6-panel summary figure.

Examples
--------
    python scripts/make_figure.py                       # bundled checkpoint
    python scripts/make_figure.py --seed 0 --var 0.05   # quieter policy
    python scripts/make_figure.py --video out/ep.mp4    # + side-by-side movie
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
sys.path.insert(0, str(ROOT))

from reacher_burst import checkpoint as ckpt        # noqa: E402
from reacher_burst import config as C               # noqa: E402
from reacher_burst.burst import calibrate_offset    # noqa: E402
from reacher_burst.envs import make_env             # noqa: E402
from reacher_burst.figures import make_figure       # noqa: E402
from reacher_burst.rollout import run_episode       # noqa: E402
from reacher_burst.train import CURVE_STRIDE        # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default=str(DATA / C.CKPT_FILE))
    p.add_argument('--traj', default=str(DATA / C.CURVE_FILE),
                   help='learning-curve .npy for the top-left panel')
    p.add_argument('--out', default=C.TRAIN_FIG)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--episode-index', type=int, default=None,
                   help='episode number driving the noise/torque schedules '
                        '(default: CURVE_STRIDE × length of the learning curve)')
    p.add_argument('--var', type=float, default=None,
                   help='override exploration noise (e.g. 0.0 = deterministic)')
    p.add_argument('--act-scale', type=float, default=None,
                   help='override torque amplitude (default: schedule value)')
    p.add_argument('--permutation', type=int, nargs=4, default=None,
                   help='fixed goal order for the 4 blocks, e.g. --permutation 0 1 2 3')
    p.add_argument('--video', default=None, help='also write an mp4 here')
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    p_rest, mu_offset = calibrate_offset()
    print(f'calibrated rest p_burst(V)={p_rest:.4f}  MU_OFFSET={mu_offset:.4f}')

    r_avg = ckpt.load_learning_curve(args.traj)
    episode_idx = (args.episode_index if args.episode_index is not None
                   else (CURVE_STRIDE * len(r_avg) if r_avg is not None else 0))

    agent = ckpt.load_agent(args.ckpt, mu_offset)
    agent.var_override = args.var
    agent.act_scale_override = args.act_scale
    print(f'loaded {args.ckpt}  |  episode index {episode_idx}  '
          f'(noise={agent.var_override if args.var is not None else C.expl_var(episode_idx):.3f}, '
          f'torque={agent.act_scale_override if args.act_scale is not None else C.action_limit(episode_idx):.4f})')

    env = make_env(render_mode='rgb_array' if args.video else None)
    recorder = None
    if args.video:
        from reacher_burst.render import VideoRecorder
        recorder = VideoRecorder(env, mu_offset)

    trace = run_episode(env, agent, episode_idx, learn=False, record=True,
                        permutation=args.permutation, step_cb=recorder)
    env.close()

    print(f'episode reward {trace.r_tot:.2f}  |  '
          f'mean fingertip-target distance {trace.dist.mean():.4f}  |  '
          f'final-100-steps distance {trace.dist[-100:].mean():.4f}')

    make_figure(trace, r_avg, agent.delta_b, mu_offset, episode_idx,
                out_path=args.out)
    print(f'figure → {args.out}')

    if recorder is not None:
        path = recorder.close(args.video)
        print(f'video  → {path}')


if __name__ == '__main__':
    main()
