#!/usr/bin/env python3
"""Evaluate a trained checkpoint per goal and write the Figure 2F data.

    python scripts/eval_policy.py                       # shipped checkpoint
    python scripts/eval_policy.py --n-eval 20 --seed 0
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reacher_burst import checkpoint as ckpt        # noqa: E402
from reacher_burst import config as C               # noqa: E402
from reacher_burst import evaluate as ev            # noqa: E402
from reacher_burst.burst import calibrate_offset    # noqa: E402
from reacher_burst.envs import make_env             # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default=str(ROOT / C.CKPT_FILE))
    p.add_argument('--out', default=str(ROOT / C.EVAL_FILE))
    p.add_argument('--n-eval', type=int, default=20, help='episodes per goal')
    p.add_argument('--k-best', type=int, default=5, help='trajectories highlighted per goal')
    p.add_argument('--steps', type=int, default=C.GOAL_BLOCK, help='steps per episode')
    p.add_argument('--noise', type=float, default=0.0, help='exploration noise (0 = deterministic mean)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    p_rest, mu_offset = calibrate_offset()
    print(f'calibrated rest p_burst(V)={p_rest:.4f}  MU_OFFSET={mu_offset:.4f}')

    agent = ckpt.load_agent(args.ckpt, mu_offset)
    env = make_env(episode_duration=args.steps)
    print(f'loaded {args.ckpt}\nrunning {args.n_eval} episodes per goal '
          f'({args.steps} steps, noise={args.noise}) …')

    eval_data = ev.evaluate(env, agent, n_eval=args.n_eval, k_best=args.k_best,
                            ep_steps=args.steps, noise=args.noise,
                            verbose=not args.quiet)
    env.close()

    for k in range(C.NUM_G):
        r = eval_data[k]['rewards']
        print(f'  g{k}: mean r={np.mean(r):7.2f}   best r={np.max(r):7.2f}')

    print(f'saved → {ev.save(eval_data, args.out)}')


if __name__ == '__main__':
    main()
