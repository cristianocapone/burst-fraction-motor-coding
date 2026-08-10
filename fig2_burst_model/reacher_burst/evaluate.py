"""
Per-goal evaluation of a trained policy — the data behind Figure 2F.

For each of the 4 cardinal goals, run ``n_eval`` deterministic 100-step episodes
(one goal block: arm re-posed folded at the centre, target moved, no exploration
noise, torque amplitude at its final schedule value ``ACT_LIMIT_END``) and record
the fingertip trajectory and total reward.  The figure shows the best ``k_best``
trajectories per goal plus the mean over all episodes.
"""
import pickle
from pathlib import Path

import numpy as np

from . import config as C
from .rollout import run_episode


def evaluate(env, agent, n_eval=20, k_best=5, ep_steps=C.GOAL_BLOCK,
             noise=0.0, act_scale=C.ACT_LIMIT_END, num_g=C.NUM_G, verbose=True):
    """Run the evaluation; returns ``{goal: {'top5', 'avg', 'rewards', 'all'}}``.

    Deterministic by default (``noise=0``): the burst neurons remain stochastic,
    so episodes still differ, which is what the per-goal spread in Fig. 2F shows.
    """
    var_prev, scale_prev = agent.var_override, agent.act_scale_override
    agent.var_override = noise
    agent.act_scale_override = act_scale

    all_trajs = {k: [] for k in range(num_g)}
    all_rewards = {k: [] for k in range(num_g)}
    try:
        for k_act in range(num_g):
            for ep in range(n_eval):
                # a single-goal block: every "block" of the episode is goal k_act
                tr = run_episode(env, agent, episode_idx=0, learn=False,
                                 record=False, permutation=[k_act] * num_g,
                                 episode_duration=ep_steps)
                all_trajs[k_act].append(tr.positions)
                all_rewards[k_act].append(tr.r_tot)
                if verbose:
                    tgt = C.TARGETS[k_act]
                    fin = np.hypot(tr.positions[-1, 0] - tgt[0],
                                   tr.positions[-1, 1] - tgt[1])
                    print(f'  g{k_act} ep{ep:02d}  r={tr.r_tot:.2f}  final dist={fin:.4f}')
    finally:
        agent.var_override, agent.act_scale_override = var_prev, scale_prev

    eval_data = {}
    for k_act in range(num_g):
        best_idx = np.argsort(all_rewards[k_act])[-k_best:][::-1]
        stacked = [np.asarray(t) for t in all_trajs[k_act]]
        min_len = min(len(t) for t in stacked)
        eval_data[k_act] = {
            'top5':    [all_trajs[k_act][ep] for ep in best_idx],
            'avg':     np.mean([t[:min_len] for t in stacked], axis=0),
            'rewards': all_rewards[k_act],
            'all':     all_trajs[k_act],
        }
    return eval_data


def save(eval_data, path):
    """Write the pickle consumed by ``scripts/make_figure2.py`` (panel F)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(eval_data, fh)
    return path
