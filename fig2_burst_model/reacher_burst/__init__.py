"""
reacher_burst — 4-goal MuJoCo Reacher controlled by bursting two-compartment
neurons, trained with a burst-based policy gradient.

Reference implementation for Figure 2 of Capone, Falorsi, Ciardiello, Manneschi
& Gigante, "Burst-selective motor coding inspires multiplicative gating for
efficient reinforcement learning".  Public entry points::

    from reacher_burst import BurstAgent, run_episode, make_env, make_figure
    from reacher_burst.train import train
    from reacher_burst.evaluate import evaluate
"""
from . import config
from . import evaluate
from .agent import BurstAgent
from .burst import (calibrate_offset, init_burst_state, learn_basal_hpg,
                    run_burst_subpop)
from .envs import fingertip_xy, make_env, set_pose
from .figures import make_figure
from .network import GOAL_GRADED
from .optimizer import Adam
from .rollout import EpisodeTrace, run_episode

__all__ = [
    'config', 'evaluate', 'BurstAgent', 'GOAL_GRADED', 'Adam',
    'run_burst_subpop', 'init_burst_state', 'learn_basal_hpg', 'calibrate_offset',
    'make_env', 'set_pose', 'fingertip_xy',
    'run_episode', 'EpisodeTrace', 'make_figure',
]
__version__ = '0.1.0'
