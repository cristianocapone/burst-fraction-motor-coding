"""
MuJoCo Reacher wrapper for the 4-goal directional task.

obs (11): [cosθ1,cosθ2, sinθ1,sinθ2, tgt_x,tgt_y, θ1dot,θ2dot, ftip-tgt(x,y,z)]
fingertip pos:  fx = obs[4]+obs[8],  fy = obs[5]+obs[9]

The 4 goals are the 4 cardinal targets in :data:`reacher_burst.config.TARGETS`;
each 100-step block resets the arm to the folded pose (tip at the centre) and
moves the target, so a single 400-step episode visits all 4 goals in a random
order.
"""
import numpy as np

from . import config as C


def make_env(render_mode=None, episode_duration=C.EPISODE_DURATION):
    """Reacher-v4 with the TimeLimit overridden so 4×100 goal blocks fit."""
    # numpy ≥2.0 removed bool8; gym 0.26's passive_env_checker still uses it
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    import gym
    return gym.make(C.ENV_ID, render_mode=render_mode,
                    max_episode_steps=episode_duration + 5)


def fingertip_xy(obs):
    """Reacher fingertip position = target + (fingertip - target) vector."""
    return obs[4] + obs[8], obs[5] + obs[9]


def set_pose(env, tgt_xy):
    """Reset the arm to the fixed folded pose (tip at centre) and set target."""
    u = env.unwrapped
    qpos = u.data.qpos.copy()
    qpos[0:2] = C.ARM_Q0
    qpos[2:4] = tgt_xy
    qvel = np.zeros_like(u.data.qvel)
    u.set_state(qpos, qvel)


def get_obs(env):
    return env.unwrapped._get_obs()


def reward(obs, a_int, tgt):
    """-distance − control cost − joint-velocity penalty.

    The control and velocity terms suppress the sustained-torque spiral that a
    pure ``-dist`` reward otherwise encourages.
    """
    fx, fy = fingertip_xy(obs)
    dist_tgt = np.hypot(fx - tgt[0], fy - tgt[1])
    ctrl_cost = float(np.sum(a_int ** 2))            # |torque|²
    jvel_pen = float(obs[6] ** 2 + obs[7] ** 2)      # joint angular speed²
    return -dist_tgt - C.W_CTRL * ctrl_cost - C.W_JVEL * jvel_pen
