"""
One episode of the 4-goal Reacher task, shared by training and figure making.

An episode is ``EPISODE_DURATION`` env steps split into 4 blocks of
``GOAL_BLOCK`` steps.  Each block picks one goal (random order per episode),
re-poses the arm at the centre, moves the target, and clears the joint-history
buffer.  Everything needed by the figure is recorded in :class:`EpisodeTrace`.
"""
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import envs


@dataclass
class EpisodeTrace:
    r_tot: float = 0.0
    ep_len: int = 0
    permutation: np.ndarray = None
    positions: np.ndarray = None          # (steps, 2) fingertip xy
    bf_log: np.ndarray = None             # (steps, num_g) p_burst of display joint
    mu_log: np.ndarray = None             # (steps, out_dim)
    Vs: np.ndarray = None                 # (steps*N_BURST_STEPS, num_g, N_B)
    Vd: np.ndarray = None
    spk: np.ndarray = None
    brst: np.ndarray = None
    goal_of_step: np.ndarray = None       # (steps,) active goal index
    dist: np.ndarray = None               # (steps,) fingertip-target distance
    extras: dict = field(default_factory=dict)


def run_episode(env, agent, episode_idx, learn=True, record=True,
                permutation=None, step_cb=None,
                episode_duration=C.EPISODE_DURATION):
    """Run one episode; returns an :class:`EpisodeTrace`.

    Parameters
    ----------
    episode_idx : int
        Drives the exploration-noise and torque-amplitude schedules.
    learn : bool
        Accumulate the policy gradient (the caller decides when to
        :meth:`~reacher_burst.agent.BurstAgent.update`).
    step_cb : callable or None
        ``step_cb(k, k_act, mu, p_burst_pop, traces)`` — used by the video renderer.
    """
    num_g, out_dim, n_b = agent.num_g, agent.out_dim, agent.n_b
    nbs = C.N_BURST_STEPS

    env.reset()
    obs, _, _, _, _ = env.step(np.zeros(out_dim))
    agent.reset_episode()

    if permutation is None:
        permutation = np.random.permutation(num_g)

    tr = EpisodeTrace(permutation=np.asarray(permutation))
    positions, bf_log, mu_log, goals, dists = [], [], [], [], []
    if record:
        tr.Vs = np.zeros((episode_duration * nbs, num_g, n_b))
        tr.Vd = np.zeros((episode_duration * nbs, num_g, n_b))
        tr.spk = np.zeros((episode_duration * nbs, num_g, n_b), dtype=bool)
        tr.brst = np.zeros((episode_duration * nbs, num_g, n_b), dtype=bool)

    jhist = np.zeros((C.HIST_LEN, C.HIST_DIM))   # joint-position history buffer
    r_tot = 0.0

    for k in range(episode_duration):
        k_act = int(permutation[k // C.GOAL_BLOCK])
        g = np.zeros(num_g)
        g[k_act] = 1.

        if k % C.GOAL_BLOCK == 0:        # new goal block → re-pose arm, move target
            envs.set_pose(env, C.TARGETS[k_act])
            obs = envs.get_obs(env)
            jhist[:] = 0.0               # clear history on context reset

        target_vector = np.zeros(num_g)
        target_vector[k_act] = C.TARGET_INTENSITY
        obs_aug = np.concatenate((obs, target_vector, jhist.ravel()))

        action, a_int, mu, p_burst_pop, traces = agent.step(
            obs_aug, g, k_act, episode_idx)

        if record:
            sl = slice(k * nbs, (k + 1) * nbs)
            tr.Vs[sl], tr.Vd[sl], tr.spk[sl], tr.brst[sl] = traces
        bf_log.append(p_burst_pop[C.BF_DISP_JOINT].copy())
        mu_log.append(mu.copy())
        goals.append(k_act)
        if step_cb is not None:
            step_cb(k, k_act, mu, p_burst_pop, traces)

        obs, _, done, _, _ = env.step(action)
        jhist = np.roll(jhist, -1, axis=0)
        jhist[-1] = obs[0:4]             # push joint cos/sin

        fx, fy = envs.fingertip_xy(obs)
        positions.append([fx, fy])
        tgt = C.TARGETS[k_act]
        dists.append(float(np.hypot(fx - tgt[0], fy - tgt[1])))

        r = envs.reward(obs, a_int, tgt)
        r_tot += r
        if learn:
            agent.learn_step(k_act, p_burst_pop, a_int, mu, r)

        tr.ep_len = k + 1
        if done:                         # run the full episode → cycle all 4 goals
            break

    tr.r_tot = r_tot
    tr.positions = np.array(positions)
    tr.bf_log = np.array(bf_log)
    tr.mu_log = np.array(mu_log)
    tr.goal_of_step = np.array(goals)
    tr.dist = np.array(dists)
    return tr
