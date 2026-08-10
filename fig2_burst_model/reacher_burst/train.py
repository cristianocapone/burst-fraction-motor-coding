"""
Training loop: burst policy gradient on the 4-goal Reacher.

One episode = 4 goal blocks; the eligibility-trace gradient is accumulated over
the whole episode and applied once at the end (Adam, per-goal readout).  Every
``plot_every`` episodes the summary figure and a checkpoint are written.
"""
from pathlib import Path

import numpy as np

from . import config as C
from . import checkpoint as ckpt
from .agent import BurstAgent
from .burst import calibrate_offset
from .envs import make_env
from .figures import make_figure
from .rollout import run_episode


# episodes per point of the saved learning curve (the cadence the shipped
# checkpoint was written with); used to infer where a resumed run restarts
CURVE_STRIDE = 100


def _progress(n, use_tqdm=True):
    if use_tqdm:
        try:
            from tqdm import trange
            return trange(n)
        except ImportError:
            pass
    return range(n)


def train(n_episodes=50000, outdir='.', ckpt_path=None,
          resume=False, plot_every=100, seed=None, lr=C.LR_ADAM,
          episode_duration=C.EPISODE_DURATION, progress=True, mu_offset=None,
          start_episode=None):
    """Train (or resume) a run; returns ``(agent, R_AVG)``.

    All outputs land in ``outdir`` under the standard names, so separate runs
    (seeds/repeats) need separate ``outdir``s.

    ``start_episode`` offsets the exploration-noise / torque-amplitude
    schedules.  When resuming it defaults to ``CURVE_STRIDE × len(learning
    curve)`` so a continued run does not fall back to the episode-0 (tiny
    torque, huge noise) regime.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # checkpoints are always *written* inside outdir, never back onto the
    # checkpoint a run was resumed from
    ckpt_out = outdir / C.CKPT_FILE
    ckpt_in = Path(ckpt_path) if ckpt_path is not None else ckpt_out
    traj_path = outdir / C.CURVE_FILE
    fig_path = outdir / C.TRAIN_FIG

    if seed is not None:
        np.random.seed(seed)

    if mu_offset is None:
        p_rest, mu_offset = calibrate_offset()
        print(f'calibrated rest p_burst(V)={p_rest:.4f}  MU_OFFSET={mu_offset:.4f}')

    env = make_env(episode_duration=episode_duration)
    print(f'obs: {env.observation_space.shape}  act: {env.action_space.shape}')

    R_AVG = []
    if resume:
        agent = ckpt.load_agent(ckpt_in, mu_offset)
        # learning curve sits next to the checkpoint it belongs to
        prev = ckpt.load_learning_curve(ckpt_in.parent / C.CURVE_FILE)
        if prev is not None:
            R_AVG = list(prev)
        if start_episode is None:
            # the curve holds one point per CURVE_STRIDE episodes of the run
            # that produced it (not per --plot-every of *this* run)
            start_episode = CURVE_STRIDE * len(R_AVG)
        print(f'resumed from {ckpt_in}  (schedules start at episode {start_episode})')
    else:
        agent = BurstAgent.new(mu_offset, lr=lr)
    start_episode = start_episode or 0

    REWARDS = []

    for it in _progress(n_episodes, progress):
        trace = run_episode(env, agent, start_episode + it, learn=True, record=True,
                            episode_duration=episode_duration)
        REWARDS.append(trace.r_tot)

        if it > 0:
            agent.update()

        if plot_every and (it % plot_every == 0) and (it > 0):
            R_AVG.append(float(np.mean(REWARDS[-plot_every:])))
            make_figure(trace, R_AVG, agent.delta_b, agent.mu_offset,
                        start_episode + it, out_path=fig_path, rewards=REWARDS)
            ckpt.save(ckpt_out, agent)
            np.save(traj_path, np.array(R_AVG))

    np.save(traj_path, np.array(R_AVG))
    if REWARDS:
        print(f'  done, final avg reward (last 100 ep): {np.mean(REWARDS[-100:]):.2f}')
    env.close()
    return agent, R_AVG
