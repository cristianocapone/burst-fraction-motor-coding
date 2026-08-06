"""
The agent: reservoir trunk + one burst sub-population per (joint, goal).

Only the goal-specific readouts ``Jout_sub[g] : (out_dim, N)`` are learned; the
recurrent reservoir is frozen.  Learning is the burst policy gradient of
:func:`reacher_burst.burst.learn_basal_hpg`, applied once per episode with Adam.
"""
import numpy as np

from . import config as C
from .burst import init_burst_state, learn_basal_hpg, run_burst_subpop
from .network import GOAL_GRADED
from .optimizer import Adam


class BurstAgent:
    def __init__(self, network, Jout_sub, adam_sub, mu_offset,
                 out_dim=C.OUT_DIM, num_g=C.NUM_G, n_b=C.N_B):
        self.network = network
        self.Jout_sub = Jout_sub
        self.adam_sub = adam_sub
        self.mu_offset = mu_offset

        self.out_dim = out_dim
        self.num_g = num_g
        self.n_b = n_b
        self.M = out_dim * num_g * n_b

        self.dJfilt_sub = [np.zeros((out_dim, network.N)) for _ in range(num_g)]
        self.dJagg_sub = [np.zeros((out_dim, network.N)) for _ in range(num_g)]

        self.delta_b = np.full((out_dim, num_g), C.DB_INIT)
        self.r_baseline = 0.0
        self.bstate = init_burst_state(self.M)

        # optional evaluation overrides of the annealing schedules (None = follow
        # the training schedule for the given episode index)
        self.var_override = None
        self.act_scale_override = None

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def new(cls, mu_offset, par=None, out_dim=C.OUT_DIM, num_g=C.NUM_G,
            n_b=C.N_B, lr=C.LR_ADAM):
        par = par if par is not None else C.build_par()
        network = GOAL_GRADED(par)
        network.gamma = C.GAMMA
        network.out_dim = out_dim
        network.num_g = num_g
        Jout_sub = [np.random.normal(0, par['sigma_output'], (out_dim, par['N']))
                    for _ in range(num_g)]
        adam_sub = [Adam(alpha=lr, drop=1.0, drop_time=10000) for _ in range(num_g)]
        return cls(network, Jout_sub, adam_sub, mu_offset, out_dim, num_g, n_b)

    # ── episode lifecycle ────────────────────────────────────────────────────
    def reset_episode(self):
        self.network.reset()
        for k in range(self.num_g):
            self.dJfilt_sub[k] *= 0
        self.bstate = init_burst_state(self.M)

    def step(self, obs_aug, g, k_act, episode_idx):
        """One environment step of the policy.

        Returns ``(action, a_int, mu, p_burst_pop, traces)`` where ``action`` is
        the physical torque (internal action × ramped amplitude) and ``traces``
        are the burst voltage/spike traces of the display joint.
        """
        self.network.step_(obs_aug)
        S = self.network.S

        y_subs = [self.Jout_sub[ki] @ S for ki in range(self.num_g)]
        p_burst_pop, Vs_tr, Vd_tr, spk_tr, brst_tr = run_burst_subpop(
            y_subs, g, self.bstate, self.delta_b, self.n_b)

        mu = C.P_BURST_SCALE * p_burst_pop[:, k_act] - self.mu_offset
        var = (C.expl_var(episode_idx) if self.var_override is None
               else self.var_override)                # annealed exploration noise
        a_int = np.clip(mu + np.random.normal(0., var, self.out_dim), -1., 1.)
        act_scale = (C.action_limit(episode_idx) if self.act_scale_override is None
                     else self.act_scale_override)
        action = act_scale * a_int                    # scaled physical torque

        self._last = (S, y_subs)
        return action, a_int, mu, p_burst_pop, (Vs_tr, Vd_tr, spk_tr, brst_tr)

    def learn_step(self, k_act, p_burst_pop, a_int, mu, r):
        """Baseline update + eligibility-trace accumulation for one step."""
        S, y_subs = self._last
        self.r_baseline += C.BASELINE_LR * (r - self.r_baseline)
        learn_basal_hpg(S, k_act, y_subs[k_act], p_burst_pop[:, k_act],
                        a_int, mu, r - self.r_baseline,
                        self.dJfilt_sub, self.dJagg_sub, self.network.gamma)

    def update(self):
        """Apply the accumulated episode gradient to every goal readout."""
        for k in range(self.num_g):
            if np.any(self.dJagg_sub[k] != 0):
                np.clip(self.dJagg_sub[k], -C.GRAD_CLIP, C.GRAD_CLIP,
                        out=self.dJagg_sub[k])
                self.Jout_sub[k] = self.adam_sub[k].step(self.Jout_sub[k],
                                                         self.dJagg_sub[k])
            self.dJagg_sub[k] *= 0
