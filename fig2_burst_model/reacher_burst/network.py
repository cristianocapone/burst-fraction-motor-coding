"""
© 2021 This work is licensed under a CC-BY-NC-SA license.
Authors: Cristiano Capone

Recurrent rate reservoir used as the shared "cortical" trunk of the agent.

Extracted from the repository-level ``goal.py``, reduced to the code path the
Reacher burst experiment actually exercises (``__init__`` / ``step_`` /
``reset``).  The random draws in ``__init__`` are kept in their original order
and shapes so that a given seed reproduces the original network exactly, and
:mod:`reacher_burst.checkpoint` aliases this module as top-level ``goal`` so
that pickles written by the original script still load.
"""
import numpy as np


def sigm(x, dv):
    if dv < 1 / 30:
        return x > 0
    y = x / dv
    return 1.5 * (1. / (1. + np.exp(-y * 3.)) - .5)


def gaussian(x, mu, sig):
    return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


class GOAL_GRADED:
    """Rate reservoir with a distribution of membrane time constants.

    The state ``S`` (N-dim) is the feature vector the goal-specific readouts
    ``Jout_sub[g]`` project from; the recurrent matrix is fixed (only readouts
    are trained by the burst policy gradient).
    """

    def __init__(self, par):
        # network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape']

        self.dt = par['dt']
        self.itau_m = self.dt / par['tau_m']
        self.tau_m = par['tau_m']
        # log-spaced membrane constants → multi-timescale memory
        self.tau_m = np.logspace(np.log(par['tau_m'] * .5),
                                 np.log(par['tau_m'] * 2.), self.N)
        self.num_g = par['num_g']

        self.itau_s = np.exp(-self.dt / par['tau_s'])
        self.itau_ro = np.exp(-self.dt / par['tau_ro'])

        self.dv = par['dv']

        # recurrent connectivity
        self.J = np.random.normal(0., par['sigma_rec'], size=(self.N, self.N))

        # input, teach and output matrices
        self.Jin = np.random.normal(0., par['sigma_input'], size=(self.N, self.I))
        self.Jteach = np.random.normal(0., par['sigma_teach'], size=(self.N, self.O))
        self.Jout = np.random.normal(0.0, par['sigma_output'], size=(self.O, self.N))
        self.Jout_g = np.random.normal(0.0, par['sigma_output'], size=(self.num_g, self.N))
        self.JoutV = np.random.normal(0.0, par['sigma_output'], size=(1, self.N))
        self.Jsigmaout = np.random.normal(0., par['sigma_output'], size=(self.O, self.N))

        self.dJfilt = np.zeros(np.shape(self.J))
        self.dJfilt_out = np.zeros(np.shape(self.Jout))
        self.dJfilt_out_g = np.zeros(np.shape(self.Jout_g))
        self.dJfilt_sigma_out = np.zeros(np.shape(self.Jsigmaout))

        self.dJoutV_filt = np.zeros(np.shape(self.JoutV))
        self.dJoutV_aggregate = np.zeros(np.shape(self.JoutV))

        self.dJout_aggregate = np.zeros(np.shape(self.Jout))
        self.dJout_g_aggregate = np.zeros(np.shape(self.Jout_g))
        self.dJout_sigma_aggregate = np.zeros(np.shape(self.Jsigmaout))
        self.dJ_aggregate = np.zeros(np.shape(self.J))
        self.dh_aggregate = np.zeros(self.N, )

        self.S_filt_tot = np.zeros(self.N, )

        self.value = 0
        self.r = 0
        self.r_old = 0

        np.fill_diagonal(self.J, 0.)          # no self-connections

        self.name = 'model'

        # reset kernel applied after a spike (only used in the spiking variant)
        self.s_inh = -par['s_inh']
        self.Jreset = np.diag(np.ones(self.N) * self.s_inh)

        h = par['h']
        assert type(h) in (np.ndarray, float, int)
        self.h = h if isinstance(h, np.ndarray) else np.ones(self.N) * h

        self.H = np.ones(self.N) * par['Vo']   # membrane potential
        self.Vo = par['Vo']

        self.S = np.zeros(self.N)
        self.S_hat = np.zeros(self.N)
        self.dH = np.zeros(self.N)

        self.state_out = np.zeros(self.N)
        self.state_out_p = np.zeros(self.N)

        self.par = par

    def _sigm(self, x, dv):
        return sigm(x, dv)

    def step_(self, inp, if_spk=False, if_tanh=False):
        itau_m = self.dt / self.tau_m
        exp_tau_s = np.exp(-self.dt / self.tau_m)

        if if_spk:
            self.S_hat = self.S_hat * self.itau_s + self.S * (1. - self.itau_s)
        else:
            self.S_hat = self.S_hat * exp_tau_s + self.S * (1. - exp_tau_s)

        self.dH = self.dH * np.exp(-itau_m) + np.tanh((1. - np.exp(-itau_m)) * self.S_hat)

        if if_spk:
            self.H = self.H * np.exp(-itau_m) + (1. - np.exp(-itau_m)) * (
                self.J @ self.S_hat + self.Jin @ inp + self.h) + self.Jreset @ self.S
            self.S = self._sigm(self.H, dv=self.dv) - 0.5 > 0.
        else:
            if if_tanh:
                self.H = self.H * np.exp(-itau_m) + (1. - np.exp(-itau_m)) * np.tanh(
                    self.J @ self.S + self.Jin @ inp)
            else:
                self.H = self.H * np.exp(-itau_m) + (1. - np.exp(-itau_m)) * (
                    self.J @ self.S + self.Jin @ inp + self.h)
            self.S = np.copy(self.H)

        self.y = self.Jout @ self.S
        self.sigma = np.sqrt(np.clip(self.Jsigmaout @ self.S, 0.05 ** 2, 0.5 ** 2))

    def reset(self, init=None):
        self.S = init if init else np.zeros(self.N)
        self.S_hat = self.S * self.itau_s if init else np.zeros(self.N)
        self.state_out *= 0
        self.state_out_p *= 0
        self.H *= 0.
        self.H += self.Vo
        self.dJfilt *= 0
        self.dJfilt_out *= 0
        self.dJfilt_out_g *= 0

        self.value = 0
        self.value_old = 0

        self.dJoutV_filt *= 0
        self.dJoutV_aggregate *= 0

        self.dJfilt_sigma_out *= 0
        self.dJout_aggregate *= 0
