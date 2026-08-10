"""
Two-compartment burst neuron with a V-based probabilistic product gate.

One sub-population of ``N_B`` neurons exists per (joint, goal) pair, i.e.
``M = out_dim * num_g * N_B`` neurons in total.

  * basal (somatic) compartment  V_s  ← readout of the reservoir, y_g = Jout_g·S
                                        plus a homeostatic bias δb
  * apical (dendritic) compartment V_d ← the goal one-hot (context signal)

A *burst* is emitted with probability  p_s(V_s) · p_d(V_d): it requires the
soma to be depolarised **and** the apical dendrite to be driven by the active
goal.  That product is the gate that makes one sub-population per goal control
the action while the others stay silent.  The burst probability averaged over
the ``N_BURST_STEPS`` sub-steps of an environment step is the policy mean:

    mu = P_BURST_SCALE * p_burst - MU_OFFSET

so the action is read out of a *rate of bursts*, not of a continuous unit.
"""
import numpy as np

from .config import (
    DT_B, TAU_S_B, TAU_D_B, V_THRESH_B, V_RESET_B, V_RESET_BURST_B, V_BURST_B,
    BETA_S_B, BETA_D_B, T_REF_B, T_REF_BURST_B, T_BURST_REF_D, N_BURST_STEPS,
    I_BAS_OFFSET, I_BAS_SCALE, I_API_SCALE, P_BURST_SCALE, DB_INIT,
    BF_DISP_JOINT, NUM_G, OUT_DIM, N_B, _sigm_prob,
)

_DT_TAUS = DT_B / TAU_S_B
_DT_TAUD = DT_B / TAU_D_B


def init_burst_state(M):
    """Fresh (de-synchronised) burst-neuron state for ``M`` neurons."""
    return dict(
        Vs=np.random.uniform(0, V_THRESH_B * 0.95, M),
        Vd=np.zeros(M),
        ref=np.zeros(M),
        burst_t=np.zeros(M),
        burst_ref=np.zeros(M),
    )


def run_burst_subpop(y_subs, g, bstate, delta_b, N_B):
    """Advance every burst neuron by ``N_BURST_STEPS`` sub-steps (1 env step).

    Parameters
    ----------
    y_subs : list of ``num_g`` arrays, shape (out_dim,)
        Goal-specific readouts of the reservoir state.
    g : array, shape (num_g,)
        Goal one-hot; drives the apical compartment.
    bstate : dict
        Persistent burst state, mutated in place.
    delta_b : array, shape (out_dim, num_g)
        Homeostatic basal bias.

    Returns
    -------
    p_burst_pop : (out_dim, num_g)   window-averaged burst probability
    Vs_tr, Vd_tr : (N_BURST_STEPS, num_g, N_B)  voltage traces (display joint)
    spk_tr, brst_tr : same shape, bool  spike / burst-onset rasters
    """
    num_g_ = len(y_subs)
    out_dim_ = y_subs[0].size
    M = out_dim_ * num_g_ * N_B

    # ── drives ───────────────────────────────────────────────────────────────
    y_mat = np.stack(y_subs, axis=1)                       # (out_dim, num_g)
    I_bas_mat = I_BAS_OFFSET + delta_b + I_BAS_SCALE * np.tanh(y_mat)
    I_bas_flat = np.repeat(I_bas_mat.ravel(), N_B)
    I_api_mat = np.tile(g * I_API_SCALE, (out_dim_, 1))
    I_api_flat = np.repeat(I_api_mat.ravel(), N_B)

    Vs, Vd = bstate['Vs'], bstate['Vd']
    ref = bstate['ref']
    burst_t = bstate['burst_t']
    burst_ref = bstate['burst_ref']

    Vs_tr = np.zeros((N_BURST_STEPS, num_g_, N_B))
    Vd_tr = np.zeros((N_BURST_STEPS, num_g_, N_B))
    spk_tr = np.zeros((N_BURST_STEPS, num_g_, N_B), dtype=bool)
    brst_tr = np.zeros((N_BURST_STEPS, num_g_, N_B), dtype=bool)

    p_burst_accum = np.zeros((out_dim_, num_g_))

    # ── burst onset: product gate p_s(V_s) · p_d(V_d), once per env step ──────
    p_s_now = _sigm_prob(Vs - V_THRESH_B, BETA_S_B)
    p_d_now = _sigm_prob(Vd - V_BURST_B, BETA_D_B)
    burst_ref = np.maximum(burst_ref - N_BURST_STEPS * DT_B, 0.0)
    can_burst = burst_ref <= 0
    new_burst = can_burst & (np.random.rand(M) < p_s_now * p_d_now)
    burst_t[new_burst] = N_BURST_STEPS * DT_B
    burst_ref[new_burst] = T_BURST_REF_D
    Vd[new_burst] = 0.0
    brst_3d = new_burst.reshape(out_dim_, num_g_, N_B)
    brst_tr[0] = brst_3d[BF_DISP_JOINT]

    # ── sub-step integration ─────────────────────────────────────────────────
    for step in range(N_BURST_STEPS):
        Vd += _DT_TAUD * (-Vd + I_api_flat)

        burst_t = np.maximum(burst_t - DT_B, 0.0)
        in_burst = burst_t > 0

        ref[ref > 0] -= DT_B
        can = ref <= 0
        Vs[can] += _DT_TAUS * (-Vs[can] + I_bas_flat[can])

        p_fire = _sigm_prob(Vs - V_THRESH_B, BETA_S_B) * DT_B
        spiked = can & (np.random.rand(M) < p_fire)

        # inside a burst: short refractory + high reset → doublet/triplet bursts
        ref[spiked & in_burst] = T_REF_BURST_B
        ref[spiked & ~in_burst] = T_REF_B
        Vs[spiked & in_burst] = V_RESET_BURST_B
        Vs[spiked & ~in_burst] = V_RESET_B

        Vs_3d = Vs.reshape(out_dim_, num_g_, N_B)
        Vd_3d = Vd.reshape(out_dim_, num_g_, N_B)
        p_s_v = _sigm_prob(Vs_3d - V_THRESH_B, BETA_S_B)
        p_d_v = _sigm_prob(Vd_3d - V_BURST_B, BETA_D_B)
        p_burst_accum += (p_s_v * p_d_v).mean(axis=2)

        spk_3d = spiked.reshape(out_dim_, num_g_, N_B)
        spk_tr[step] = spk_3d[BF_DISP_JOINT]
        Vs_tr[step] = Vs_3d[BF_DISP_JOINT]
        Vd_tr[step] = Vd_3d[BF_DISP_JOINT]

    p_burst_pop = p_burst_accum / N_BURST_STEPS

    bstate['burst_t'] = burst_t
    bstate['burst_ref'] = burst_ref
    return p_burst_pop, Vs_tr, Vd_tr, spk_tr, brst_tr


def learn_basal_hpg(S, k_act, y_k, pb_k, action, mu,
                    r, dJfilt_sub, dJagg_sub, gamma):
    """Policy gradient on the basal readout of the *active* goal.

    ``dphi = d mu / d y`` is the analytic derivative of the burst transfer
    function; ``eps = action - mu`` is the exploration perturbation.  The
    per-step outer product is low-pass filtered into an eligibility trace and
    accumulated weighted by the baseline-subtracted reward.
    """
    alpha_J = 1.0 - gamma
    sech2_y = 1.0 - np.tanh(y_k) ** 2
    pb_grad = pb_k * (1.0 - pb_k)
    dphi = P_BURST_SCALE * pb_grad * I_BAS_SCALE * sech2_y / BETA_S_B
    eps = action - mu
    dJ_step = np.outer(eps * dphi, S)
    dJfilt_sub[k_act] = dJfilt_sub[k_act] * (1.0 - alpha_J) + dJ_step
    dJagg_sub[k_act] += r * dJfilt_sub[k_act]


def calibrate_offset(n_warm=300, out_dim=OUT_DIM, num_g=NUM_G, n_b=N_B):
    """Measure the resting burst probability and return ``(p_rest, mu_offset)``.

    The window-averaged V-based ``p_burst`` differs from the equilibrium target,
    so it is measured empirically and subtracted; without this a constant-sign
    torque bias makes the arm spiral from step 0.
    """
    M = out_dim * num_g * n_b
    bs = init_burst_state(M)
    y0 = [np.zeros(out_dim) for _ in range(num_g)]
    g0 = np.zeros(num_g)
    g0[0] = 1.0
    db0 = np.full((out_dim, num_g), DB_INIT)
    acc = []
    for _ in range(n_warm):
        p, *_ = run_burst_subpop(y0, g0, bs, db0, n_b)
        acc.append(p[:, 0])
    p_rest = float(np.mean(acc))
    return p_rest, P_BURST_SCALE * p_rest
