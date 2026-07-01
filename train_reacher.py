"""
train_reacher.py
Trains the two-compartment burst-gating agent on a 4-goal directional reaching
task in the MuJoCo Reacher-v4 environment (Fig. 2D-F of the paper).

Each output unit is a probabilistic two-compartment neuron whose burst probability
p_burst = <p_s(V_s) p_d(V_d)> multiplicatively gates a state-dependent basal drive
by a goal-selective apical drive; the motor command is linear in p_burst.  Weights
are learned online with a three-factor (reward-modulated Hebbian) rule.

Reacher mapping of the 4-goal directional task:
  obs (11): [cosθ1,cosθ2, sinθ1,sinθ2, tgt_x,tgt_y, θ1dot,θ2dot, ftip-tgt(x,y,z)]
  fingertip pos:  fx = obs[4]+obs[8],  fy = obs[5]+obs[9]
  4 goals = push fingertip toward +X / -X / +Y / -Y
  reward(g) = -distance to target - control cost - joint-velocity penalty

Outputs (every 100 episodes): reacher_checkpoint.pickle (weights),
reacher_learning_curve.npy (100-episode running-mean reward),
reacher_training_progress.png (diagnostics).
"""
import gym
import imageio
import gc
import matplotlib
matplotlib.use('Agg')
import numpy as np
# numpy >=2.0 removed bool8; gym passive_env_checker still uses it
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import goal
from goal import GOAL_GRADED
import mujoco
from optimizer import Adam
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from tqdm import trange

# ── Environment — 2-D Reacher (TimeLimit overridden so 4×100 goal blocks fit) ──
episode_duration = 400
env = gym.make('Reacher-v4', render_mode=None,
               max_episode_steps=episode_duration + 5)
print(f'obs: {env.observation_space.shape}  act: {env.action_space.shape}')

target_intensity = 10.
n_episodes       = 50000
if_load          = False   # set True to resume from reacher_checkpoint.pickle

# ── Exploration / action scaling (anti-spin) ─────────────────────────────────
ACT_LIMIT_START = 0.002  # near-zero physical torque at episode 0
ACT_LIMIT_END   = 0.1    # full physical torque after the startup ramp
TAU_ACT         = 750.0  # episodes; torque-amplitude ramp time-constant
VAR0            = 2.2    # initial internal noise (deliberately very high)
VAR_END         = 0.2    # final internal noise
TAU_EXPL        = 2500.0 # episodes; noise decay time-constant (longer)

def action_limit(it):
    """Ramp torque amplitude without suppressing early stochasticity."""
    return ACT_LIMIT_END - (ACT_LIMIT_END - ACT_LIMIT_START) * np.exp(-it / TAU_ACT)

def expl_var(it):
    return VAR_END + (VAR0 - VAR_END) * np.exp(-it / TAU_EXPL)

num_g   = 4
out_dim = 2          # Reacher action dim (was 8 for Ant)
N_B     = 10
N       = 500
HIST_LEN = 10        # remember joint positions from 10 steps in the past
HIST_DIM = 4         # joint state per step: obs[0:4] = cos/sin of 2 joints
I_DIM   = 11 + 4 + HIST_LEN * HIST_DIM   # obs + goal one-hot + joint history
T       = 100
shape   = (N, I_DIM, 1, T)

dt     = 0.001
tau_m  = 5.*dt;  tau_s = 2.*dt;  tau_ro = 3.*dt
beta_ro = np.exp(-dt / tau_ro)

par = dict(tau_m=tau_m, tau_s=tau_s, tau_ro=tau_ro, beta_ro=beta_ro,
           dv=1/5., alpha=0.5, Vo=0., h=0., s_inh=20,
           N=N, T=T, dt=dt, offT=1, alpha_rout=0.0005,
           sigma_input=0.01, sigma_teach=5.0,
           shape=shape, sigma_output=0.001)
par['sigma_rec'] = 0.9 / np.sqrt(N)   # spectral radius ≈0.9 → long reservoir memory
par['num_g']     = num_g

# ── Burst neuron (V-based probabilistic product gate) — identical to Ant ──────
DT_B = 1.0
TAU_S_B = 5.0;   TAU_D_B = 5.0
V_THRESH_B = 1.0;   V_RESET_B = 0.0;   V_RESET_BURST_B = 0.9
V_BURST_B  = 0.75
BETA_S_B   = 0.5
BETA_D_B   = 0.2
T_REF_B    = 5.0
T_REF_BURST_B = 1.5
T_BURST_REF_D = 30.0
N_BURST_STEPS = 15
SPIKE_H_B  = 2.0

_DT_TAUS = DT_B / TAU_S_B
_DT_TAUD = DT_B / TAU_D_B

I_BAS_OFFSET = 0.5;   I_BAS_SCALE = 1.3
I_API_SCALE  = 4.0

def _sigm_prob(x, beta):
    return 1.0 / (1.0 + np.exp(-x / beta))

_F_D_ACT   = 1.0 / (1.0 + np.exp(-(I_API_SCALE - V_BURST_B) / BETA_D_B))
_F_S_TGT   = min(0.5 / max(_F_D_ACT, 1e-6), 0.999)
_VS_TGT    = V_THRESH_B + BETA_S_B * np.log(_F_S_TGT / (1 - _F_S_TGT))
DB_INIT    = float(np.clip(_VS_TGT - I_BAS_OFFSET, 0.0, 1.5))
DB_CLIP    = 1.5

P_BURST_SCALE  = 7.0
P_BURST_TARGET = 1.0 / P_BURST_SCALE

BASELINE_LR   = 0.5
BURST_TRACE_MS = 500
GOAL_COLORS    = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
SPIKE_DISP_H   = 2.0
BF_DISP_JOINT  = 0


def init_burst_state(M):
    Vs_init = np.random.uniform(0, V_THRESH_B * 0.95, M)
    return dict(
        Vs      = Vs_init,
        Vd      = np.zeros(M),
        ref     = np.zeros(M),
        burst_t   = np.zeros(M),
        burst_ref = np.zeros(M),
    )


def run_burst_subpop(y_subs, g, bstate, delta_b, N_B):
    num_g_   = len(y_subs)
    out_dim_ = y_subs[0].size
    M = out_dim_ * num_g_ * N_B

    y_mat      = np.stack(y_subs, axis=1)
    I_bas_mat  = I_BAS_OFFSET + delta_b + I_BAS_SCALE * np.tanh(y_mat)
    I_bas_flat = np.repeat(I_bas_mat.ravel(), N_B)
    I_api_mat  = np.tile(g * I_API_SCALE, (out_dim_, 1))
    I_api_flat = np.repeat(I_api_mat.ravel(), N_B)

    Vs, Vd    = bstate['Vs'], bstate['Vd']
    ref       = bstate['ref']
    burst_t   = bstate['burst_t']
    burst_ref = bstate['burst_ref']

    Vs_tr   = np.zeros((N_BURST_STEPS, num_g_, N_B))
    Vd_tr   = np.zeros((N_BURST_STEPS, num_g_, N_B))
    spk_tr  = np.zeros((N_BURST_STEPS, num_g_, N_B), dtype=bool)
    brst_tr = np.zeros((N_BURST_STEPS, num_g_, N_B), dtype=bool)

    p_burst_accum = np.zeros((out_dim_, num_g_))

    p_s_now   = _sigm_prob(Vs - V_THRESH_B, BETA_S_B)
    p_d_now   = _sigm_prob(Vd - V_BURST_B,  BETA_D_B)
    burst_ref = np.maximum(burst_ref - N_BURST_STEPS * DT_B, 0.0)
    can_burst = burst_ref <= 0
    new_burst = can_burst & (np.random.rand(M) < p_s_now * p_d_now)
    burst_t[new_burst]   = N_BURST_STEPS * DT_B
    burst_ref[new_burst] = T_BURST_REF_D
    Vd[new_burst] = 0.0
    brst_3d   = new_burst.reshape(out_dim_, num_g_, N_B)
    brst_tr[0] = brst_3d[BF_DISP_JOINT]

    for step in range(N_BURST_STEPS):
        Vd += _DT_TAUD * (-Vd + I_api_flat)

        burst_t  = np.maximum(burst_t - DT_B, 0.0)
        in_burst = burst_t > 0

        ref[ref > 0] -= DT_B
        can = ref <= 0
        Vs[can] += _DT_TAUS * (-Vs[can] + I_bas_flat[can])

        p_fire = _sigm_prob(Vs - V_THRESH_B, BETA_S_B) * DT_B
        spiked = can & (np.random.rand(M) < p_fire)

        ref[spiked &  in_burst] = T_REF_BURST_B
        ref[spiked & ~in_burst] = T_REF_B
        Vs[spiked &  in_burst]  = V_RESET_BURST_B
        Vs[spiked & ~in_burst]  = V_RESET_B

        Vs_3d  = Vs.reshape(out_dim_, num_g_, N_B)
        Vd_3d  = Vd.reshape(out_dim_, num_g_, N_B)
        p_s_v  = _sigm_prob(Vs_3d - V_THRESH_B, BETA_S_B)
        p_d_v  = _sigm_prob(Vd_3d - V_BURST_B,  BETA_D_B)
        p_burst_accum += (p_s_v * p_d_v).mean(axis=2)

        spk_3d      = spiked.reshape(out_dim_, num_g_, N_B)
        spk_tr[step]  = spk_3d[BF_DISP_JOINT]
        Vs_tr[step]   = Vs_3d[BF_DISP_JOINT]
        Vd_tr[step]   = Vd_3d[BF_DISP_JOINT]

    p_burst_pop = p_burst_accum / N_BURST_STEPS

    bstate['burst_t']   = burst_t
    bstate['burst_ref'] = burst_ref
    return p_burst_pop, Vs_tr, Vd_tr, spk_tr, brst_tr


def learn_basal_hpg(S, k_act, y_k, pb_k, action, mu,
                    r, dJfilt_sub, dJagg_sub, gamma):
    alpha_J = 1.0 - gamma
    sech2_y = 1.0 - np.tanh(y_k) ** 2
    pb_grad = pb_k * (1.0 - pb_k)
    dphi    = P_BURST_SCALE * pb_grad * I_BAS_SCALE * sech2_y / BETA_S_B
    eps     = action - mu
    dJ_step           = np.outer(eps * dphi, S)
    dJfilt_sub[k_act] = dJfilt_sub[k_act] * (1.0 - alpha_J) + dJ_step
    dJagg_sub[k_act] += r * dJfilt_sub[k_act]


def _add_spike_excursions(V_arr, spk_arr, threshold, disp_h):
    V_disp = V_arr.copy()
    for t in np.where(spk_arr > 0)[0]:
        V_disp[t] = threshold + disp_h
    return V_disp


def master_plot(R_AVG, REWARDS, ep_Vs_tr, ep_Vd_tr, ep_spk_tr, ep_brst_tr, ep_len,
                POSITIONS, permutation, delta_b, bf_log, iteration):
    n_bt   = ep_len * N_BURST_STEPS
    t_bt   = np.arange(n_bt) * DT_B
    n_show = min(BURST_TRACE_MS, n_bt)
    t_show = t_bt[-n_show:]
    Vs_win   = ep_Vs_tr[-n_show:]
    Vd_win   = ep_Vd_tr[-n_show:]
    spk_win  = ep_spk_tr[-n_show:]
    brst_win = ep_brst_tr[-n_show:]

    fig = plt.figure(iteration, figsize=(18, 10))
    fig.clf()
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(R_AVG, color='k', lw=1.4)
    ax.set_xlabel('Episode (×100)'); ax.set_ylabel('Avg reward')
    ax.set_title('Learning curve', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    for ki in range(num_g):
        _lane_s = ki * (V_THRESH_B + SPIKE_DISP_H + 0.5)
        Vs_one  = _add_spike_excursions(Vs_win[:, ki, 0], spk_win[:, ki, 0],
                                        V_THRESH_B, SPIKE_DISP_H)
        ax.plot(t_show, Vs_one + _lane_s, color=GOAL_COLORS[ki],
                lw=0.9, alpha=0.9, label=f'g{ki}')
        ax.axhline(_lane_s + V_THRESH_B, color=GOAL_COLORS[ki],
                   lw=0.4, ls='--', alpha=0.35)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('$V_s$')
    ax.set_title(f'$V_s$ — joint {BF_DISP_JOINT}, last {BURST_TRACE_MS} ms',
                 fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    for ki in range(num_g):
        _lane_d = ki * (I_API_SCALE + 0.5)
        Vd_one  = _add_spike_excursions(Vd_win[:, ki, 0], brst_win[:, ki, 0],
                                        V_BURST_B, SPIKE_DISP_H * 0.5)
        ax.plot(t_show, Vd_one + _lane_d, color=GOAL_COLORS[ki],
                lw=0.9, alpha=0.9, label=f'g{ki}')
        ax.axhline(_lane_d + V_BURST_B, color=GOAL_COLORS[ki],
                   lw=0.4, ls='--', alpha=0.35)
        ax.axhline(_lane_d + I_API_SCALE, color=GOAL_COLORS[ki],
                   lw=0.4, ls=':', alpha=0.2)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('$V_d$')
    ax.set_title(f'$V_d$ — joint {BF_DISP_JOINT}, last {BURST_TRACE_MS} ms\n'
                 f'dashed=burst thr, dotted=I_API eq.', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    POS = np.array(POSITIONS)
    for seg in range(4):
        sl = slice(seg * 100, min((seg + 1) * 100, len(POS)))
        gi = permutation[seg]
        ax.plot(POS[sl, 0], POS[sl, 1], color=GOAL_COLORS[gi],
                lw=1.2, alpha=0.85, label=f'g{gi}')
        if sl.start < len(POS):
            ax.plot(POS[sl.start, 0], POS[sl.start, 1], 'o',
                    color=GOAL_COLORS[gi], ms=5)
    ax.set_xlabel('fingertip X'); ax.set_ylabel('fingertip Y')
    ax.set_title('Fingertip trajectory', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax_fr = fig.add_subplot(gs[1, 1])
    bf_arr = np.array(bf_log)
    for ki in range(num_g):
        ax_fr.plot(bf_arr[:, ki], color=GOAL_COLORS[ki],
                   label=f'g{ki}', lw=0.9, alpha=0.85)
    ax_fr.axhline(P_BURST_TARGET, color='k', lw=0.8, ls='--', alpha=0.6,
                  label=f'p_rest≈{P_BURST_TARGET:.2f}')
    ax_fr.set_xlabel('Env step'); ax_fr.set_ylabel('p_burst (V)')
    ax_fr.set_title(f'V-based p_burst + mu — joint {BF_DISP_JOINT}',
                    fontweight='bold')
    ax_fr.set_ylim(-0.05, 0.5)
    ax_fr.legend(fontsize=7, framealpha=0.7, ncol=2, loc='upper left')
    ax_fr.spines['top'].set_visible(False)
    ax_mu = ax_fr.twinx()
    for ki in range(num_g):
        ax_mu.plot(P_BURST_SCALE * bf_arr[:, ki] - MU_OFFSET,
                   color=GOAL_COLORS[ki], lw=1.2, alpha=0.5, ls='--')
    ax_mu.axhline(0, color='k', lw=0.6, ls=':', alpha=0.5)
    ax_mu.set_ylabel('mu (dashed)', color='gray')
    ax_mu.tick_params(axis='y', labelcolor='gray')
    ax_mu.set_ylim(-1.05, 1.05)

    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(delta_b.T, aspect='auto', cmap='RdBu_r',
                   vmin=-DB_CLIP, vmax=DB_CLIP)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    ax.set_xlabel('Joint'); ax.set_ylabel('Goal')
    ax.set_yticks(range(num_g))
    ax.set_title('Homeostatic δb', fontweight='bold')

    fig.suptitle(f'[Reacher] Episode {iteration}  |  avg reward (last 100): '
                 f'{np.mean(REWARDS[-100:]):.1f}'
                 f'  [I_API={I_API_SCALE}  scale={P_BURST_SCALE}  win={BURST_TRACE_MS}ms]',
                 fontsize=11, fontweight='bold')
    plt.savefig('reacher_training_progress.png', dpi=120, bbox_inches='tight')


def save_video(frames, path, fps=30):
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, f in enumerate(frames):
            imageio.imwrite(os.path.join(tmpdir, f'frame_{i:05d}.png'), f)
        subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y',
                        '-r', str(fps),
                        '-i', f'{tmpdir}/frame_%05d.png',
                        '-vcodec', 'libx264', '-crf', '23',
                        '-pix_fmt', 'yuv420p', path],
                       capture_output=True)


def _make_render_frame(gym_frame, roll_Vs, roll_Vd, roll_spk, roll_brst,
                       k_act, mu_vec, p_burst_mat, roll_mu):
    H, W  = gym_frame.shape[:2]
    dpi   = 100
    PW    = int(W * 1.8)
    t_ms  = np.arange(roll_Vs.shape[0]) * DT_B
    t_mu  = np.arange(roll_mu.shape[0]) * DT_B
    _N_SHOW   = min(4, N_B)
    _lane_gap = V_THRESH_B + SPIKE_DISP_H + 0.6

    fig = plt.figure(figsize=(PW / dpi, H / dpi), dpi=dpi, facecolor='k')
    gs  = fig.add_gridspec(num_g + 1, 1,
                           height_ratios=[0.72 / num_g] * num_g + [0.28],
                           hspace=0.07, left=0.04, right=0.97,
                           top=0.97, bottom=0.06)

    for ki in range(num_g):
        ax = fig.add_subplot(gs[ki])
        for ni in range(_N_SHOW):
            Vs_one = _add_spike_excursions(roll_Vs[:, ki, ni], roll_spk[:, ki, ni],
                                           V_THRESH_B, SPIKE_DISP_H)
            ax.plot(t_ms, Vs_one + ni * _lane_gap, color=GOAL_COLORS[ki],
                    lw=0.8, alpha=0.9 - ni * 0.15)
        Vd_one = _add_spike_excursions(roll_Vd[:, ki, 0], roll_brst[:, ki, 0],
                                       V_BURST_B, SPIKE_DISP_H * 0.35)
        ax.plot(t_ms, Vd_one - 0.5, color='white', lw=0.5, alpha=0.3, ls='--')
        ax.set_facecolor('k'); ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylim(-0.7, _N_SHOW * _lane_gap)
        lw_b = 2.5 if ki == k_act else 0.5
        for sp in ax.spines.values():
            sp.set_edgecolor(GOAL_COLORS[ki]); sp.set_linewidth(lw_b)
        pb = p_burst_mat[BF_DISP_JOINT, ki]
        ax.text(0.005, 0.95, f'g{ki}  p(V)={pb:.3f}  μ={P_BURST_SCALE*pb-MU_OFFSET:.2f}',
                transform=ax.transAxes, color=GOAL_COLORS[ki], fontsize=7, va='top')

    ax_mu = fig.add_subplot(gs[num_g])
    ax_mu.set_facecolor('k')
    cmap = plt.get_cmap('tab10')
    for ji in range(out_dim):
        ax_mu.plot(t_mu, roll_mu[:, ji], color=cmap(ji / out_dim),
                   lw=0.75, alpha=0.85, label=f'j{ji}')
    ax_mu.axhline(0,  color='white', lw=0.5, alpha=0.5)
    ax_mu.axhline( 1, color='white', lw=0.3, alpha=0.2, ls=':')
    ax_mu.axhline(-1, color='white', lw=0.3, alpha=0.2, ls=':')
    for ji in range(out_dim):
        ax_mu.scatter([t_mu[-1]], [mu_vec[ji]], color=cmap(ji / out_dim), s=12, zorder=5)
    _mu_max = max(np.abs(roll_mu).max(), 0.15)
    ax_mu.set_ylim(-_mu_max * 1.15, _mu_max * 1.15)
    ax_mu.set_xlim(t_mu[0], t_mu[-1])
    ax_mu.tick_params(colors='white', labelsize=5)
    ax_mu.set_ylabel('μ', color='white', fontsize=7)
    ax_mu.set_xlabel('ms', color='white', fontsize=6)
    ax_mu.set_title(f'action trajectory  (goal {k_act} active)',
                    color='white', fontsize=7, pad=2)
    for sp in ax_mu.spines.values():
        sp.set_edgecolor('dimgray'); sp.set_linewidth(0.5)
    ax_mu.legend(ncol=out_dim, loc='upper left', fontsize=4.5,
                 facecolor='k', labelcolor='white', framealpha=0.4,
                 handlelength=1, columnspacing=0.5, borderpad=0.2)

    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    gc.collect()
    PH, PW2 = panel.shape[:2]
    if PH != H:
        panel = panel[:H] if PH > H else np.vstack(
            [panel, np.zeros((H - PH, PW2, 3), np.uint8)])
    return np.hstack([gym_frame, panel])


def fingertip_xy(obs):
    """Reacher fingertip position = target + (fingertip-target) vector."""
    return obs[4] + obs[8], obs[5] + obs[9]


# ── 4 fixed cardinal targets at double distance ──────────────────────────────
# Reacher reach max ≈ 0.21; default target radius ≤ 0.1 → double ≈ 0.19 (reachable).
TGT_DIST = 0.19
TARGETS = np.array([
    [ TGT_DIST,  0.0],   # g0 right (+X)
    [-TGT_DIST,  0.0],   # g1 left  (-X)
    [ 0.0,  TGT_DIST],   # g2 top   (+Y)
    [ 0.0, -TGT_DIST],   # g3 bottom(-Y)
])
W_CTRL  = 0.005           # control-cost weight (penalise large/sustained torque)
W_JVEL  = 0.008           # joint-velocity penalty (penalise spinning)

ARM_Q0 = np.array([0.0, np.pi])   # folded pose → fingertip at center (0,0)

def set_pose(env, tgt_xy):
    """Reset arm to fixed folded pose (tip at center) + set target."""
    u = env.unwrapped
    qpos = u.data.qpos.copy(); qpos[0:2] = ARM_Q0; qpos[2:4] = tgt_xy
    qvel = np.zeros_like(u.data.qvel)
    u.set_state(qpos, qvel)


# ── Training loop ──────────────────────────────────────────────────────────────
M = out_dim * num_g * N_B

# ── Calibrate rest p_burst → centre transfer fn (mu=0 at y=0) ─────────────────
# V-based mean p_burst over the window differs from the equilibrium target, so
# measure it empirically and offset mu, otherwise a constant-sign torque bias
# makes the arm spiral from step 0.
def _calibrate_offset(n_warm=300):
    bs = init_burst_state(M)
    y0 = [np.zeros(out_dim) for _ in range(num_g)]
    g0 = np.zeros(num_g); g0[0] = 1.0
    db0 = np.full((out_dim, num_g), DB_INIT)
    acc = []
    for _ in range(n_warm):
        p, *_ = run_burst_subpop(y0, g0, bs, db0, N_B)
        acc.append(p[:, 0])
    return float(np.mean(acc))

P_REST    = _calibrate_offset()
MU_OFFSET = P_BURST_SCALE * P_REST       # mu = P_BURST_SCALE*p_burst - MU_OFFSET
print(f"calibrated rest p_burst(V)={P_REST:.4f}  MU_OFFSET={MU_OFFSET:.4f}")

n_repeats = 1
for repeat in range(n_repeats):
    print(f"\n--- repeat {repeat} ---")

    network = GOAL_GRADED(par)
    network.gamma   = 0.99
    network.out_dim = out_dim
    network.num_g   = num_g

    Jout_sub   = [np.random.normal(0, par['sigma_output'], (out_dim, N))
                  for _ in range(num_g)]
    adam_sub   = [Adam(alpha=7e-5, drop=1.0, drop_time=10000) for _ in range(num_g)]
    dJfilt_sub = [np.zeros((out_dim, N)) for _ in range(num_g)]
    dJagg_sub  = [np.zeros((out_dim, N)) for _ in range(num_g)]

    if if_load:
        with open('reacher_checkpoint.pickle', 'rb') as fh:
            st = pickle.load(fh)
            network  = st['network']
            Jout_sub = st['Jout_sub']
            adam_sub = st['adam_sub']

    delta_b    = np.full((out_dim, num_g), DB_INIT)
    r_baseline = 0.0
    REWARDS = [];  R_AVG = []

    for iterations in trange(n_episodes):

        render_ep = False
        _tmpdir_obj = None
        if render_ep:
            import tempfile
            _tmpdir_obj = tempfile.TemporaryDirectory()
            _tmpdir = _tmpdir_obj.name
            _frame_idx = 0
            _RLEN = 800
            roll_Vs   = np.zeros((_RLEN, num_g, N_B))
            roll_Vd   = np.zeros((_RLEN, num_g, N_B))
            roll_spk  = np.zeros((_RLEN, num_g, N_B), dtype=bool)
            roll_brst = np.zeros((_RLEN, num_g, N_B), dtype=bool)
            roll_mu   = np.zeros((_RLEN, out_dim))
        env.reset()
        obs, r, done, _, _ = env.step(np.zeros(out_dim))
        r_tot = 0

        network.reset()
        for k in range(num_g):
            dJfilt_sub[k] *= 0
        bstate = init_burst_state(M)

        permutation = np.random.permutation(4)
        POSITIONS = []
        jhist = np.zeros((HIST_LEN, HIST_DIM))   # joint-position history buffer
        ep_Vs_tr  = np.zeros((episode_duration * N_BURST_STEPS, num_g, N_B))
        ep_Vd_tr  = np.zeros((episode_duration * N_BURST_STEPS, num_g, N_B))
        ep_spk_tr  = np.zeros((episode_duration * N_BURST_STEPS, num_g, N_B), dtype=bool)
        ep_brst_tr = np.zeros((episode_duration * N_BURST_STEPS, num_g, N_B), dtype=bool)
        bf_log   = []
        ep_len   = 0

        for k in range(episode_duration):
            index = k // 100
            k_act = int(permutation[index])
            g     = np.zeros(num_g);  g[k_act] = 1.

            if k % 100 == 0:                       # new goal block → reset arm to center + set target
                set_pose(env, TARGETS[k_act])
                obs = env.unwrapped._get_obs()
                jhist[:] = 0.0                     # clear history on context reset

            target_vector = np.zeros(4);  target_vector[k_act] = target_intensity
            obs_aug = np.concatenate((obs, target_vector, jhist.ravel()))

            network.step_(obs_aug)
            S = network.S

            y_subs = [Jout_sub[ki] @ S for ki in range(num_g)]

            p_burst_pop, Vs_tr, Vd_tr, spk_tr, brst_tr = run_burst_subpop(
                y_subs, g, bstate, delta_b, N_B)

            mu     = P_BURST_SCALE * p_burst_pop[:, k_act] - MU_OFFSET
            var    = expl_var(iterations)          # annealed exploration noise
            a_int  = np.clip(mu + np.random.normal(0., var, out_dim), -1., 1.)
            act_scale = action_limit(iterations)   # tiny initially; noise stays high
            action = act_scale * a_int             # scaled physical torque

            sl = slice(k * N_BURST_STEPS, (k + 1) * N_BURST_STEPS)
            ep_Vs_tr[sl]  = Vs_tr
            ep_Vd_tr[sl]  = Vd_tr
            ep_spk_tr[sl]  = spk_tr
            ep_brst_tr[sl] = brst_tr
            bf_log.append(p_burst_pop[BF_DISP_JOINT])
            ep_len = k + 1

            if render_ep:
                _sl = min(N_BURST_STEPS, _RLEN)
                roll_Vs   = np.roll(roll_Vs,   -_sl, axis=0); roll_Vs[-_sl:]   = Vs_tr[:_sl]
                roll_Vd   = np.roll(roll_Vd,   -_sl, axis=0); roll_Vd[-_sl:]   = Vd_tr[:_sl]
                roll_spk  = np.roll(roll_spk,  -_sl, axis=0); roll_spk[-_sl:]  = spk_tr[:_sl]
                roll_brst = np.roll(roll_brst, -_sl, axis=0); roll_brst[-_sl:] = brst_tr[:_sl]
                roll_mu   = np.roll(roll_mu,   -_sl, axis=0)
                roll_mu[-_sl:] = mu[np.newaxis, :]

            obs, r, done, _, _ = env.step(action)
            fx, fy = fingertip_xy(obs)
            jhist = np.roll(jhist, -1, axis=0); jhist[-1] = obs[0:4]   # push joints

            if render_ep:
                _fr = _make_render_frame(
                    env.render(), roll_Vs, roll_Vd, roll_spk, roll_brst,
                    k_act, mu, p_burst_pop, roll_mu)
                imageio.imwrite(f'{_tmpdir}/frame_{_frame_idx:05d}.png', _fr)
                _frame_idx += 1

            POSITIONS.append([fx, fy])

            # reach reward: -dist  - control cost  - joint-velocity penalty
            #   control + velocity terms suppress the sustained-torque spiral
            tgt = TARGETS[k_act]
            dist_tgt = np.hypot(fx - tgt[0], fy - tgt[1])
            ctrl_cost = float(np.sum(a_int ** 2))           # |torque|²
            jvel_pen  = float(obs[6] ** 2 + obs[7] ** 2)    # joint angular speed²
            r = -dist_tgt - W_CTRL * ctrl_cost - W_JVEL * jvel_pen
            r_tot += r

            r_baseline += BASELINE_LR * (r - r_baseline)
            learn_basal_hpg(S, k_act, y_subs[k_act], p_burst_pop[:, k_act],
                            a_int, mu, r - r_baseline, dJfilt_sub, dJagg_sub,
                            network.gamma)

            if done:                 # run full episode → cycle all 4 goals
                break

        REWARDS.append(r_tot)
        if render_ep and _frame_idx > 0:
            import subprocess, os
            os.makedirs('movies', exist_ok=True)
            subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-r', '30',
                            '-i', f'{_tmpdir}/frame_%05d.png',
                            '-vcodec', 'libx264', '-crf', '23',
                            '-pix_fmt', 'yuv420p',
                            f'movies/burst_vprob_reacher_ep{iterations:06d}.mp4'],
                           capture_output=True)
        if _tmpdir_obj is not None:
            _tmpdir_obj.cleanup()

        if iterations > 0:
            for k in range(num_g):
                if np.any(dJagg_sub[k] != 0):
                    np.clip(dJagg_sub[k], -0.05, 0.05, out=dJagg_sub[k])
                    Jout_sub[k] = adam_sub[k].step(Jout_sub[k], dJagg_sub[k])
                dJagg_sub[k] *= 0

        if (iterations % 100 == 0) and (iterations > 0):
            R_AVG.append(np.mean(REWARDS[-100:]))
            master_plot(R_AVG, REWARDS, ep_Vs_tr, ep_Vd_tr, ep_spk_tr, ep_brst_tr, ep_len,
                        POSITIONS, permutation, delta_b, bf_log, iterations)
            with open('reacher_checkpoint.pickle', 'wb') as fh:
                pickle.dump({'network': network,
                             'Jout_sub': Jout_sub,
                             'adam_sub': adam_sub}, fh)
            np.save('reacher_learning_curve.npy', np.array(R_AVG))

    np.save('reacher_learning_curve.npy', np.array(R_AVG))
    print(f"  rep {repeat} done, final avg reward: {np.mean(REWARDS[-100:]):.2f}")
