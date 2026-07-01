"""
eval_reacher.py
Load reacher_checkpoint.pickle and evaluate the trained burst-gating policy.
Runs N_EVAL episodes per goal direction, keeps the best K by total reward, and
writes reacher_eval_trajectories.pickle (consumed by burst_fraction_model.py,
Fig. 2F) plus a standalone best-trajectory figure.
"""
import pickle
import numpy as np
# numpy ≥2.0 removed bool8; gym passive_env_checker still uses it
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
from goal import GOAL_GRADED

# ── Parameters (must match training script exactly) ───────────────────────────
episode_duration = 400
N_EVAL  = 20    # episodes per direction
K_BEST  = 5     # best trajectories to highlight

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
_DT_TAUS = DT_B / TAU_S_B
_DT_TAUD = DT_B / TAU_D_B

I_BAS_OFFSET = 0.5;   I_BAS_SCALE = 1.3
I_API_SCALE  = 4.0
P_BURST_SCALE  = 7.0

num_g   = 4
out_dim = 2
N_B     = 10
N       = 500
HIST_LEN = 10
HIST_DIM = 4
I_DIM   = 11 + 4 + HIST_LEN * HIST_DIM
T       = 100

dt     = 0.001
tau_m  = 5.*dt;  tau_s = 2.*dt;  tau_ro = 3.*dt
beta_ro = np.exp(-dt / tau_ro)
par = dict(tau_m=tau_m, tau_s=tau_s, tau_ro=tau_ro, beta_ro=beta_ro,
           dv=1/5., alpha=0.5, Vo=0., h=0., s_inh=20,
           N=N, T=T, dt=dt, offT=1, alpha_rout=0.0005,
           sigma_input=0.01, sigma_teach=5.0,
           shape=(N, I_DIM, 1, T), sigma_output=0.001)
par['sigma_rec'] = 0.9 / np.sqrt(N)
par['num_g']     = num_g

target_intensity = 10.
TGT_DIST = 0.19
TARGETS = np.array([
    [ TGT_DIST,  0.0],
    [-TGT_DIST,  0.0],
    [ 0.0,  TGT_DIST],
    [ 0.0, -TGT_DIST],
])
W_CTRL = 0.005
W_JVEL = 0.008
ARM_Q0 = np.array([0.0, np.pi])
ACT_LIMIT_END = 0.1
DB_CLIP = 1.5

_F_D_ACT = 1.0 / (1.0 + np.exp(-(I_API_SCALE - V_BURST_B) / BETA_D_B))
_F_S_TGT = min(0.5 / max(_F_D_ACT, 1e-6), 0.999)
_VS_TGT  = V_THRESH_B + BETA_S_B * np.log(_F_S_TGT / (1 - _F_S_TGT))
DB_INIT  = float(np.clip(_VS_TGT - I_BAS_OFFSET, 0.0, 1.5))

GOAL_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
GOAL_NAMES  = ['+X (right)', '-X (left)', '+Y (top)', '-Y (bottom)']

# light versions for individual trajectories (high alpha)
LIGHT_COLORS = ['#f4a0a0', '#a0c4f4', '#a0f4a0', '#f4d0a0']


def _sigm_prob(x, beta):
    return 1.0 / (1.0 + np.exp(-x / beta))


def init_burst_state(M):
    return dict(
        Vs      = np.random.uniform(0, V_THRESH_B * 0.95, M),
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

    p_burst_accum = np.zeros((out_dim_, num_g_))

    p_s_now   = _sigm_prob(Vs - V_THRESH_B, BETA_S_B)
    p_d_now   = _sigm_prob(Vd - V_BURST_B,  BETA_D_B)
    burst_ref = np.maximum(burst_ref - N_BURST_STEPS * DT_B, 0.0)
    can_burst = burst_ref <= 0
    new_burst = can_burst & (np.random.rand(M) < p_s_now * p_d_now)
    burst_t[new_burst]   = N_BURST_STEPS * DT_B
    burst_ref[new_burst] = T_BURST_REF_D
    Vd[new_burst] = 0.0

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
        Vs_3d = Vs.reshape(out_dim_, num_g_, N_B)
        Vd_3d = Vd.reshape(out_dim_, num_g_, N_B)
        p_s_v = _sigm_prob(Vs_3d - V_THRESH_B, BETA_S_B)
        p_d_v = _sigm_prob(Vd_3d - V_BURST_B,  BETA_D_B)
        p_burst_accum += (p_s_v * p_d_v).mean(axis=2)

    bstate['burst_t']   = burst_t
    bstate['burst_ref'] = burst_ref
    return p_burst_accum / N_BURST_STEPS


def fingertip_xy(obs):
    return obs[4] + obs[8], obs[5] + obs[9]


def set_pose(env, tgt_xy):
    u = env.unwrapped
    qpos = u.data.qpos.copy(); qpos[0:2] = ARM_Q0; qpos[2:4] = tgt_xy
    qvel = np.zeros_like(u.data.qvel)
    u.set_state(qpos, qvel)


def _calibrate_offset(network, Jout_sub, delta_b, n_warm=300):
    M = out_dim * num_g * N_B
    bs = init_burst_state(M)
    y0 = [np.zeros(out_dim) for _ in range(num_g)]
    g0 = np.zeros(num_g); g0[0] = 1.0
    acc = []
    for _ in range(n_warm):
        p = run_burst_subpop(y0, g0, bs, delta_b, N_B)
        acc.append(p[:, 0])
    return float(np.mean(acc))


def run_episode(env, network, Jout_sub, delta_b, MU_OFFSET, k_act,
                ep_steps=100, noise=0.0):
    """Run one 100-step episode for goal k_act. Returns (trajectory, total_reward)."""
    M = out_dim * num_g * N_B
    network.reset()
    bstate = init_burst_state(M)
    jhist = np.zeros((HIST_LEN, HIST_DIM))

    set_pose(env, TARGETS[k_act])
    obs = env.unwrapped._get_obs()

    g = np.zeros(num_g); g[k_act] = 1.
    traj = []
    r_tot = 0.0

    for _ in range(ep_steps):
        target_vector = np.zeros(4); target_vector[k_act] = target_intensity
        obs_aug = np.concatenate((obs, target_vector, jhist.ravel()))
        network.step_(obs_aug)
        S = network.S
        y_subs = [Jout_sub[ki] @ S for ki in range(num_g)]
        p_burst_pop = run_burst_subpop(y_subs, g, bstate, delta_b, N_B)
        mu = P_BURST_SCALE * p_burst_pop[:, k_act] - MU_OFFSET
        a_int = np.clip(mu + np.random.normal(0., noise, out_dim), -1., 1.)
        action = ACT_LIMIT_END * a_int
        obs, _, done, _, _ = env.step(action)
        fx, fy = fingertip_xy(obs)
        jhist = np.roll(jhist, -1, axis=0); jhist[-1] = obs[0:4]
        traj.append([fx, fy])
        tgt = TARGETS[k_act]
        dist_tgt = np.hypot(fx - tgt[0], fy - tgt[1])
        ctrl_cost = float(np.sum(a_int ** 2))
        jvel_pen  = float(obs[6] ** 2 + obs[7] ** 2)
        r_tot += -dist_tgt - W_CTRL * ctrl_cost - W_JVEL * jvel_pen
        if done:
            break

    return np.array(traj), r_tot


# ── Load checkpoint ───────────────────────────────────────────────────────────
print("Loading checkpoint …")
with open('reacher_checkpoint.pickle', 'rb') as fh:
    st = pickle.load(fh)
network  = st['network']
Jout_sub = st['Jout_sub']
adam_sub = st['adam_sub']

delta_b = np.full((out_dim, num_g), DB_INIT)

print("Calibrating MU_OFFSET …")
MU_OFFSET = P_BURST_SCALE * _calibrate_offset(network, Jout_sub, delta_b)
print(f"  MU_OFFSET = {MU_OFFSET:.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
env = gym.make('Reacher-v4', render_mode="rgb_array",
               max_episode_steps=episode_duration + 5)
env.reset()

ep_steps = 100   # one goal block
all_trajs   = {k: [] for k in range(num_g)}
all_rewards = {k: [] for k in range(num_g)}

print(f"Running {N_EVAL} eval episodes per goal direction …")
for k_act in range(num_g):
    for ep in range(N_EVAL):
        traj, r = run_episode(env, network, Jout_sub, delta_b, MU_OFFSET,
                              k_act, ep_steps=ep_steps, noise=0.0)
        all_trajs[k_act].append(traj)
        all_rewards[k_act].append(r)
        print(f"  g{k_act} ep{ep:02d}  r={r:.2f}  "
              f"final dist={np.hypot(traj[-1,0]-TARGETS[k_act,0], traj[-1,1]-TARGETS[k_act,1]):.4f}")

env.close()

# ── Pick best K per direction ─────────────────────────────────────────────────
best_idx = {k: np.argsort(all_rewards[k])[-K_BEST:][::-1] for k in range(num_g)}

# ── Save eval data for composite figure ──────────────────────────────────────
eval_data = {}
for k_act in range(num_g):
    trajs = [all_trajs[k_act][ep] for ep in best_idx[k_act]]
    stacked = [np.array(all_trajs[k_act][ep]) for ep in range(N_EVAL)]
    min_len = min(len(t) for t in stacked)
    avg_traj = np.mean([t[:min_len] for t in stacked], axis=0)
    eval_data[k_act] = {
        'top5':    trajs,
        'avg':     avg_traj,
        'rewards': all_rewards[k_act],
    }
with open('reacher_eval_trajectories.pickle', 'wb') as fh:
    pickle.dump(eval_data, fh)
print("Saved reacher_eval_trajectories.pickle")

# ── Render one reacher frame ──────────────────────────────────────────────────
_renv = gym.make('Reacher-v4', render_mode='rgb_array',
                 max_episode_steps=episode_duration + 5)
_renv.reset()
_renv.step(np.zeros(out_dim))
_frame = _renv.render()
_renv.close()
import imageio
imageio.imwrite('reacher_render.png', _frame)
print("Saved reacher_render.png")

# ── Plot — single panel, all directions overlapped ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

theta = np.linspace(0, 2*np.pi, 200)
ax.plot(TGT_DIST * np.cos(theta), TGT_DIST * np.sin(theta),
        'k--', lw=0.5, alpha=0.2, zorder=0)
ax.axhline(0, color='gray', lw=0.4, alpha=0.3)
ax.axvline(0, color='gray', lw=0.4, alpha=0.3)

for k_act in range(num_g):
    tgt = TARGETS[k_act]

    # all 20 in very light color (background)
    for ep in range(N_EVAL):
        traj = all_trajs[k_act][ep]
        ax.plot(traj[:, 0], traj[:, 1],
                color=LIGHT_COLORS[k_act], lw=0.5, alpha=0.20, zorder=1)

    # best K in saturated goal color
    for rank, ep in enumerate(best_idx[k_act]):
        traj = all_trajs[k_act][ep]
        alpha_val = 0.90 - rank * 0.12
        label = f'g{k_act} {GOAL_NAMES[k_act]}' if rank == 0 else None
        ax.plot(traj[:, 0], traj[:, 1],
                color=GOAL_COLORS[k_act], lw=1.4, alpha=alpha_val,
                zorder=3, label=label)
        ax.plot(traj[0, 0],  traj[0, 1],  'o',
                color=GOAL_COLORS[k_act], ms=3.5, alpha=alpha_val, zorder=4)
        ax.plot(traj[-1, 0], traj[-1, 1], 's',
                color=GOAL_COLORS[k_act], ms=3.5, alpha=alpha_val, zorder=4)

    # target star
    ax.plot(tgt[0], tgt[1], '*', color='gold', ms=12, zorder=5,
            markeredgecolor=GOAL_COLORS[k_act], markeredgewidth=0.8)

ax.set_aspect('equal')
ax.set_xlim(-0.30, 0.30); ax.set_ylim(-0.30, 0.30)
ax.set_xlabel('fingertip X'); ax.set_ylabel('fingertip Y')
ax.legend(fontsize=8, framealpha=0.8, loc='upper left',
          handlelength=1.5, labelspacing=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

best_rs = [max(all_rewards[k]) for k in range(num_g)]
ax.set_title(
    f'Reacher eval — best {K_BEST}/{N_EVAL} per direction\n'
    f'best r: g0={best_rs[0]:.1f}  g1={best_rs[1]:.1f}  '
    f'g2={best_rs[2]:.1f}  g3={best_rs[3]:.1f}',
    fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig('figure2_eval_trajectories.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figure2_eval_trajectories.png', dpi=150, bbox_inches='tight')
print("Saved figure2_eval_trajectories.pdf / .png")
