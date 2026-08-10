"""
Hyper-parameters for the V-prob burst-neuron Reacher experiment.

Every constant here is byte-for-byte the value used in `burst_vprob_reacher.py`
(the script this package was extracted from), so results are reproducible.
"""
import numpy as np

# ── Task / environment ────────────────────────────────────────────────────────
ENV_ID           = 'Reacher-v4'
EPISODE_DURATION = 400          # env steps per episode (4 goal blocks × 100)
GOAL_BLOCK       = 100          # steps spent on each goal
NUM_G            = 4            # number of goals / sub-policies
OUT_DIM          = 2            # Reacher action dim (2 joint torques)

TARGET_INTENSITY = 10.          # amplitude of the goal one-hot fed to the reservoir

# 4 fixed cardinal targets. Reacher reach max ≈ 0.21, default target radius ≤ 0.1,
# so 0.19 is "double distance" and still reachable.
TGT_DIST = 0.19
TARGETS = np.array([
    [ TGT_DIST,  0.0],   # g0 right  (+X)
    [-TGT_DIST,  0.0],   # g1 left   (-X)
    [ 0.0,  TGT_DIST],   # g2 top    (+Y)
    [ 0.0, -TGT_DIST],   # g3 bottom (-Y)
])
ARM_Q0 = np.array([0.0, np.pi])   # folded pose → fingertip at the centre (0, 0)

W_CTRL = 0.005    # control-cost weight       (penalise large / sustained torque)
W_JVEL = 0.008    # joint-velocity penalty    (penalise spinning)

# ── Exploration / action scaling (anti-spin) ─────────────────────────────────
ACT_LIMIT_START = 0.002   # near-zero physical torque at episode 0
ACT_LIMIT_END   = 0.1     # full physical torque after the startup ramp
TAU_ACT         = 750.0   # episodes; torque-amplitude ramp time-constant
VAR0            = 2.2     # initial internal noise (deliberately very high)
VAR_END         = 0.2     # final internal noise
TAU_EXPL        = 2500.0  # episodes; noise decay time-constant (longer)


def action_limit(it):
    """Ramp torque amplitude without suppressing early stochasticity."""
    return ACT_LIMIT_END - (ACT_LIMIT_END - ACT_LIMIT_START) * np.exp(-it / TAU_ACT)


def expl_var(it):
    """Annealed exploration noise on the internal (pre-scaling) action."""
    return VAR_END + (VAR0 - VAR_END) * np.exp(-it / TAU_EXPL)


# ── Reservoir (GOAL_GRADED) ──────────────────────────────────────────────────
N        = 500          # reservoir units
N_B      = 10           # burst neurons per (joint, goal) sub-population
HIST_LEN = 10           # remember joint positions from 10 steps in the past
HIST_DIM = 4            # joint state per step: obs[0:4] = cos/sin of the 2 joints
OBS_DIM  = 11           # Reacher observation
I_DIM    = OBS_DIM + NUM_G + HIST_LEN * HIST_DIM
T        = 100

dt     = 0.001
tau_m  = 5. * dt
tau_s  = 2. * dt
tau_ro = 3. * dt

GAMMA   = 0.99          # eligibility-trace decay of the policy gradient
LR_ADAM = 7e-5
GRAD_CLIP = 0.05

def build_par():
    """Parameter dict consumed by :class:`reacher_burst.network.GOAL_GRADED`."""
    par = dict(
        tau_m=tau_m, tau_s=tau_s, tau_ro=tau_ro, beta_ro=np.exp(-dt / tau_ro),
        dv=1 / 5., alpha=0.5, Vo=0., h=0., s_inh=20,
        N=N, T=T, dt=dt, offT=1, alpha_rout=0.0005,
        sigma_input=0.01, sigma_teach=5.0,
        shape=(N, I_DIM, 1, T), sigma_output=0.001,
    )
    par['sigma_rec'] = 0.9 / np.sqrt(N)   # spectral radius ≈0.9 → long reservoir memory
    par['num_g']     = NUM_G
    return par


# ── Burst neuron (V-based probabilistic product gate) ────────────────────────
DT_B          = 1.0     # ms; burst-neuron integration step
TAU_S_B       = 5.0     # somatic time constant   (ms)
TAU_D_B       = 5.0     # dendritic time constant (ms)
V_THRESH_B    = 1.0     # somatic spike threshold
V_RESET_B     = 0.0     # reset after a regular spike
V_RESET_BURST_B = 0.9   # reset after a spike inside a burst (→ high-freq doublets)
V_BURST_B     = 0.75    # dendritic (apical) burst threshold
BETA_S_B      = 0.5     # somatic sigmoid sharpness
BETA_D_B      = 0.2     # dendritic sigmoid sharpness
T_REF_B       = 5.0     # refractory, regular spike (ms)
T_REF_BURST_B = 1.5     # refractory, in-burst spike (ms)
T_BURST_REF_D = 30.0    # dendritic burst refractory (ms)
N_BURST_STEPS = 15      # burst-neuron sub-steps per env step
SPIKE_H_B     = 2.0

I_BAS_OFFSET = 0.5      # basal drive at y = 0
I_BAS_SCALE  = 1.3      # basal drive gain on tanh(y)
I_API_SCALE  = 4.0      # apical (goal) drive when the goal is active

P_BURST_SCALE  = 7.0    # p_burst → mu gain
P_BURST_TARGET = 1.0 / P_BURST_SCALE

BASELINE_LR = 0.5       # reward-baseline low-pass rate
DB_CLIP     = 1.5       # homeostatic basal-bias clip


def _sigm_prob(x, beta):
    return 1.0 / (1.0 + np.exp(-x / beta))


# Homeostatic basal bias initialisation: pick δb so that, with the apical
# compartment fully driven, the product gate p_s·p_d sits at ≈0.5.
_F_D_ACT = 1.0 / (1.0 + np.exp(-(I_API_SCALE - V_BURST_B) / BETA_D_B))
_F_S_TGT = min(0.5 / max(_F_D_ACT, 1e-6), 0.999)
_VS_TGT  = V_THRESH_B + BETA_S_B * np.log(_F_S_TGT / (1 - _F_S_TGT))
DB_INIT  = float(np.clip(_VS_TGT - I_BAS_OFFSET, 0.0, 1.5))

# ── Default file names (kept identical to the ones the paper repo ships) ─────
CKPT_FILE  = 'reacher_checkpoint.pickle'         # trained network weights
CURVE_FILE = 'reacher_learning_curve.npy'        # 100-episode running-mean reward (Fig. 2D)
EVAL_FILE  = 'reacher_eval_trajectories.pickle'  # best-of-20 eval trajectories (Fig. 2F)
TRAIN_FIG  = 'reacher_training_progress.png'     # training diagnostics (not a paper figure)
FIG2_FILE  = 'figure2_model.pdf'                 # Figure 2

# ── Plotting ─────────────────────────────────────────────────────────────────
GOAL_COLORS   = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
BURST_TRACE_MS = 500    # width of the voltage window shown in the figure
SPIKE_DISP_H   = 2.0    # cosmetic spike height added on top of threshold
BF_DISP_JOINT  = 0      # which joint's sub-population is displayed
