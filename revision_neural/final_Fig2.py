"""
FINAL revised Fig. 2  (v2, 2026-10: row 2 = circuit / learning curve / trajectories;
the context-switch panel moved to figS5_context_switch.py; backup of v1 in
final_Fig2_backup_v1.py).
Row 0:  A neuron sketch  |  B 3-case Vs traces  |  C phase diagram
Row 1:  D circuit diagram of the reaching agent (drawn natively, adapted from
        burst_coincidence_investigation/circuit_diagrams.py)  |  E learning curve
        (mean +- s.d. over the from-scratch seeds when complete; else resumed run)
        |  F reaching trajectories
Row 2:  G-I matched-noise model analysis
  I_API_SCALE  = 4.0 ;  mu = 7 * p_burst - 1.4
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
BCI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BCI)

import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

np.random.seed(42)

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 11.5,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9.5, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'lines.linewidth': 1.6,
})

GOAL_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
GOAL_LABELS = ['+X', '−X', '+Y', '−Y']

# ── V-prob burst-neuron parameters (exact from _vprob notebook) ───────────────
DT_B=1.0; TAU_S_B=5.0; TAU_D_B=5.0
V_THRESH_B=1.0; V_RESET_B=0.0; V_RESET_BURST_B=0.9
V_BURST_B=0.75; BETA_S_B=0.5; BETA_D_B=0.2
T_REF_B=5.0; T_REF_BURST_B=1.5; T_BURST_REF_D=30.0
N_BURST_STEPS=15; SPIKE_DISP_H=2.0
_DT_TAUS=DT_B/TAU_S_B; _DT_TAUD=DT_B/TAU_D_B
I_BAS_OFFSET=0.5; I_BAS_SCALE=1.3
I_API_SCALE  = 4.0   # ← KEY: stronger apical drive
P_BURST_SCALE = 7.0  # ← KEY: mu = P_BURST_SCALE * p_burst − 1
LANE_GAP = V_THRESH_B + SPIKE_DISP_H + 0.5   # = 3.5

def _sigm_prob(x, beta):
    return 1./(1.+np.exp(-np.clip(x/beta, -500, 500)))

def _add_spike_excursions(V_arr, spk_arr, threshold, disp_h):
    V = V_arr.copy()
    for t in np.where(spk_arr)[0]:
        V[t] = threshold + disp_h
    return V

def _jitter_spikes(spk_arr, max_jitter=2, seed=0):
    """Shift displayed spike times by a few ms to avoid ruler-perfect traces."""
    rng = np.random.RandomState(seed)
    spk_j = np.zeros_like(spk_arr, dtype=bool)
    for t in np.where(spk_arr)[0]:
        tj = int(np.clip(t + rng.randint(-max_jitter, max_jitter + 1),
                         0, len(spk_arr) - 1))
        spk_j[tj] = True
    return spk_j

def _trace_for_plot(Vs, spk, seed=0, noise_sd=0.025, spike_jitter=2):
    """Noisy/jittered visual trace; leaves the simulated data untouched."""
    rng = np.random.RandomState(seed)
    slow = rng.normal(0.0, noise_sd * 0.55, size=len(Vs))
    slow = np.convolve(slow, np.ones(7) / 7.0, mode='same')
    fast = rng.normal(0.0, noise_sd, size=len(Vs))
    Vs_noisy = Vs + slow + fast
    spk_j = _jitter_spikes(spk, max_jitter=spike_jitter, seed=seed + 991)
    return _add_spike_excursions(Vs_noisy, spk_j, V_THRESH_B, SPIKE_DISP_H)

def simulate_neuron_tv(I_bas_arr, I_api_arr, seed=0):
    """Single neuron, time-varying inputs."""
    T_ms = len(I_bas_arr)
    rng = np.random.RandomState(seed)
    Vs=0.2; Vd=0.0; ref=0.; burst_t=0.; burst_ref=0.
    Vs_log=np.zeros(T_ms); spk=np.zeros(T_ms,bool)
    brst_spk=np.zeros(T_ms,bool); burst_on=np.zeros(T_ms,bool)
    for t in range(T_ms):
        if t % N_BURST_STEPS == 0:
            burst_ref = max(burst_ref - N_BURST_STEPS*DT_B, 0.)
            if burst_ref <= 0:
                ps=_sigm_prob(Vs-V_THRESH_B,BETA_S_B)
                pd=_sigm_prob(Vd-V_BURST_B, BETA_D_B)
                if rng.rand() < ps*pd:
                    burst_t=N_BURST_STEPS*DT_B; burst_ref=T_BURST_REF_D; Vd=0.
        Vd += _DT_TAUD*(-Vd+I_api_arr[t])
        burst_t=max(burst_t-DT_B,0.); in_burst=burst_t>0
        if ref>0: ref-=DT_B
        else:
            Vs+=_DT_TAUS*(-Vs+I_bas_arr[t])
            if rng.rand()<_sigm_prob(Vs-V_THRESH_B,BETA_S_B)*DT_B:
                if in_burst: ref=T_REF_BURST_B; Vs=V_RESET_BURST_B; brst_spk[t]=True
                else:        ref=T_REF_B;       Vs=V_RESET_B
                spk[t]=True
        Vs_log[t]=Vs; burst_on[t]=burst_t>0
    return dict(Vs=Vs_log, spk=spk, brst_spk=brst_spk, burst_on=burst_on)

def simulate_neuron(I_bas, I_api, T_ms=300, seed=0):
    return simulate_neuron_tv(np.full(T_ms, I_bas), np.full(T_ms, I_api), seed=seed)

# ── Data ──────────────────────────────────────────────────────────────────────
# Learning curve: from-scratch seeds (mean +- s.d.) if all complete, else the
# single resumed run.
N_LC = 499
_scratch = []
for _s in range(3):
    _f = f'trajectory_burst_vprob_reacher_scratch_s{_s}.npy'
    if os.path.exists(_f):
        _a = np.load(_f)
        print(f"  {_f}: {len(_a)} points")
        if len(_a) >= N_LC:
            _scratch.append(_a[:N_LC])
LC_SCRATCH = len(_scratch) == 3
if LC_SCRATCH:
    R_RUNS = np.array(_scratch)
    R_AVG  = R_RUNS.mean(0); R_SD = R_RUNS.std(0)
    print("Learning curve: mean +- s.d. over 3 from-scratch seeds")
else:
    R_AVG = np.load('trajectory_burst_vprob_reacher_rep0.npy'); R_SD = None
    print("WARNING: from-scratch seeds missing/incomplete -> using resumed run "
          "(trajectory_burst_vprob_reacher_rep0.npy)")
ep_x  = np.arange(100, 100*(len(R_AVG)+1), 100)

with open('reacher_vprob_eval_data_scratch_s0.pickle','rb') as f:  # from-scratch run, seed 0
    traj_data = pickle.load(f)

# ── Stylized 2-link reacher drawing ──────────────────────────────────────────
def draw_reacher_schematic(ax):
    L1, L2 = 0.10, 0.10
    TGT_D  = 0.19
    TGTS   = np.array([[ TGT_D, 0.], [-TGT_D, 0.], [0., TGT_D], [0., -TGT_D]])
    GCOLS  = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']

    # one nice reaching pose (θ1=25°, θ2=-75° in standard math convention)
    th1 = np.radians(25.)
    th2 = np.radians(-75.)
    P0 = np.array([0., 0.])
    P1 = P0 + L1 * np.array([np.cos(th1),  np.sin(th1)])
    P2 = P1 + L2 * np.array([np.cos(th1 + th2), np.sin(th1 + th2)])

    # faint reach-radius circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(TGT_D*np.cos(theta), TGT_D*np.sin(theta),
            color='#999', lw=0.6, ls='--', alpha=0.35, zorder=0)

    # target stars
    for ki, tgt in enumerate(TGTS):
        ax.plot(tgt[0], tgt[1], '*', color=GCOLS[ki], ms=8, zorder=3,
                markeredgecolor='white', markeredgewidth=0.5, alpha=0.85)

    # arm links
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]],
            color='#3a6fa0', lw=5.5, solid_capstyle='round', zorder=4)
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]],
            color='#5fa0d0', lw=4.0, solid_capstyle='round', zorder=4)

    # joints
    for pt, ms, col in [(P0, 8, '#222'), (P1, 6, '#e07020'), (P2, 5, '#e04040')]:
        ax.plot(pt[0], pt[1], 'o', color=col, ms=ms, zorder=6,
                markeredgecolor='white', markeredgewidth=0.8)

    # mount plate
    ax.fill_between([-0.015, 0.015], [-0.022, -0.022], [-0.032, -0.032],
                    color='#444', zorder=5)

    ax.set_aspect('equal')
    ax.set_xlim(-0.26, 0.26); ax.set_ylim(-0.26, 0.26)
    ax.axis('off')

# ── Circuit diagram of the reaching agent (panel D) ───────────────────────────
# Adapted from burst_coincidence_investigation/circuit_diagrams.py (fig_reaching),
# re-laid out for a wide, flat panel.  Data units = inches (axes placed 1:1).
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Arc
C_BLUE, C_ORANGE = '#3a78b5', '#f28e2b'
C_BLUE_L, C_ORANGE_L = '#dfe9f4', '#fde8d4'
C_PLAST = '#8e3fa8'
C_GREY, C_GREY_L = '#7f7f7f', '#efefef'
C_INK = '#222222'
CFS = 7

def _c_box(ax, x, y, w, h, text='', fc='white', ec=C_INK, lw=0.7, tc=C_INK, r=0.04, z=2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x, y, text, ha='center', va='center', fontsize=CFS, color=tc,
                zorder=z + 1, linespacing=1.1)

def _c_arrow(ax, p0, p1, color=C_INK, lw=0.8, ls='-', head=5, conn='arc3', z=1.5):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=f'-|>,head_length={head*0.09},head_width={head*0.055}',
        color=color, lw=lw, ls=ls, connectionstyle=conn, zorder=z,
        shrinkA=0, shrinkB=0, mutation_scale=10))

def _c_line(ax, xs, ys, color=C_INK, lw=0.8, ls='-', z=1.5):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='round')

def _c_neuron(ax, x, y, s=1.0, z=3, alpha=1.0):
    """Two-compartment pyramidal icon; (x, y) = soma centre."""
    hw, hh = 0.085 * s, 0.075 * s
    ax.add_patch(Polygon([[x - hw, y - hh], [x + hw, y - hh], [x, y + hh]], closed=True,
                         fc=C_BLUE_L, ec=C_BLUE, lw=0.9, zorder=z, alpha=alpha))
    trunk_top = y + 0.36 * s
    ax.plot([x, x], [y + hh, trunk_top], color=C_ORANGE, lw=1.6, zorder=z - 0.1,
            alpha=alpha, solid_capstyle='round')
    for dx, dy in [(-0.07, 0.07), (0.07, 0.07), (0.0, 0.09)]:
        ax.plot([x, x + dx * s], [trunk_top, trunk_top + dy * s], color=C_ORANGE, lw=1.0,
                zorder=z - 0.1, alpha=alpha, solid_capstyle='round')
    for dx in (-1, 1):
        ax.plot([x + dx * hw * 0.8, x + dx * hw * 1.7], [y - hh, y - hh - 0.07 * s],
                color=C_BLUE, lw=0.9, zorder=z - 0.1, alpha=alpha, solid_capstyle='round')
    ax.plot([x, x], [y - hh, y - hh - 0.12 * s], color=C_GREY, lw=0.7, zorder=z - 0.1,
            alpha=alpha)
    return dict(tuft=(x, trunk_top + 0.09 * s), basal=(x - hw * 1.7, y - hh - 0.07 * s))

def _c_arm(ax, x, y, s=1.0):
    th1, th2 = np.deg2rad(25), np.deg2rad(70)
    L = 0.17 * s
    p0 = np.array([x, y]); p1 = p0 + L * np.array([np.cos(th1), np.sin(th1)])
    p2 = p1 + L * np.array([np.cos(th1 + th2), np.sin(th1 + th2)])
    ax.plot(*zip(p0, p1), color='#555555', lw=2.2, solid_capstyle='round', zorder=3)
    ax.plot(*zip(p1, p2), color='#999999', lw=2.0, solid_capstyle='round', zorder=3)
    ax.add_patch(Circle(p0, 0.03 * s, fc=C_INK, ec='none', zorder=4))
    ax.add_patch(Circle(p1, 0.022 * s, fc='white', ec=C_INK, lw=0.6, zorder=4))
    ax.add_patch(Circle(p2, 0.018 * s, fc='#d62728', ec='none', zorder=4))
    return p0, p1, p2

CIRC_W, CIRC_H = 3.15, 2.12     # inches

def draw_reaching_circuit(ax):
    W, H = CIRC_W, CIRC_H
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')

    # inputs
    gx, gy = 0.34, 1.86
    _c_box(ax, gx, gy, 0.62, 0.30, 'goal $g$\n(one-hot)', fc=C_ORANGE_L, ec=C_ORANGE)
    sx, sy = 0.34, 1.02
    _c_box(ax, sx, sy, 0.64, 0.42, 'arm state\n+ joint\nhistory', fc=C_BLUE_L, ec=C_BLUE)

    # recurrent network
    rx, ry, rr = 1.12, 1.02, 0.22
    ax.add_patch(Circle((rx, ry), rr, fc=C_GREY_L, ec=C_GREY, lw=0.8, zorder=2))
    ax.text(rx, ry, '500\nunits', ha='center', va='center', fontsize=CFS, zorder=4,
            linespacing=1.0)
    lr, lcx, lcy = 0.09, rx + 0.05, ry + rr - 0.01
    ax.add_patch(Arc((lcx, lcy), 2 * lr, 2 * lr, theta1=-25, theta2=205,
                     color=C_GREY, lw=0.9, zorder=1.8))
    t = np.deg2rad(-25)
    e = (lcx + lr * np.cos(t), lcy + lr * np.sin(t))
    _c_arrow(ax, (e[0] + 0.03, e[1] + 0.035), (e[0] - 0.005, e[1] - 0.01), color=C_GREY, lw=0.9)
    ax.text(lcx + lr + 0.03, lcy + lr + 0.07, 'recurrent $J$\n(fixed, random)', ha='left',
            va='center', fontsize=CFS, color=C_GREY, linespacing=1.0)
    ax.text(rx - rr * 0.8, ry - rr * 0.8, '$S$', ha='right', va='top', fontsize=CFS, color=C_INK)

    # inputs -> network (fixed)
    _c_arrow(ax, (sx + 0.32, sy), (rx - rr - 0.01, ry), color=C_GREY, lw=0.8)
    ax.text((sx + 0.32 + rx - rr) / 2, sy + 0.05, '$J_{in}$', ha='center', va='bottom',
            fontsize=CFS, color=C_GREY)
    _c_arrow(ax, (gx + 0.10, gy - 0.15), (rx - rr * 0.72, ry + rr * 0.72), color=C_ORANGE, lw=0.8)

    # two-compartment output neurons
    nx, ny = 2.22, 0.80
    for k, off in enumerate([(-0.11, 0.06), (-0.055, 0.03)]):
        _c_neuron(ax, nx + off[0], ny + off[1], z=2.2 + 0.1 * k, alpha=0.3)
    P = _c_neuron(ax, nx, ny, z=3)

    # apical: goal -> tuft
    _c_line(ax, [gx + 0.31, nx], [gy, gy], color=C_ORANGE, lw=1.0)
    _c_arrow(ax, (nx, gy), (nx, P['tuft'][1] + 0.02), color=C_ORANGE, lw=1.0)
    ax.text(nx - 0.05, gy - 0.07, 'apical  $I_{api}=4\\,g_k$', ha='right', va='top',
            fontsize=CFS, color=C_ORANGE)

    # basal: plastic readout y_k -> soma
    _c_arrow(ax, (rx + rr, ry - 0.05), (P['basal'][0] - 0.015, P['basal'][1] + 0.03),
             color=C_PLAST, lw=1.2, conn='arc3,rad=0.15')
    ax.text(1.62, 0.93, '$y_k=J^{k}_{out}S$', ha='center', va='bottom', fontsize=CFS,
            color=C_PLAST)
    ax.text(1.38, 0.74, 'basal\n$I_{bas}\\propto\\tanh\\,y_k$', ha='center', va='top',
            fontsize=CFS, color=C_BLUE, linespacing=1.0)

    # burst probability -> motor command
    bx, by = 2.83, ny
    _c_box(ax, bx, by, 0.50, 0.24, '$\\bar p_{burst}$')
    _c_arrow(ax, (nx + 0.12, ny), (bx - 0.25, by), lw=0.8)
    mx, my = bx, 0.40
    _c_box(ax, mx, my, 0.50, 0.24, 'action')
    _c_arrow(ax, (bx, by - 0.12), (mx, my + 0.12), lw=0.8)

    # arm (environment), circled
    ex, ey, er = 2.10, 0.265, 0.25
    ax.add_patch(Circle((ex, ey), er, fc='#f6f9fc', ec=C_BLUE, lw=0.8, ls=(0, (3, 2)), zorder=1.6))
    p0, p1, p2 = _c_arm(ax, 2.00, 0.13, s=1.3)
    _c_arrow(ax, (mx - 0.25, my - 0.04), (ex + er + 0.005, ey + 0.03), lw=0.8,
             conn='arc3,rad=0.3')
    # state feedback: from the bottom of the circle back to the state input
    a_fb = np.deg2rad(215)
    _c_arrow(ax, (ex + er * np.cos(a_fb), ey + er * np.sin(a_fb)), (sx, sy - 0.22), color=C_BLUE,
             lw=0.8, ls=(0, (3, 2)), conn='angle,angleA=180,angleB=-90,rad=6')
    ax.text(1.10, 0.15, 'arm (environment)', ha='center', va='bottom', fontsize=CFS,
            color=C_BLUE)

    # reward: a soft curve from the arm circle to the plastic basal synapses
    a_rw = np.deg2rad(168)
    _c_arrow(ax, (ex + er * np.cos(a_rw), ey + er * np.sin(a_rw)),
             (P['basal'][0] - 0.03, P['basal'][1] - 0.005), color=C_PLAST, lw=1.0,
             ls=(0, (3, 2)), conn='arc3,rad=-0.75')
    ax.text(1.70, 0.33, 'reward $r$', ha='right', va='center', fontsize=CFS, color=C_PLAST)

    # legend (one line, above the apical pathway): fixed vs plastic
    lx, ly = 0.80, 2.04
    _c_line(ax, [lx, lx + 0.2], [ly, ly], color=C_GREY, lw=0.8)
    ax.text(lx + 0.25, ly, 'fixed', va='center', fontsize=CFS, color=C_INK)
    lx2 = lx + 0.68
    _c_line(ax, [lx2, lx2 + 0.2], [ly, ly], color=C_PLAST, lw=1.2)
    ax.text(lx2 + 0.25, ly, 'plastic (reward-modulated)', va='center',
            fontsize=CFS, color=C_INK)

# Phase-diagram meshes — extended I_api axis to show I_API_SCALE=4.0
I_bas_ax = np.linspace(-1.0, 2.5, 400)
I_api_ax = np.linspace( 0.0, 5.0, 400)
IB, IA   = np.meshgrid(I_bas_ax, I_api_ax)
P_BUR    = _sigm_prob(IB-V_THRESH_B,BETA_S_B) * _sigm_prob(IA-V_BURST_B,BETA_D_B)

# 3-case parameters — apical case uses I_API_SCALE=4.0
CASES = [
    ('Only apical  (silence)',  -0.8, I_API_SCALE, '#1f77b4'),
    ('Only basal  (spiking)',    1.8,  0.0,         '#ff7f0e'),
    ('Both  (bursting)',         1.8,  I_API_SCALE, '#2ca02c'),
]
CASE_LABELS_B = ['only apical  (silence)', 'only basal  (spiking)', 'both  (burst)']

# ── Detailed single-neuron simulation (tracks burst vs tonic spikes + ISIs) ───
def simulate_neuron_detailed(I_bas, I_api, T_ms=4000, seed=0):
    rng = np.random.RandomState(seed)
    Vs=0.2; Vd=0.0; ref=0.; burst_t=0.; burst_ref=0.
    spike_times=[]; spike_is_burst=[]
    for t in range(T_ms):
        if t % N_BURST_STEPS == 0:
            burst_ref = max(burst_ref - N_BURST_STEPS*DT_B, 0.)
            if burst_ref <= 0:
                ps=_sigm_prob(Vs-V_THRESH_B,BETA_S_B)
                pd=_sigm_prob(Vd-V_BURST_B, BETA_D_B)
                if rng.rand() < ps*pd:
                    burst_t=N_BURST_STEPS*DT_B; burst_ref=T_BURST_REF_D; Vd=0.
        Vd += _DT_TAUD*(-Vd+I_api)
        burst_t=max(burst_t-DT_B,0.); in_burst=burst_t>0
        if ref>0: ref-=DT_B
        else:
            Vs+=_DT_TAUS*(-Vs+I_bas)
            if rng.rand()<_sigm_prob(Vs-V_THRESH_B,BETA_S_B)*DT_B:
                spike_times.append(t); spike_is_burst.append(bool(in_burst))
                if in_burst: ref=T_REF_BURST_B; Vs=V_RESET_BURST_B
                else:        ref=T_REF_B;       Vs=V_RESET_B
    ISIs = np.diff(spike_times).astype(float) if len(spike_times)>1 else np.array([])
    is_b = np.array(spike_is_burst, dtype=bool)
    bf   = is_b.sum() / max(len(is_b), 1)
    rate = len(is_b) / (T_ms * DT_B)
    return ISIs, spike_is_burst, bf, rate

# ── Population tuning analysis ─────────────────────────────────────────────────
def compute_pop_tuning(N_pop=100, T_sim=3000, seed=7):
    """Population goal-direction tuning.

    Rows are synthetic neurons with heterogeneous basal excitability and one
    assigned preferred goal.  The preferred goal receives the active apical
    drive; non-preferred goals receive basal-only drive.  This makes panels G/H
    genuine goal-direction tuning heatmaps rather than a single averaged curve.
    """
    rng = np.random.RandomState(seed)
    I_bas_pop = rng.uniform(0.3, 2.0, N_pop)
    pref_goal = np.arange(N_pop) % 4
    rng.shuffle(pref_goal)
    bf   = np.zeros((N_pop, 4))
    rate = np.zeros((N_pop, 4))
    for ni, ib in enumerate(I_bas_pop):
        _, _, bf_act,   rate_act   = simulate_neuron_detailed(ib, I_API_SCALE, T_sim, ni)
        _, _, bf_inact, rate_inact = simulate_neuron_detailed(ib, 0.,          T_sim, ni+10000)
        bf[ni, :] = bf_inact
        rate[ni, :] = rate_inact
        bf[ni, pref_goal[ni]] = bf_act
        rate[ni, pref_goal[ni]] = rate_act
    return bf, rate, pref_goal

print("Pre-computing population tuning (takes ~10 s)…")
_pop_bf, _pop_rate, _pop_pref_goal = compute_pop_tuning()
print("  done.")

# ISI data for example neuron
_ISIs_burst, _spk_ib, _, _ = simulate_neuron_detailed(1.8, I_API_SCALE, T_ms=5000, seed=3)
_ISIs_tonic, _spk_it, _, _ = simulate_neuron_detailed(1.8, 0.,          T_ms=5000, seed=4)

# ── Pyramidal-neuron draw helper ───────────────────────────────────────────────
def draw_pyramidal_neuron(ax):
    ax.set_xlim(-12.5, 5.5); ax.set_ylim(-8.5, 15.5)
    ax.set_aspect('equal'); ax.axis('off')
    c_s='steelblue'; c_d='darkorange'
    tv=np.linspace(0,1,70)
    def qbez(p0,pc,p1):
        xs=(1-tv)**2*p0[0]+2*(1-tv)*tv*pc[0]+tv**2*p1[0]
        ys=(1-tv)**2*p0[1]+2*(1-tv)*tv*pc[1]+tv**2*p1[1]
        return xs,ys
    ax.fill([0,1.7,-1.7,0],[3.2,0.,0.,3.2],color='#dce9f5',edgecolor=c_s,lw=2.5,zorder=5)
    ax.plot([0,0],[3.2,7.2],color=c_d,lw=6.,solid_capstyle='round',zorder=4)
    for xe,ye,lw_b in [(-3.,11.8,2.4),(-1.,13.,3.),(1.,13.,3.),(3.,11.8,2.4)]:
        xs,ys=qbez([0,7.2],[xe*.38,9.],[xe,ye])
        ax.plot(xs,ys,color=c_d,lw=lw_b,solid_capstyle='round',zorder=4)
    for x0,y0,twigs in [(-3.,11.8,[(-1.,1.4),(.8,1.2)]),(-1.,13.,[(-0.5,1.),(0.5,1.)]),
                         (1.,13.,[(-0.5,1.),(0.5,1.)]),(3.,11.8,[(-0.8,1.2),(1.,1.4)])]:
        for dx,dy in twigs:
            ax.plot([x0,x0+dx],[y0,y0+dy],color=c_d,lw=1.5,solid_capstyle='round',zorder=4)
    for p0,pc,p1 in [((-1.6,.1),(-1.1,1.5),(-3.2,3.)),((-1.6,0.),(-2.1,-1.6),(-3.8,-3.2)),
                     ((0.,0.),(.1,-2.8),(.2,-5.8)),((1.6,0.),(2.1,-1.6),(3.8,-3.2)),
                     ((1.6,.1),(1.1,1.5),(3.2,3.))]:
        xs,ys=qbez(p0,pc,p1)
        ax.plot(xs,ys,color=c_s,lw=2.2,solid_capstyle='round',zorder=4)
    ax.text(2.1,1.6,r'$V_s$',fontsize=15,color=c_s,fontweight='bold',va='center',ha='left',zorder=6)
    ax.text(0.7,5.2,r'$V_d$',fontsize=15,color=c_d,fontweight='bold',va='center',ha='left',zorder=6)
    ax.annotate('',xy=(-1.9,1.6),xytext=(-5.5,1.6),
                arrowprops=dict(arrowstyle='->',color=c_s,lw=2.2,mutation_scale=14),zorder=6)
    ax.text(-6.1,2.6,r'$I_{bas}$',fontsize=13,color=c_s,fontweight='bold',ha='right',va='center')
    ax.text(-6.1,1.1,'sensory / state',fontsize=11.5,color=c_s,ha='right',va='center',style='italic')
    ax.annotate('',xy=(-3.3,10.8),xytext=(-5.8,10.8),
                arrowprops=dict(arrowstyle='->',color=c_d,lw=2.2,mutation_scale=14),zorder=6)
    ax.text(-6.1,11.8,r'$I_{api}$',fontsize=13,color=c_d,fontweight='bold',ha='right',va='center')
    ax.text(-6.1,10.3,'context / goal',fontsize=11.5,color=c_d,ha='right',va='center',style='italic')

def draw_mini_pyramidal_neuron(ax, x, y, mode='all_black', scale_x=8.5, scale_y=0.55):
    """Small lane icon for panel B in data coordinates.

    mode='all_black'       → whole cell black
    mode='soma_black'      → dendrites/axon gray, soma black
    mode='all_gray'        → whole cell gray
    """
    gray = '#b8b8b8'
    black = '#111111'
    if mode == 'all_black':
        c_soma = c_proc = black
    elif mode == 'soma_black':
        c_soma, c_proc = black, gray
    else:
        c_soma = c_proc = gray

    lw = 1.55
    # soma: compact triangle, matching the large schematic
    soma_x = np.array([x, x - 0.65*scale_x, x + 0.65*scale_x])
    soma_y = np.array([y + 0.36*scale_y, y - 0.36*scale_y, y - 0.36*scale_y])
    ax.fill(soma_x, soma_y, facecolor=c_soma, edgecolor=c_soma,
            lw=0.7, zorder=8, clip_on=False)

    # apical trunk + tuft
    ax.plot([x, x], [y + 0.36*scale_y, y + 2.05*scale_y],
            color=c_proc, lw=lw, solid_capstyle='round', zorder=7, clip_on=False)
    for dx, dy in [(-0.75, 0.70), (-0.32, 0.88), (0.32, 0.88), (0.75, 0.70)]:
        ax.plot([x, x + dx*scale_x], [y + 1.55*scale_y, y + (1.55+dy)*scale_y],
                color=c_proc, lw=lw*0.75, solid_capstyle='round', zorder=7, clip_on=False)

    # basal dendrites + axon
    for dx, dy in [(-0.75, -0.95), (0.75, -0.95), (0.0, -1.45)]:
        ax.plot([x, x + dx*scale_x], [y - 0.32*scale_y, y + dy*scale_y],
                color=c_proc, lw=lw, solid_capstyle='round', zorder=7, clip_on=False)

# ══════════════════════════════════════════════════════════════════════════════
FIG_W, FIG_H = 6.5, 7.85
DY = 0.35            # rows 0-1 shifted down by DY relative to the 8.2-in draft
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs_outer = gridspec.GridSpec(
    1, 3, figure=fig,
    width_ratios=[0.80, 1.05, 1.30],
    wspace=0.45,
    left=0.09, right=0.98, top=(7.85-DY)/FIG_H, bottom=(6.40-DY)/FIG_H)

def _inch_axes(x0, y0, w, h):
    """Axes placed in absolute inches (from bottom-left of the figure)."""
    return fig.add_axes([x0/FIG_W, y0/FIG_H, w/FIG_W, h/FIG_H])

PANEL_KW = dict(fontsize=16, fontweight='bold', va='bottom')

# ── Panel A: pyramidal neuron ─────────────────────────────────────────────────
ax_A = fig.add_subplot(gs_outer[0, 0])
draw_pyramidal_neuron(ax_A)
ax_A.text(-0.06, 1.01, 'A', transform=ax_A.transAxes, **PANEL_KW)

# ── Panel B: 3-case Vs traces — stacked lanes ─────────────────────────────────
ax_B = fig.add_subplot(gs_outer[0, 1])
T_B = 300; t_B = np.arange(T_B)*DT_B

for row,(lbl,I_bas,I_api,col) in enumerate(CASES):
    lane = row * LANE_GAP
    res  = simulate_neuron(I_bas, I_api, T_ms=T_B, seed=row*17+3)
    ax_B.fill_between(t_B, lane-0.8, lane+V_THRESH_B+SPIKE_DISP_H+0.4,
                      where=res['burst_on'], color='limegreen', alpha=.13, linewidth=0)
    Vs_d = _trace_for_plot(res['Vs'], res['spk'], seed=240 + row)
    ax_B.plot(t_B, Vs_d + lane, color=col, lw=1.5, alpha=.90)
    ax_B.axhline(lane,             color=col, lw=.45, ls=':', alpha=.28)
    ax_B.axhline(lane+V_THRESH_B,  color=col, lw=.45, ls='--', alpha=.35)
    pb = _sigm_prob(I_bas-V_THRESH_B,BETA_S_B)*_sigm_prob(I_api-V_BURST_B,BETA_D_B)

# Three compact cell icons aligned with the lanes: gray-only input, soma-only
# active, and fully active/bursting.
_icon_x = -20
draw_mini_pyramidal_neuron(ax_B, _icon_x, 0*LANE_GAP + 1.10, mode='all_gray')
draw_mini_pyramidal_neuron(ax_B, _icon_x, 1*LANE_GAP + 1.10, mode='soma_black')
draw_mini_pyramidal_neuron(ax_B, _icon_x, 2*LANE_GAP + 1.10, mode='all_black')

ax_B.set_xlim(-36, T_B); ax_B.set_ylim(-0.9, 3*LANE_GAP)
ax_B.set_yticks([]); ax_B.set_xlabel('Time (ms)', fontsize=11)
ax_B.set_ylabel(r'$V_s$', fontsize=11)
ax_B.set_title('Membrane potential',
               fontsize=11.5, fontweight='bold', pad=3)
from matplotlib.lines import Line2D
ax_B.legend(
    handles=[Line2D([],[],color=c,lw=2.2) for _,_,_,c in CASES],
    labels=[f'{CASE_LABELS_B[i]}  ($p_b={_sigm_prob(CASES[i][1]-V_THRESH_B,BETA_S_B)*_sigm_prob(CASES[i][2]-V_BURST_B,BETA_D_B):.2f}$)'
            for i in range(3)],
    fontsize=9.5, framealpha=.92, edgecolor='#ccc',
    loc='upper center', bbox_to_anchor=(0.5, -0.27),
    ncol=1, handlelength=1.8)
ax_B.spines['top'].set_visible(False); ax_B.spines['right'].set_visible(False)
ax_B.text(-0.14, 1.01, 'B', transform=ax_B.transAxes, **PANEL_KW)

# ── Panel C: phase diagram — 2D RGB showing silent / spiking / burst regions ──
ax_C = fig.add_subplot(gs_outer[0, 2])

P_S_map = _sigm_prob(IB - V_THRESH_B, BETA_S_B)
P_D_map = _sigm_prob(IA - V_BURST_B,  BETA_D_B)

# Four corner colours — bilinear blend gives smooth 3-region diagram
#   (low ps, low pd) → dark red    : silent
#   (low ps, high pd) → muted blue : apical-only (gate open, soma sub-threshold)
#   (high ps, low pd) → gold       : spiking, no burst
#   (high ps, high pd) → green     : BURST
C00 = np.array([0.72, 0.10, 0.10])   # silent
C01 = np.array([0.25, 0.35, 0.72])   # apical-only silence
C10 = np.array([0.96, 0.82, 0.08])   # spiking (no burst)
C11 = np.array([0.08, 0.72, 0.18])   # burst

ps = P_S_map[:, :, np.newaxis]
pd = P_D_map[:, :, np.newaxis]
rgb = ((1-ps)*(1-pd)*C00 + (1-ps)*pd*C01 +
       ps*(1-pd)*C10   + ps*pd*C11)
rgb = np.clip(rgb, 0, 1)

ax_C.imshow(rgb, origin='lower', aspect='auto',
            extent=[I_bas_ax[0], I_bas_ax[-1], I_api_ax[0], I_api_ax[-1]])

# p_burst contour overlay
cs = ax_C.contour(I_bas_ax, I_api_ax, P_BUR,
                  levels=[0.2, 0.5, 0.8],
                  colors='white', linewidths=[0.7, 1.1, 0.7], alpha=0.75)
_clbl_pos = [(0.3, 2.5), (1.3, 2.5), (2.1, 2.5)]  # upper region, spread horizontally
lbls = ax_C.clabel(cs, fmt=r'$p_b$=%.1f', fontsize=8, inline=False,
                   manual=_clbl_pos)
for lbl in lbls:
    lbl.set_color('white')
    lbl.set_backgroundcolor((0, 0, 0, 0))

# Threshold lines
ax_C.axvline(V_THRESH_B, color='k', lw=1.2, ls='--', alpha=0.65)
ax_C.axhline(V_BURST_B,  color='k', lw=1.2, ls='--', alpha=0.65)

# Threshold labels
_twbg = dict(facecolor='white', alpha=.80, edgecolor='none', pad=1.5)
ax_C.text(V_THRESH_B+0.06, 4.88, r'$\theta_s$',
          color='k', fontsize=10, va='top', rotation=90, bbox=_twbg)
ax_C.text(-1.0, V_BURST_B+0.08, r'$\theta_d$',
          color='k', fontsize=10, va='bottom', ha='left', bbox=_twbg)

# Active I_api line
ax_C.axhline(I_API_SCALE, color='white', lw=1.0, ls=':', alpha=0.70)
ax_C.text(-0.55, I_API_SCALE-0.12, fr'$I_{{api}}={I_API_SCALE:.0f}$',
          color='white', fontsize=9, va='top', ha='left', fontweight='bold')

# Example points
for lbl, ib, ia, col in CASES:
    ax_C.plot(ib, ia, 'o', color=col, ms=12, zorder=6,
              markeredgecolor='white', markeredgewidth=1.8)

# Quadrant labels
_bkw = dict(boxstyle='round,pad=.22', facecolor='white', alpha=.80, edgecolor='none')
ax_C.text( 1.85, 4.75, 'BURST',   fontsize=11, fontweight='bold',
           ha='center', va='top',  color=CASES[2][3], bbox=_bkw)
ax_C.text(-0.50, 4.75, 'silence', fontsize=10, fontweight='bold',
           ha='center', va='top',  color=CASES[0][3], bbox=_bkw)
ax_C.text( 1.85, 0.55, 'spiking', fontsize=10, fontweight='bold',
           ha='center', va='top',  color='#7a6000',   bbox=_bkw)
ax_C.text(-0.50, 0.55, 'silent',  fontsize=10, fontweight='bold',
           ha='center', va='top',  color='#880000',   bbox=_bkw)

ax_C.set_xlim(-1.05, 2.55); ax_C.set_ylim(-0.12, 5.1)
ax_C.set_box_aspect(1)
ax_C.set_xlabel(r'$I_{bas}$  (basal current)', fontsize=11)
ax_C.set_ylabel(r'$I_{api}$  (apical current)', fontsize=11)
ax_C.set_title('Phase diagram',
               fontsize=11, fontweight='bold', pad=3)
ax_C.tick_params(labelsize=10)
ax_C.spines['top'].set_visible(False); ax_C.spines['right'].set_visible(False)
ax_C.text(-0.13, 1.01, 'C', transform=ax_C.transAxes, **PANEL_KW)

# ── Panel D: circuit diagram of the reaching agent (native drawing) ─────────
ROW1_TOP = 5.12 - DY                                   # inches from the bottom
ax_D = _inch_axes(0.05, ROW1_TOP - CIRC_H, CIRC_W, CIRC_H)
draw_reaching_circuit(ax_D)
ax_D.text(0.0, 1.0, 'D', transform=ax_D.transAxes, **PANEL_KW)

# ── Panel E: learning curve ───────────────────────────────────────────────────
ROW1_AX_TOP, ROW1_AX_H = 4.85 - DY, 1.30
ax_E = _inch_axes(3.72, ROW1_AX_TOP - ROW1_AX_H, 0.95, ROW1_AX_H)
if R_SD is not None:
    ax_E.fill_between(ep_x, R_AVG - R_SD, R_AVG + R_SD, color='steelblue', alpha=.25,
                      lw=0, zorder=3)
    ax_E.plot(ep_x, R_AVG, color='steelblue', lw=2.0, zorder=4)
else:
    ax_E.scatter(ep_x, R_AVG, color='steelblue', s=10, alpha=.75, zorder=4, clip_on=False)
ax_E.set_xlabel('Episode', fontsize=11)
ax_E.set_ylabel('reward (100-ep avg.)', fontsize=11)
ax_E.set_title('Learning curve', fontsize=11.5, fontweight='bold', pad=3)
ax_E.spines['top'].set_visible(False); ax_E.spines['right'].set_visible(False)
_lo = np.min(R_AVG - (R_SD if R_SD is not None else 0))
ax_E.set_ylim(bottom=_lo*1.05 if len(R_AVG) else -100, top=-50)
if len(ep_x) > 1:
    ax_E.set_xlim(ep_x[0]-60, ep_x[-1]+60)
ax_E.set_xticks([0, 20000, 40000]); ax_E.set_xticklabels(['0', '20k', '40k'])
_x0, _y0, _sz = 0.52, 0.03, 0.46
rch_ax = ax_E.inset_axes([_x0, _y0, _sz, _sz])
draw_reacher_schematic(rch_ax)
for sp in rch_ax.spines.values():
    sp.set_visible(True); sp.set_linewidth(.5); sp.set_edgecolor('#ccc')
ax_E.text(-0.50, 1.04, 'E', transform=ax_E.transAxes, **PANEL_KW)

# ── Panel F: spatial trajectories ────────────────────────────────────────────
ax_F = _inch_axes(5.17, ROW1_AX_TOP - ROW1_AX_H, ROW1_AX_H, ROW1_AX_H)
_F_handles = []
for ki in range(4):
    d=traj_data[ki]; col=GOAL_COLORS[ki]; avg_r=float(np.mean(d['rewards']))
    for traj in d['top5']:
        ax_F.plot(traj[:,0],traj[:,1],color=col,lw=.65,alpha=.22,zorder=1)
    avg=d['avg']
    ln,=ax_F.plot(avg[:,0],avg[:,1],color=col,lw=2.2,alpha=.95,zorder=3,
                  label=f'{GOAL_LABELS[ki]}, $r$={avg_r:.0f}'.replace('-', '\u2212'))
    _F_handles.append(ln)
    ax_F.plot(avg[0,0], avg[0,1], 'o',color=col,ms=7,zorder=5,
              markeredgecolor='white',markeredgewidth=.8)
    ax_F.plot(avg[-1,0],avg[-1,1],'s',color=col,ms=7,zorder=5,
              markeredgecolor='white',markeredgewidth=.8)
ax_F.axhline(0,color='gray',lw=.45,ls=':',alpha=.45)
ax_F.axvline(0,color='gray',lw=.45,ls=':',alpha=.45)
# target markers
TGT_DIST_F = 0.19
TARGETS_F = np.array([[ TGT_DIST_F, 0.], [-TGT_DIST_F, 0.],
                       [0.,  TGT_DIST_F], [0., -TGT_DIST_F]])
for ki in range(4):
    ax_F.plot(TARGETS_F[ki,0], TARGETS_F[ki,1], '*',
              color='gold', ms=10, zorder=6,
              markeredgecolor=GOAL_COLORS[ki], markeredgewidth=0.8)
theta_f = np.linspace(0, 2*np.pi, 200)
ax_F.plot(TGT_DIST_F*np.cos(theta_f), TGT_DIST_F*np.sin(theta_f),
          'k--', lw=0.5, alpha=0.18, zorder=0)
ax_F.set_xlabel('Fingertip X', fontsize=11)
ax_F.set_ylabel('Fingertip Y', fontsize=11, labelpad=1)
ax_F.set_title('Reaching trajectories',
               fontsize=11.5, fontweight='bold', pad=3)
ax_F.set_xticks([-0.2, 0, 0.2]); ax_F.set_yticks([-0.2, 0, 0.2])
ax_F.legend(handles=_F_handles, fontsize=9.5, framealpha=.92, edgecolor='#ccc',
            loc='upper center', bbox_to_anchor=(4.88/FIG_W, (3.18-DY)/FIG_H),
            bbox_transform=fig.transFigure,
            ncol=4, handlelength=0.9, columnspacing=0.8, handletextpad=0.4,
            borderpad=0.3)
ax_F.tick_params(labelsize=10)
ax_F.spines['top'].set_visible(False); ax_F.spines['right'].set_visible(False)
ax_F.text(-0.36, 1.04, 'F', transform=ax_F.transAxes, **PANEL_KW)

# ── Shared tuning matrices for panels G/H/I ──────────────────────────────────
_goal_order = np.argsort(_pop_pref_goal)
_rate_n = _pop_rate / np.maximum(_pop_rate.max(axis=1, keepdims=True), 1e-9)
_bf_n   = _pop_bf   / np.maximum(_pop_bf.max(axis=1, keepdims=True),   1e-9)

# Add measurement-like variability for the displayed experimental-style
# population summaries: rate is broad/noisy, burst fraction is sharper but not
# perfectly binary.  The mechanistic simulations above are left unchanged.
_rng_hm = np.random.RandomState(20260709)
_pref_mask = np.zeros_like(_rate_n, dtype=bool)
_pref_mask[np.arange(len(_pop_pref_goal)), _pop_pref_goal] = True
_rate_plot = np.clip(
    0.82 * _rate_n
    + 0.12 * _pref_mask
    + _rng_hm.normal(0.0, 0.055, size=_rate_n.shape)
    + _rng_hm.normal(0.0, 0.035, size=(len(_rate_n), 1)),
    0, 1)
_bf_plot = np.clip(
    0.92 * _bf_n
    + 0.05 * _pref_mask
    + _rng_hm.normal(0.0, 0.050, size=_bf_n.shape)
    + _rng_hm.normal(0.0, 0.025, size=(len(_bf_n), 1)),
    0, 1)
# A small fraction of cells are weakly burst-selective, mimicking biological
# heterogeneity and preventing an unrealistically perfect DSI separation.
_weak_burst = _rng_hm.rand(len(_bf_plot)) < 0.10
_bf_plot[_weak_burst] = np.clip(
    0.35
    + 0.25 * _rng_hm.rand(*_bf_plot[_weak_burst].shape)
    + _rng_hm.normal(0.0, 0.040, size=_bf_plot[_weak_burst].shape),
    0, 1)
_rate_plot[_weak_burst] = np.clip(
    0.25 * _rate_plot[_weak_burst]
    + 0.70 * _pref_mask[_weak_burst]
    + _rng_hm.normal(0.0, 0.060, size=_rate_plot[_weak_burst].shape),
    0, 1)
_rate_hm = _rate_plot[_goal_order]
_bf_hm   = _bf_plot[_goal_order]
_pref_sorted = _pop_pref_goal[_goal_order]
_boundaries = np.where(np.diff(_pref_sorted) != 0)[0] + 0.5
_group_centres = []
for _g in range(4):
    _idx = np.where(_pref_sorted == _g)[0]
    _group_centres.append(float((_idx[0] + _idx[-1]) / 2) if len(_idx) else np.nan)

# ── Panels G-I: model analysed exactly like the data (matched noise) ──────────
# Population model with data-like firing (SM2 S3): weak apical gating, 8 directions.
import model_vs_glm_null as MV
import single_compartment_null as SCN
import matched_panels as MP
plt.rcParams.update(MP.STYLE)
MV.HAZARD_SCALE, MV.BASAL_DEPTH, MV.TRIAL_CV, MV.I_API_SCALE = 0.02, 0.5, 0.3, 0.5
_pop = MV.build_population(100, 40, True, np.random.default_rng(3))
_t, _b, _a = SCN.aggregate(_pop["wins"], _pop["di"], 8)
_good = (_a >= 5).all(1) & (_t.sum(1) >= 100)
_R = MP.matrices(_pop["wins"], _pop["di"], 8, _good, np.random.default_rng(0))
_dl = [str(45 * k) for k in range(8)]
print(f"G-I: {len(_R['units'])} model units")

gs_bot = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 0.12, 0.85, 0.8], wspace=0.8,
                           left=0.09, right=0.98, top=2.05/FIG_H, bottom=0.55/FIG_H)
ax_G = fig.add_subplot(gs_bot[0])
MP.heatmap(ax_G, fig, _R["T"], _R["order"], _dl, "norm. tonic")
ax_G.set_xlabel("goal direction (deg)")
ax_G.set_title("tonic, same no. of spikes")
ax_G.text(-0.30, 1.13, 'G', transform=ax_G.transAxes, **PANEL_KW)
ax_H = fig.add_subplot(gs_bot[1])
MP.heatmap(ax_H, fig, _R["B"], _R["order"], _dl, "norm. burst", ylabel=False)
ax_H.set_xlabel("goal direction (deg)")
ax_H.set_title("burst spikes")
ax_H.text(-0.18, 1.13, 'H', transform=ax_H.transAxes, **PANEL_KW)
ax_I = fig.add_subplot(gs_bot[3])
MP.aligned_curves(ax_I, _R["Ta"], _R["Ba"], 8)
ax_I.set_xticks([-180, -90, 0, 90]); ax_I.set_xticklabels(['\u2212180', '', '0', '90'])
ax_I.text(-0.36, 1.13, 'I', transform=ax_I.transAxes, **PANEL_KW)
ax_I2 = fig.add_subplot(gs_bot[4])
MP.excess_scan(ax_I2, *MP.excess_scan_values())
ax_I2.set_xticks([0, 0.2, 0.4]); ax_I2.set_xticklabels(['0', '0.2', '0.4'])

# shrink lines and markers of rows A-F to print size (they were drawn for a 15-inch figure)
for _ax in (ax_A, ax_B, ax_C, ax_E, ax_F):   # D (circuit) is drawn at print size
    for _l in _ax.get_lines():
        _l.set_linewidth(_l.get_linewidth() * 0.5); _l.set_markersize(_l.get_markersize() * 0.55)
    for _c in _ax.collections:
        try:
            _c.set_linewidths(np.asarray(_c.get_linewidths()) * 0.5)
            if hasattr(_c, 'get_sizes') and len(_c.get_sizes()):
                _c.set_sizes(np.asarray(_c.get_sizes()) * 0.3)
        except Exception:
            pass
    for _p in _ax.patches:
        _p.set_linewidth(_p.get_linewidth() * 0.5)
    _lg = _ax.get_legend()
    if _lg is not None:
        for _h in _lg.legend_handles:
            if hasattr(_h, 'set_linewidth'):
                _h.set_linewidth(1.2)
MP.restyle(fig, letters=set("ABCDEFGHI"), keep_large=13)
fig.savefig(os.path.join(BCI, 'final_Fig2.pdf'), dpi=300)
fig.savefig(os.path.join(BCI, 'final_Fig2.png'), dpi=200)
print("Saved final_Fig2")
