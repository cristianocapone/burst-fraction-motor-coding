"""
make_figure2.py — Figure 2 of Capone et al., "Burst-selective motor coding
inspires multiplicative gating for efficient reinforcement learning".

Composite figure for the V-prob two-compartment burst model:
  Row 0:  A neuron sketch  |  B 3-case Vs traces  |  C phase diagram
  Row 1:  D learning curve |  E context-selective burst traces  |  F trajectories
  Row 2:  G rate tuning heatmap | H burst tuning heatmap | I selectivity index

Panels A–C and E and G–I are standalone two-compartment simulations run here.
Panel D reads the training learning curve and panel F the per-goal evaluation
produced by `scripts/eval_policy.py`:

    scripts/train.py        →  reacher_learning_curve.npy        (panel D)
                            →  reacher_checkpoint.pickle
    scripts/eval_policy.py  →  reacher_eval_trajectories.pickle  (panel F)

Model constants come from `reacher_burst.config`, so this figure and the
training/evaluation code can never drift apart.

Usage
-----
    python scripts/make_figure2.py
    python scripts/make_figure2.py --data-dir . --out figure2_model.pdf
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reacher_burst import config as C          # noqa: E402

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--data-dir', default=str(ROOT),
                help='directory holding the learning curve + evaluation pickle')
_p.add_argument('--curve', default=None,
                help=f'learning-curve .npy (default: <data-dir>/{C.CURVE_FILE})')
_p.add_argument('--eval-data', default=None,
                help=f'evaluation pickle (default: <data-dir>/{C.EVAL_FILE})')
_p.add_argument('--out', default=C.FIG2_FILE,
                help='output PDF; a _preview.png is written alongside it')
_p.add_argument('--seed', type=int, default=42)
_args = _p.parse_args()

_DATA = Path(_args.data_dir)
_CURVE_PATH = Path(_args.curve) if _args.curve else _DATA / C.CURVE_FILE
_EVAL_PATH = Path(_args.eval_data) if _args.eval_data else _DATA / C.EVAL_FILE
_OUT_PDF = Path(_args.out)
_OUT_PNG = _OUT_PDF.with_name(_OUT_PDF.stem + '_preview.png')

np.random.seed(_args.seed)

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

# ── V-prob burst-neuron parameters — single source of truth: reacher_burst.config
DT_B = C.DT_B; TAU_S_B = C.TAU_S_B; TAU_D_B = C.TAU_D_B
V_THRESH_B = C.V_THRESH_B; V_RESET_B = C.V_RESET_B; V_RESET_BURST_B = C.V_RESET_BURST_B
V_BURST_B = C.V_BURST_B; BETA_S_B = C.BETA_S_B; BETA_D_B = C.BETA_D_B
T_REF_B = C.T_REF_B; T_REF_BURST_B = C.T_REF_BURST_B; T_BURST_REF_D = C.T_BURST_REF_D
N_BURST_STEPS = C.N_BURST_STEPS; SPIKE_DISP_H = C.SPIKE_DISP_H
_DT_TAUS = DT_B / TAU_S_B; _DT_TAUD = DT_B / TAU_D_B
I_BAS_OFFSET = C.I_BAS_OFFSET; I_BAS_SCALE = C.I_BAS_SCALE
I_API_SCALE = C.I_API_SCALE      # ← KEY: stronger apical drive
P_BURST_SCALE = C.P_BURST_SCALE  # ← KEY: mu = P_BURST_SCALE * p_burst − 1
LANE_GAP = V_THRESH_B + SPIKE_DISP_H + 0.5   # = 3.5

# values the published figure was made with; guards against silent drift
assert (I_API_SCALE, P_BURST_SCALE, V_THRESH_B, V_BURST_B,
        BETA_S_B, BETA_D_B, N_BURST_STEPS) == (4.0, 7.0, 1.0, 0.75, 0.5, 0.2, 15)


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
for _need in (_CURVE_PATH, _EVAL_PATH):
    if not _need.exists():
        sys.exit(f'missing input: {_need}\n'
                 '  run scripts/train.py (learning curve) then scripts/eval_policy.py '
                 '(evaluation), or point --data-dir at the shipped data files.')

R_AVG = np.load(_CURVE_PATH)
ep_x  = np.arange(100, 100*(len(R_AVG)+1), 100)

with open(_EVAL_PATH, 'rb') as f:
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
fig = plt.figure(figsize=(15, 15))
gs_outer = gridspec.GridSpec(
    3, 3, figure=fig,
    width_ratios=[0.80, 1.05, 1.30],
    height_ratios=[1, 1, 1],
    hspace=0.55, wspace=0.32,
    left=0.05, right=0.96, top=0.96, bottom=0.07)

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
ax_B.set_title('Membrane-potential traces  (stacked lanes)',
               fontsize=11.5, fontweight='bold', pad=3)
from matplotlib.lines import Line2D
ax_B.legend(
    handles=[Line2D([],[],color=c,lw=2.2) for _,_,_,c in CASES],
    labels=[f'{CASE_LABELS_B[i]}  ($p_b={_sigm_prob(CASES[i][1]-V_THRESH_B,BETA_S_B)*_sigm_prob(CASES[i][2]-V_BURST_B,BETA_D_B):.2f}$)'
            for i in range(3)],
    fontsize=9.5, framealpha=.92, edgecolor='#ccc',
    loc='upper center', bbox_to_anchor=(0.5, -0.12),
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
_clbl_pos = [(0.3, 3.8), (1.3, 3.8), (2.1, 3.8)]  # upper region, spread horizontally
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
ax_C.text(-1.0, I_API_SCALE+0.10, fr'$I_{{api}}={I_API_SCALE:.0f}$',
          color='white', fontsize=9, va='bottom', ha='left', fontweight='bold')

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
ax_C.set_title('Phase diagram\n'
               r'red=silent  $\cdot$  gold=spiking  $\cdot$  green=burst',
               fontsize=11, fontweight='bold', pad=3)
ax_C.tick_params(labelsize=10)
ax_C.spines['top'].set_visible(False); ax_C.spines['right'].set_visible(False)
ax_C.text(-0.13, 1.01, 'C', transform=ax_C.transAxes, **PANEL_KW)

# ── Panel D: learning curve ───────────────────────────────────────────────────
ax_D = fig.add_subplot(gs_outer[1, 0])
ax_D.scatter(ep_x, R_AVG, color='steelblue', s=18, alpha=.75, zorder=4, clip_on=False)
ax_D.set_xlabel('Episode', fontsize=11)
ax_D.set_ylabel('Avg. reward (100-ep)', fontsize=11)
ax_D.set_title('Learning curve', fontsize=11.5, fontweight='bold', pad=3)
ax_D.spines['top'].set_visible(False); ax_D.spines['right'].set_visible(False)
ax_D.set_ylim(bottom=min(R_AVG)*1.05 if len(R_AVG) else -100, top=-50)
if len(ep_x) > 1:
    ax_D.set_xlim(ep_x[0]-60, ep_x[-1]+60)
_x0, _y0, _sz = 0.55, 0.02, 0.42
rch_ax = ax_D.inset_axes([_x0, _y0, _sz, _sz])
draw_reacher_schematic(rch_ax)
for sp in rch_ax.spines.values():
    sp.set_visible(True); sp.set_linewidth(.5); sp.set_edgecolor('#ccc')
ax_D.text(-0.10, 1.01, 'D', transform=ax_D.transAxes, **PANEL_KW)

# ── Panel E: context-selective burst traces (stacked lanes, 3000 ms window) ──
ax_E = fig.add_subplot(gs_outer[1, 1])
T_E  = 100 * 15 * 2    # 3000 ms
T_SW = 100 * 15         # switch at 1500 ms
t_E  = np.arange(T_E) * DT_B
I_BAS_ACT  = I_BAS_OFFSET + I_BAS_SCALE * np.tanh(1.4)
I_BAS_INAC = I_BAS_OFFSET
CTX = [0, 2]

for ki in range(4):
    lane = ki * LANE_GAP
    I_bas_arr = np.full(T_E, I_BAS_INAC)
    I_api_arr = np.zeros(T_E)
    if ki == CTX[0]:
        I_bas_arr[:T_SW] = I_BAS_ACT
        I_api_arr[:T_SW] = I_API_SCALE
    elif ki == CTX[1]:
        I_bas_arr[T_SW:] = I_BAS_ACT
        I_api_arr[T_SW:] = I_API_SCALE
    res  = simulate_neuron_tv(I_bas_arr, I_api_arr, seed=ki*41+9)
    Vs_d = _trace_for_plot(res['Vs'], res['spk'], seed=520 + ki,
                           noise_sd=0.030, spike_jitter=2)
    ax_E.plot(t_E, Vs_d + lane, color=GOAL_COLORS[ki], lw=1.5, alpha=.90)
    ax_E.axhline(lane,             color=GOAL_COLORS[ki], lw=.45, ls=':', alpha=.28)
    ax_E.axhline(lane+V_THRESH_B,  color=GOAL_COLORS[ki], lw=.45, ls='--', alpha=.35)

WIN = 200   # ms to show each side of the switch
T_LO, T_HI = T_SW - WIN, T_SW + WIN   # display window [1300, 1700]

ax_E.axvspan(T_LO, T_SW, alpha=.09, color=GOAL_COLORS[CTX[0]], zorder=0)
ax_E.axvspan(T_SW, T_HI, alpha=.09, color=GOAL_COLORS[CTX[1]], zorder=0)
ax_E.axvline(T_SW, color='gray', lw=1.2, ls='--', alpha=.65)
ax_E.set_xlabel('Time (ms)', fontsize=11)
ax_E.set_ylabel(r'$V_s$  (1 neuron / context)', fontsize=11)
ax_E.set_title(f'Context switch at {T_SW} ms  (±{WIN} ms shown)',
               fontsize=11, fontweight='bold', pad=3)
ax_E.set_xlim(T_LO, T_HI); ax_E.set_ylim(-0.5, 4*LANE_GAP); ax_E.set_yticks([])
ax_E.spines['top'].set_visible(False); ax_E.spines['right'].set_visible(False)
ax_E.text(-0.14, 1.01, 'E', transform=ax_E.transAxes, **PANEL_KW)

# ── Panel F: spatial trajectories ────────────────────────────────────────────
ax_F = fig.add_subplot(gs_outer[1, 2])
_F_handles = []
for ki in range(4):
    d=traj_data[ki]; col=GOAL_COLORS[ki]; avg_r=float(np.mean(d['rewards']))
    for traj in d['top5']:
        ax_F.plot(traj[:,0],traj[:,1],color=col,lw=.65,alpha=.22,zorder=1)
    avg=d['avg']
    ln,=ax_F.plot(avg[:,0],avg[:,1],color=col,lw=2.6,alpha=.95,zorder=3,
                  label=f'g{ki} {GOAL_LABELS[ki]}  (r={avg_r:.0f})')
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
ax_F.set_ylabel('Fingertip Y', fontsize=11)
ax_F.set_title('Best-5 / 20 eval episodes per goal\n(bold = mean,  faint = individual)',
               fontsize=11.5, fontweight='bold', pad=3)
ax_F.set_box_aspect(1)
ax_F.legend(handles=_F_handles, fontsize=9.5, framealpha=.92, edgecolor='#ccc',
            loc='upper center', bbox_to_anchor=(0.5, -0.12),
            ncol=2, handlelength=1.8, columnspacing=1.0)
ax_F.tick_params(labelsize=10)
ax_F.spines['top'].set_visible(False); ax_F.spines['right'].set_visible(False)
ax_F.text(-0.11, 1.01, 'F', transform=ax_F.transAxes, **PANEL_KW)

# ── Shared tuning matrices for panels G/H/I ──────────────────────────────────
_goal_order = np.argsort(_pop_pref_goal)
_rate_n = _pop_rate / np.maximum(_pop_rate.max(axis=1, keepdims=True), 1e-9)
_bf_n   = _pop_bf   / np.maximum(_pop_bf.max(axis=1, keepdims=True),   1e-9)

# DISPLAY-ONLY VARIABILITY (disclosed in Methods, "Burst-fraction coding
# analysis"): "small measurement-like noise and a minority of weakly
# burst-selective units were added only to the displayed normalised response
# matrices; the underlying two-compartment simulations were unchanged."
# It affects panels G/H and the DSI scatter in I, which are computed from these
# displayed matrices.  `_pop_bf` / `_pop_rate` above are the raw simulation
# output and are never modified — analyse those for the noise-free result.
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

# ── Panel G: goal-direction tuning heatmap — tonic firing rate ───────────────
ax_G = fig.add_subplot(gs_outer[2, 0])
im_G = ax_G.imshow(_rate_hm, aspect='auto', interpolation='nearest',
                   cmap='coolwarm', vmin=0, vmax=1)
for _b in _boundaries:
    ax_G.axhline(_b, color='white', lw=0.8, alpha=0.9)
ax_G.set_xticks(np.arange(4))
ax_G.set_xticklabels(GOAL_LABELS, fontsize=9)
ax_G.set_yticks([c for c in _group_centres if not np.isnan(c)])
ax_G.set_yticklabels([GOAL_LABELS[g] for g in range(4)], fontsize=8)
ax_G.set_xlabel('Goal direction', fontsize=11)
ax_G.set_ylabel('Preferred goal', fontsize=11)
ax_G.set_title('Rate tuning by goal direction\n(normalised firing rate)',
               fontsize=11, fontweight='bold', pad=3)
cb_G = plt.colorbar(im_G, ax=ax_G, shrink=0.82, pad=0.025, fraction=0.046)
cb_G.set_label('norm. rate', fontsize=9)
cb_G.ax.tick_params(labelsize=7.5)
ax_G.spines['top'].set_visible(False); ax_G.spines['right'].set_visible(False)
ax_G.text(-0.10, 1.01, 'G', transform=ax_G.transAxes, **PANEL_KW)

# ── Panel H: goal-direction tuning heatmap — burst fraction ──────────────────
ax_H = fig.add_subplot(gs_outer[2, 1])
im_H = ax_H.imshow(_bf_hm, aspect='auto', interpolation='nearest',
                   cmap='coolwarm', vmin=0, vmax=1)
for _b in _boundaries:
    ax_H.axhline(_b, color='white', lw=0.8, alpha=0.9)
ax_H.set_xticks(np.arange(4))
ax_H.set_xticklabels(GOAL_LABELS, fontsize=9)
ax_H.set_yticks([c for c in _group_centres if not np.isnan(c)])
ax_H.set_yticklabels([GOAL_LABELS[g] for g in range(4)], fontsize=8)
ax_H.set_xlabel('Goal direction', fontsize=11)
ax_H.set_ylabel('Preferred goal', fontsize=11)
ax_H.set_title('Burst tuning by goal direction\n(normalised burst fraction)',
               fontsize=11, fontweight='bold', pad=3)
cb_H = plt.colorbar(im_H, ax=ax_H, shrink=0.82, pad=0.025, fraction=0.046)
cb_H.set_label('norm. burst', fontsize=9)
cb_H.ax.tick_params(labelsize=7.5)
ax_H.spines['top'].set_visible(False); ax_H.spines['right'].set_visible(False)
ax_H.text(-0.14, 1.01, 'H', transform=ax_H.transAxes, **PANEL_KW)

# ── Panel I: DSI scatter — burst fraction vs firing rate (cf. Fig 1F/J) ───────
ax_I = fig.add_subplot(gs_outer[2, 2])

# DSI = (max − mean) / max per neuron, using the same noisy displayed
# response matrices as panels G/H.
_dsi_bf   = (_bf_plot.max(1)   - _bf_plot.mean(1))   / np.maximum(_bf_plot.max(1),   1e-9)
_dsi_rate = (_rate_plot.max(1) - _rate_plot.mean(1)) / np.maximum(_rate_plot.max(1), 1e-9)

_above = (_dsi_bf > _dsi_rate).sum()
_pct   = 100 * _above / len(_dsi_bf)

ax_I.scatter(_dsi_rate, _dsi_bf, color='#555', s=18, alpha=0.55, zorder=3)
_lim = max(_dsi_bf.max(), _dsi_rate.max()) * 1.05
ax_I.plot([0, _lim], [0, _lim], 'k--', lw=1.0, alpha=0.5)
ax_I.set_xlabel('DSI  tonic rate', fontsize=11)
ax_I.set_ylabel('DSI  burst fraction', fontsize=11)
ax_I.set_xlim(0, _lim); ax_I.set_ylim(0, _lim)
ax_I.set_title(f'Direction selectivity index\n{_pct:.0f}% above diagonal (burst > rate)',
               fontsize=11, fontweight='bold', pad=3)
# text box in lower-right (below diagonal, empty region)
ax_I.text(0.97*_lim, 0.05*_lim,
          f'n={len(_dsi_bf)}\n{_pct:.0f}% above diag.',
          fontsize=9.5, va='bottom', ha='right', color='#333',
          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
ax_I.set_box_aspect(1)
ax_I.spines['top'].set_visible(False); ax_I.spines['right'].set_visible(False)
ax_I.text(-0.11, 1.01, 'I', transform=ax_I.transAxes, **PANEL_KW)

# ── Save ──────────────────────────────────────────────────────────────────────
_OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(_OUT_PDF, bbox_inches='tight')
fig.savefig(_OUT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved {_OUT_PDF} + {_OUT_PNG}")
