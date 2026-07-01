"""
Publication-ready figure: burst fraction encodes reach direction
independently of tonic spike rate in MC_Maze motor cortex.

Panels (single row)
------
A  Activity matrix — tonic rate
B  Activity matrix — burst fraction
C  DSI scatter
D  Control: mean aligned tuning curves

Output: burst_fraction_summary.pdf
"""

import os, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pynwb import NWBHDF5IO

NWB_FILE   = 'mc_maze_train.nwb'
WIN_START  = 0.0; WIN_END = 0.500; WIN_DUR = WIN_END - WIN_START
ISI_THRESH = 0.010
MIN_TRIALS = 5

print('Loading NWB...')
io      = NWBHDF5IO(NWB_FILE, 'r')
nwb     = io.read()
trials  = nwb.trials.to_dataframe()
units   = nwb.units.to_dataframe()
n_units = len(units)

trial_angles = []
for _, row in trials.iterrows():
    tp = row['target_pos']; ai = int(row['active_target']); xy = tp[ai]
    trial_angles.append(np.degrees(np.arctan2(float(xy[1]), float(xy[0]))))
trial_angles = np.array(trial_angles)
dir_quant = np.round(trial_angles / 45) * 45
dir_quant = ((dir_quant + 180) % 360) - 180
trials['direction_deg'] = dir_quant.astype(int)
directions     = sorted(trials['direction_deg'].unique())
n_dirs         = len(directions)
dir_angles_deg = np.array(directions, dtype=float)

all_spk      = [np.array(units.iloc[u]['spike_times']) for u in range(n_units)]
total_count  = np.zeros((n_units, n_dirs))
burst_count  = np.zeros((n_units, n_dirs))
trial_counts = np.zeros((n_units, n_dirs))

for _, trial in trials.iterrows():
    d_val = trial['direction_deg']
    if d_val not in directions: continue
    di    = directions.index(d_val)
    t_ref = trial['go_cue_time']
    t0, t1 = t_ref + WIN_START, t_ref + WIN_END
    f_vec = np.zeros(n_units)
    for u, spk in enumerate(all_spk):
        ws = spk[(spk >= t0) & (spk < t1)]
        ns = len(ws)
        if ns == 0: continue
        total_count[u, di]  += ns
        trial_counts[u, di] += 1
        if ns > 1:
            isi = np.diff(ws)
            nb = np.concatenate([[False], isi < ISI_THRESH]).sum()
            burst_count[u, di] += nb
            f_vec[u] = nb / ns

tonic_count = total_count - burst_count
with np.errstate(invalid='ignore', divide='ignore'):
    mean_tonic = np.where(trial_counts > 0,
                          tonic_count / (trial_counts * WIN_DUR), 0.)
    mean_frac  = np.where(total_count > 0,
                          burst_count / total_count, 0.)

good   = ((trial_counts >= MIN_TRIALS).all(1)) & \
         ((mean_tonic + mean_frac).sum(1) > 0.2)
rate_g = mean_tonic[good]
frac_g = mean_frac[good]
N      = good.sum()
print(f'  N = {N} neurons, {n_dirs} directions')
io.close()

# ── DSI ────────────────────────────────────────────────────────────────────────
def dsi(X):
    mn, mx = X.min(1), X.max(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(mn + mx > 0, (mx - mn) / (mx + mn), 0.)

dsi_rate  = dsi(rate_g)
dsi_frac  = dsi(frac_g)
_, p_main = wilcoxon(dsi_frac, dsi_rate, alternative='greater')

# ── Activity matrices ──────────────────────────────────────────────────────────
pref_frac  = np.argmax(frac_g, axis=1)
pref_rate  = np.argmax(rate_g, axis=1)
sort_order = np.argsort(pref_frac)

def activity_matrix(tuning, sort_idx):
    mat = tuning[sort_idx]
    mu  = mat.mean(1, keepdims=True)
    sd  = mat.std(1,  keepdims=True)
    return (mat - mu) / (sd + 1e-9)

am_rate = activity_matrix(rate_g, sort_order)
am_frac = activity_matrix(frac_g, sort_order)

# ── Mean aligned tuning curves ─────────────────────────────────────────────────
def aligned_tuning(tuning, pref_idx):
    aligned = np.zeros_like(tuning)
    half    = n_dirs // 2
    for i in range(len(tuning)):
        rolled = np.roll(tuning[i], half - pref_idx[i])
        aligned[i] = rolled / (rolled.max() + 1e-9)
    return aligned

al_rate      = aligned_tuning(rate_g, pref_rate)
al_frac      = aligned_tuning(frac_g, pref_frac)
mean_rate_tc = al_rate.mean(0);  sem_rate_tc = al_rate.std(0) / np.sqrt(N)
mean_frac_tc = al_frac.mean(0);  sem_frac_tc = al_frac.std(0) / np.sqrt(N)
rel_dirs     = (np.arange(n_dirs) - n_dirs // 2) * 45

COL_RATE = '#2166ac'
COL_FRAC = '#b2182b'

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# =============================================================================
# FIGURE — single row, 4 square panels
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 4),
                         gridspec_kw=dict(wspace=0.45,
                                         width_ratios=[1, 1, 1, 1]))
fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.20)

tick_labels = [f'{int(d)}°' for d in directions]

# ── A & B: activity matrices ───────────────────────────────────────────────────
for ax, mat, col, lbl in [
    (axes[0], am_rate, COL_RATE, 'A'),
    (axes[1], am_frac, COL_FRAC, 'B')]:

    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                   vmin=-2.5, vmax=2.5, interpolation='nearest')
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cb.set_label('z-score', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_xticks(range(n_dirs))
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_xlabel('Direction', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(1.5)
    ax.set_box_aspect(1)
    ax.text(-0.10, 1.04, lbl, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom')

axes[0].set_ylabel('Neuron', fontsize=8)

# ── C: DSI scatter ─────────────────────────────────────────────────────────────
ax = axes[2]
sc = ax.scatter(dsi_rate, dsi_frac, c=dsi_frac - dsi_rate,
                cmap='RdBu_r', vmin=-0.8, vmax=0.8,
                s=14, alpha=0.6, linewidths=0)
lim = max(dsi_rate.max(), dsi_frac.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.9, alpha=0.6)
ax.set_xlabel('DSI  tonic rate', fontsize=8)
ax.set_ylabel('DSI  burst fraction', fontsize=8)
ax.text(0.05, 0.93,
        f'p = {p_main:.1e}\n{(dsi_frac > dsi_rate).mean()*100:.0f}% above diag.',
        transform=ax.transAxes, fontsize=7,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85))
cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
cb.set_label('ΔDSI', fontsize=7); cb.ax.tick_params(labelsize=6)
despine(ax); ax.tick_params(labelsize=7)
ax.set_box_aspect(1)
ax.text(-0.14, 1.04, 'C', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='bottom')

# ── D: Mean aligned tuning curves ─────────────────────────────────────────────
ax = axes[3]
for mean, sem, col, lbl in [
    (mean_rate_tc, sem_rate_tc, COL_RATE, 'Tonic rate'),
    (mean_frac_tc, sem_frac_tc, COL_FRAC, 'Burst fraction')]:
    ax.plot(rel_dirs, mean, 'o-', color=col, lw=2.0, ms=4, label=lbl)
    ax.fill_between(rel_dirs, mean - sem, mean + sem, color=col, alpha=0.15)

ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Offset from preferred (deg)', fontsize=8)
ax.set_ylabel('Normalised response', fontsize=8)
ax.set_xticks(rel_dirs)
ax.set_xticklabels([f'{int(d)}' for d in rel_dirs], fontsize=7, rotation=45)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc='lower center', ncol=2, framealpha=0.85)
despine(ax)
ax.set_box_aspect(1)
ax.text(-0.14, 1.04, 'D', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='bottom')

plt.savefig('Fig0b.pdf', bbox_inches='tight', dpi=200)
plt.close(fig)
print('Saved burst_fraction_summary.pdf')
