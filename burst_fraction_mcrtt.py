"""
Burst-fraction vs tonic-rate direction selectivity — MC_RTT (DANDI 000129).

Adaptation notes vs MC_Maze:
- No delay / go cue: trial start = movement onset; use full 500 ms window.
- Target directions are continuous; binned to 8 x 45-deg sectors.
- Direction computed as angle of target position relative to workspace centre.
- Only 'train' split trials used.

Output: Fig0b_mcrtt.pdf
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

NWB_FILE   = 'mc_rtt_train.nwb'
WIN_START  = 0.0
WIN_END    = 0.500
WIN_DUR    = WIN_END - WIN_START
ISI_THRESH = 0.010
MIN_TRIALS = 5
N_DIRS     = 8

print('Loading NWB...')
io      = NWBHDF5IO(NWB_FILE, 'r')
nwb     = io.read()
trials  = nwb.trials.to_dataframe()
units   = nwb.units.to_dataframe()
n_units = len(units)

beh      = nwb.processing['behavior']
tgt_ts   = beh['target_pos']
cur_ts   = beh['cursor_pos']
ts_t0    = tgt_ts.starting_time
ts_rate  = tgt_ts.rate
tgt_data = tgt_ts.data[:]
cur_data = cur_ts.data[:]

# ── Keep only train-split trials ──────────────────────────────────────────────
trials = trials[trials['split'] == 'train'].copy()

# ── Compute reach direction for each trial ────────────────────────────────────
# Direction = angle of (target - cursor) at trial start; skip if already close
MIN_DIST = 20.0   # mm — exclude trials where monkey is already at target

def snap(arr, t):
    idx = int(round((t - ts_t0) * ts_rate))
    return arr[max(0, min(idx, len(arr) - 1))]

vecs   = np.array([snap(tgt_data, r.start_time) - snap(cur_data, r.start_time)
                   for _, r in trials.iterrows()])
dists  = np.linalg.norm(vecs, axis=1)
angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))

# Bin to N_DIRS sectors; drop already-at-target trials
bin_size  = 360 / N_DIRS
dir_quant = np.round(angles / bin_size) * bin_size
dir_quant = ((dir_quant + 180) % 360) - 180
trials['direction_deg'] = dir_quant.astype(int)
trials['reach_dist']    = dists
trials = trials[trials['reach_dist'] > MIN_DIST].copy()
directions     = sorted(trials['direction_deg'].unique())
n_dirs         = len(directions)
dir_angles_deg = np.array(directions, dtype=float)
print(f'  {n_dirs} direction bins, {len(trials)} train trials')

# ── Spike extraction ───────────────────────────────────────────────────────────
all_spk      = [np.array(units.iloc[u]['spike_times']) for u in range(n_units)]
total_count  = np.zeros((n_units, n_dirs))
burst_count  = np.zeros((n_units, n_dirs))
trial_counts = np.zeros((n_units, n_dirs))

for _, trial in trials.iterrows():
    d_val = trial['direction_deg']
    if d_val not in directions: continue
    di   = directions.index(d_val)
    t0   = trial['start_time'] + WIN_START
    t1   = trial['start_time'] + WIN_END
    f_vec = np.zeros(n_units)
    for u, spk in enumerate(all_spk):
        ws = spk[(spk >= t0) & (spk < t1)]
        ns = len(ws)
        if ns == 0: continue
        total_count[u, di]  += ns
        trial_counts[u, di] += 1
        if ns > 1:
            isi = np.diff(ws)
            nb  = np.concatenate([[False], isi < ISI_THRESH]).sum()
            burst_count[u, di] += nb

io.close()

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
print(f'  N = {N} neurons pass quality filter')

# ── DSI ────────────────────────────────────────────────────────────────────────
def dsi(X):
    mn, mx = X.min(1), X.max(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(mn + mx > 0, (mx - mn) / (mx + mn), 0.)

dsi_rate  = dsi(rate_g)
dsi_frac  = dsi(frac_g)
_, p_main = wilcoxon(dsi_frac, dsi_rate, alternative='greater')
pct_above = (dsi_frac > dsi_rate).mean() * 100
print(f'  Wilcoxon p = {p_main:.2e},  {pct_above:.0f}% neurons above diagonal')

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
rel_dirs     = (np.arange(n_dirs) - n_dirs // 2) * (360 // n_dirs)

COL_RATE = '#2166ac'
COL_FRAC = '#b2182b'

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# =============================================================================
# FIGURE — single row, 4 square panels (mirrors Fig0b)
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 4),
                         gridspec_kw=dict(wspace=0.45,
                                          width_ratios=[1, 1, 1, 1]))
fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.20)

tick_labels = [f'{int(d)}°' for d in directions]

for ax, mat, col, lbl in [
    (axes[0], am_rate, COL_RATE, 'A'),
    (axes[1], am_frac, COL_FRAC, 'B')]:
    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                   vmin=-2.5, vmax=2.5, interpolation='nearest')
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cb.set_label('z-score', fontsize=7); cb.ax.tick_params(labelsize=6)
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

ax = axes[2]
sc = ax.scatter(dsi_rate, dsi_frac, c=dsi_frac - dsi_rate,
                cmap='RdBu_r', vmin=-0.8, vmax=0.8,
                s=14, alpha=0.6, linewidths=0)
lim = max(dsi_rate.max(), dsi_frac.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.9, alpha=0.6)
ax.set_xlabel('DSI  tonic rate', fontsize=8)
ax.set_ylabel('DSI  burst fraction', fontsize=8)
ax.text(0.05, 0.93,
        f'p = {p_main:.1e}\n{pct_above:.0f}% above diag.',
        transform=ax.transAxes, fontsize=7,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85))
cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
cb.set_label('ΔDSI', fontsize=7); cb.ax.tick_params(labelsize=6)
despine(ax); ax.tick_params(labelsize=7); ax.set_box_aspect(1)
ax.text(-0.14, 1.04, 'C', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='bottom')

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
despine(ax); ax.set_box_aspect(1)
ax.text(-0.14, 1.04, 'D', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='bottom')

plt.savefig('Fig0b_mcrtt.pdf', bbox_inches='tight', dpi=200)
plt.close(fig)
print('Saved Fig0b_mcrtt.pdf')
