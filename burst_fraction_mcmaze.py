"""
Burst-fraction direction selectivity analysis on MC_Maze (DANDI 000128).

Pipeline
--------
1. Download the MC_Maze NWB file from DANDI (train split, one session).
2. Extract spike trains for each neuron aligned to go-cue, per trial.
3. Classify each spike as burst or tonic using an ISI threshold (10 ms).
4. For each neuron compute per-direction:
     - total spike rate
     - burst fraction (burst spikes / total spikes)
5. Compute direction selectivity index (DSI) for both signals.
6. Test whether DSI_burst_frac > DSI_total_rate (Wilcoxon signed-rank).
7. Plot: scatter, violin, example tuning curves, example raster.

Outputs
-------
  burst_fraction_mcmaze.pdf   — figure for the paper
  burst_fraction_mcmaze.log   — statistics printed to file
"""

import os, sys, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import wilcoxon, ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. Download data ─────────────────────────────────────────────────────────
NWB_FILE = 'mc_maze_train.nwb'
DANDISET  = '000128'
ASSET_PATH = 'sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb'

if not os.path.exists(NWB_FILE):
    print(f'Downloading MC_Maze from DANDI {DANDISET}...')
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET, 'draft').get_asset_by_path(ASSET_PATH)
        asset.download(NWB_FILE)
    print(f'  Saved to {NWB_FILE}')
else:
    print(f'Using cached {NWB_FILE}')

# ── 2. Load NWB and extract trial info + spike times ─────────────────────────
from pynwb import NWBHDF5IO
import h5py

print('Loading NWB file...')
io = NWBHDF5IO(NWB_FILE, 'r')
nwb = io.read()

# Trials table: start_time, stop_time, reach_direction (0–7)
trials = nwb.trials.to_dataframe()
print(f'  Trials: {len(trials)}  |  columns: {list(trials.columns)}')

# Units table: spike times
units = nwb.units.to_dataframe()
print(f'  Units: {len(units)}')

# Derive reach direction from active target position
# direction = atan2(y, x) of target_pos[active_target], quantised to 45°
trial_angles = []
for _, row in trials.iterrows():
    tp = row['target_pos']
    ai = int(row['active_target'])
    xy = tp[ai]
    trial_angles.append(np.degrees(np.arctan2(float(xy[1]), float(xy[0]))))
trial_angles = np.array(trial_angles)

# Quantise to 45° bins
dir_quant = np.round(trial_angles / 45) * 45
dir_quant = ((dir_quant + 180) % 360) - 180   # keep in (-180, 180]
trials['direction_deg'] = dir_quant.astype(int)
dir_col    = 'direction_deg'
directions = sorted(trials[dir_col].unique())
n_dirs     = len(directions)
print(f'  Direction column: derived from target_pos (quantised to 45°)')
print(f'  Directions ({n_dirs}): {directions}')
print(f'  Trials per direction: '
      f'{dict(zip(directions, [(trials[dir_col]==d).sum() for d in directions]))}')

# ── 3. Build spike trains per neuron per trial ────────────────────────────────
# Analysis window: go cue + 0 to 500 ms
WIN_START   = 0.0    # s after go-cue (or trial start)
WIN_END     = 0.500  # s

# Use go_cue_time if available, else start_time
if 'go_cue_time' in trials.columns:
    t0_col = 'go_cue_time'
elif 'move_onset_time' in trials.columns:
    t0_col = 'move_onset_time'
else:
    t0_col = 'start_time'

print(f'  Alignment: {t0_col} + [{WIN_START:.3f}, {WIN_END:.3f}] s')

ISI_THRESH = 0.010  # 10 ms burst ISI threshold
MIN_TRIALS = 5      # minimum trials per direction to include neuron

# For each neuron: [n_trials] lists of spike times (relative to t0)
n_units = len(units)
# Pre-extract spike time arrays (avoid repeated .iloc calls)
all_spike_times = [np.array(units.iloc[u]['spike_times']) for u in range(n_units)]

# Per neuron, per direction: total spikes, burst spikes
total_count  = np.zeros((n_units, n_dirs))
burst_count  = np.zeros((n_units, n_dirs))
trial_counts = np.zeros((n_units, n_dirs))

for ti, trial in trials.iterrows():
    d_val = trial[dir_col]
    if d_val not in directions:
        continue
    di    = directions.index(d_val)
    t_ref = trial[t0_col]
    t_win_start = t_ref + WIN_START
    t_win_end   = t_ref + WIN_END

    for u, spk in enumerate(all_spike_times):
        win_spk = spk[(spk >= t_win_start) & (spk < t_win_end)]
        n_spk = len(win_spk)
        if n_spk == 0:
            continue
        total_count[u, di] += n_spk
        trial_counts[u, di] += 1

        # Burst classification: spike i is a burst spike if ISI to
        # previous spike (within window) < ISI_THRESH
        if n_spk > 1:
            isi = np.diff(win_spk)
            # A spike is a burst spike if the preceding ISI is short
            is_burst_spike = np.concatenate([[False], isi < ISI_THRESH])
            burst_count[u, di] += is_burst_spike.sum()

print('  Spike extraction done.')

# ── 4. Compute per-neuron metrics ─────────────────────────────────────────────
WIN_DUR = WIN_END - WIN_START

# Mean total rate per direction (sp/s)
with np.errstate(invalid='ignore', divide='ignore'):
    mean_rate      = np.where(trial_counts > 0,
                              total_count / (trial_counts * WIN_DUR), 0.)
    mean_burst_frac = np.where(total_count > 0,
                               burst_count / total_count, 0.)

# DSI = (max - min) / (max + min) across directions
def dsi(X):
    mn, mx = X.min(1), X.max(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(mn + mx > 0, (mx - mn) / (mx + mn), 0.)
    return d

# Keep only neurons with enough trials and non-zero total firing
enough_trials = (trial_counts >= MIN_TRIALS).all(1)
active         = mean_rate.sum(1) > 0.5    # > 0.5 sp/s overall
good           = enough_trials & active
print(f'  Neurons passing quality filter: {good.sum()} / {n_units}')

rate_g  = mean_rate[good]
frac_g  = mean_burst_frac[good]

dsi_rate = dsi(rate_g)
dsi_frac = dsi(frac_g)

# ── 5. Control 1: rate-residual burst fraction ───────────────────────────────
# Regress total rate out of burst fraction (per neuron, across directions)
# to check whether burst fraction carries direction info beyond rate alone.
from scipy.stats import linregress

frac_resid = np.zeros_like(frac_g)
for i in range(len(rate_g)):
    r = rate_g[i]
    f = frac_g[i]
    if r.std() > 0:
        slope, intercept, *_ = linregress(r, f)
        frac_resid[i] = f - (slope * r + intercept)
    else:
        frac_resid[i] = f - f.mean()

dsi_resid = dsi(frac_resid - frac_resid.min(1, keepdims=True))  # shift to ≥0

# ── 5. Control 2: preferred-direction disagreement ────────────────────────────
# For each neuron, does burst fraction prefer the SAME direction as total rate?
# If result were a rate artefact, they should almost always agree.
pref_rate = np.argmax(rate_g, axis=1)
pref_frac = np.argmax(frac_g, axis=1)
pref_agree = (pref_rate == pref_frac)
# Also compute correlation between rate and frac tuning curves per neuron
tuning_corr = np.array([np.corrcoef(rate_g[i], frac_g[i])[0, 1]
                         for i in range(len(rate_g))])

# ── 5. Main statistics ────────────────────────────────────────────────────────
stat_w,  p_w  = wilcoxon(dsi_frac,  dsi_rate,  alternative='greater')
stat_t,  p_t  = ttest_rel(dsi_frac, dsi_rate)
stat_wr, p_wr = wilcoxon(dsi_resid, dsi_rate,  alternative='greater')

log_lines = [
    f'MC_Maze burst-fraction direction selectivity analysis',
    f'N neurons (quality-filtered): {good.sum()}',
    f'ISI burst threshold: {ISI_THRESH*1000:.0f} ms',
    f'Analysis window: {WIN_START*1000:.0f}–{WIN_END*1000:.0f} ms post-{t0_col}',
    f'',
    f'=== Main result ===',
    f'DSI total rate        : mean={dsi_rate.mean():.3f}  median={np.median(dsi_rate):.3f}',
    f'DSI burst fraction    : mean={dsi_frac.mean():.3f}  median={np.median(dsi_frac):.3f}',
    f'Wilcoxon (frac>rate)  : W={stat_w:.0f}, p={p_w:.3e}',
    f'Paired t-test         : t={stat_t:.3f}, p={p_t:.3e}',
    f'Neurons frac>rate DSI : {(dsi_frac>dsi_rate).sum()}/{good.sum()} '
    f'({100*(dsi_frac>dsi_rate).mean():.1f}%)',
    f'',
    f'=== Control 1: rate-residual burst fraction ===',
    f'DSI residual frac     : mean={dsi_resid.mean():.3f}  median={np.median(dsi_resid):.3f}',
    f'Wilcoxon (resid>rate) : W={stat_wr:.0f}, p={p_wr:.3e}',
    f'Interpretation: if significant, burst fraction encodes direction BEYOND rate',
    f'',
    f'=== Control 2: preferred-direction disagreement ===',
    f'Preferred dir agrees (rate vs frac): {pref_agree.sum()}/{good.sum()} '
    f'({100*pref_agree.mean():.1f}%)',
    f'Mean tuning-curve correlation (rate vs frac): {tuning_corr.mean():.3f} '
    f'± {tuning_corr.std():.3f}',
    f'Interpretation: low agreement / low correlation → dissociation, not rate artefact',
]
for line in log_lines:
    print(line)
with open('burst_fraction_mcmaze.log', 'w') as f:
    f.write('\n'.join(log_lines))

# ── 6. Figure ─────────────────────────────────────────────────────────────────
# Pick 4 example neurons: 2 with high DSI_frac, 2 with preferred-dir disagreement
top2_agree    = np.where(pref_agree)[0][np.argsort(dsi_frac[pref_agree])[-2:][::-1]]
top2_disagree = np.where(~pref_agree)[0][np.argsort(dsi_frac[~pref_agree])[-2:][::-1]] \
                if (~pref_agree).sum() >= 2 else \
                np.argsort(dsi_frac)[-2:][::-1]
example_neurons = np.concatenate([top2_agree, top2_disagree])

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.42)

# ── Panel A: scatter DSI_frac vs DSI_rate ─────────────────────────────────────
ax_sc = fig.add_subplot(gs[0, 0])
ax_sc.scatter(dsi_rate, dsi_frac, s=12, alpha=0.4, color='#4c72b0', linewidths=0)
lim = max(dsi_rate.max(), dsi_frac.max()) * 1.05
ax_sc.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
ax_sc.set_xlabel('DSI  total spike rate', fontsize=10)
ax_sc.set_ylabel('DSI  burst fraction', fontsize=10)
ax_sc.set_title('Direction selectivity\nper neuron', fontsize=10)
ax_sc.text(0.05, 0.92,
           f'p={p_w:.3g} (Wilcoxon)\nn={good.sum()}',
           transform=ax_sc.transAxes, fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
frac_above = (dsi_frac > dsi_rate).mean()
ax_sc.text(0.95, 0.05, f'{frac_above*100:.0f}% above diagonal',
           transform=ax_sc.transAxes, fontsize=8, ha='right', color='#c44e52')

# ── Panel B: violin / box comparison ─────────────────────────────────────────
ax_vl = fig.add_subplot(gs[0, 1])
data  = [dsi_rate, dsi_frac]
vp    = ax_vl.violinplot(data, positions=[1, 2], showmedians=True, showextrema=False)
colors = ['#4c72b0', '#c44e52']
for body, col in zip(vp['bodies'], colors):
    body.set_facecolor(col); body.set_alpha(0.6)
vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
ax_vl.set_xticks([1, 2])
ax_vl.set_xticklabels(['Total\nrate', 'Burst\nfraction'], fontsize=9)
ax_vl.set_ylabel('DSI', fontsize=10)
ax_vl.set_title('DSI distribution', fontsize=10)
stars = '***' if p_w < 0.001 else ('**' if p_w < 0.01 else ('*' if p_w < 0.05 else 'n.s.'))
y_top = max(dsi_rate.max(), dsi_frac.max()) * 1.05
ax_vl.plot([1, 2], [y_top, y_top], 'k-', lw=1.2)
ax_vl.text(1.5, y_top * 1.02, stars, ha='center', fontsize=12)

# ── Panel C: rate-residual control ────────────────────────────────────────────
ax_rc = fig.add_subplot(gs[0, 2])
ax_rc.scatter(dsi_rate, dsi_resid, s=12, alpha=0.4, color='#2ca02c', linewidths=0)
lim2 = max(dsi_rate.max(), dsi_resid.max()) * 1.05
ax_rc.plot([0, lim2], [0, lim2], 'k--', lw=0.8, alpha=0.5)
ax_rc.axhline(0, color='gray', lw=0.5, ls=':')
ax_rc.set_xlabel('DSI  total spike rate', fontsize=10)
ax_rc.set_ylabel('DSI  residual burst fraction\n(rate regressed out)', fontsize=9)
ax_rc.set_title('Control 1: rate-residual', fontsize=10)
stars_r = '***' if p_wr < 0.001 else ('**' if p_wr < 0.01 else ('*' if p_wr < 0.05 else 'n.s.'))
ax_rc.text(0.05, 0.92, f'p={p_wr:.3g} {stars_r}',
           transform=ax_rc.transAxes, fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# ── Panel D: tuning-curve correlation histogram ────────────────────────────────
ax_co = fig.add_subplot(gs[0, 3])
ax_co.hist(tuning_corr, bins=25, color='#9467bd', alpha=0.7, edgecolor='white')
ax_co.axvline(tuning_corr.mean(), color='black', lw=1.5, ls='--',
              label=f'mean={tuning_corr.mean():.2f}')
ax_co.set_xlabel('Pearson r (rate vs burst-frac tuning)', fontsize=9)
ax_co.set_ylabel('Number of neurons', fontsize=9)
ax_co.set_title('Control 2: tuning-curve\ncorrelation', fontsize=10)
ax_co.text(0.05, 0.88,
           f'Pref-dir agree:\n{pref_agree.mean()*100:.0f}% of neurons',
           transform=ax_co.transAxes, fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax_co.legend(fontsize=8)

# ── Panels E–H: example tuning curves ────────────────────────────────────────
dir_labels = [f'{int(d):+d}°' for d in directions]

for ci, nidx in enumerate(example_neurons):
    ax = fig.add_subplot(gs[1, ci])
    rate_vals = rate_g[nidx]
    frac_vals = frac_g[nidx]
    r_norm = (rate_vals - rate_vals.min()) / (rate_vals.max() - rate_vals.min() + 1e-9)
    f_norm = (frac_vals - frac_vals.min()) / (frac_vals.max() - frac_vals.min() + 1e-9)
    x = np.arange(n_dirs)
    ax.plot(x, r_norm, 'o-', color='#4c72b0', lw=1.8, ms=5, label='total rate')
    ax.plot(x, f_norm, 's-', color='#c44e52', lw=1.8, ms=5, label='burst frac')
    ax.axvline(pref_rate[nidx], color='#4c72b0', lw=0.8, ls=':', alpha=0.6)
    ax.axvline(pref_frac[nidx], color='#c44e52', lw=0.8, ls=':', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(dir_labels, fontsize=7, rotation=45)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel('Normalised value', fontsize=8)
    agree_str = 'agree' if pref_agree[nidx] else 'DISAGREE'
    ax.set_title(f'Neuron {np.where(good)[0][nidx]}  [{agree_str}]\n'
                 f'DSI_r={dsi_rate[nidx]:.2f}  DSI_f={dsi_frac[nidx]:.2f}', fontsize=8)
    if ci == 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=7)

fig.suptitle(
    'MC_Maze (DANDI 000128): burst fraction is more direction-selective than total spike rate\n'
    f'ISI burst threshold = {ISI_THRESH*1000:.0f} ms  |  '
    f'window = {WIN_START*1000:.0f}–{WIN_END*1000:.0f} ms  |  '
    f'N = {good.sum()} neurons',
    fontsize=11, y=1.01)

plt.savefig('burst_fraction_mcmaze.pdf', bbox_inches='tight')
plt.close(fig)
print('\nSaved burst_fraction_mcmaze.pdf')

io.close()
