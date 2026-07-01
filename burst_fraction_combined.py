"""
Combined figure: burst-fraction vs tonic-rate direction selectivity
across two independent datasets (MC_Maze and MC_RTT).

Layout: 2 rows x 4 columns
  Row 0  (MC_Maze): A act.rate | B act.frac | C DSI scatter | D tuning curves
  Row 1  (MC_RTT):  E act.rate | F act.frac | G DSI scatter | H tuning curves

Controls computed and printed (not shown in figure):
  - Shuffle test (200 permutations of direction labels)
  - Rate-residual burst fraction DSI (burst frac after regressing out tonic rate)

Output: figure1_burst_fraction.pdf
"""

import os, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import wilcoxon, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

ISI_THRESH = 0.010
MIN_TRIALS = 5
N_SHUFFLE  = 200
COL_RATE   = '#2166ac'
COL_FRAC   = '#b2182b'
COL_RESID  = '#1a7837'

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def dsi(X):
    mn, mx = X.min(1), X.max(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(mn + mx > 0, (mx - mn) / (mx + mn), 0.)

def dsi_signed(X):
    """DSI for signed residuals: peak-to-peak range / (2 * mean |response|)."""
    rng = X.max(1) - X.min(1)
    mu  = np.abs(X).mean(1)
    return np.where(mu > 1e-9, rng / (2 * mu), 0.)

def activity_matrix(tuning, sort_idx):
    mat = tuning[sort_idx]
    mu  = mat.mean(1, keepdims=True)
    sd  = mat.std(1,  keepdims=True)
    return (mat - mu) / (sd + 1e-9)

def aligned_tuning(tuning, pref_idx, n_dirs):
    aligned = np.zeros_like(tuning)
    half    = n_dirs // 2
    for i in range(len(tuning)):
        rolled = np.roll(tuning[i], half - pref_idx[i])
        aligned[i] = rolled / (rolled.max() + 1e-9)
    return aligned

# =============================================================================
# Population-level bilinear structure (cross-validated dPCA-style analysis)
# =============================================================================
# The per-neuron DSI test (DSI panels of figure1_burst_fraction.pdf) shows burst fraction is more direction-
# selective than tonic rate. A skeptic could attribute this to burst spikes
# simply being high-rate spikes. The bilinear model B(n,d) = sum_k G_k(d) Y_k(n)
# makes a stronger, population-level prediction: the *direction-dependent* part
# of the burst-fraction matrix should be (i) a larger share of reliable variance
# and (ii) lower-dimensional (a shared low-rank gate G) than for tonic rate.
# We test this with cross-validation across trial halves so the comparison is
# not biased by burst fraction being the noisier signal.

def cvpca_spectrum(M1, M2):
    """Cross-validated PCA spectrum (Stringer et al. 2019).
    M1, M2: N x D matrices from two independent trial halves.
    Returns cross-validated variance per component (noise -> ~0)."""
    U, _, _ = np.linalg.svd(M1, full_matrices=False)
    proj1 = U.T @ M1
    proj2 = U.T @ M2
    return np.sum(proj1 * proj2, axis=1)

def _half_means(tr_dir, tot, brt, idx, n_dirs, win_dur):
    """Aggregate a subset of trials (idx) into mean tonic rate / burst frac (N x D)."""
    N = tot.shape[1]
    sum_tot = np.zeros((N, n_dirs)); sum_brt = np.zeros((N, n_dirs))
    cnt     = np.zeros(n_dirs)
    for t in idx:
        d = tr_dir[t]
        sum_tot[:, d] += tot[t]; sum_brt[:, d] += brt[t]; cnt[d] += 1
    cnt = np.maximum(cnt, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_tonic = (sum_tot - sum_brt) / (cnt * win_dur)
        mean_frac  = np.where(sum_tot > 0, sum_brt / sum_tot, 0.)
    return mean_tonic, mean_frac

def bilinear_analysis(tr_dir, tr_tot, tr_brt, good, n_dirs, win_dur,
                      n_splits=100, seed=0):
    """Cross-validated population decomposition for rate and burst fraction.
    For each random trial-half split, compute the reliable direction-dependent
    variance fraction, the participation ratio (dimensionality) of the direction
    code, and the share of direction variance in the first component (rank-1)."""
    tot = tr_tot[:, good]; brt = tr_brt[:, good]
    rng = np.random.default_rng(seed)
    by_dir = {d: np.where(tr_dir == d)[0] for d in range(n_dirs)}

    acc = {k: [] for k in
           ('fdir_rate', 'fdir_frac', 'pr_rate', 'pr_frac',
            'pc1_rate', 'pc1_frac', 'cum_rate', 'cum_frac')}

    for _ in range(n_splits):
        h1, h2 = [], []
        for d, ix in by_dir.items():
            ix = ix.copy(); rng.shuffle(ix)
            m = len(ix) // 2
            h1.extend(ix[:m]); h2.extend(ix[m:2 * m])
        h1 = np.array(h1); h2 = np.array(h2)

        rt1, fr1 = _half_means(tr_dir, tot, brt, h1, n_dirs, win_dur)
        rt2, fr2 = _half_means(tr_dir, tot, brt, h2, n_dirs, win_dur)

        for tag, A1, A2 in (('rate', rt1, rt2), ('frac', fr1, fr2)):
            # condition-independent (per-neuron mean) vs direction-dependent
            d1 = A1 - A1.mean(1, keepdims=True)
            d2 = A2 - A2.mean(1, keepdims=True)
            var_tot = np.sum(A1 * A2)            # reliable total variance
            var_dir = np.sum(d1 * d2)            # reliable direction variance
            fdir    = var_dir / var_tot if var_tot > 1e-12 else np.nan

            cv = cvpca_spectrum(d1, d2)
            cv = np.clip(cv, 0, None)
            s  = cv.sum()
            pr   = (s ** 2) / np.sum(cv ** 2) if np.sum(cv ** 2) > 1e-12 else np.nan
            pc1  = cv[0] / s if s > 1e-12 else np.nan
            cum  = np.cumsum(cv) / s if s > 1e-12 else np.full_like(cv, np.nan)

            acc[f'fdir_{tag}'].append(fdir)
            acc[f'pr_{tag}'].append(pr)
            acc[f'pc1_{tag}'].append(pc1)
            acc[f'cum_{tag}'].append(cum)

    out = {}
    for k in ('fdir_rate', 'fdir_frac', 'pr_rate', 'pr_frac',
              'pc1_rate', 'pc1_frac'):
        v = np.array(acc[k], dtype=float)
        out[k] = np.nanmean(v); out[k + '_sd'] = np.nanstd(v)
    out['cum_rate'] = np.nanmean(np.array(acc['cum_rate']), axis=0)
    out['cum_frac'] = np.nanmean(np.array(acc['cum_frac']), axis=0)
    return out

# =============================================================================
# Analysis functions per dataset
# =============================================================================

def analyse_mcmaze(nwb_file):
    WIN_DUR = 0.500
    print(f'  Loading {nwb_file}...')
    io     = NWBHDF5IO(nwb_file, 'r')
    nwb    = io.read()
    trials = nwb.trials.to_dataframe()
    units  = nwb.units.to_dataframe()
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
    tr_dir, tr_tot, tr_brt = [], [], []

    for _, trial in trials.iterrows():
        d_val = trial['direction_deg']
        if d_val not in directions: continue
        di = directions.index(d_val)
        t0 = trial['go_cue_time']
        t1 = t0 + WIN_DUR
        tot_v = np.zeros(n_units); brt_v = np.zeros(n_units)
        for u, spk in enumerate(all_spk):
            ws = spk[(spk >= t0) & (spk < t1)]
            ns = len(ws)
            if ns == 0: continue
            total_count[u, di]  += ns
            trial_counts[u, di] += 1
            tot_v[u] = ns
            if ns > 1:
                nb = np.concatenate([[False], np.diff(ws) < ISI_THRESH]).sum()
                burst_count[u, di] += nb
                brt_v[u] = nb
        tr_dir.append(di); tr_tot.append(tot_v); tr_brt.append(brt_v)

    io.close()
    return _compute_results(total_count, burst_count, trial_counts,
                            WIN_DUR, directions, dir_angles_deg,
                            np.array(tr_dir), np.array(tr_tot), np.array(tr_brt))

def analyse_mcrtt(nwb_file):
    WIN_DUR  = 0.500
    MIN_DIST = 20.0
    N_DIRS   = 8
    print(f'  Loading {nwb_file}...')
    io     = NWBHDF5IO(nwb_file, 'r')
    nwb    = io.read()
    trials = nwb.trials.to_dataframe()
    units  = nwb.units.to_dataframe()
    n_units = len(units)

    beh      = nwb.processing['behavior']
    tgt_ts   = beh['target_pos']; cur_ts = beh['cursor_pos']
    ts_t0    = tgt_ts.starting_time; ts_rate = tgt_ts.rate
    tgt_data = tgt_ts.data[:];       cur_data = cur_ts.data[:]
    trials   = trials[trials['split'] == 'train'].copy()

    def snap(arr, t):
        idx = int(round((t - ts_t0) * ts_rate))
        return arr[max(0, min(idx, len(arr) - 1))]

    vecs   = np.array([snap(tgt_data, r.start_time) - snap(cur_data, r.start_time)
                       for _, r in trials.iterrows()])
    dists  = np.linalg.norm(vecs, axis=1)
    angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    bin_sz = 360 / N_DIRS
    dq     = ((np.round(angles / bin_sz) * bin_sz + 180) % 360) - 180
    trials['direction_deg'] = dq.astype(int)
    trials['reach_dist']    = dists
    trials     = trials[trials['reach_dist'] > MIN_DIST].copy()
    directions     = sorted(trials['direction_deg'].unique())
    n_dirs         = len(directions)
    dir_angles_deg = np.array(directions, dtype=float)

    all_spk      = [np.array(units.iloc[u]['spike_times']) for u in range(n_units)]
    total_count  = np.zeros((n_units, n_dirs))
    burst_count  = np.zeros((n_units, n_dirs))
    trial_counts = np.zeros((n_units, n_dirs))
    tr_dir, tr_tot, tr_brt = [], [], []

    for _, trial in trials.iterrows():
        d_val = trial['direction_deg']
        if d_val not in directions: continue
        di = directions.index(d_val)
        t0 = trial['start_time']
        t1 = t0 + WIN_DUR
        tot_v = np.zeros(n_units); brt_v = np.zeros(n_units)
        for u, spk in enumerate(all_spk):
            ws = spk[(spk >= t0) & (spk < t1)]
            ns = len(ws)
            if ns == 0: continue
            total_count[u, di]  += ns
            trial_counts[u, di] += 1
            tot_v[u] = ns
            if ns > 1:
                nb = np.concatenate([[False], np.diff(ws) < ISI_THRESH]).sum()
                burst_count[u, di] += nb
                brt_v[u] = nb
        tr_dir.append(di); tr_tot.append(tot_v); tr_brt.append(brt_v)

    io.close()
    return _compute_results(total_count, burst_count, trial_counts,
                            WIN_DUR, directions, dir_angles_deg,
                            np.array(tr_dir), np.array(tr_tot), np.array(tr_brt))

def _compute_results(total_count, burst_count, trial_counts,
                     win_dur, directions, dir_angles_deg,
                     tr_dir=None, tr_tot=None, tr_brt=None):
    n_dirs = len(directions)
    tonic_count = total_count - burst_count
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_tonic = np.where(trial_counts > 0,
                              tonic_count / (trial_counts * win_dur), 0.)
        mean_frac  = np.where(total_count > 0,
                              burst_count / total_count, 0.)

    good   = ((trial_counts >= MIN_TRIALS).all(1)) & \
             ((mean_tonic + mean_frac).sum(1) > 0.2)
    rate_g = mean_tonic[good]
    frac_g = mean_frac[good]
    N      = good.sum()

    dsi_rate  = dsi(rate_g)
    dsi_frac  = dsi(frac_g)
    _, p_main = wilcoxon(dsi_frac, dsi_rate, alternative='greater')
    pct_above = (dsi_frac > dsi_rate).mean() * 100

    # ── Rate-residual burst fraction DSI ──────────────────────────────────────
    frac_resid = np.zeros_like(frac_g)
    for i in range(N):
        r, f = rate_g[i], frac_g[i]
        if r.std() > 1e-9:
            sl, ic, *_ = linregress(r, f)
            frac_resid[i] = f - (sl * r + ic)
        else:
            frac_resid[i] = f - f.mean()

    dsi_resid  = dsi_signed(frac_resid)
    _, p_resid = wilcoxon(dsi_resid, dsi_rate, alternative='greater')
    pct_resid  = (dsi_resid > dsi_rate).mean() * 100

    # ── Bootstrap CI on % above diagonal (resample neurons) ──────────────────
    rng_boot   = np.random.default_rng(0)
    boot_pcts  = np.array([
        (dsi(frac_g[rng_boot.integers(0, N, N)])
         > dsi(rate_g[rng_boot.integers(0, N, N)])).mean() * 100
        for _ in range(N_SHUFFLE)
    ])
    boot_ci_lo = np.percentile(boot_pcts, 2.5)
    boot_ci_hi = np.percentile(boot_pcts, 97.5)

    # ── PC embedding of direction-averaged population (ring plot) ─────────────
    def _pc_embed(B):
        Bd = B - B.mean(1, keepdims=True)          # direction-dependent part
        U, S, _ = np.linalg.svd(Bd, full_matrices=False)
        ev = (S ** 2) / (S ** 2).sum()
        return (U[:, :2].T @ Bd).T, ev[:2]         # (D x 2 scores), (2 EVR)
    pc_rate, ev_rate = _pc_embed(rate_g)
    pc_frac, ev_frac = _pc_embed(frac_g)

    # Circular correlation between true direction and PC-plane angle
    # (how faithfully the population ring preserves direction order;
    #  invariant to rotation, and we take the better of the two reflections)
    def _circ_corr(a, b):
        a = a - np.angle(np.exp(1j * a).mean())
        b = b - np.angle(np.exp(1j * b).mean())
        num = (np.sin(a) * np.sin(b)).sum()
        den = np.sqrt((np.sin(a) ** 2).sum() * (np.sin(b) ** 2).sum())
        return num / den if den > 1e-12 else 0.
    true_ang = np.radians(dir_angles_deg)
    def _ring_corr(pc):
        pa = np.arctan2(pc[:, 1], pc[:, 0])
        return max(_circ_corr(true_ang, pa), _circ_corr(true_ang, -pa))
    ring_corr_rate = _ring_corr(pc_rate)
    ring_corr_frac = _ring_corr(pc_frac)

    # ── Activity matrices and tuning curves ───────────────────────────────────
    pref_frac  = np.argmax(frac_g, axis=1)
    pref_rate  = np.argmax(rate_g, axis=1)
    sort_order = np.argsort(pref_frac)

    am_rate = activity_matrix(rate_g, sort_order)
    am_frac = activity_matrix(frac_g, sort_order)

    al_rate      = aligned_tuning(rate_g, pref_rate, n_dirs)
    al_frac      = aligned_tuning(frac_g, pref_frac, n_dirs)
    mean_rate_tc = al_rate.mean(0);  sem_rate_tc = al_rate.std(0) / np.sqrt(N)
    mean_frac_tc = al_frac.mean(0);  sem_frac_tc = al_frac.std(0) / np.sqrt(N)
    rel_dirs     = (np.arange(n_dirs) - n_dirs // 2) * (360 // n_dirs)

    print(f'    N={N}  p={p_main:.1e}  {pct_above:.0f}% above diag  '
          f'boot 95%CI=[{boot_ci_lo:.0f}%,{boot_ci_hi:.0f}%]  '
          f'resid {pct_resid:.0f}% above diag (p={p_resid:.1e})')

    # ── Population-level bilinear structure (cross-validated) ─────────────────
    bil = None
    if tr_dir is not None:
        bil = bilinear_analysis(tr_dir, tr_tot, tr_brt, good, n_dirs, win_dur)
        print(f'    [bilinear]  dir-var fraction  rate={bil["fdir_rate"]:.2f} '
              f'frac={bil["fdir_frac"]:.2f}  |  participation ratio  '
              f'rate={bil["pr_rate"]:.2f} frac={bil["pr_frac"]:.2f}  |  '
              f'rank-1 share  rate={bil["pc1_rate"]:.2f} frac={bil["pc1_frac"]:.2f}')

    return dict(N=N, directions=directions, dir_angles_deg=dir_angles_deg,
                dsi_rate=dsi_rate, dsi_frac=dsi_frac,
                p_main=p_main, pct_above=pct_above,
                boot_ci_lo=boot_ci_lo, boot_ci_hi=boot_ci_hi,
                dsi_resid=dsi_resid, p_resid=p_resid, pct_resid=pct_resid,
                am_rate=am_rate, am_frac=am_frac,
                mean_rate_tc=mean_rate_tc, sem_rate_tc=sem_rate_tc,
                mean_frac_tc=mean_frac_tc, sem_frac_tc=sem_frac_tc,
                rel_dirs=rel_dirs, bil=bil,
                pc_rate=pc_rate, pc_frac=pc_frac,
                ev_rate=ev_rate, ev_frac=ev_frac,
                ring_corr_rate=ring_corr_rate, ring_corr_frac=ring_corr_frac)

# =============================================================================
# Run both analyses
# =============================================================================
print('MC_Maze (DANDI 000128)')
res_maze = analyse_mcmaze('mc_maze_train.nwb')
print('MC_RTT  (DANDI 000129)')
res_rtt  = analyse_mcrtt('mc_rtt_train.nwb')

# =============================================================================
# Summary table
# =============================================================================
print()
print(f'{"Dataset":<10} {"N":>4} {"p_main":>10} {"% above":>8} '
      f'{"boot 95%CI":>16} {"resid%":>7} {"p_resid":>10}')
print('-' * 72)
for res, label in [(res_maze,'MC_Maze'), (res_rtt,'MC_RTT')]:
    print(f'{label:<10} {res["N"]:>4} {res["p_main"]:>10.1e} '
          f'{res["pct_above"]:>7.0f}% '
          f'[{res["boot_ci_lo"]:>5.0f}%,{res["boot_ci_hi"]:>5.0f}%] '
          f'{res["pct_resid"]:>6.0f}% '
          f'{res["p_resid"]:>10.1e}')
print()

# =============================================================================
# FIGURE  —  2 rows x 4 columns
# =============================================================================
datasets = [
    (res_maze, 'MC_Maze', list('ABCD')),
    (res_rtt,  'MC_RTT',  list('EFGH')),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 8),
                         gridspec_kw=dict(wspace=0.45, hspace=0.55,
                                          width_ratios=[1, 1, 1, 1]))
fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.10)

for row, (res, ds_label, labels) in enumerate(datasets):
    tick_labels = [f'{int(d)}°' for d in res['directions']]
    n_dirs      = len(res['directions'])

    # ── A/F and B/G: activity matrices ────────────────────────────────────────
    for col, (mat, col_rgb, lbl) in enumerate([
        (res['am_rate'], COL_RATE, labels[0]),
        (res['am_frac'], COL_FRAC, labels[1])]):
        ax = axes[row, col]
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                       vmin=-2.5, vmax=2.5, interpolation='nearest')
        cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
        cb.set_label('z-score', fontsize=6); cb.ax.tick_params(labelsize=5)
        ax.set_xticks(range(n_dirs))
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_xlabel('Direction', fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(col_rgb); spine.set_linewidth(1.5)
        ax.set_box_aspect(1)
        ax.text(-0.10, 1.04, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom')

    axes[row, 0].set_ylabel('Neuron', fontsize=7)
    axes[row, 0].text(-0.32, 0.5, ds_label, transform=axes[row, 0].transAxes,
                      fontsize=8, fontweight='bold', va='center', rotation=90)

    # ── C/H: DSI scatter with shuffle annotation ──────────────────────────────
    ax = axes[row, 2]
    sc = ax.scatter(res['dsi_rate'], res['dsi_frac'],
                    c=res['dsi_frac'] - res['dsi_rate'],
                    cmap='RdBu_r', vmin=-0.8, vmax=0.8,
                    s=12, alpha=0.6, linewidths=0)
    lim = max(res['dsi_rate'].max(), res['dsi_frac'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.9, alpha=0.6)
    ax.set_xlabel('DSI  tonic rate', fontsize=7)
    ax.set_ylabel('DSI  burst fraction', fontsize=7)
    ax.text(0.05, 0.93,
            f'p={res["p_main"]:.0e}\n{res["pct_above"]:.0f}% above diag.',
            transform=ax.transAxes, fontsize=6,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85))
    cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
    cb.set_label('ΔDSI', fontsize=6); cb.ax.tick_params(labelsize=5)
    despine(ax); ax.tick_params(labelsize=6); ax.set_box_aspect(1)
    ax.text(-0.14, 1.04, labels[2], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

    # ── D/I: tuning curves ────────────────────────────────────────────────────
    ax = axes[row, 3]
    for mean, sem, col_rgb, lbl in [
        (res['mean_rate_tc'], res['sem_rate_tc'], COL_RATE, 'Tonic rate'),
        (res['mean_frac_tc'], res['sem_frac_tc'], COL_FRAC, 'Burst frac')]:
        ax.plot(res['rel_dirs'], mean, 'o-', color=col_rgb,
                lw=1.8, ms=4, label=lbl)
        ax.fill_between(res['rel_dirs'], mean - sem, mean + sem,
                        color=col_rgb, alpha=0.15)
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Offset from preferred (deg)', fontsize=7)
    ax.set_ylabel('Normalised response', fontsize=7)
    ax.set_xticks(res['rel_dirs'])
    ax.set_xticklabels([f'{int(d)}' for d in res['rel_dirs']],
                       fontsize=6, rotation=45)
    ax.tick_params(labelsize=6)
    if row == 0:
        ax.legend(fontsize=6, loc='lower center', ncol=2, framealpha=0.85)
    despine(ax); ax.set_box_aspect(1)
    ax.text(-0.14, 1.04, labels[3], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')


plt.savefig('figure1_burst_fraction.pdf', bbox_inches='tight', dpi=200)
plt.close(fig)
print('Saved figure1_burst_fraction.pdf')

# =============================================================================
# SUPPLEMENTARY FIGURE — Population-level bilinear structure (figure1_population_structure.pdf)
#   Col 0: fraction of reliable variance that is direction-dependent (rate vs frac)
#   Col 1: cumulative cross-validated direction variance vs # components
#          (lower-dimensional / faster-rising = more bilinear)
# =============================================================================
print()
print(f'{"Dataset":<10} {"dir-var frac":>22} {"particip. ratio":>22} '
      f'{"rank-1 share":>22}')
print(f'{"":<10} {"rate / frac":>22} {"rate / frac":>22} {"rate / frac":>22}')
print('-' * 80)
for res, label in [(res_maze, 'MC_Maze'), (res_rtt, 'MC_RTT')]:
    b = res['bil']
    print(f'{label:<10} '
          f'{b["fdir_rate"]:.2f} / {b["fdir_frac"]:.2f}'.rjust(22) + ' '
          f'{b["pr_rate"]:.2f} / {b["pr_frac"]:.2f}'.rjust(22) + ' '
          f'{b["pc1_rate"]:.2f} / {b["pc1_frac"]:.2f}'.rjust(22))
print()
for res, label in [(res_maze, 'MC_Maze'), (res_rtt, 'MC_RTT')]:
    print(f'  {label}: ring circular-corr  rate={res["ring_corr_rate"]:.2f}  '
          f'frac={res["ring_corr_frac"]:.2f}')
print()

from matplotlib.lines import Line2D

figc, axc = plt.subplots(2, 3, figsize=(11, 7),
                         gridspec_kw=dict(wspace=0.5, hspace=0.55))
figc.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.10)

for row, (res, ds_label, labels) in enumerate(
        [(res_maze, 'MC_Maze', ['A', 'B', 'C']),
         (res_rtt,  'MC_RTT',  ['D', 'E', 'F'])]):
    b = res['bil']

    # ── direction-variance fraction bar ──────────────────────────────────────
    ax = axc[row, 0]
    vals = [b['fdir_rate'], b['fdir_frac']]
    err  = [b['fdir_rate_sd'], b['fdir_frac_sd']]
    ax.bar([0, 1], vals, yerr=err, capsize=4,
           color=[COL_RATE, COL_FRAC], width=0.6, alpha=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Tonic\nrate', 'Burst\nfrac'],
                                              fontsize=7)
    ax.set_ylabel('Direction-dependent\nvariance fraction (cv)', fontsize=7)
    ax.set_ylim(0, max(vals) * 1.6); ax.tick_params(labelsize=6)
    despine(ax); ax.set_box_aspect(1)
    ax.text(-0.28, 1.04, labels[0], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')
    ax.text(-0.52, 0.5, ds_label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='center', rotation=90)

    # ── cumulative cv variance (dimensionality) ──────────────────────────────
    ax = axc[row, 1]
    for cum, col_rgb, lbl, pr in [
        (b['cum_rate'], COL_RATE, 'Tonic rate', b['pr_rate']),
        (b['cum_frac'], COL_FRAC, 'Burst frac', b['pr_frac'])]:
        k = np.arange(1, len(cum) + 1)
        ax.plot(k, cum, 'o-', color=col_rgb, lw=1.8, ms=4,
                label=f'{lbl} (PR={pr:.1f})')
    ax.set_xlabel('# components', fontsize=7)
    ax.set_ylabel('Cumulative direction\nvariance (cv)', fontsize=7)
    ax.set_ylim(0, 1.02); ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc='lower right', framealpha=0.85)
    despine(ax); ax.set_box_aspect(1)
    ax.text(-0.28, 1.04, labels[1], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

    # ── PC1-PC2 ring: mean population position per direction ──────────────────
    ax = axc[row, 2]
    angs  = res['dir_angles_deg']
    order = np.argsort(angs)
    for pc, mk, ls, lbl in [(res['pc_rate'], 'x', '--', 'Tonic rate'),
                            (res['pc_frac'], 'o', '-',  'Burst frac')]:
        P = pc[order]
        P = P / np.sqrt((P ** 2).sum(1).mean() + 1e-12)   # normalise radius
        Pl = np.vstack([P, P[0]])                          # close the loop
        ax.plot(Pl[:, 0], Pl[:, 1], color='0.65', lw=1.0, ls=ls, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], c=angs[order], cmap='hsv',
                   vmin=-180, vmax=180, s=45, marker=mk,
                   edgecolors='k', linewidths=0.3, zorder=2)
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlabel('PC1', fontsize=7); ax.set_ylabel('PC2', fontsize=7)
    ax.tick_params(labelsize=6); despine(ax); ax.set_box_aspect(1)
    handles = [Line2D([0], [0], color='0.5', marker='o', ls='-',  label='Burst frac'),
               Line2D([0], [0], color='0.5', marker='x', ls='--', label='Tonic rate')]
    ax.legend(handles=handles, fontsize=6, loc='best', framealpha=0.85)
    ax.text(0.02, 0.02,
            f'ring $\\rho_{{circ}}$\nfrac={res["ring_corr_frac"]:.2f}\n'
            f'rate={res["ring_corr_rate"]:.2f}',
            transform=ax.transAxes, fontsize=6, va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85))
    ax.text(-0.28, 1.04, labels[2], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

plt.savefig('figure1_population_structure.pdf', bbox_inches='tight', dpi=200)
plt.close(figc)
print('Saved figure1_population_structure.pdf')
