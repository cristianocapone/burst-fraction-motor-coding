"""Burst vs tonic direction selectivity at matched noise.

Burst spikes are few, so any tuning measure based on them is noisier than one
based on the (many) tonic spikes. To compare like with like, for each unit we
draw at random as many tonic spikes as the unit has burst spikes (same trials
pool), and compare the direction tuning of the two equal-sized spike sets:
  - DSI of per-direction counts (normalised by trials per direction)
  - cross-validated tuning depth (preferred direction from one half of the
    trials, depth measured on the other half)
  - split-half agreement of the preferred direction
Same noise level -> differences reflect how much direction information each
kind of spike carries ("information per spike").
Run on all monkey C sessions (and the model, with --model).
"""
import argparse
import glob

import numpy as np

import audit_monkeyC_nulls as A
import single_compartment_null as S

N_REP = 50


def spike_sets(wins_u, isi):
    """Per trial: arrays of burst-spike and tonic-spike times (burst spike = ISI < isi)."""
    b_tr, t_tr = [], []
    for w in wins_u:
        if len(w) > 1:
            short = np.r_[False, np.diff(w) < isi]
        else:
            short = np.zeros(len(w), bool)
        b_tr.append(int(short.sum())); t_tr.append(int((~short).sum()))
    return np.array(b_tr), np.array(t_tr)


def matched_counts(b_tr, t_tr, rng):
    """Subsample tonic spikes (uniformly over all tonic spikes) to the burst-spike total."""
    nb, nt = b_tr.sum(), t_tr.sum()
    if nb == 0 or nt < nb:
        return None
    owner = np.repeat(np.arange(len(t_tr)), t_tr)          # trial index of each tonic spike
    pick = rng.choice(nt, nb, replace=False)
    return np.bincount(owner[pick], minlength=len(t_tr))


def dsi_vec(x):
    hi, lo = x.max(), x.min()
    return (hi - lo) / (hi + lo) if hi + lo > 0 else 0.0


def tuning(counts_tr, di, n_dir, idx):
    return np.array([counts_tr[idx][di[idx] == d].mean() if np.any(di[idx] == d) else 0 for d in range(n_dir)])


def unit_metrics(b_tr, t_tr, di, n_dir, rng):
    out = {"dsi_b": [], "dsi_t": [], "cv_b": [], "cv_t": [], "agree_b": [], "agree_t": []}
    for _ in range(N_REP):
        t_m = matched_counts(b_tr, t_tr, rng)
        if t_m is None:
            return None
        perm = rng.permutation(len(di)); h1, h2 = perm[::2], perm[1::2]
        for key, c in (("b", b_tr), ("t", t_m)):
            full = tuning(c, di, n_dir, np.arange(len(di)))
            out["dsi_" + key].append(dsi_vec(full))
            t1, t2 = tuning(c, di, n_dir, h1), tuning(c, di, n_dir, h2)
            p = t1.argmax()
            m = t2.max() if t2.max() > 0 else 1.0
            out["cv_" + key].append((t2[p] - np.delete(t2, p).mean()) / m)      # cross-validated depth
            out["agree_" + key].append(float(t1.argmax() == t2.argmax()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def run_units(wins, di, n_dir, good, rng):
    rows = []
    for u in np.flatnonzero(good):
        b_tr, t_tr = spike_sets(wins[u], A.ISI)
        if b_tr.sum() < 20:
            continue
        m = unit_metrics(b_tr, t_tr, di, n_dir, rng)
        if m:
            rows.append(m)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    allr = []
    if args.model:
        import model_vs_glm_null as M
        M.HAZARD_SCALE, M.BASAL_DEPTH, M.TRIAL_CV = 0.02, 0.5, 0.3
        for api in (0.0, 0.5):
            M.I_API_SCALE = api
            for seed in range(3):
                pop = M.build_population(100, 40, True, np.random.default_rng(seed))
                t, b, a = S.aggregate(pop["wins"], pop["di"], 8)
                good = (a >= 5).all(1) & (t.sum(1) >= 100)
                r = run_units(pop["wins"], pop["di"], 8, good, rng)
                print(f"model api={api} seed={seed}: N={len(r['dsi_b'])} | DSI burst {r['dsi_b'].mean():.3f} vs tonic(matched) {r['dsi_t'].mean():.3f} | "
                      f"CV depth burst {r['cv_b'].mean():.3f} vs tonic {r['cv_t'].mean():.3f} | pref agree burst {r['agree_b'].mean():.2f} vs tonic {r['agree_t'].mean():.2f}", flush=True)
        return
    for f in sorted(glob.glob("dandi000688_C_cache/*CO*.nwb")):
        sp, st, y = A.load(f)
        dirs = np.sort(np.unique(y))
        t, b, a, _, di, w = A.observed(sp, st, y, dirs)
        good = (a >= 5).all(1) & (t.sum(1) >= 100)
        r = run_units(w, di, len(dirs), good, rng)
        tag = f.split("_ses-CO-")[1][:8]
        allr.append({k: v.mean() for k, v in r.items()})
        print(f"{tag}: N={len(r['dsi_b'])} | DSI burst {r['dsi_b'].mean():.3f} vs tonic(matched) {r['dsi_t'].mean():.3f} | "
              f"CV depth burst {r['cv_b'].mean():.3f} vs tonic {r['cv_t'].mean():.3f} | pref agree burst {r['agree_b'].mean():.2f} vs tonic {r['agree_t'].mean():.2f}", flush=True)
    from scipy.stats import wilcoxon
    for k1, k2, lab in (("dsi_b", "dsi_t", "DSI"), ("cv_b", "cv_t", "cross-validated depth"), ("agree_b", "agree_t", "pref. agreement")):
        x = np.array([r[k1] - r[k2] for r in allr])
        print(f"ALL {len(allr)} sessions, {lab}: burst - matched tonic = {x.mean():+.3f}, >0 in {np.sum(x > 0)}/{len(x)}, Wilcoxon p={wilcoxon(x).pvalue:.2g}")


if __name__ == "__main__":
    main()
