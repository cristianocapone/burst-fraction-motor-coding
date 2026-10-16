"""Single-compartment GLM null WITH trial-to-trial gain variability.

Why: a two-compartment model with NO apical gating but with trial-to-trial
variability of its input exceeds the plain GLM null by +0.036 dDSI (the size of
the effect in monkey C), because the plain GLM has a fixed rate per direction
and underestimates the trial-to-trial variance of spike counts. Bursts grow
nonlinearly with rate, so trials with high input contribute extra bursts.

Fix: after fitting the GLM, estimate for each unit the extra count variance not
explained by the GLM and model it as a multiplicative gain on the GLM rate,
drawn once per trial from a gamma distribution with that CV:
    cv^2 = max(0, sum_d n_d (Var_data,d - Var_glm,d)) / sum_d n_d Mean_data,d^2
Then simulate the GLM with that gain and score as before.

Usage:
  python glm_gain_null.py model --i-api 0.0 --trial-cv 0.3        (calibration on the model)
  python glm_gain_null.py data --session 20150309 --check out.json
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

import audit_monkeyC_nulls as A
import single_compartment_null as S

N_PILOT = 10


def simulate_glm_gain(coef, intercept, T, H, di, n_dir, rng, cv):
    """As S.simulate_glm, with a per-trial multiplicative gain (gamma, mean 1, CV cv)."""
    n_bins, n_lag = T.shape[0], H.shape[0]
    n_pre = int(round(S.PRE / S.DT))
    h_rev = (H @ coef[n_dir * S.N_TIME_BASIS:])[::-1]
    base = np.stack([np.exp(intercept + T @ coef[d * S.N_TIME_BASIS:(d + 1) * S.N_TIME_BASIS]) for d in range(n_dir)])
    out = [None] * len(di)
    for d in range(n_dir):
        ix = np.flatnonzero(di == d)
        g = rng.gamma(1 / cv ** 2, cv ** 2, len(ix)) if cv > 0 else np.ones(len(ix))
        Sx = np.zeros((len(ix), n_bins + n_lag))
        for t in range(n_bins):
            mu = np.minimum(base[d, t] * g * np.exp(Sx[:, t:t + n_lag] @ h_rev), S.LAM_CAP * S.DT)
            spike = rng.random(len(ix)) < 1 - np.exp(-mu)
            spike &= Sx[:, t + n_lag - 1] == 0
            Sx[:, t + n_lag] = spike
        for j, ti in enumerate(ix):
            bins = np.flatnonzero(Sx[j, n_lag:]) - n_pre
            w = (bins + rng.random(len(bins))) * S.DT
            out[ti] = w[(w >= 0) & (w < S.WIN)]
    return out


def counts(wins_u):
    return np.array([len(w) for w in wins_u], float)


def gain_cv(data_counts, sim_counts_list, di, n_dir):
    num = den = 0.0
    for d in range(n_dir):
        m = di == d
        cd = data_counts[m]
        if cd.mean() <= 0:
            continue
        vs = np.mean([sc[m].var(ddof=1) for sc in sim_counts_list])
        num += m.sum() * (cd.var(ddof=1) - vs)
        den += m.sum() * cd.mean() ** 2
    return float(np.sqrt(max(num, 0.0) / den)) if den > 0 else 0.0


def test(spikes, starts, wins, di, n_dir, good, n_surr, rng, label):
    t0 = time.time()
    total, burst, active = S.aggregate(wins, di, n_dir)
    obs = A.summ(total, burst, active, good)[0]
    fits, cvs = [], np.zeros(len(spikes))
    for u in range(len(spikes)):
        if not good[u]:
            fits.append(None); continue
        X, yy, T, H = S.glm_design(spikes[u], starts, di, n_dir)
        coef, b0, _, _ = S.fit_glm(X, yy)
        fits.append((coef, b0, T, H))
        pilot = [counts(simulate_glm_gain(coef, b0, T, H, di, n_dir, rng, 0.0)) for _ in range(N_PILOT)]
        cvs[u] = gain_cv(counts(wins[u]), pilot, di, n_dir)
    res = {}
    for name, use_gain in (("glm", False), ("glm_gain", True)):
        nl = []
        for _ in range(n_surr):
            w = [simulate_glm_gain(*fits[u], di, n_dir, rng, cvs[u] if use_gain else 0.0) if fits[u] is not None
                 else [np.empty(0)] * len(di) for u in range(len(spikes))]
            t, b, a = S.aggregate(w, di, n_dir)
            nl.append(A.summ(t, b, a, good)[0])
        nl = np.array(nl)
        res[name] = float(nl.mean()); res["p_" + name] = float((1 + np.sum(nl >= obs)) / (n_surr + 1))
    res.update(obs=float(obs), n_units=int(good.sum()), gain_cv_median=float(np.median(cvs[good])))
    print(f"{label}: N={good.sum()} obs {obs:.3f} | GLM {res['glm']:.3f} (excess {obs-res['glm']:+.3f}, p={res['p_glm']:.3g}) | "
          f"GLM+gain {res['glm_gain']:.3f} (excess {obs-res['glm_gain']:+.3f}, p={res['p_glm_gain']:.3g}) | gain CV median {res['gain_cv_median']:.2f} | {time.time()-t0:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "data"])
    ap.add_argument("--i-api", type=float, default=0.0)
    ap.add_argument("--trial-cv", type=float, default=0.3)
    ap.add_argument("--session", default=None)
    ap.add_argument("--n-surr", type=int, default=100)
    ap.add_argument("--check", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(5)
    if args.what == "model":
        import model_vs_glm_null as M
        M.HAZARD_SCALE, M.BASAL_DEPTH, M.TRIAL_CV, M.I_API_SCALE = 0.02, 0.5, args.trial_cv, args.i_api
        pop = M.build_population(100, 40, True, rng)
        t, b, a = S.aggregate(pop["wins"], pop["di"], 8)
        good = (a >= 5).all(1) & (t.sum(1) >= S.MIN_TOTAL_SPIKES)
        r = test(pop["spikes"], pop["starts"], pop["wins"], pop["di"], 8, good, args.n_surr, rng,
                 f"model api={args.i_api} trialCV={args.trial_cv}")
        if args.check:
            p = Path(args.check); c = json.loads(p.read_text()) if p.exists() else {}
            c[f"api{args.i_api}_tcv{args.trial_cv}"] = r; p.write_text(json.dumps(c, indent=1))
    else:
        p = Path(args.check); c = json.loads(p.read_text()) if p.exists() else {}
        for name, (spikes, starts, y) in S.load_sessions("C", args.session):
            if name in c:
                continue
            dirs = np.sort(np.unique(y))
            total, burst, active, binned, di, wins = A.observed(spikes, starts, y, dirs)
            good = (active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES)
            c[name] = test(spikes, starts, wins, di, len(dirs), good, args.n_surr, rng, name)
            p.write_text(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
