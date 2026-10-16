"""Gate fit, second round: direction-specificity test, gate x drive interaction,
and a burst-propensity regression at matched rate. Null control with trial gain.

Per unit (monkey C sessions, or a model population):
  GLM family (1-ms bins, as single_compartment_null):
    M0   log rate = b + f_d(t) + h * s                        (null, 108 parameters)
    Mc   M0 + g * B(t)                                        (burst term common to all directions)
    Md   M0 + g_d * B(t)                                      (direction-specific burst term)
    Mdk0 Md + k * B(t) * z(t)                                 (common gate x drive interaction)
    Mdk  Md + k_d * B(t) * z(t)                               (direction-specific interaction)
  B(t) = 1 if the unit fired in the previous 10 ms; z(t) = the unit's stimulus-driven
  log rate from M0 (b + f_d(t), no history), centred.
  Tests (likelihood ratio, chi2):
    LR8  = M0 vs Md, 8 df (the published test)
    LR7  = Mc vs Md, 7 df (is the burst term direction-specific?)
    LRk7 = Mdk0 vs Mdk, 7 df (does the direction effect change with drive?)
  A multiplicative (AND) gate is a constant factor in log scale: k_d - mean(k) ~ 0.
  An additive gate loses relative weight as drive grows: k_d below the mean at the
  preferred direction.

  Regression at matched rate (per spike k, scored spikes at t < WIN - 10 ms):
    y_k = 1 if the next spike follows within 10 ms
    L_k = log of the leave-one-trial-out rate of the unit at t_k in direction d_k
    x_k = cos(d_k - theta_hat), theta_hat from the other half of the trials
    logit P(y=1) = cubic(L) + gamma2 * x + gamma3 * x * L
    gamma2 > 0: more bursts in the preferred direction at the same rate.
    gamma3: change of that effect with rate (multiplicative ~ 0, additive < 0).

  Null control: spikes simulated from the unit's M0 with a per-trial gain
  (gamma, CV matched to the excess count variance of the unit), refitted and
  regressed exactly like the data.

Usage:
  python gated_glm2.py --session 20150309 --out gated2/C_20150309.json
  python gated_glm2.py --model-api 0.5 --model-api-cv 1 --out gated2/model_api0.5_cv1.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

import audit_monkeyC_nulls as A
import single_compartment_null as S

BURST_MS = 10
RIDGE = 1e-3


def fit(X, y, theta0=None):
    """Poisson MLE with ridge (as S.fit_glm), optional warm start. Returns (theta, nll).
    Newton/IRLS (glm_newton.py); the L-BFGS version is kept as fit_lbfgs."""
    from glm_newton import fit_newton
    return fit_newton(X, y, theta0, ridge=RIDGE)


def fit_lbfgs(X, y, theta0=None):
    n, p = X.shape
    y = y.astype(np.float64); ysum = y.sum()

    def f(th):
        beta, b0 = th[:p], th[p]
        eta = (X @ beta.astype(np.float32)).astype(np.float64) + b0
        mu = np.exp(np.minimum(eta, 10.0))
        r = (mu - y).astype(np.float32)
        g = np.empty(p + 1)
        g[:p] = (X.T @ r).astype(np.float64) + RIDGE * beta
        g[p] = mu.sum() - ysum
        return mu.sum() - (y * eta).sum() + 0.5 * RIDGE * (beta ** 2).sum(), g

    if theta0 is None:
        theta0 = np.zeros(p + 1); theta0[p] = np.log(max(ysum, 1) / n)
    sol = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000, gtol=1e-6, ftol=1e-12))
    th = sol.x
    eta = (X @ th[:p].astype(np.float32)).astype(np.float64) + th[p]
    nll = float(np.exp(np.minimum(eta, 10.0)).sum() - (y * eta).sum())
    return th, nll


def burst_cov(y, n_tr, n_bins):
    r = y.reshape(n_tr, n_bins)
    B = np.zeros_like(r)
    for lag in range(1, BURST_MS + 1):
        B[:, lag:] = np.maximum(B[:, lag:], r[:, :-lag])
    return B.reshape(-1)


def warm(th_prev, n_new):
    """Insert n_new zero coefficients before the intercept."""
    return np.concatenate([th_prev[:-1], np.zeros(n_new), th_prev[-1:]])


def glm_family(spk, starts, di, n_dir):
    X, y, T, H = S.glm_design(spk, starts, di, n_dir)
    n_tr, n_bins = len(di), T.shape[0]
    p_rate = n_dir * T.shape[1]
    th0, nll0 = fit(X, y)
    B = burst_cov(y, n_tr, n_bins).astype(np.float32)
    D = np.zeros((len(y), n_dir), np.float32)
    D[np.arange(len(y)), np.repeat(di, n_bins)] = 1.0
    z = (X[:, :p_rate] @ th0[:p_rate].astype(np.float32)).astype(np.float64) + th0[-1]
    z = (z - z.mean()).astype(np.float32)
    Xc = np.concatenate([X, B[:, None]], 1)
    thc, nllc = fit(Xc, y, warm(th0, 1))
    Xd = np.concatenate([X, B[:, None] * D], 1)
    thd, nlld = fit(Xd, y, warm(th0, n_dir))
    Xdk0 = np.concatenate([Xd, (B * z)[:, None]], 1)
    thdk0, nlldk0 = fit(Xdk0, y, warm(thd, 1))
    Xdk = np.concatenate([Xd, (B * z)[:, None] * D], 1)
    thdk, nlldk = fit(Xdk, y, warm(thd, n_dir))
    lr8, lr7, lrk7 = 2 * (nll0 - nlld), 2 * (nllc - nlld), 2 * (nlldk0 - nlldk)
    return dict(g=thd[-1 - n_dir:-1].tolist(), k=thdk[-1 - n_dir:-1].tolist(), g_common=float(thc[-2]),
                lr8=float(lr8), p8=float(chi2.sf(max(lr8, 0), n_dir)),
                lr7=float(lr7), p7=float(chi2.sf(max(lr7, 0), n_dir - 1)),
                lrk7=float(lrk7), pk7=float(chi2.sf(max(lrk7, 0), n_dir - 1))), (th0, T, H)


def gain_cv(wins, di, n_dir):
    """CV of a per-trial gain from the excess count variance over Poisson (method of moments)."""
    c = np.array([len(w) for w in wins], float)
    v = []
    for d in range(n_dir):
        x = c[di == d]
        if len(x) > 2 and x.mean() > 0:
            v.append((x.var(ddof=1) - x.mean()) / x.mean() ** 2)
    return float(np.sqrt(np.clip(np.mean(v) if v else 0.0, 0.0, 0.25)))


def simulate_with_gain(th0, T, H, di, n_dir, cv, rng):
    """M0 simulation (as S.simulate_glm, 1-ms absolute refractory, rate cap) with a gamma trial gain."""
    p_rate = n_dir * T.shape[1]
    coef, b0 = th0[:-1], th0[-1]
    n_bins, n_lag = T.shape[0], H.shape[0]
    n_pre = int(round(S.PRE / S.DT))
    h_rev = (H @ coef[p_rate:])[::-1]
    gains = rng.gamma(1 / cv ** 2, cv ** 2, len(di)) if cv > 0 else np.ones(len(di))
    out = [None] * len(di)
    for d in range(n_dir):
        ix = np.flatnonzero(di == d)
        base = np.exp(b0 + T @ coef[d * T.shape[1]:(d + 1) * T.shape[1]])
        G = gains[ix][:, None]
        Sm = np.zeros((len(ix), n_bins + n_lag))
        for t in range(n_bins):
            mu = np.minimum(G[:, 0] * base[t] * np.exp(Sm[:, t:t + n_lag] @ h_rev), S.LAM_CAP * S.DT)
            spike = (rng.random(len(ix)) < 1 - np.exp(-mu)) & (Sm[:, t + n_lag - 1] == 0)
            Sm[:, t + n_lag] = spike
        for j, ti in enumerate(ix):
            bins = np.flatnonzero(Sm[j, n_lag:]) - n_pre
            out[ti] = (bins + rng.random(len(bins))) * S.DT
    return out


def matched_rate_regression(trial_spikes, di, n_dir, rng):
    """gamma2, gamma3 of logit P(next spike < 10 ms) = cubic(L) + g2 x + g3 x L (two folds)."""
    WIN, bw = A.WIN, 0.010
    edges = np.arange(0, WIN + bw / 2, bw); nb = len(edges) - 1
    counts = np.array([np.histogram(s[(s >= 0) & (s < WIN)], edges)[0] for s in trial_spikes], float)
    k = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2); k /= k.sum()   # 20-ms gaussian on 10-ms bins
    ang = 2 * np.pi * np.arange(n_dir) / n_dir
    tri = np.arange(len(di)); folds = (tri % 2 == 0, tri % 2 == 1)
    rows = []
    for fit_mask, tune_mask in (folds, folds[::-1]):
        tc = np.array([counts[tune_mask & (di == d)].sum(1).mean() if np.any(tune_mask & (di == d)) else 0 for d in range(n_dir)])
        theta = np.angle(np.sum((tc - tc.mean()) * np.exp(1j * ang)))
        for d in range(n_dir):
            idx = np.flatnonzero(di == d)
            tot = counts[idx].sum(0)
            for ti in idx[fit_mask[idx]]:
                if len(idx) < 3:
                    continue
                loo = np.convolve((tot - counts[ti]) / ((len(idx) - 1) * bw), k, mode="same")
                s = np.sort(trial_spikes[ti]); s_in = s[(s >= 0) & (s < WIN - 0.010)]
                if len(s_in) == 0:
                    continue
                nxt = np.searchsorted(s, s_in, side="right")
                gap = np.where(nxt < len(s), s[np.minimum(nxt, len(s) - 1)] - s_in, np.inf)
                b = np.minimum((s_in / bw).astype(int), nb - 1)
                L = np.log(np.maximum(loo[b], 0.5))
                x = np.cos(ang[d] - theta)
                rows.append(np.column_stack([(gap < 0.010).astype(float), L, np.full(len(s_in), x)]))
    if not rows:
        return None
    R = np.concatenate(rows)
    y, L, x = R[:, 0], R[:, 1], R[:, 2]
    if y.sum() < 20 or (1 - y).sum() < 20:
        return dict(n=int(len(y)), n_burst=int(y.sum()), g2=None, g3=None)
    Lc = (L - L.mean()) / (L.std() + 1e-9)
    Z = np.column_stack([np.ones_like(Lc), Lc, Lc ** 2, Lc ** 3, x, x * Lc])

    def f(w):
        e = Z @ w; p = 1 / (1 + np.exp(-e))
        return -(y * e - np.logaddexp(0, e)).sum() + 0.5e-3 * (w[1:] ** 2).sum(), Z.T @ (p - y) + np.r_[0, 1e-3 * w[1:]]

    w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return dict(n=int(len(y)), n_burst=int(y.sum()), g2=float(w[4]), g3=float(w[5]))


def run_unit(spk_abs, wins_u, starts, di, n_dir, rng):
    fam, (th0, T, H) = glm_family(spk_abs, starts, di, n_dir)
    res = dict(data=fam, reg=matched_rate_regression(wins_u, di, n_dir, rng))
    cv = gain_cv(wins_u, di, n_dir)
    w = simulate_with_gain(th0, T, H, di, n_dir, cv, rng)
    spk = np.sort(np.concatenate([starts[ti] + w[ti] for ti in range(len(di))]))
    res["null"], _ = glm_family(spk, starts, di, n_dir)
    res["null_reg"] = matched_rate_regression([x[x >= 0] for x in w], di, n_dir, rng)
    res["gain_cv"] = cv
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--dataset", default="C", choices=["C", "M", "T", "J", "indy"])
    ap.add_argument("--subsample-trials", type=int, default=0,
                    help="keep at most this many random trials per direction (power matching, e.g. 40 as in Indy)")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--thin", type=float, default=0.0,
                    help="keep each spike with this probability (power matching in spike/burst count, e.g. 0.6 to reach Indy's bursts per unit)")
    ap.add_argument("--model-api", type=float, default=None)
    ap.add_argument("--model-api-cv", type=float, default=0.0)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--model-trials", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0, help="process only units with rank %% nparts == part")
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--skip", nargs="*", default=[], help="json files whose units are already done")
    args = ap.parse_args()
    skip = {}
    for f in args.skip:
        if Path(f).exists():
            for nm, units in json.loads(Path(f).read_text()).items():
                skip.setdefault(nm, set()).update(units)
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    rng = np.random.default_rng(5)
    if args.model_api is not None:
        import model_vs_glm_null as M
        M.HAZARD_SCALE, M.BASAL_DEPTH, M.TRIAL_CV, M.I_API_SCALE = 0.02, 0.5, 0.3, args.model_api
        M.API_CV = args.model_api_cv
        pop = M.build_population(100, args.model_trials, True, np.random.default_rng(args.model_seed))
        n_dir = 8
        name = f"model_api{args.model_api}_cv{args.model_api_cv:g}_seed{args.model_seed}" + (f"_trials{args.model_trials}" if args.model_trials != 40 else "")
        total, burst, active = S.aggregate(pop["wins"], pop["di"], n_dir)
        items = [(name, pop["spikes"], pop["starts"], pop["di"], n_dir, pop["wins"], total, active)]
    else:
        items = []
        for name, (spikes, starts, y) in S.load_sessions(args.dataset, args.session):
            dirs = np.sort(np.unique(y)); n_dir = len(dirs)
            if args.subsample_trials:
                rs = np.random.default_rng(args.subsample_seed)
                keep = np.sort(np.concatenate([rs.choice(np.flatnonzero(y == d), min(args.subsample_trials, int(np.sum(y == d))), replace=False) for d in dirs]))
                starts, y = np.asarray(starts)[keep], np.asarray(y)[keep]
                name = f"{name}_sub{args.subsample_trials}"
            if args.thin:
                rs = np.random.default_rng(args.subsample_seed + 1)
                spikes = [np.asarray(s)[rs.random(len(s)) < args.thin] for s in spikes]
                name = f"{name}_thin{args.thin:g}"
            total, burst, active, _, di, wins = A.observed(spikes, starts, y, dirs)
            items.append((name, spikes, starts, di, n_dir, wins, total, active))
    for name, spikes, starts, di, n_dir, wins, total, active in items:
        good = np.flatnonzero((active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES))
        good = np.array([u for u in good if str(int(u)) not in skip.get(name, set())], int)[args.part::args.nparts]
        done = res.get(name, {}); t0 = time.time()
        for u in good:
            if str(int(u)) in done:
                continue
            r = run_unit(np.asarray(spikes[u]), wins[u], starts, di, n_dir, rng)
            r["tonic"] = ((total[u]) / np.maximum(active[u], 1)).tolist()
            done[str(int(u))] = r; res[name] = done; out.write_text(json.dumps(res))
        v = list(done.values())
        if not v:
            continue
        f = lambda key, grp: 100 * np.mean([x[grp][key] < 0.01 for x in v])
        print(f"{name}: N={len(v)} | p8<0.01 data {f('p8','data'):.0f}% null {f('p8','null'):.0f}% | "
              f"p7<0.01 data {f('p7','data'):.0f}% null {f('p7','null'):.0f}% | "
              f"pk7<0.01 data {f('pk7','data'):.0f}% null {f('pk7','null'):.0f}% | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
