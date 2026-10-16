"""Multiplicative (m) vs autonomous (s) apical gate, fitted per spike on top of the additive GLM.

For every scored spike k (t in [0, WIN-10 ms)), y_k = 1 if another spike follows within 10 ms.
rho_k = expected number of spikes in the 10 ms after spike k under the ADDITIVE model
        (GLM with direction-specific rate, shared history filter and a common burst term g),
        with the history evaluated as if no spike followed spike k (counterfactual history).
a_k   = (1 + cos(d_k - theta)) / 2, theta = rate-preferred direction of the unit.

    P(y_k = 1) = 1 - exp(-mu_k),   mu_k = c * rho_k * (1 + m * a_k) + s * a_k

  c : free scale (calibration of rho)        m : apical multiplies the basal-driven bursting (coincidence)
  s : apical adds bursts on its own          additive context = (m, s) = (0, 0)
Variants fitted by maximum likelihood: base (c), M (c, m), S (c, s), MS (c, m, s); LR against base.
Null: the whole pipeline on spikes simulated from the unit's ungated GLM with trial-to-trial gain.

Usage: python gate_ms.py --session 20150309 --out gatems/C_20150309.json
       python gate_ms.py --dataset indy --session indy_20160921_01 --out gatems/I_indy_20160921_01.json
       python gate_ms.py --model-api 0.5 --model-api-cv 1 --out gatems/m110_api0.5_cv1.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

import audit_monkeyC_nulls as A
import single_compartment_null as S
import gated_glm2 as G

L = 10   # burst window, ms


def pref_angle(trial_spikes, di, n_dir):
    ang = 2 * np.pi * np.arange(n_dir) / n_dir
    c = np.array([len(s[(s >= 0) & (s < A.WIN)]) for s in trial_spikes], float)
    mrate = np.array([c[di == d].mean() for d in range(n_dir)])
    return float(np.angle(np.sum((mrate - mrate.mean()) * np.exp(1j * ang))))


def per_spike_rho(X, y, T, H, th_c, di, n_dir, theta):
    """rho_k, a_k, y_k for every scored spike (counterfactual history after spike k)."""
    n_tr, n_bins = len(di), T.shape[0]
    p_rate = n_dir * T.shape[1]
    coef, b0 = th_c[:-1], th_c[-1]
    h = H @ coef[p_rate:p_rate + H.shape[1]]          # per-lag history weight, index = lag - 1
    g = coef[-1]
    n_lag = len(h)
    eta_rate = ((X[:, :p_rate] @ coef[:p_rate].astype(np.float32)).astype(np.float64) + b0).reshape(n_tr, n_bins)
    rasters = y.reshape(n_tr, n_bins)
    n_pre = int(round(S.PRE / S.DT))
    Hcf = np.zeros((L, n_lag))
    for l in range(1, L + 1):
        i = np.arange(n_lag - l + 1)
        Hcf[l - 1, i] = h[l + i - 1]
    ang = 2 * np.pi * np.arange(n_dir) / n_dir
    rho, a, yy = [], [], []
    for ti in range(n_tr):
        r = rasters[ti]
        pad = np.concatenate([np.zeros(n_lag), r])
        for b in np.flatnonzero(r[n_pre:n_bins - L]) + n_pre:
            past = pad[b + n_lag - np.arange(n_lag)]              # r[b], r[b-1], ..., r[b-99]
            hist_cf = Hcf @ past
            lam = np.exp(np.minimum(eta_rate[ti, b + 1:b + L + 1] + hist_cf + g, 10.0))
            rho.append(lam.sum()); yy.append(float(r[b + 1:b + L + 1].any()))
            a.append(0.5 * (1 + np.cos(ang[di[ti]] - theta)))
    return np.array(rho), np.array(a), np.array(yy)


def fit_ms(rho, a, y):
    def nll(params, use_m, use_s):
        c = np.exp(params[0]); m = params[1] if use_m else 0.0; s = params[2] if use_s else 0.0
        mu = np.maximum(c * rho * (1 + m * a) + s * a, 1e-12)
        p1 = -np.expm1(-mu)
        return -(y * np.log(np.maximum(p1, 1e-300)) - (1 - y) * mu).sum()
    out = {}
    for name, use_m, use_s in (("base", False, False), ("M", True, False), ("S", False, True), ("MS", True, True)):
        bounds = [(-6, 6), (-1.0, 20.0) if use_m else (0, 0), (0.0, 5.0) if use_s else (0, 0)]
        best = None
        for m0, s0 in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.05)):
            x0 = np.array([0.0, m0 if use_m else 0.0, s0 if use_s else 0.0])
            sol = minimize(nll, x0, args=(use_m, use_s), method="L-BFGS-B", bounds=bounds)
            if best is None or sol.fun < best.fun:
                best = sol
        out[name] = dict(c=float(np.exp(best.x[0])), m=float(best.x[1]), s=float(best.x[2]), nll=float(best.fun))
    for v in ("M", "S", "MS"):
        out[v]["lr"] = float(2 * (out["base"]["nll"] - out[v]["nll"]))
    out["dll_M_minus_S"] = float(out["S"]["nll"] - out["M"]["nll"])
    return out


def pipeline(spk_abs, trial_spikes, starts, di, n_dir):
    X, y, T, H = S.glm_design(spk_abs, starts, di, n_dir)
    th0, _ = G.fit(X, y)
    B = G.burst_cov(y, len(di), T.shape[0]).astype(np.float32)
    thc, _ = G.fit(np.concatenate([X, B[:, None]], 1), y, G.warm(th0, 1))
    theta = pref_angle(trial_spikes, di, n_dir)
    rho, a, yy = per_spike_rho(X, y, T, H, thc, di, n_dir, theta)
    res = fit_ms(rho, a, yy)
    res.update(n_spk=int(len(yy)), n_burst=int(yy.sum()), rho_mean=float(rho.mean()), theta=theta)
    return res, (th0, T, H)


def run_unit(spk_abs, wins_u, starts, di, n_dir, rng):
    data, (th0, T, H) = pipeline(spk_abs, [np.asarray(s, float) for s in wins_u], starts, di, n_dir)
    cv = G.gain_cv(wins_u, di, n_dir)
    w = G.simulate_with_gain(th0, T, H, di, n_dir, cv, rng)
    spk = np.sort(np.concatenate([starts[ti] + w[ti] for ti in range(len(di))]))
    null, _ = pipeline(spk, [x[x >= 0] for x in w], starts, di, n_dir)
    return dict(data=data, null=null, gain_cv=cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--dataset", default="C", choices=["C", "M", "T", "J", "indy"])
    ap.add_argument("--model-api", type=float, default=None)
    ap.add_argument("--model-api-cv", type=float, default=0.0)
    ap.add_argument("--model-trials", type=int, default=110)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    rng = np.random.default_rng(13)
    if args.model_api is not None:
        import model_vs_glm_null as M
        M.HAZARD_SCALE, M.BASAL_DEPTH, M.TRIAL_CV, M.I_API_SCALE = 0.02, 0.5, 0.3, args.model_api
        M.API_CV = args.model_api_cv
        pop = M.build_population(100, args.model_trials, True, np.random.default_rng(0))
        name = f"model_api{args.model_api}_cv{args.model_api_cv:g}_trials{args.model_trials}"
        total, burst, active = S.aggregate(pop["wins"], pop["di"], 8)
        items = [(name, pop["spikes"], pop["starts"], pop["di"], 8, pop["wins"], total, active)]
    else:
        items = []
        for name, (spikes, starts, y) in S.load_sessions(args.dataset, args.session):
            dirs = np.sort(np.unique(y))
            total, burst, active, _, di, wins = A.observed(spikes, starts, y, dirs)
            items.append((name, spikes, starts, di, len(dirs), wins, total, active))
    for name, spikes, starts, di, n_dir, wins, total, active in items:
        good = np.flatnonzero((active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES))
        done = res.get(name, {}); t0 = time.time()
        for u in good:
            if str(int(u)) in done:
                continue
            done[str(int(u))] = run_unit(np.asarray(spikes[u]), wins[u], starts, di, n_dir, rng)
            res[name] = done; out.write_text(json.dumps(res))
        v = list(done.values())
        md = np.median([x["data"]["M"]["m"] for x in v]); mn = np.median([x["null"]["M"]["m"] for x in v])
        sd = np.median([x["data"]["S"]["s"] for x in v]); sn = np.median([x["null"]["S"]["s"] for x in v])
        print(f"{name}: N={len(v)} | m data {md:+.3f} null {mn:+.3f} | s data {sd:.4f} null {sn:.4f} | "
              f"LR_M>6.6: data {100*np.mean([x['data']['M']['lr']>6.63 for x in v]):.0f}% null {100*np.mean([x['null']['M']['lr']>6.63 for x in v]):.0f}% | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
