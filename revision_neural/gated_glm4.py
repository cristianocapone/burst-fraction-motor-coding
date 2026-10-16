"""Two checks on the direction-specific burst term g_d (7-df test of gated_glm2.py):

(1) Null with DIRECTION-SPECIFIC trial-to-trial gain. In free reaching (Indy) the direction classes
    differ in kinematics, so count variance can differ between directions. A single gain CV may then
    leave direction-dependent overdispersion that the g_d term absorbs. Here the gamma gain CV is
    estimated separately for each direction and used in the null simulation.
(2) Split-half consistency of the gate direction. Md is refitted on odd and on even trials; the phase
    of the first Fourier component of g_d is compared between halves. A real tuned gate gives
    consistent phases (|dphi| small); noise gives |dphi| uniform (mean 90 deg). Same on the null.

Output per unit: p7, g (full), g_odd, g_even for data and null, gain CV per direction.
Usage: python gated_glm4.py --session 20150309 --out gated4/C_20150309.json
       python gated_glm4.py --dataset indy --session indy_20160921_01 --out gated4/I_indy_20160921_01.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2

import audit_monkeyC_nulls as A
import single_compartment_null as S
import gated_glm2 as G


def gain_cv_dir(wins, di, n_dir):
    c = np.array([len(w) for w in wins], float)
    out = np.zeros(n_dir)
    for d in range(n_dir):
        x = c[di == d]
        if len(x) > 2 and x.mean() > 0:
            out[d] = np.sqrt(np.clip((x.var(ddof=1) - x.mean()) / x.mean() ** 2, 0.0, 0.25))
    return out


def simulate_dir_gain(th0, T, H, di, n_dir, cv_dir, rng):
    p_rate = n_dir * T.shape[1]
    coef, b0 = th0[:-1], th0[-1]
    n_bins, n_lag = T.shape[0], H.shape[0]
    n_pre = int(round(S.PRE / S.DT))
    h_rev = (H @ coef[p_rate:])[::-1]
    out = [None] * len(di)
    for d in range(n_dir):
        ix = np.flatnonzero(di == d)
        cv = cv_dir[d]
        gains = rng.gamma(1 / cv ** 2, cv ** 2, len(ix)) if cv > 0 else np.ones(len(ix))
        base = np.exp(b0 + T @ coef[d * T.shape[1]:(d + 1) * T.shape[1]])
        Sm = np.zeros((len(ix), n_bins + n_lag))
        for t in range(n_bins):
            mu = np.minimum(gains * base[t] * np.exp(Sm[:, t:t + n_lag] @ h_rev), S.LAM_CAP * S.DT)
            spike = (rng.random(len(ix)) < 1 - np.exp(-mu)) & (Sm[:, t + n_lag - 1] == 0)
            Sm[:, t + n_lag] = spike
        for j, ti in enumerate(ix):
            bins = np.flatnonzero(Sm[j, n_lag:]) - n_pre
            out[ti] = (bins + rng.random(len(bins))) * S.DT
    return out


def fits(spk_abs, starts, di, n_dir):
    X, y, T, H = S.glm_design(spk_abs, starts, di, n_dir)
    n_tr, n_bins = len(di), T.shape[0]
    th0, nll0 = G.fit(X, y)
    B = G.burst_cov(y, n_tr, n_bins).astype(np.float32)
    D = np.zeros((len(y), n_dir), np.float32); D[np.arange(len(y)), np.repeat(di, n_bins)] = 1.0
    Xc = np.concatenate([X, B[:, None]], 1); thc, nllc = G.fit(Xc, y, G.warm(th0, 1))
    Xd = np.concatenate([X, B[:, None] * D], 1); thd, nlld = G.fit(Xd, y, G.warm(th0, n_dir))
    lr7 = 2 * (nllc - nlld)
    res = dict(g=thd[-1 - n_dir:-1].tolist(), lr7=float(lr7), p7=float(chi2.sf(max(lr7, 0), n_dir - 1)))
    # split halves (rows of odd / even trials), warm start from the full fit
    rows = np.repeat(np.arange(n_tr) % 2, n_bins)
    for name, half in (("g_odd", 1), ("g_even", 0)):
        m = rows == half
        thh, _ = G.fit(Xd[m], y[m], thd)
        res[name] = thh[-1 - n_dir:-1].tolist()
    return res, (th0, T, H)


def run_unit(spk_abs, wins_u, starts, di, n_dir, rng):
    data, (th0, T, H) = fits(spk_abs, starts, di, n_dir)
    cv_dir = gain_cv_dir(wins_u, di, n_dir)
    w = simulate_dir_gain(th0, T, H, di, n_dir, cv_dir, rng)
    spk = np.sort(np.concatenate([starts[ti] + w[ti] for ti in range(len(di))]))
    null, _ = fits(spk, starts, di, n_dir)
    return dict(data=data, null=null, cv_dir=cv_dir.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--dataset", default="C", choices=["C", "M", "T", "J", "indy"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    rng = np.random.default_rng(17)
    for name, (spikes, starts, y) in S.load_sessions(args.dataset, args.session):
        dirs = np.sort(np.unique(y)); n_dir = len(dirs)
        total, burst, active, _, di, wins = A.observed(spikes, starts, y, dirs)
        good = np.flatnonzero((active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES))
        done = res.get(name, {}); t0 = time.time()
        for u in good:
            if str(int(u)) in done:
                continue
            r = run_unit(np.asarray(spikes[u]), wins[u], starts, di, n_dir, rng)
            r["tonic"] = (total[u] / np.maximum(active[u], 1)).tolist()
            done[str(int(u))] = r; res[name] = done; out.write_text(json.dumps(res))
        v = list(done.values())
        print(f"{name}: N={len(v)} | p7<0.01 data {100*np.mean([x['data']['p7']<0.01 for x in v]):.1f}% null(dir gain) {100*np.mean([x['null']['p7']<0.01 for x in v]):.1f}% | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
