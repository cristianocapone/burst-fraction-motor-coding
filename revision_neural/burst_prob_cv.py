"""Cross-validated burst probability at matched expected rate, aligned to each unit's OWN gate direction.

No assumption that the gate follows the rate tuning. For every scored spike k (as in gate_ms.py):
y_k = 1 if another spike follows within 10 ms; rho_k = expected spikes in those 10 ms under the
additive GLM (direction-specific rate, shared history, common burst term; counterfactual history).
The gate direction theta_gate is estimated on one half of the trials (odd or even) from the
observed / expected burst ratio per direction, R_d = sum y / sum (1 - exp(-rho)), as the phase of
the first Fourier component of log R_d. On the OTHER half, spikes are binned by rho and by
direction class relative to theta_gate (gate-preferred: |d - theta| <= 60 deg; opposite: >= 120 deg).
Both splits are pooled. Same pipeline on spikes simulated from the ungated GLM with trial gain.
A stable gate gives P(burst | rho, gate-preferred) > P(burst | rho, opposite) on held-out trials,
above the null; a multiplicative gate gives a ratio roughly constant across rho.

Usage: python burst_prob_cv.py --dataset M --session 20140929 --out bpmcv/M_20140929.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np

import audit_monkeyC_nulls as A
import single_compartment_null as S
import gated_glm2 as G
import gate_ms as MS

EDGES = np.logspace(np.log10(0.02), np.log10(5.0), 13)


def per_spike(X, y, T, H, th_c, di, n_dir):
    """rho_k, direction index d_k, trial index t_k, y_k for every scored spike."""
    ang = 2 * np.pi * np.arange(n_dir) / n_dir
    rho, a, yy = MS.per_spike_rho(X, y, T, H, th_c, di, n_dir, 0.0)
    # recover direction from a = (1 + cos(d)) / 2 is ambiguous; recompute d and trial order directly
    n_tr, n_bins = len(di), T.shape[0]
    rasters = y.reshape(n_tr, n_bins)
    n_pre = int(round(S.PRE / S.DT))
    d_k, t_k = [], []
    for ti in range(n_tr):
        nb = len(np.flatnonzero(rasters[ti][n_pre:n_bins - MS.L]))
        d_k += [di[ti]] * nb; t_k += [ti] * nb
    return rho, np.array(d_k), np.array(t_k), yy


def theta_gate(rho, d, y, mask, n_dir):
    ang = 2 * np.pi * np.arange(n_dir) / n_dir
    exp_b = -np.expm1(-rho)
    R = np.array([(y[mask & (d == k)].sum() + 0.5) / (exp_b[mask & (d == k)].sum() + 0.5) for k in range(n_dir)])
    L = np.log(R)
    return float(np.angle(np.sum((L - L.mean()) * np.exp(1j * ang))))


def hist_cv(rho, d, t, y, n_dir):
    ang = 2 * np.pi * np.arange(n_dir) / n_dir
    n = np.zeros((3, len(EDGES) - 1)); k = np.zeros_like(n)
    for fit_half in (0, 1):
        fit = (t % 2) == fit_half
        th = theta_gate(rho, d, y, fit, n_dir)
        test = ~fit
        a = 0.5 * (1 + np.cos(ang[d[test]] - th))
        cls = np.where(a >= 0.75, 0, np.where(a <= 0.25, 2, 1))
        b = np.clip(np.searchsorted(EDGES, rho[test], side="right") - 1, 0, len(EDGES) - 2)
        np.add.at(n, (cls, b), 1); np.add.at(k, (cls, b), y[test])
    return dict(n=n.tolist(), k=k.tolist(), n_spk=int(len(y)), n_burst=int(y.sum()))


def pipeline(spk_abs, starts, di, n_dir):
    X, y, T, H = S.glm_design(spk_abs, starts, di, n_dir)
    th0, _ = G.fit(X, y)
    B = G.burst_cov(y, len(di), T.shape[0]).astype(np.float32)
    thc, _ = G.fit(np.concatenate([X, B[:, None]], 1), y, G.warm(th0, 1))
    rho, d, t, yy = per_spike(X, y, T, H, thc, di, n_dir)
    return hist_cv(rho, d, t, yy, n_dir), (th0, T, H)


def run_unit(spk_abs, wins_u, starts, di, n_dir, rng):
    data, (th0, T, H) = pipeline(spk_abs, starts, di, n_dir)
    cv = G.gain_cv(wins_u, di, n_dir)
    w = G.simulate_with_gain(th0, T, H, di, n_dir, cv, rng)
    spk = np.sort(np.concatenate([starts[ti] + w[ti] for ti in range(len(di))]))
    null, _ = pipeline(spk, starts, di, n_dir)
    return dict(data=data, null=null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--dataset", default="C", choices=["C", "M", "T", "J", "indy"])
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    rng = np.random.default_rng(29)
    for name, (spikes, starts, y) in S.load_sessions(args.dataset, args.session):
        dirs = np.sort(np.unique(y)); n_dir = len(dirs)
        total, burst, active, _, di, wins = A.observed(spikes, starts, y, dirs)
        good = np.flatnonzero((active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES))
        if args.max_units:
            good = good[:args.max_units]
        done = res.get(name, {}); t0 = time.time()
        for u in good:
            if str(int(u)) in done:
                continue
            done[str(int(u))] = run_unit(np.asarray(spikes[u]), wins[u], starts, di, n_dir, rng)
            res[name] = done; out.write_text(json.dumps(res))
        v = list(done.values())
        if not v:
            continue
        P = lambda g, c: (sum(np.array(x[g]["k"])[c] for x in v).sum() / max(sum(np.array(x[g]["n"])[c] for x in v).sum(), 1))
        print(f"{name}: N={len(v)} | held-out P(burst) gate-pref/opp data {P('data',0):.3f}/{P('data',2):.3f} null {P('null',0):.3f}/{P('null',2):.3f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
