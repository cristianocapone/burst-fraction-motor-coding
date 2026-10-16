"""Burst probability at matched expected rate, preferred vs opposite direction.

For every scored spike k: y_k = 1 if another spike follows within 10 ms; rho_k = expected
spikes in those 10 ms under the ADDITIVE model (GLM with direction-specific rate, shared
history, common burst term; counterfactual history), as in gate_ms.py. Spikes are binned by
rho_k and by direction class relative to the unit's rate-preferred direction theta:
  pref: |d - theta| <= 60 deg (a_k >= 0.75), opp: |d - theta| >= 120 deg (a_k <= 0.25), mid: else.
Per unit and bin we store the number of spikes n and of followed spikes k, for the data and
for spikes simulated from the unit's ungated GLM with trial gain (null). Pooled over units,
P(burst | rho, class) in the data vs the null shows a gate directly: at the same expected
rate, a higher burst probability in the preferred direction. A multiplicative gate gives a
roughly constant ratio pref/opp across rho; an additive one a ratio that falls towards 1.

Usage: python burst_prob_matched.py --dataset C --session 20150309 --out bpm/C_20150309.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np

import audit_monkeyC_nulls as A
import single_compartment_null as S
import gated_glm2 as G
import gate_ms as MS

EDGES = np.logspace(np.log10(0.02), np.log10(5.0), 13)   # 12 bins of rho


def hist(rho, a, y):
    b = np.clip(np.searchsorted(EDGES, rho, side="right") - 1, 0, len(EDGES) - 2)
    cls = np.where(a >= 0.75, 0, np.where(a <= 0.25, 2, 1))
    n = np.zeros((3, len(EDGES) - 1)); k = np.zeros_like(n)
    np.add.at(n, (cls, b), 1); np.add.at(k, (cls, b), y)
    return dict(n=n.tolist(), k=k.tolist(), n_spk=int(len(y)), n_burst=int(y.sum()), rho_mean=float(rho.mean()))


def pipeline(spk_abs, trial_spikes, starts, di, n_dir):
    X, y, T, H = S.glm_design(spk_abs, starts, di, n_dir)
    th0, _ = G.fit(X, y)
    B = G.burst_cov(y, len(di), T.shape[0]).astype(np.float32)
    thc, _ = G.fit(np.concatenate([X, B[:, None]], 1), y, G.warm(th0, 1))
    theta = MS.pref_angle(trial_spikes, di, n_dir)
    rho, a, yy = MS.per_spike_rho(X, y, T, H, thc, di, n_dir, theta)
    res = hist(rho, a, yy); res["theta"] = theta
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
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    rng = np.random.default_rng(23)
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
        print(f"{name}: N={len(v)} | P(burst) pref/opp data {P('data',0):.3f}/{P('data',2):.3f} null {P('null',0):.3f}/{P('null',2):.3f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
