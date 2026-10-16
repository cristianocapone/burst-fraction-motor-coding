"""Robustness of the GLM-null result to the burst ISI threshold (5, 10, 15 ms).

The GLM fit does not depend on the burst definition, so each unit is fitted
once, its surrogates are simulated once, and observed data and surrogates are
scored at every threshold. Unit mask: burst-independent (>=5 active trials per
direction and >=100 spikes), which the self-calibration showed to be unbiased.

Usage: python glm_threshold_robustness.py --dataset C [--session 20150309] --check out.json
       python glm_threshold_robustness.py --dataset rtt --check out.json
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

import audit_monkeyC_nulls as A
import single_compartment_null as S

THRESHOLDS_MS = (5, 10, 15)


def score_at(wins, di, n_dir, good, isi_s):
    n = len(wins)
    total = np.zeros((n, n_dir)); burst = np.zeros((n, n_dir)); active = np.zeros((n, n_dir))
    for u in range(n):
        for ti, w in enumerate(wins[u]):
            d = di[ti]; total[u, d] += len(w); active[u, d] += (len(w) > 0)
            if len(w) > 1:
                burst[u, d] += int(np.sum(np.diff(w) < isi_s))
    return A.summ(total, burst, active, good)[0]


def run_session(name, spikes, starts, y, n_surr, seed):
    t0 = time.time()
    directions = np.sort(np.unique(y)); n_dir = len(directions)
    total, burst, active, binned, di, wins = A.observed(spikes, starts, y, directions)
    good = (active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES)
    rng = np.random.default_rng(seed)
    fits = []
    for u in range(len(spikes)):
        if not good[u]:
            fits.append(None); continue
        X, yy, T, H = S.glm_design(spikes[u], starts, di, n_dir)
        coef, b0, _, _ = S.fit_glm(X, yy)
        fits.append((coef, b0, T, H))
    obs = {ms: score_at(wins, di, n_dir, good, ms / 1000) for ms in THRESHOLDS_MS}
    null = {ms: [] for ms in THRESHOLDS_MS}
    for _ in range(n_surr):
        w_sim = [S.simulate_glm(*fits[u], di, n_dir, rng) if fits[u] is not None
                 else [np.empty(0)] * len(di) for u in range(len(spikes))]
        for ms in THRESHOLDS_MS:
            null[ms].append(score_at(w_sim, di, n_dir, good, ms / 1000))
    res = dict(session=name, n_units=int(good.sum()))
    line = f"{name}: N={good.sum()}"
    for ms in THRESHOLDS_MS:
        nl = np.array(null[ms])
        res[f"obs_{ms}"] = obs[ms]; res[f"glm_{ms}"] = float(nl.mean())
        res[f"p_{ms}"] = float((1 + np.sum(nl >= obs[ms])) / (n_surr + 1))
        line += f" | {ms}ms: obs {obs[ms]:.3f} GLM {nl.mean():.3f} excess {obs[ms]-nl.mean():+.3f} p={res[f'p_{ms}']:.3g}"
    print(line + f" | {time.time()-t0:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C", choices=["C", "rtt", "maze"])
    ap.add_argument("--session", default=None)
    ap.add_argument("--n-surr", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--check", required=True)
    args = ap.parse_args()
    path = Path(args.check)
    check = json.loads(path.read_text()) if path.exists() else {}
    for name, (spikes, starts, y) in S.load_sessions(args.dataset, args.session, set(check)):
        check[name] = run_session(name, spikes, starts, y, args.n_surr, args.seed)
        path.write_text(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
