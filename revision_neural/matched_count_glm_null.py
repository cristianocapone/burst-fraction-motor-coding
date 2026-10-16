"""Noise-matched burst-vs-tonic selectivity (matched_count_selectivity.py) tested
against the fitted single-compartment GLM null: is the per-spike advantage of
burst spikes larger in the data than in a neuron with the same rate tuning and
fixed intrinsic dynamics? Monkey C sessions; burst-independent unit selection.
Usage: python matched_count_glm_null.py --session 201503 --check out.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import audit_monkeyC_nulls as A
import single_compartment_null as S
import matched_count_selectivity as MC

N_SURR = 20


def session_metric(wins, di, n_dir, good, rng):
    r = MC.run_units(wins, di, n_dir, good, rng)
    return {k: float(v.mean()) for k, v in r.items()} if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--check", required=True)
    ap.add_argument("--model-api", type=float, default=None, help="run on the recalibrated model population with this apical drive")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--model-api-cv", type=float, default=0.0)
    args = ap.parse_args()
    MC.N_REP = 20
    p = Path(args.check); c = json.loads(p.read_text()) if p.exists() else {}
    rng = np.random.default_rng(1)
    if args.model_api is not None:
        import model_vs_glm_null as M
        M.HAZARD_SCALE, M.BASAL_DEPTH, M.TRIAL_CV, M.I_API_SCALE = 0.02, 0.5, 0.3, args.model_api
        M.API_CV = args.model_api_cv
        pop = M.build_population(100, 40, True, np.random.default_rng(args.model_seed))
        sessions = [(f"model_api{args.model_api}_seed{args.model_seed}" + (f"_apicv{args.model_api_cv:g}" if args.model_api_cv else ""), pop)]
    else:
        sessions = S.load_sessions("C", args.session)
    for name, item in sessions:
        if name in c:
            continue
        t0 = time.time()
        if args.model_api is not None:
            spikes, starts, di, wins = item["spikes"], item["starts"], item["di"], item["wins"]
            n_dir = 8
            total, burst, active = S.aggregate(wins, di, n_dir)
        else:
            spikes, starts, y = item
            dirs = np.sort(np.unique(y)); n_dir = len(dirs)
            total, burst, active, _, di, wins = A.observed(spikes, starts, y, dirs)
        good = (active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES)
        obs = session_metric(wins, di, n_dir, good, rng)
        fits = []
        for u in range(len(spikes)):
            if not good[u]:
                fits.append(None); continue
            X, yy, T, H = S.glm_design(spikes[u], starts, di, n_dir)
            coef, b0, _, _ = S.fit_glm(X, yy)
            fits.append((coef, b0, T, H))
        nulls = []
        for _ in range(N_SURR):
            w = [S.simulate_glm(*fits[u], di, n_dir, rng) if fits[u] is not None else [np.empty(0)] * len(di)
                 for u in range(len(spikes))]
            m = session_metric(w, di, n_dir, good, rng)
            if m:
                nulls.append(m)
        res = dict(obs=obs, null={k: float(np.mean([n[k] for n in nulls])) for k in obs})
        c[name] = res; p.write_text(json.dumps(c, indent=1))
        o, n = res["obs"], res["null"]
        print(f"{name}: per-spike advantage (burst - matched tonic), cross-validated depth: data {o['cv_b']-o['cv_t']:+.3f} vs GLM null "
              f"{n['cv_b']-n['cv_t']:+.3f} | pref agreement: data {o['agree_b']-o['agree_t']:+.3f} vs null {n['agree_b']-n['agree_t']:+.3f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
