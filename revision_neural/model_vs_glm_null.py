"""Does the two-compartment (V-prob) model's coincidence signature survive the
same single-compartment GLM null that we apply to the data?

The paper's Fig. 2G-I population (N=100 units, basal drive U(0.3,2.0), apical
drive 4.0 for the preferred goal and 0 otherwise) is re-simulated here with the
exact neuron dynamics of figure_combined_vprob_tuning_heatmaps.py, but scored
with the DATA pipeline instead of the figure's displayed matrices:
  * trials of 500 ms after a 200 ms warm-up, n trials per direction
  * burst spike = ISI < 10 ms (not the model's internal burst flag)
  * DSI = (max-min)/(max+min) on tonic rate and burst fraction (audit summ)
  * jitter null and fitted single-compartment GLM null (single_compartment_null)
This gives the model's expected dDSI excess over the GLM null, to compare with
the ~+0.03-0.07 measured in monkey C.

  --graded  cosine-graded apical drive over 8 directions instead of one-hot
            over 4 goals (closer to the recorded tuning)
Usage: python model_vs_glm_null.py [--graded] [--n-trials 40] [--n-surr 100]
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

from audit_monkeyC_nulls import summ, jitter_null, WIN, ISI
from single_compartment_null import (aggregate, glm_design, fit_glm, simulate_glm, isi_stats,
                                     fmt_isi, pooled_burst_fraction, score, N_TIME_BASIS, PRE)

ROOT = Path(__file__).resolve().parent

# V-prob burst-neuron parameters (figure_combined_vprob_tuning_heatmaps.py)
DT = 1.0; TAU_S = 5.0; TAU_D = 5.0
V_THRESH = 1.0; V_RESET = 0.0; V_RESET_BURST = 0.9
V_BURST = 0.75; BETA_S = 0.5; BETA_D = 0.2
T_REF = 5.0; T_REF_BURST = 1.5; T_BURST_REF_D = 30.0
N_BURST_STEPS = 15
I_API_SCALE = 4.0
HAZARD_SCALE = 1.0          # multiplies the per-ms somatic spike probability; 1.0 = published model
BASAL_DEPTH = 0.0           # cosine modulation depth of the basal drive across directions (0 = published)
TRIAL_CV = 0.0              # trial-to-trial variability of the basal drive (CV of a gamma gain); 0 = none
API_CV = 0.0                # heterogeneity of the apical drive across units (CV of a gamma distribution with mean I_API_SCALE); 0 = same drive for all units
G_COUPLE = 0.0              # ohmic soma-dendrite coupling g_c (Reviewer 1, comment 5); 0 = published model
WARM_MS = 200
TRIAL_GAP = 10.0            # s between synthetic trial onsets (absolute times for the GLM)


def sigm(x, beta):
    return 1.0 / (1.0 + np.exp(-np.clip(x / beta, -500, 500)))


def simulate_trials(I_bas, I_api, n_trials, T_ms, rng):
    """Vectorised port of simulate_neuron_tv: n_trials independent runs of one
    neuron with constant drives. Returns a list of spike-time arrays (ms)."""
    n = n_trials
    I_bas = np.broadcast_to(np.asarray(I_bas, float), (n,))
    Vs = np.full(n, 0.2); Vd = np.zeros(n); ref = np.zeros(n)
    burst_t = np.zeros(n); burst_ref = np.zeros(n)
    spikes = [[] for _ in range(n)]
    for t in range(T_ms):
        if t % N_BURST_STEPS == 0:
            burst_ref = np.maximum(burst_ref - N_BURST_STEPS * DT, 0.0)
            ps = sigm(Vs - V_THRESH, BETA_S); pd = sigm(Vd - V_BURST, BETA_D)
            start = (burst_ref <= 0) & (rng.random(n) < ps * pd)
            burst_t[start] = N_BURST_STEPS * DT; burst_ref[start] = T_BURST_REF_D; Vd[start] = 0.0
        Vs_prev = Vs.copy()
        Vd += DT / TAU_D * (-Vd + I_api + G_COUPLE * (Vs_prev - Vd))
        burst_t = np.maximum(burst_t - DT, 0.0); in_burst = burst_t > 0
        refractory = ref > 0
        ref[refractory] -= DT
        free = ~refractory
        Vs[free] += DT / TAU_S * (-Vs[free] + I_bas[free] + G_COUPLE * (Vd[free] - Vs[free]))
        fire = free & (rng.random(n) < sigm(Vs - V_THRESH, BETA_S) * DT * HAZARD_SCALE)
        fb = fire & in_burst; ft = fire & ~in_burst
        ref[fb] = T_REF_BURST; Vs[fb] = V_RESET_BURST
        ref[ft] = T_REF; Vs[ft] = V_RESET
        for i in np.flatnonzero(fire):
            spikes[i].append(t)
    return [np.asarray(s, float) for s in spikes]


def build_population(n_units, n_trials, graded, rng, i_bas_range=(0.3, 2.0)):
    """Spike windows for every unit/trial, plus absolute spike times and trial
    onsets so the GLM machinery can be reused unchanged. i_bas_range=(0.3,2.0)
    is the published population; at these values the units fire >100 Hz, so a
    lower range (e.g. -1.5..-0.3) gives cortical tonic rates of ~5-40 Hz."""
    n_dir = 8 if graded else 4
    directions = np.arange(n_dir)
    I_bas = rng.uniform(i_bas_range[0], i_bas_range[1], n_units)
    api_scale = I_API_SCALE * (rng.gamma(1.0 / API_CV ** 2, API_CV ** 2, n_units) if API_CV > 0 else np.ones(n_units))
    pref = rng.integers(0, n_dir, n_units)
    di = np.repeat(np.arange(n_dir), n_trials)
    starts = np.arange(len(di)) * TRIAL_GAP
    T_ms = WARM_MS + int(WIN * 1000)
    wins = [[None] * len(di) for _ in range(n_units)]
    spikes_abs = []
    for u in range(n_units):
        allspk = []
        for d in range(n_dir):
            if graded:
                api = api_scale[u] * max(0.0, np.cos(2 * np.pi * (d - pref[u]) / n_dir))
            else:
                api = I_API_SCALE if d == pref[u] else 0.0
            # optional cosine tuning of the basal (state) drive around the same
            # preferred direction, so that tonic rate is direction-tuned as in M1
            bas = I_bas[u] * (1.0 + BASAL_DEPTH * np.cos(2 * np.pi * (d - pref[u]) / n_dir))
            if TRIAL_CV > 0:   # per-trial gain on the basal drive (state varies from trial to trial)
                bas = bas * rng.gamma(1.0 / TRIAL_CV ** 2, TRIAL_CV ** 2, n_trials)
            trials = simulate_trials(bas, api, n_trials, T_ms, rng)
            for j, ti in enumerate(np.flatnonzero(di == d)):
                s = (trials[j] + rng.random(len(trials[j])) - WARM_MS) / 1000.0   # s, window at 0
                wins[u][ti] = s[(s >= 0) & (s < WIN)]
                allspk.append(starts[ti] + s[s >= -PRE])
        spikes_abs.append(np.sort(np.concatenate(allspk)) if allspk else np.empty(0))
    return dict(wins=wins, spikes=spikes_abs, starts=starts, di=di, directions=directions,
                I_bas=I_bas, pref=pref, api_scale=api_scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-units", type=int, default=100)
    ap.add_argument("--n-trials", type=int, default=40, help="trials per direction")
    ap.add_argument("--n-surr", type=int, default=100)
    ap.add_argument("--graded", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--i-bas", type=float, nargs=2, default=(0.3, 2.0), metavar=("LO", "HI"),
                    help="uniform range of the basal drive (published: 0.3 2.0)")
    ap.add_argument("--i-api", type=float, default=4.0, help="apical drive for the preferred goal (published: 4.0)")
    ap.add_argument("--api-cv", type=float, default=0.0, help="CV of a gamma distribution of the apical drive across units (0 = homogeneous)")
    ap.add_argument("--hazard-scale", type=float, default=1.0,
                    help="scale of the per-ms somatic spike probability; 0.1 brings tonic rates to ~10-20 Hz")
    ap.add_argument("--g-couple", type=float, default=0.0, help="ohmic soma-dendrite coupling g_c")
    ap.add_argument("--trial-cv", type=float, default=0.0, help="trial-to-trial CV of the basal drive")
    ap.add_argument("--out", default="model_vs_glm_null.json", help="results json (use distinct files for parallel runs)")
    ap.add_argument("--basal-depth", type=float, default=0.0,
                    help="cosine tuning depth of the basal drive (0 = untuned, as published)")
    args = ap.parse_args()
    global I_API_SCALE, HAZARD_SCALE, BASAL_DEPTH, G_COUPLE, TRIAL_CV, API_CV
    G_COUPLE = args.g_couple
    TRIAL_CV = args.trial_cv
    I_API_SCALE = args.i_api
    API_CV = args.api_cv
    HAZARD_SCALE = args.hazard_scale
    BASAL_DEPTH = args.basal_depth
    rng = np.random.default_rng(args.seed)
    tag = (("graded8" if args.graded else "onehot4") + f"_ibas{args.i_bas[0]:g}to{args.i_bas[1]:g}"
           + f"_iapi{args.i_api:g}_haz{args.hazard_scale:g}_bas{args.basal_depth:g}"
           + (f"_gc{args.g_couple:g}" if args.g_couple else "") + (f"_tcv{args.trial_cv:g}" if args.trial_cv else "") + (f"_apicv{args.api_cv:g}" if args.api_cv else "")
           + (f"_ib{args.i_bas[0]:g}-{args.i_bas[1]:g}" if tuple(args.i_bas) != (0.3, 2.0) else ""))

    t0 = time.time()
    pop = build_population(args.n_units, args.n_trials, args.graded, rng, tuple(args.i_bas))
    wins, di, directions = pop["wins"], pop["di"], pop["directions"]
    n_dir = len(directions)
    total, burst, active = aggregate(wins, di, n_dir)
    # burst-independent unit selection (selecting on burst counts biases the test; see SM2 S2)
    good = (active >= 5).all(1) & (total.sum(1) >= 100)
    obs = summ(total, burst, active, good)
    res = dict(model=tag, n_units=int(good.sum()), n_trials_per_dir=args.n_trials,
               obs_d=obs[0], obs_p=obs[1], obs_bf=pooled_burst_fraction(total, burst, good),
               obs_isi=isi_stats(wins, good))
    print(f"\nV-prob model [{tag}]: N={good.sum()}/{args.n_units} units, {args.n_trials} trials/dir, "
          f"{n_dir} dirs, simulated in {time.time()-t0:.0f}s")
    print(f"  observed    : dDSI={obs[0]:.3f} %above={obs[1]:.0f} burst fraction={res['obs_bf']:.3f}")
    print(f"  observed    : {fmt_isi(res['obs_isi'])}")

    jit = np.array([summ(*jitter_null(wins, di, directions, rng), good) for _ in range(args.n_surr)])
    res.update(jit_d=jit[:, 0].mean(), jit_p=jit[:, 1].mean(),
               p_jit_d=(1 + np.sum(jit[:, 0] >= obs[0])) / (args.n_surr + 1))
    print(f"  jitter null : dDSI={res['jit_d']:.3f} (p={res['p_jit_d']:.3g})  %above={res['jit_p']:.0f}")

    t0 = time.time(); fits = []; n_fail = 0
    for u in range(args.n_units):
        if not good[u]:
            fits.append(None); continue
        X, yy, T, H = glm_design(pop["spikes"][u], pop["starts"], di, n_dir)
        coef, b0, nit, ok = fit_glm(X, yy)
        n_fail += (not ok)
        fits.append((coef, b0, T, H))
    print(f"  glm fitted for {good.sum()} units in {time.time()-t0:.0f}s ({n_fail} not converged)")
    sims = []
    for i in range(args.n_surr):
        w_sim = [simulate_glm(*fits[u], di, n_dir, rng) if fits[u] is not None
                 else [np.empty(0)] * len(di) for u in range(args.n_units)]
        if i == 0:
            res["glm_isi"] = isi_stats(w_sim, good)
        (dd, pp), bf = score(w_sim, di, n_dir, good)
        sims.append((dd, pp, bf))
    sims = np.array(sims)
    h_all = np.array([f[3] @ f[0][n_dir * N_TIME_BASIS:] for f in fits if f is not None])
    hm = np.median(h_all, axis=0)
    res.update(glm_d=sims[:, 0].mean(), glm_p=sims[:, 1].mean(), glm_bf=sims[:, 2].mean(),
               p_glm_d=(1 + np.sum(sims[:, 0] >= obs[0])) / (args.n_surr + 1),
               excess_over_glm=obs[0] - sims[:, 0].mean(), excess_over_jitter=obs[0] - jit[:, 0].mean())
    print(f"  glm fit     : {fmt_isi(res['glm_isi'])}")
    print(f"  glm null    : dDSI={res['glm_d']:.3f} (p={res['p_glm_d']:.3g})  %above={res['glm_p']:.0f}  "
          f"burst fraction={res['glm_bf']:.3f} | history filter 1-2ms={hm[:2].mean():+.2f}, "
          f"3-8ms={hm[2:8].mean():+.2f}, 10-30ms={hm[10:30].mean():+.2f}")
    print(f"  => model excess over GLM null {res['excess_over_glm']:+.3f}, over jitter null "
          f"{res['excess_over_jitter']:+.3f}  (monkey C data: +0.036 / +0.089)")

    out = ROOT / args.out
    allres = json.load(open(out)) if out.exists() else {}
    allres[f"{tag}_trials{args.n_trials}_seed{args.seed}"] = res
    out.write_text(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
