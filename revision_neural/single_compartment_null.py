"""Single-compartment (no-coincidence) nulls for the burst-fraction dDSI claim.

Reviewer 2 asks whether a conventional single-compartment neuron with a
direction-tuned rate and FIXED intrinsic ISI dynamics can reproduce
dDSI = DSI(burst fraction) - DSI(tonic rate) as observed. The paper's nulls do
not test this directly: the Poisson-PSTH null has no intrinsic dynamics, and
the jitter null destroys them. Two fitted nulls close the gap:

  gamma  Rate-modulated renewal process. Per unit: direction-specific PSTH
         (10-ms bins) gives lambda_d(t); ISIs rescaled by the integrated rate
         are fitted with a gamma distribution whose shape kappa is SHARED across
         directions (kappa<1 bursty, kappa=1 Poisson, kappa>1 regular). A
         per-trial gain matches the observed spike-count variance so the null
         has the same count noise (hence the same DSI noise bias) as the data.
  glm    Poisson GLM (1-ms bins) with direction-specific time kernels and a
         spike-history filter (piecewise-constant, 1-100 ms) SHARED across
         directions. Captures refractoriness and intrinsic burstiness; only the
         rate depends on direction.

Both nulls are simulated on the real trial structure and scored with the exact
Fig. S1 pipeline (audit_monkeyC_nulls.observed/summ, same good-unit mask).

--calib additionally runs a calibration of the paper's jitter test: synthetic
non-bursting renewal neurons (fixed kappa, real PSTHs and trial structure) are
treated as data and tested against their own jitter null. If they "beat" it,
the jitter test is anti-conservative for regular/refractory cells.

Usage (homebrew python has numpy/scipy/sklearn/pynwb):
  python single_compartment_null.py --dataset C --session 20150309 --calib
  python single_compartment_null.py --dataset C            # all 15 monkey C CO sessions
  python single_compartment_null.py --dataset maze         # MC_Maze
  python single_compartment_null.py --dataset rtt          # MC_RTT
Outputs single_compartment_null_checkpoint.json (+ per-run log to stdout).
"""
from pathlib import Path
import argparse
import glob
import json
import time
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
from audit_monkeyC_nulls import load as load_C, observed, summ, jitter_null, WIN, ISI, BIN, NB

ROOT = Path(__file__).resolve().parent
CACHE_C = ROOT / "dandi000688_C_cache"
CHECK = ROOT / "single_compartment_null_checkpoint.json"

RATE_FLOOR = 0.5       # spikes/s, keeps the integrated rate strictly increasing
PSTH_SMOOTH = 1.0      # bins of 10 ms
WARM = 0.2             # s of pre-window simulation so the window starts in steady state
DT = 0.001             # GLM bin
PRE = 0.1              # s of pre-window bins included in the GLM fit
HIST_EDGES_MS = np.array([1, 2, 3, 4, 6, 8, 11, 15, 20, 30, 50, 100])
N_TIME_BASIS = 12
GOOD_MODE = "paper"    # unit mask: "paper" (active>=5 & bursts>=5, as Fig. S1) or "total" (burst-independent)
AREA = ""              # "" (all units), "M1" or "PMd" (DANDI 000688 datasets only)
MIN_TOTAL_SPIKES = 100
LAM_CAP = 400.0        # spikes/s, guards GLM simulation against runaway self-excitation
ABS_REFRACTORY_BINS = 1  # no spike in the bin right after a spike (1 ms absolute refractory period)


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def aggregate(wins, di, n_dir):
    """total / burst / active per unit x direction from per-trial spike windows."""
    n = len(wins)
    total = np.zeros((n, n_dir)); burst = np.zeros((n, n_dir)); active = np.zeros((n, n_dir))
    for u in range(n):
        for ti, w in enumerate(wins[u]):
            d = di[ti]; total[u, d] += len(w); active[u, d] += (len(w) > 0)
            if len(w) > 1:
                burst[u, d] += int(np.sum(np.diff(w) < ISI))
    return total, burst, active


def pooled_burst_fraction(total, burst, good):
    return float(burst[good].sum() / max(total[good].sum(), 1))


ISI_BANDS_MS = (0, 3, 10, 30, 1e9)


def isi_stats(wins, good):
    """Fit diagnostics: pooled ISI fractions per band and mean spikes/window
    over the good units. Lets us check that a null reproduces the short-ISI
    regime it is supposed to control for."""
    isi, n_spk, n_win = [], 0, 0
    for u in np.flatnonzero(good):
        for w in wins[u]:
            n_spk += len(w); n_win += 1
            if len(w) > 1:
                isi.append(np.diff(w))
    isi = np.concatenate(isi) * 1000 if isi else np.empty(0)
    frac = [float(np.mean((isi >= a) & (isi < b))) if len(isi) else 0.0
            for a, b in zip(ISI_BANDS_MS[:-1], ISI_BANDS_MS[1:])]
    return dict(isi_frac=frac, spikes_per_window=n_spk / max(n_win, 1))


def fmt_isi(s):
    f = s["isi_frac"]
    return (f"ISI <3ms {100*f[0]:.1f}% | 3-10 {100*f[1]:.1f}% | 10-30 {100*f[2]:.1f}% | >30 {100*f[3]:.1f}% "
            f"| {s['spikes_per_window']:.1f} spk/window")


def psth_rates(binned, di, n_dir):
    """Direction-specific rate profiles, units x directions x NB, spikes/s."""
    n = binned.shape[0]
    lam = np.zeros((n, n_dir, NB))
    for d in range(n_dir):
        ix = np.flatnonzero(di == d)
        lam[:, d] = binned[:, ix].mean(1) / BIN
    if PSTH_SMOOTH > 0:
        lam = gaussian_filter1d(lam, PSTH_SMOOTH, axis=2, mode="nearest")
    return np.maximum(lam, RATE_FLOOR)


def cumulative_rate(lam_ud):
    """Integrated rate on edges of [-WARM, WIN), with the first bin's rate
    extended backwards over the warm-up. Returns (t_edges, Lambda_edges)."""
    n_warm = int(round(WARM / BIN))
    rates = np.concatenate([np.full(n_warm, lam_ud[0]), lam_ud])
    t_edges = np.arange(len(rates) + 1) * BIN - WARM
    Lam = np.concatenate([[0.0], np.cumsum(rates) * BIN])
    return t_edges, Lam


# ----------------------------------------------------------------------------
# gamma renewal null
# ----------------------------------------------------------------------------
def fit_gamma_shape(tau):
    """MLE shape of a unit-mean gamma on rescaled ISIs tau (>0)."""
    tau = np.maximum(np.asarray(tau, float), 1e-6)
    lt, mt = np.log(tau).mean(), tau.mean()

    def nll(logk):
        k = np.exp(logk)
        return -(k * np.log(k) - gammaln(k) + (k - 1) * lt - k * mt)
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.1), np.log(30.0)), method="bounded").x))


def fit_renewal(wins, di, lam):
    """Per-unit shared gamma shape (time-rescaled ISIs pooled over directions)
    and per-unit trial-gain CV matching the spike-count variance."""
    n, n_dir, _ = lam.shape
    kappa = np.ones(n); cvg = np.zeros(n); n_isi = np.zeros(n, int)
    for u in range(n):
        tau = []
        counts = np.array([len(w) for w in wins[u]], float)
        for d in range(n_dir):
            edges = np.arange(NB + 1) * BIN
            Lam = np.concatenate([[0.0], np.cumsum(lam[u, d]) * BIN])
            for ti in np.flatnonzero(di == d):
                w = wins[u][ti]
                if len(w) > 1:
                    tau.append(np.diff(np.interp(w, edges, Lam)))
        if tau:
            tau = np.concatenate(tau)
            n_isi[u] = len(tau)
            if len(tau) >= 10:
                kappa[u] = fit_gamma_shape(tau)
        # gain variance: Var(N) = mu/kappa + cv^2 mu^2, pooled over directions
        num = den = 0.0
        for d in range(n_dir):
            c = counts[di == d]
            if len(c) >= 3 and c.mean() > 0:
                mu, var = c.mean(), c.var(ddof=1)
                num += (var - mu / kappa[u]) * len(c)
                den += mu ** 2 * len(c)
        cvg[u] = np.sqrt(max(num, 0.0) / den) if den > 0 else 0.0
    return kappa, cvg, n_isi


def simulate_renewal(lam, kappa, cvg, di, rng):
    """Spike windows for every unit/trial from the fitted renewal model."""
    n, n_dir, _ = lam.shape
    wins = [[None] * len(di) for _ in range(n)]
    for u in range(n):
        k = kappa[u]
        for d in range(n_dir):
            ix = np.flatnonzero(di == d)
            t_edges, Lam = cumulative_rate(lam[u, d])
            L_tot = Lam[-1]
            gain = rng.gamma(1 / cvg[u] ** 2, cvg[u] ** 2, len(ix)) if cvg[u] > 0 else np.ones(len(ix))
            n_max = int(np.ceil(L_tot * gain.max() + 8 * np.sqrt(L_tot * gain.max() / k) + 20))
            tau = rng.gamma(k, 1.0 / k, (len(ix), n_max)) / gain[:, None]
            cum = np.cumsum(tau, axis=1)
            times = np.interp(cum, Lam, t_edges, right=np.inf)
            for j, ti in enumerate(ix):
                w = times[j]
                wins[u][ti] = w[(w >= 0) & (w < WIN)]
    return wins


# ----------------------------------------------------------------------------
# GLM with shared spike-history filter
# ----------------------------------------------------------------------------
def raised_cosines(n_bins, n_basis):
    t = np.arange(n_bins)
    centers = np.linspace(0, n_bins - 1, n_basis)
    width = centers[1] - centers[0]
    B = np.zeros((n_bins, n_basis))
    for j, c in enumerate(centers):
        x = np.clip((t - c) / width, -1, 1)
        B[:, j] = 0.5 * (1 + np.cos(np.pi * x))
    return B


def history_basis():
    """Piecewise-constant history basis over lags 1..100 ms (lag index 0 = 1 ms)."""
    n_lag = int(HIST_EDGES_MS[-1])
    H = np.zeros((n_lag, len(HIST_EDGES_MS) - 1))
    for j in range(len(HIST_EDGES_MS) - 1):
        H[HIST_EDGES_MS[j] - 1:HIST_EDGES_MS[j + 1] - 1, j] = 1.0
    return H


def glm_design(spikes_u, starts, di, n_dir):
    """Binary spike raster over [-PRE, WIN) per trial and the design matrix."""
    n_pre, n_win = int(round(PRE / DT)), int(round(WIN / DT))
    n_bins = n_pre + n_win
    n_lag = int(HIST_EDGES_MS[-1])
    T = raised_cosines(n_bins, N_TIME_BASIS)
    H = history_basis()
    rasters = np.zeros((len(starts), n_bins), np.float32)
    for ti, t0 in enumerate(starts):
        a, b = np.searchsorted(spikes_u, [t0 - PRE, t0 + WIN])
        idx = np.floor((spikes_u[a:b] - (t0 - PRE)) / DT).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        rasters[ti, idx] = 1.0            # duplicates in one 1-ms bin count once
    # history covariates: lagged spike indicators projected on the basis
    hist = np.zeros((len(starts), n_bins, n_lag), np.float32)
    for lag in range(1, n_lag + 1):
        hist[:, lag:, lag - 1] = rasters[:, :-lag]
    Xh = hist.reshape(-1, n_lag) @ H.astype(np.float32)
    Xt = np.zeros((len(starts), n_bins, n_dir * N_TIME_BASIS), np.float32)
    for ti in range(len(starts)):
        d = di[ti]
        Xt[ti, :, d * N_TIME_BASIS:(d + 1) * N_TIME_BASIS] = T
    X = np.concatenate([Xt.reshape(-1, n_dir * N_TIME_BASIS), Xh], axis=1)
    return X, rasters.reshape(-1), T, H


def fit_glm(X, y, ridge=1e-3):
    """Poisson MLE (log link) by L-BFGS with analytic gradient. X is float32;
    products are done in float32, sums in float64. Returns (coef, intercept)."""
    from scipy.optimize import minimize
    n, p = X.shape
    y = y.astype(np.float64)
    ysum = y.sum()

    def f(theta):
        beta, b0 = theta[:p], theta[p]
        eta = (X @ beta.astype(np.float32)).astype(np.float64) + b0
        mu = np.exp(np.minimum(eta, 10.0))
        nll = mu.sum() - (y * eta).sum() + 0.5 * ridge * (beta ** 2).sum()
        r = (mu - y).astype(np.float32)
        g = np.empty(p + 1)
        g[:p] = (X.T @ r).astype(np.float64) + ridge * beta
        g[p] = mu.sum() - ysum
        return nll, g

    theta0 = np.zeros(p + 1)
    theta0[p] = np.log(max(ysum, 1) / n)
    sol = minimize(f, theta0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=3000, maxfun=6000, gtol=1e-6, ftol=1e-12))
    return sol.x[:p], float(sol.x[p]), sol.nit, bool(sol.success)


def simulate_glm(coef, intercept, T, H, di, n_dir, rng, keep_pre=False):
    """Sequential 1-ms simulation of the fitted GLM on the real trial structure.
    keep_pre=True also returns the pre-window spikes (times in [-PRE, WIN)),
    needed when the synthetic data are re-fitted for calibration."""
    n_bins, n_lag = T.shape[0], H.shape[0]
    n_pre = int(round(PRE / DT))
    h = H @ coef[n_dir * N_TIME_BASIS:]                    # per-lag history weight
    h_rev = h[::-1]
    base = np.zeros((n_dir, n_bins))
    for d in range(n_dir):
        base[d] = np.exp(intercept + T @ coef[d * N_TIME_BASIS:(d + 1) * N_TIME_BASIS])
    out = [None] * len(di)
    for d in range(n_dir):
        ix = np.flatnonzero(di == d)
        S = np.zeros((len(ix), n_bins + n_lag))           # left-padded with n_lag zeros
        for t in range(n_bins):
            drive = S[:, t:t + n_lag] @ h_rev
            # base is the fitted expected count per 1-ms bin (the GLM was fitted
            # on per-bin counts), so no further DT scaling here.
            mu = np.minimum(base[d, t] * np.exp(drive), LAM_CAP * DT)
            spike = rng.random(len(ix)) < 1 - np.exp(-mu)
            if ABS_REFRACTORY_BINS:
                spike &= S[:, t + n_lag - ABS_REFRACTORY_BINS:t + n_lag].sum(1) == 0
            S[:, t + n_lag] = spike
        spk = S[:, n_lag:]
        for j, ti in enumerate(ix):
            bins = np.flatnonzero(spk[j]) - n_pre
            w = (bins + rng.random(len(bins))) * DT
            out[ti] = w if keep_pre else w[(w >= 0) & (w < WIN)]
    return out


def glm_calibration(fits, good, starts, di, n_dir, directions, n_rep, n_surr, rng):
    """Is the GLM test itself calibrated? Simulate data from the fitted GLMs
    (no direction-specific dynamics), treat them as observed, re-fit the GLM on
    them, and test against the re-fitted null. Returns per-replicate p-values."""
    out = []
    for rep in range(n_rep):
        t0 = time.time()
        syn_full = [simulate_glm(*fits[u], di, n_dir, rng, keep_pre=True) if fits[u] is not None
                    else [np.empty(0)] * len(di) for u in range(len(fits))]
        syn = [[w[(w >= 0) & (w < WIN)] for w in wu] for wu in syn_full]
        tot_s, bur_s, act_s = aggregate(syn, di, n_dir)
        if GOOD_MODE == "paper":
            good_s = good & (act_s >= 5).all(1) & (bur_s.sum(1) >= 5)
        else:
            good_s = good & (act_s >= 5).all(1) & (tot_s.sum(1) >= MIN_TOTAL_SPIKES)
        o_s = summ(tot_s, bur_s, act_s, good_s)
        refits = []
        for u in range(len(fits)):
            if not good_s[u]:
                refits.append(None); continue
            spikes_u = np.sort(np.concatenate([starts[ti] + w for ti, w in enumerate(syn_full[u])]))
            X, yy, T, H = glm_design(spikes_u, starts, di, n_dir)
            coef, b0, _, _ = fit_glm(X, yy)
            refits.append((coef, b0, T, H))
        null = []
        for _ in range(n_surr):
            w_sim = [simulate_glm(*refits[u], di, n_dir, rng) if refits[u] is not None
                     else [np.empty(0)] * len(di) for u in range(len(fits))]
            null.append(score(w_sim, di, n_dir, good_s)[0][0])
        null = np.array(null)
        p = (1 + np.sum(null >= o_s[0])) / (n_surr + 1)
        out.append(dict(syn_d=o_s[0], null_d=float(null.mean()), p=float(p), n_units=int(good_s.sum())))
        print(f"  GLM-CALIB rep {rep+1}/{n_rep}: synthetic dDSI={o_s[0]:.3f} vs re-fitted GLM null "
              f"{null.mean():.3f}  p={p:.3g}  (N={good_s.sum()}, {time.time()-t0:.0f}s)", flush=True)
    return out


# ----------------------------------------------------------------------------
# session driver
# ----------------------------------------------------------------------------
def score(wins, di, n_dir, good):
    total, burst, active = aggregate(wins, di, n_dir)
    return summ(total, burst, active, good), pooled_burst_fraction(total, burst, good)


def run_session(name, spikes, starts, y, models, n_surr, seed, calib_kappas=(), n_glm_calib=0):
    t_start = time.time()
    directions = np.sort(np.unique(y)); n_dir = len(directions)
    total, burst, active, binned, di, wins = observed(spikes, starts, y, directions)
    if GOOD_MODE == "paper":
        good = (active >= 5).all(1) & (burst.sum(1) >= 5)
    else:
        # burst-independent unit selection: removes the selection bias that
        # arises when units are kept because they happened to burst
        good = (active >= 5).all(1) & (total.sum(1) >= MIN_TOTAL_SPIKES)
    obs = summ(total, burst, active, good)
    res = dict(session=name, good_mode=GOOD_MODE, n_units=int(good.sum()), n_trials=int(len(starts)),
               n_dir=int(n_dir), obs_d=obs[0], obs_p=obs[1], obs_bf=pooled_burst_fraction(total, burst, good))
    res["obs_isi"] = isi_stats(wins, good)
    print(f"\n{name}: N={good.sum()} units, {len(starts)} trials, {n_dir} dirs | "
          f"observed dDSI={obs[0]:.3f} %above={obs[1]:.0f} burst fraction={res['obs_bf']:.3f}")
    print(f"  observed    : {fmt_isi(res['obs_isi'])}")
    rng = np.random.default_rng(seed)

    # paper's jitter null on the same mask, for reference
    jit = np.array([summ(*jitter_null(wins, di, directions, rng), good) for _ in range(n_surr)])
    res.update(jit_d=jit[:, 0].mean(), jit_p=jit[:, 1].mean(),
               p_jit_d=(1 + np.sum(jit[:, 0] >= obs[0])) / (n_surr + 1))
    print(f"  jitter null : dDSI={res['jit_d']:.3f} (p={res['p_jit_d']:.3g})  %above={res['jit_p']:.0f}")

    lam = psth_rates(binned, di, n_dir)

    if "gamma" in models:
        kappa, cvg, n_isi = fit_renewal(wins, di, lam)
        sims = []
        for i in range(n_surr):
            w_sim = simulate_renewal(lam, kappa, cvg, di, rng)
            if i == 0:
                res["gamma_isi"] = isi_stats(w_sim, good)
            (dd, pp), bf = score(w_sim, di, n_dir, good)
            sims.append((dd, pp, bf))
        sims = np.array(sims)
        print(f"  gamma fit   : {fmt_isi(res['gamma_isi'])}")
        res.update(gamma_d=sims[:, 0].mean(), gamma_p=sims[:, 1].mean(), gamma_bf=sims[:, 2].mean(),
                   p_gamma_d=(1 + np.sum(sims[:, 0] >= obs[0])) / (n_surr + 1),
                   kappa_median=float(np.median(kappa[good])),
                   kappa_frac_below1=float(np.mean(kappa[good] < 1)),
                   cvg_median=float(np.median(cvg[good])))
        print(f"  gamma null  : dDSI={res['gamma_d']:.3f} (p={res['p_gamma_d']:.3g})  %above={res['gamma_p']:.0f}  "
              f"burst fraction={res['gamma_bf']:.3f} | kappa median={res['kappa_median']:.2f}, "
              f"{100*res['kappa_frac_below1']:.0f}% units kappa<1, gain CV median={res['cvg_median']:.2f}")

    if "glm" in models:
        fits = []
        t0 = time.time()
        n_fail, nits = 0, []
        for u in range(len(spikes)):
            if not good[u]:
                fits.append(None); continue
            X, yy, T, H = glm_design(spikes[u], starts, di, n_dir)
            coef, b0, nit, ok = fit_glm(X, yy)
            n_fail += (not ok); nits.append(nit)
            fits.append((coef, b0, T, H))
        print(f"  glm fitted for {good.sum()} units in {time.time()-t0:.0f}s "
              f"(median {np.median(nits):.0f} iterations, {n_fail} not converged)")
        h_all = np.array([f[3] @ f[0][n_dir * N_TIME_BASIS:] for f in fits if f is not None])
        sims = []
        for i in range(n_surr):
            w_sim = [simulate_glm(*fits[u], di, n_dir, rng) if fits[u] is not None
                     else [np.empty(0)] * len(di) for u in range(len(spikes))]
            if i == 0:
                res["glm_isi"] = isi_stats(w_sim, good)
            (dd, pp), bf = score(w_sim, di, n_dir, good)
            sims.append((dd, pp, bf))
        sims = np.array(sims)
        print(f"  glm fit     : {fmt_isi(res['glm_isi'])}")
        res.update(glm_d=sims[:, 0].mean(), glm_p=sims[:, 1].mean(), glm_bf=sims[:, 2].mean(),
                   p_glm_d=(1 + np.sum(sims[:, 0] >= obs[0])) / (n_surr + 1),
                   glm_hist_median=np.median(h_all, axis=0).tolist())
        hm = np.median(h_all, axis=0)
        print(f"  glm null    : dDSI={res['glm_d']:.3f} (p={res['p_glm_d']:.3g})  %above={res['glm_p']:.0f}  "
              f"burst fraction={res['glm_bf']:.3f} | median history filter (log-gain) at "
              f"1-2ms={hm[:2].mean():+.2f}, 3-8ms={hm[2:8].mean():+.2f}, 10-30ms={hm[10:30].mean():+.2f}")
        if n_glm_calib > 0:
            res["glm_calib"] = glm_calibration(fits, good, starts, di, n_dir, directions,
                                               n_glm_calib, n_surr, rng)

    # calibration of the jitter test on synthetic non-bursting neurons
    if calib_kappas:
        res["calib"] = {}
        kappa_fit, cvg_fit, _ = fit_renewal(wins, di, lam)
        for k in calib_kappas:
            ks = np.full(len(spikes), float(k))
            rows = []
            for _ in range(5):
                w_syn = simulate_renewal(lam, ks, cvg_fit, di, rng)
                tot_s, bur_s, act_s = aggregate(w_syn, di, n_dir)
                good_s = (act_s >= 5).all(1) & (bur_s.sum(1) >= 5)
                if good_s.sum() < 5:
                    continue
                o_s = summ(tot_s, bur_s, act_s, good_s)
                j_s = np.array([summ(*jitter_null(w_syn, di, directions, rng), good_s)[0] for _ in range(50)])
                rows.append((o_s[0], j_s.mean(), (1 + np.sum(j_s >= o_s[0])) / 51, good_s.sum()))
            if rows:
                rows = np.array(rows)
                res["calib"][str(k)] = dict(syn_d=rows[:, 0].mean(), jit_d=rows[:, 1].mean(),
                                            p_median=float(np.median(rows[:, 2])), n_units=float(rows[:, 3].mean()))
                print(f"  CALIB kappa={k:<4}: synthetic dDSI={rows[:,0].mean():.3f} vs its jitter null "
                      f"{rows[:,1].mean():.3f}  median p={np.median(rows[:,2]):.3g}  (N={rows[:,3].mean():.0f})")
    res["seconds"] = time.time() - t_start
    return res


def download_resumable(url, path, expected_size, chunk=8 << 20, max_rounds=60):
    """Download with HTTP range resume. Zenodo drops long connections at
    ~0.7 MB/s, so a 1 GB file rarely completes in one go; each round continues
    from the bytes already on disk instead of restarting from zero."""
    import os
    import requests
    for _ in range(max_rounds):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= expected_size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as r:
                if r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code}")
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(path, mode) as f:
                    for c in r.iter_content(chunk):
                        if c:
                            f.write(c)
        except Exception as e:
            print(f"    resume after {type(e).__name__} at {have/1e6:.0f} MB", flush=True)
            time.sleep(20)
    if os.path.getsize(path) != expected_size:
        raise IOError("download size mismatch")


def load_sessions(dataset, session_filter, done=()):
    """Yields (name, (spikes, starts, directions)); names in `done` are skipped
    so an interrupted run resumes from its checkpoint."""
    if dataset in ("C", "M", "T", "J"):
        # DANDI 000688 centre-out sessions: C (Chewie, the 15 sessions of the paper),
        # M (Mihili), T (MrT), J (Jaco); same NWB schema, same loader
        cache = CACHE_C if dataset == "C" else ROOT / f"dandi000688_{dataset}_cache"
        files = sorted(glob.glob(str(cache / "*CO*.nwb")))
        for f in files:
            tag = Path(f).name.split("_ses-CO-")[1][:8]
            if session_filter and session_filter not in tag:
                continue
            if f"{dataset}_CO_{tag}" in done:
                continue
            spikes, starts, y = load_C(f)
            if AREA:
                # keep only the units of one cortical area (M1 or PMd); map from dandi000688_unit_area.json
                areas = json.loads((ROOT / "dandi000688_unit_area.json").read_text())[f"{dataset}_CO_{tag}"]
                spikes = [s for s, a in zip(spikes, areas) if a == AREA]
                if len(spikes) < 5:
                    print(f"  skip {dataset}_CO_{tag}_{AREA}: {len(spikes)} units", flush=True)
                    continue
                tag = f"{tag}_{AREA}"
            if dataset != "C":
                tpd = np.unique(y, return_counts=True)[1]
                if len(tpd) < 8 or tpd.min() < 10:
                    print(f"  skip {dataset}_CO_{tag}: {len(tpd)} dirs, min {tpd.min()} trials/dir", flush=True)
                    continue
            yield f"{dataset}_CO_{tag}", (spikes, starts, y)
    elif dataset == "indy":
        # all public Indy M1 sessions (Zenodo 583331), streamed one at a time
        # into a temp file and discarded, as in mcrtt_jitter_audit.py
        import tempfile, requests
        from mcrtt_multisession_permutation import RECORD_API, read_session, reach_onsets, download
        meta_cache = ROOT / "zenodo583331_meta.json"
        meta = None
        if meta_cache.exists():
            meta = json.loads(meta_cache.read_text())
        for attempt in range(20):
            if meta is not None:
                break
            try:
                meta = requests.get(RECORD_API, timeout=120).json()
                meta_cache.write_text(json.dumps(meta))
            except Exception as e:
                print(f"  record metadata failed ({type(e).__name__}); retry {attempt+1}/20 in 3 min", flush=True)
                time.sleep(180)
        assets = sorted([f for f in meta["files"] if f["key"].startswith("indy_")], key=lambda f: f["size"])
        tmpdir = Path(tempfile.gettempdir())
        for a in assets:
            tag = a["key"].removesuffix(".mat")
            if session_filter and session_filter not in tag:
                continue
            if tag in done:
                continue
            # survive network outages: the shared download() already retries 6x;
            # on top of that wait up to ~1 h for connectivity before giving up on this file
            for attempt in range(12):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mat", dir=tmpdir) as tmp:
                        print(f"downloading {a['key']} ({a['size']/1e6:.0f} MB)", flush=True)
                        download_resumable(a["links"]["self"], tmp.name, a["size"])
                        t, cursor, target, groups, _ = read_session(tmp.name)
                    break
                except Exception as e:
                    print(f"  download failed ({type(e).__name__}); retry {attempt+1}/12 in 5 min", flush=True)
                    time.sleep(300)
            else:
                print(f"  skip {tag}: download failed repeatedly", flush=True)
                continue
            if not groups["M1"]:
                continue
            starts, direction = reach_onsets(t, cursor, target)
            directions = np.sort(np.unique(direction))
            tpd = np.bincount(np.searchsorted(directions, direction), minlength=len(directions))
            if len(directions) < 6 or tpd.min() < 5:
                print(f"  skip {tag}: {len(directions)} dirs, min {tpd.min()} trials/dir")
                continue
            yield tag, (groups["M1"], starts, direction)
    else:
        from jitter_audit_maze_rtt import load_mr
        fn, kind = ("mc_maze_train.nwb", "maze") if dataset == "maze" else ("mc_rtt_train.nwb", "rtt")
        yield ("MC_Maze" if dataset == "maze" else "MC_RTT"), load_mr(fn, kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C", choices=["C", "M", "T", "J", "maze", "rtt", "indy"])
    ap.add_argument("--session", default=None, help="substring filter, e.g. 20150309")
    ap.add_argument("--models", default="gamma,glm")
    ap.add_argument("--n-surr", type=int, default=100)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-kappas", default="0.7,1,2,4")
    ap.add_argument("--glm-calib", type=int, default=0,
                    help="replicates of the GLM self-calibration (simulate from fitted GLM, re-fit, re-test)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--good-mode", default="paper", choices=["paper", "total"])
    ap.add_argument("--area", default="", choices=["", "M1", "PMd"], help="DANDI 000688 only: restrict to one cortical area")
    ap.add_argument("--check", default=str(CHECK), help="checkpoint json (use distinct files for parallel runs)")
    args = ap.parse_args()
    global GOOD_MODE, AREA
    GOOD_MODE = args.good_mode
    AREA = args.area
    models = set(args.models.split(","))
    kappas = [float(k) for k in args.calib_kappas.split(",")] if args.calib else ()

    check_path = Path(args.check)
    check = json.load(open(check_path)) if check_path.exists() else {}
    done = {k for k, r in check.items() if all(m + "_d" in r for m in models)}
    for name, (spikes, starts, y) in load_sessions(args.dataset, args.session, done):
        res = run_session(name, spikes, starts, y, models, args.n_surr, args.seed, kappas, args.glm_calib)
        check[name] = res
        check_path.write_text(json.dumps(check, indent=1, default=float))

    prefix = {"C": "C_CO", "M": "M_CO", "T": "T_CO", "J": "J_CO", "maze": "MC_Maze", "rtt": "MC_RTT", "indy": "indy_"}[args.dataset]
    rows = [r for k, r in check.items() if k.startswith(prefix)]
    if len(rows) >= 5:
        print(f"\n{'='*64}\n{len(rows)} sessions\n{'='*64}")
        o = np.array([r["obs_d"] for r in rows])
        for key, lab in (("jit_d", "jitter"), ("gamma_d", "gamma renewal"), ("glm_d", "GLM history")):
            if all(key in r for r in rows):
                nu = np.array([r[key] for r in rows])
                pk = "p_" + key
                sig = sum(r[pk] < .05 for r in rows)
                print(f"  vs {lab:14s}: obs={o.mean():.3f} null={nu.mean():.3f} above in {np.sum(o > nu)}/{len(rows)}, "
                      f"per-session p<.05 in {sig}/{len(rows)}, session Wilcoxon p={wilcoxon(o - nu, alternative='greater').pvalue:.3g}")


if __name__ == "__main__":
    main()
