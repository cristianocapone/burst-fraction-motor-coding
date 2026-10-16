"""Audit the surprising monkey-C positive (reconstruct_bursts_ccg.py): does SORTED
per-unit burst fraction really beat the rate null, when Indy/MC_RTT said no?

Sorted units only (no reconstruction). Per unit/direction: total spikes, burst
spikes (within-unit ISI<10ms), tonic rate. Metric ΔDSI = DSI(burst frac) -
DSI(tonic rate), and %above. Tested against TWO independent nulls:
  A) Poisson-PSTH surrogate, VALIDATED construction (active recomputed from the
     draw, exactly as fig0b_null_models.surrogate_a) -- checks whether the
     reconstruct-script null (~0.20) was mis-built (too low).
  B) spike-JITTER null (+/-25 ms per-spike) -- destroys <10ms burst structure,
     preserves rate/PSTH. Independent null type.
If observed beats BOTH -> real monkey-C effect. If only a mis-built Poisson ->
artifact.
"""
from pathlib import Path
import glob
import json
import warnings
import numpy as np
from pynwb import NWBHDF5IO
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "dandi000688_C_cache"
WIN, ISI, BIN, JIT = 0.5, 0.010, 0.010, 0.025
NB, N_SURR = int(round(WIN / BIN)), 200


def load(path):
    with NWBHDF5IO(path, "r") as io:
        nwb = io.read()
        u = nwb.units.to_dataframe()
        spikes = [np.sort(np.asarray(u.iloc[i]["spike_times"], float)) for i in range(len(u))]
        tr = nwb.trials.to_dataframe()
        ok = tr[(tr["result"] == "R") & tr["go_cue_time"].notna() & tr["target_dir"].notna()]
        starts = ok["go_cue_time"].to_numpy(float)
        deg = np.degrees(ok["target_dir"].to_numpy(float))
        y = (((np.round(deg / 45) * 45 + 180) % 360) - 180).astype(int)
    return spikes, starts, y


def dsi(x):
    lo, hi = x.min(1), x.max(1)
    return np.divide(hi - lo, hi + lo, out=np.zeros_like(lo), where=(hi + lo) > 0)


def summ(total, burst, active, good):
    denom = np.maximum(active, 1)
    rate = (total - burst) / (denom * WIN)
    frac = np.divide(burst, total, out=np.zeros_like(burst), where=total > 0)
    return float(np.mean(dsi(frac[good]) - dsi(rate[good]))), float(100 * np.mean(dsi(frac[good]) > dsi(rate[good])))


def observed(spikes, starts, y, directions):
    n, nd = len(spikes), len(directions)
    di = np.searchsorted(directions, y)
    total = np.zeros((n, nd)); burst = np.zeros((n, nd)); active = np.zeros((n, nd))
    binned = np.zeros((n, len(starts), NB)); wins = [[] for _ in range(n)]
    for u, s in enumerate(spikes):
        for ti, t0 in enumerate(starts):
            a, b = np.searchsorted(s, [t0, t0 + WIN]); w = s[a:b] - t0
            d = di[ti]; total[u, d] += len(w); active[u, d] += (len(w) > 0)
            if len(w):
                binned[u, ti] = np.histogram(w, NB, (0, WIN))[0]
            if len(w) > 1:
                burst[u, d] += int(np.sum(np.diff(w) < ISI))
            wins[u].append(w)
    return total, burst, active, binned, di, wins


def poisson_null(binned, di, directions, rng):
    """VALIDATED: active recomputed from the draw."""
    n, _, nb = binned.shape; nd = len(directions)
    total = np.zeros((n, nd)); burst = np.zeros((n, nd)); active = np.zeros((n, nd))
    for d in range(nd):
        ix = np.flatnonzero(di == d); lam = binned[:, ix].mean(1)
        draw = rng.poisson(lam[None, :, :], size=(len(ix), n, nb))
        total[:, d] = draw.sum((0, 2)); active[:, d] = (draw.sum(2) > 0).sum(0)
        flat = draw.ravel(); cells = np.repeat(np.arange(flat.size), flat)
        if len(cells) < 2: continue
        bn = cells % nb; tmp = cells // nb; un = tmp % n; trn = tmp // n
        g = trn * n + un; tms = (bn + rng.random(len(cells))) * BIN
        o = np.lexsort((tms, g)); g, un, tms = g[o], un[o], tms[o]
        sh = (g[1:] == g[:-1]) & (np.diff(tms) < ISI); burst[:, d] = np.bincount(un[1:][sh], minlength=n)
    return total, burst, active


def jitter_null(wins, di, directions, rng):
    n, nd = len(wins), len(directions)
    total = np.zeros((n, nd)); burst = np.zeros((n, nd)); active = np.zeros((n, nd))
    for u in range(n):
        for ti, w in enumerate(wins[u]):
            d = di[ti]; total[u, d] += len(w); active[u, d] += (len(w) > 0)
            if len(w) > 1:
                wj = np.sort(w + rng.uniform(-JIT, JIT, len(w)))
                burst[u, d] += int(np.sum(np.diff(wj) < ISI))
    return total, burst, active


def main():
    files = sorted(glob.glob(str(CACHE / "*CO*.nwb")))
    rows = []
    for f in files:
        spikes, starts, y = load(f)
        directions = np.sort(np.unique(y))
        if len(directions) < 4: continue
        total, burst, active, binned, di, wins = observed(spikes, starts, y, directions)
        good = (active >= 5).all(1) & (burst.sum(1) >= 5)
        if good.sum() < 5: continue
        obs = summ(total, burst, active, good)
        rng = np.random.default_rng(9)
        pois = np.array([summ(*poisson_null(binned, di, directions, rng), good) for _ in range(N_SURR)])
        jit = np.array([summ(*jitter_null(wins, di, directions, rng), good) for _ in range(N_SURR)])
        rows.append(dict(session=Path(f).name, obs_d=obs[0], obs_p=obs[1],
                         pois_d=pois[:,0].mean(), pois_p=pois[:,1].mean(),
                         p_pois_d=(1+np.sum(pois[:,0]>=obs[0]))/(N_SURR+1),
                         p_pois_p=(1+np.sum(pois[:,1]>=obs[1]))/(N_SURR+1),
                         jit_d=jit[:,0].mean(), jit_p=jit[:,1].mean(),
                         p_jit_d=(1+np.sum(jit[:,0]>=obs[0]))/(N_SURR+1),
                         p_jit_p=(1+np.sum(jit[:,1]>=obs[1]))/(N_SURR+1)))
        r = rows[-1]
        print(f"  {Path(f).name.split('_ses-')[1][:8]}: obsΔ={obs[0]:.3f} %={obs[1]:.0f}  "
              f"POISSON null Δ={r['pois_d']:.3f}(p{r['p_pois_d']:.2g}) %={r['pois_p']:.0f}(p{r['p_pois_p']:.2g})  "
              f"JITTER null Δ={r['jit_d']:.3f}(p{r['p_jit_d']:.2g}) %={r['jit_p']:.0f}(p{r['p_jit_p']:.2g})")

    (ROOT / "audit_monkeyC_nulls_checkpoint.json").write_text(json.dumps(rows, indent=2))

    def agg(k_obs, k_null, k_p):
        o = np.array([r[k_obs] for r in rows]); nu = np.array([r[k_null] for r in rows])
        sig = sum(r[k_p] < .05 for r in rows)
        return o.mean(), nu.mean(), wilcoxon(o-nu, alternative='greater').pvalue, sig
    print(f"\n{'='*60}\n{len(rows)} sessions (SORTED units, no reconstruction)\n{'='*60}")
    for lab, ko, kn, kp in [("ΔDSI vs POISSON", "obs_d", "pois_d", "p_pois_d"),
                            ("%above vs POISSON", "obs_p", "pois_p", "p_pois_p"),
                            ("ΔDSI vs JITTER", "obs_d", "jit_d", "p_jit_d"),
                            ("%above vs JITTER", "obs_p", "jit_p", "p_jit_p")]:
        om, nm, p, sig = agg(ko, kn, kp)
        print(f"  {lab:20s}: obs={om:.3f}  null={nm:.3f}  session-p={p:.3g}  per-session sig {sig}/{len(rows)}")


if __name__ == "__main__":
    print("AUDIT: monkey C sorted burst-fraction vs validated Poisson + jitter nulls")
    main()
