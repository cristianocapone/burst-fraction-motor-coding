"""Reviewer 2: is the burst result driven by poorly isolated units?

Monkey C NWB files (DANDI 000688) contain the waveform of every spike. For each
unit we compute standard isolation metrics:
  - SNR: peak-to-trough amplitude of the mean waveform / (2 x SD of the
    residuals of single-spike waveforms around the mean)
  - ISI violations: fraction of ISIs shorter than 1.5 ms (refractory violations;
    contamination by another neuron increases it)
For every unit we also compute the observed dDSI (burst-fraction DSI minus
tonic-rate DSI) and its GLM-null expectation (fitted single-compartment model,
burst-independent unit selection), so the excess can be compared between
well-isolated and other units.
Usage: python isolation_glm.py --session 201503 --check out.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

import audit_monkeyC_nulls as A
import single_compartment_null as S

N_SURR = 40
MAX_WF = 2000


def isolation_metrics(nwb):
    units = nwb.units
    out = []
    for i in range(len(units)):
        st = np.sort(np.asarray(units["spike_times"][i], float))
        isi = np.diff(st)
        viol = float(np.mean(isi < 0.0015)) if len(isi) else np.nan
        wf = np.asarray(units["waveforms"][i], float)
        wf = wf.reshape(wf.shape[0], -1)
        if wf.shape[0] > MAX_WF:
            wf = wf[np.linspace(0, wf.shape[0] - 1, MAX_WF).astype(int)]
        mean = wf.mean(0)
        amp = mean.max() - mean.min()
        noise = np.std(wf - mean)
        out.append(dict(snr=float(amp / (2 * noise)) if noise > 0 else np.nan, isi_viol=viol, n_spikes=len(st)))
    return out


def per_unit_dsi(total, burst, active):
    rate = (total - burst) / (np.maximum(active, 1) * A.WIN)
    frac = np.divide(burst, total, out=np.zeros_like(burst), where=total > 0)
    return A.dsi(frac) - A.dsi(rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--check", required=True)
    args = ap.parse_args()
    p = Path(args.check); c = json.loads(p.read_text()) if p.exists() else {}
    rng = np.random.default_rng(2)
    for name, (spikes, starts, y) in S.load_sessions("C", args.session):
        if name in c:
            continue
        t0 = time.time()
        import glob
        f = [g for g in glob.glob(str(Path(__file__).resolve().parent / "dandi000688_C_cache/*.nwb")) if name.split("_")[-1] in g][0]
        with NWBHDF5IO(f, "r") as io:
            iso = isolation_metrics(io.read())
        dirs = np.sort(np.unique(y)); n_dir = len(dirs)
        total, burst, active, _, di, wins = A.observed(spikes, starts, y, dirs)
        good = (active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES)
        obs = per_unit_dsi(total, burst, active)
        fits = [None] * len(spikes)
        for u in np.flatnonzero(good):
            X, yy, T, H = S.glm_design(spikes[u], starts, di, n_dir)
            coef, b0, _, _ = S.fit_glm(X, yy)
            fits[u] = (coef, b0, T, H)
        null = np.zeros(len(spikes))
        for _ in range(N_SURR):
            w = [S.simulate_glm(*fits[u], di, n_dir, rng) if fits[u] is not None else [np.empty(0)] * len(di)
                 for u in range(len(spikes))]
            tt, bb, aa = S.aggregate(w, di, n_dir)
            null += per_unit_dsi(tt, bb, aa) / N_SURR
        units = [dict(unit=int(u), excess=float(obs[u] - null[u]), **iso[u]) for u in np.flatnonzero(good)]
        c[name] = units; p.write_text(json.dumps(c, indent=1))
        ex = np.array([x["excess"] for x in units]); snr = np.array([x["snr"] for x in units]); v = np.array([x["isi_viol"] for x in units])
        best = (snr >= np.nanmedian(snr)) & (v < 0.005)
        print(f"{name}: N={len(units)} | excess all {ex.mean():+.3f} | best-isolated (SNR>=median & ISI viol<0.5%, n={best.sum()}) {ex[best].mean():+.3f} | "
              f"others {ex[~best].mean():+.3f} | median SNR {np.nanmedian(snr):.2f}, ISI viol median {np.nanmedian(v)*100:.2f}% | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
