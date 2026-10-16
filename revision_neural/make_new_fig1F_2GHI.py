"""Data for the revised Fig. 1F and Fig. 2G-I, plus firing statistics model vs data (Reviewer 2).

1. Data, example session (monkey C, CO-20150309): per-unit DSI of tonic rate and
   burst fraction, observed and under the fitted single-compartment GLM null.
2. Model (two-compartment, cortical regime that reproduces the data: hazard 0.05,
   graded apical tuning, basal tuning depth 0.5, apical drive 0.8): same quantities,
   with its own GLM null, plus direction-tuning matrices for the heatmaps.
3. Firing statistics for model and data (all 15 monkey C sessions): firing rate,
   Fano factor of spike counts, CV of ISIs, fraction of silent 500-ms windows,
   fraction of time in long gaps (ISI > 100 ms), burst fraction.
Output: new_figs_data.npz
"""
from pathlib import Path
import glob

import numpy as np

import audit_monkeyC_nulls as A
import single_compartment_null as S
import model_vs_glm_null as M

ROOT = Path(__file__).resolve().parent
N_SURR = 20
WIN = A.WIN


def per_unit(total, burst, active, good):
    rate = (total - burst) / (np.maximum(active, 1) * WIN)
    frac = np.divide(burst, total, out=np.zeros_like(burst), where=total > 0)
    return rate[good], frac[good], A.dsi(rate[good]), A.dsi(frac[good])


def null_dsi(spikes, starts, di, n_dir, good, rng):
    fits = []
    for u in range(len(spikes)):
        if not good[u]:
            fits.append(None); continue
        X, yy, T, H = S.glm_design(spikes[u], starts, di, n_dir)
        coef, b0, _, _ = S.fit_glm(X, yy)
        fits.append((coef, b0, T, H))
    dr, db = [], []
    for _ in range(N_SURR):
        w = [S.simulate_glm(*fits[u], di, n_dir, rng) if fits[u] is not None else [np.empty(0)] * len(di)
             for u in range(len(spikes))]
        t, b, a = S.aggregate(w, di, n_dir)
        _, _, r_, f_ = per_unit(t, b, a, good)
        dr.append(r_); db.append(f_)
    return np.mean(dr, 0), np.mean(db, 0)


def firing_stats(wins, di, good):
    """Per-unit statistics over all trials (500-ms windows)."""
    out = dict(rate=[], fano=[], cv=[], silent=[], long_gap=[], bf=[])
    n_dir = di.max() + 1
    for u in np.flatnonzero(good):
        counts = np.array([len(w) for w in wins[u]])
        out["rate"].append(counts.mean() / WIN)
        # Fano per direction (removes tuning), averaged
        f = [counts[di == d].var(ddof=1) / counts[di == d].mean() for d in range(n_dir) if counts[di == d].mean() > 0]
        out["fano"].append(np.mean(f) if f else np.nan)
        isi = np.concatenate([np.diff(w) for w in wins[u] if len(w) > 1]) if any(len(w) > 1 for w in wins[u]) else np.array([])
        out["cv"].append(isi.std() / isi.mean() if len(isi) > 5 else np.nan)
        out["silent"].append(np.mean(counts == 0))
        out["long_gap"].append(isi[isi > 0.1].sum() / max(len(wins[u]) * WIN, 1e-9))
        out["bf"].append(np.mean(isi < A.ISI) if len(isi) else np.nan)
    return {k: np.asarray(v, float) for k, v in out.items()}


def main():
    rng = np.random.default_rng(3)
    res = {}
    old = dict(np.load(ROOT / "new_figs_data.npz")) if (ROOT / "new_figs_data.npz").exists() else {}
    if "data_dsi_rate" in old:   # data part does not depend on the model: reuse it
        res.update({k: v for k, v in old.items() if k.startswith("data_") or k.startswith("stat_data_")})
    else:
        res.update(data_part(rng))
    # ---- 2. model ----
    M.HAZARD_SCALE, M.BASAL_DEPTH, M.TRIAL_CV, M.I_API_SCALE = 0.02, 0.5, 0.3, 0.5
    pop = M.build_population(100, 40, True, rng)
    wins_m, di_m = pop["wins"], pop["di"]; n_dir_m = 8
    t, b, a = S.aggregate(wins_m, di_m, n_dir_m)
    good_m = (a >= 5).all(1) & (t.sum(1) >= S.MIN_TOTAL_SPIKES)
    mrate, mfrac, mdr, mdb = per_unit(t, b, a, good_m)
    mndr, mndb = null_dsi(pop["spikes"], pop["starts"], di_m, n_dir_m, good_m, rng)
    res.update(model_dsi_rate=mdr, model_dsi_bf=mdb, model_null_dsi_rate=mndr, model_null_dsi_bf=mndb,
               model_rate=mrate, model_bf=mfrac, model_pref=pop["pref"][good_m])
    print(f"model: {good_m.sum()} units, dDSI obs {np.mean(mdb - mdr):.3f}, GLM null {np.mean(mndb - mndr):.3f}", flush=True)
    stats_model = firing_stats(wins_m, di_m, good_m)
    for k in stats_model:
        res[f"stat_model_{k}"] = stats_model[k]
        dd = res[f"stat_data_{k}"]
        print(f"  {k:9s} data median {np.nanmedian(dd):.3f} | model median {np.nanmedian(stats_model[k]):.3f}", flush=True)
    np.savez(ROOT / "new_figs_data.npz", **res)
    print("saved new_figs_data.npz")


def data_part(rng):
    res = {}
    # ---- 1. data example session ----
    f = sorted(glob.glob(str(ROOT / "dandi000688_C_cache/*CO-20150309*.nwb")))[0]
    spikes, starts, y = A.load(f)
    directions = np.sort(np.unique(y)); n_dir = len(directions)
    total, burst, active, binned, di, wins = A.observed(spikes, starts, y, directions)
    good = (active >= 5).all(1) & (total.sum(1) >= S.MIN_TOTAL_SPIKES)
    rate, frac, dr, db = per_unit(total, burst, active, good)
    ndr, ndb = null_dsi(spikes, starts, di, n_dir, good, rng)
    res.update(data_dsi_rate=dr, data_dsi_bf=db, data_null_dsi_rate=ndr, data_null_dsi_bf=ndb,
               data_rate=rate, data_bf=frac, data_dirs=directions)
    print(f"data CO-20150309: {good.sum()} units, dDSI obs {np.mean(db - dr):.3f}, GLM null {np.mean(ndb - ndr):.3f}", flush=True)
    return res  # (firing statistics of the data are computed below in the original single-pass version)

    # ---- 2. model (legacy single-pass) ----
    M.HAZARD_SCALE, M.BASAL_DEPTH, M.I_API_SCALE = 0.05, 0.5, 0.8
    pop = M.build_population(100, 40, True, rng)
    wins_m, di_m = pop["wins"], pop["di"]; n_dir_m = 8
    t, b, a = S.aggregate(wins_m, di_m, n_dir_m)
    good_m = (a >= 5).all(1) & (t.sum(1) >= S.MIN_TOTAL_SPIKES)
    mrate, mfrac, mdr, mdb = per_unit(t, b, a, good_m)
    mndr, mndb = null_dsi(pop["spikes"], pop["starts"], di_m, n_dir_m, good_m, rng)
    res.update(model_dsi_rate=mdr, model_dsi_bf=mdb, model_null_dsi_rate=mndr, model_null_dsi_bf=mndb,
               model_rate=mrate, model_bf=mfrac, model_pref=pop["pref"][good_m])
    print(f"model: {good_m.sum()} units, dDSI obs {np.mean(mdb - mdr):.3f}, GLM null {np.mean(mndb - mndr):.3f}", flush=True)
    stats_model = firing_stats(wins_m, di_m, good_m)

    # ---- 3. firing statistics, all monkey C sessions ----
    agg = {k: [] for k in stats_model}
    for f in sorted(glob.glob(str(ROOT / "dandi000688_C_cache/*CO*.nwb"))):
        sp, st, yy = A.load(f)
        dirs = np.sort(np.unique(yy))
        tt, bb, aa, _, dd, ww = A.observed(sp, st, yy, dirs)
        gg = (aa >= 5).all(1) & (tt.sum(1) >= S.MIN_TOTAL_SPIKES)
        s = firing_stats(ww, dd, gg)
        for k in agg:
            agg[k].append(s[k])
    stats_data = {k: np.concatenate(v) for k, v in agg.items()}
    for k in stats_model:
        res[f"stat_model_{k}"] = stats_model[k]; res[f"stat_data_{k}"] = stats_data[k]
        print(f"  {k:9s} data median {np.nanmedian(stats_data[k]):.3f} (IQR {np.nanpercentile(stats_data[k],25):.3f}-{np.nanpercentile(stats_data[k],75):.3f}) | "
              f"model median {np.nanmedian(stats_model[k]):.3f} (IQR {np.nanpercentile(stats_model[k],25):.3f}-{np.nanpercentile(stats_model[k],75):.3f})", flush=True)
    np.savez(ROOT / "new_figs_data.npz", **res)
    print("saved new_figs_data.npz")


if __name__ == "__main__":
    main()
