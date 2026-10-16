"""Rate-controlled + ISI-permutation burst analysis across all public Indy RTT
sessions. Extends mcrtt_multisession_robust.py (surrogate A / LOO-Poisson PSTH
only) with surrogate B (pooled-ISI permutation, from fig0b_null_models.py),
which is the null model that directly tests whether burst events are
non-randomly allocated across directions given a neuron's own fixed total
burst count.

Files are downloaded one at a time from Zenodo record 583331 into a temporary
file, read without waveform data, analysed, and discarded. Results are
checkpointed after every session in mcrtt_multisession_permutation_checkpoint.json
(separate from the surrogate-A-only checkpoint so that existing results are
not disturbed).
"""

from pathlib import Path
import json
import tempfile
import time
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RECORD_API = "https://zenodo.org/api/records/583331"
CHECKPOINT = ROOT / "mcrtt_multisession_permutation_checkpoint.json"
S1_CHECKPOINT = ROOT / "mcrtt_multisession_permutation_s1_checkpoint.json"
OUT_PDF = ROOT / "Fig0b_mcrtt_multisession_permutation.pdf"
ISI = .010
WIN = .500
BIN = .010
N_NULL = 100
SEED = 20260706
warnings.filterwarnings("ignore", category=DeprecationWarning)


def dsi(x):
    lo, hi = x.min(1), x.max(1)
    return np.divide(hi - lo, hi + lo, out=np.zeros_like(lo), where=hi + lo > 0)


def decode_matlab_string(f, ref):
    return "".join(chr(int(v)) for v in np.asarray(f[ref]).ravel())


def read_session(path):
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"]).ravel()
        cursor = np.asarray(f["cursor_pos"]).T
        target = np.asarray(f["target_pos"]).T
        labels = [decode_matlab_string(f, f["chan_names"][0, j])
                  for j in range(f["chan_names"].shape[1])]
        spike_cell = f["spikes"]
        spikes_by_region = {}
        channel_counts = {}
        for region in ("M1", "S1"):
            channels = [j for j, label in enumerate(labels)
                        if label.strip().upper().startswith(region)]
            channel_counts[region] = len(channels)
            spikes = []
            # MATLAB row 0 is the unsorted/hash unit; retain sorted units only.
            for ch in channels:
                for unit_row in range(1, spike_cell.shape[0]):
                    ref = spike_cell[unit_row, ch]
                    a = np.asarray(f[ref])
                    if a.dtype.kind == "f" and a.size >= 20:
                        s = np.sort(a.ravel())
                        s = s[np.isfinite(s)]
                        if len(s) >= 20:
                            spikes.append(s)
            spikes_by_region[region] = spikes
    return t, cursor, target, spikes_by_region, channel_counts


def reach_onsets(t, cursor, target):
    valid = np.isfinite(target).all(1) & np.isfinite(cursor).all(1)
    change = np.linalg.norm(np.diff(target, axis=0), axis=1) > 1e-6
    idx = np.flatnonzero(change) + 1
    idx = idx[valid[idx]]
    vec = target[idx] - cursor[idx]
    keep = np.linalg.norm(vec, axis=1) > 20.
    idx, vec = idx[keep], vec[keep]
    angle = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    direction = (((np.round(angle / 45) * 45 + 180) % 360) - 180).astype(int)
    return t[idx], direction


def extract(spikes, starts, direction):
    directions = np.sort(np.unique(direction))
    di = np.searchsorted(directions, direction)
    n_units, n_trials, n_bins = len(spikes), len(starts), int(WIN / BIN)
    counts = np.zeros((n_units, n_trials), np.int16)
    binned = np.zeros((n_units, n_trials, n_bins), np.int16)
    short = []
    for u, s in enumerate(spikes):
        unit_short = []
        for k, t0 in enumerate(starts):
            a, b = np.searchsorted(s, [t0, t0 + WIN])
            w = s[a:b] - t0
            counts[u, k] = len(w)
            if len(w):
                binned[u, k], _ = np.histogram(w, n_bins, (0, WIN))
            unit_short.append(np.diff(w) < ISI)
        short.append(unit_short)
    return dict(counts=counts, binned=binned, short=short, di=di,
                directions=directions, trials_per_dir=np.bincount(di, minlength=len(directions)))


def observed_arrays(data):
    n_units = data["counts"].shape[0]
    n_dir = len(data["directions"])
    total = np.zeros((n_units, n_dir), float)
    burst = np.zeros_like(total)
    for d in range(n_dir):
        ix = np.flatnonzero(data["di"] == d)
        total[:, d] = data["counts"][:, ix].sum(1)
        for u in range(n_units):
            burst[u, d] = sum(np.count_nonzero(data["short"][u][k]) for k in ix)
    return total, burst


def summarize(total, burst, n_trials, good=None):
    rate = (total - burst) / (n_trials[None, :] * WIN)
    frac = np.divide(burst, total, out=np.zeros_like(burst), where=total > 0)
    if good is None:
        good = (total.sum(1) > 0) & ((rate + frac).sum(1) > .2)
    dr, db = dsi(rate[good]), dsi(frac[good])
    return good, float(np.mean(db - dr)), float(100 * np.mean(db > dr))


def poisson_loo(data, rng):
    """Surrogate A: each trial's spikes redrawn from a leave-one-trial-out
    Poisson PSTH for its direction. No true bursting; rate-tuning only."""
    xall = data["binned"]
    n_units, _, n_bins = xall.shape
    n_dir = len(data["directions"])
    total = np.zeros((n_units, n_dir), float)
    burst = np.zeros_like(total)
    for d in range(n_dir):
        ix = np.flatnonzero(data["di"] == d)
        x = xall[:, ix, :]
        if len(ix) < 2:
            continue
        lam = (x.sum(1, keepdims=True) - x) / (len(ix) - 1)
        draw = rng.poisson(np.transpose(lam, (1, 0, 2)))
        total[:, d] = draw.sum((0, 2))
        flat = draw.ravel()
        cells = np.repeat(np.arange(flat.size), flat)
        if len(cells) < 2:
            continue
        bins = cells % n_bins
        tmp = cells // n_bins
        units = tmp % n_units
        trials = tmp // n_units
        groups = trials * n_units + units
        times = (bins + rng.random(len(cells))) * BIN
        order = np.lexsort((times, groups))
        groups, units, times = groups[order], units[order], times[order]
        is_short = (groups[1:] == groups[:-1]) & (np.diff(times) < ISI)
        burst[:, d] = np.bincount(units[1:][is_short], minlength=n_units)
    return total, burst


def permutation_null(data, rng):
    """Surrogate B: shuffle each unit's own pooled burst/tonic labels across
    its fixed transition slots (real trial-direction structure and total
    burst count preserved; only direction-specific allocation randomized)."""
    counts = data["counts"]
    n_units = counts.shape[0]
    n_dir = len(data["directions"])
    total = np.zeros((n_units, n_dir), float)
    burst = np.zeros_like(total)
    for d in range(n_dir):
        ix = data["di"] == d
        total[:, d] = counts[:, ix].sum(1)
    for u in range(n_units):
        labels = np.concatenate(data["short"][u]) if counts.shape[1] else np.empty(0, bool)
        labels = rng.permutation(labels)
        slots = np.repeat(data["di"], np.maximum(counts[u] - 1, 0))
        burst[u] = np.bincount(slots, weights=labels, minlength=n_dir)
    return total, burst


def analyse_arrays(t, cursor, target, spikes, name, region, channel_count):
    starts, direction = reach_onsets(t, cursor, target)
    data = extract(spikes, starts, direction)
    if len(data["directions"]) < 6 or data["trials_per_dir"].min() < 5:
        raise ValueError(f"insufficient directions/trials: {len(data['directions'])}, {data['trials_per_dir']}")
    total, burst = observed_arrays(data)
    good, obs_delta, obs_pct = summarize(total, burst, data["trials_per_dir"])
    rng = np.random.default_rng(SEED + sum(name.encode()))
    null_a = np.zeros((N_NULL, 2))
    null_b = np.zeros((N_NULL, 2))
    for i in range(N_NULL):
        nt, nb = poisson_loo(data, rng)
        _, null_a[i, 0], null_a[i, 1] = summarize(nt, nb, data["trials_per_dir"], good)
        pt, pb = permutation_null(data, rng)
        _, null_b[i, 0], null_b[i, 1] = summarize(pt, pb, data["trials_per_dir"], good)
    p_delta_a = (1 + np.count_nonzero(null_a[:, 0] >= obs_delta)) / (N_NULL + 1)
    p_pct_a = (1 + np.count_nonzero(null_a[:, 1] >= obs_pct)) / (N_NULL + 1)
    p_delta_b = (1 + np.count_nonzero(null_b[:, 0] >= obs_delta)) / (N_NULL + 1)
    p_pct_b = (1 + np.count_nonzero(null_b[:, 1] >= obs_pct)) / (N_NULL + 1)
    return dict(session=name.removesuffix(".mat"), region=region,
                channels=int(channel_count), sorted_units=len(spikes),
                units=int(good.sum()), reaches=len(starts),
                directions=len(data["directions"]), observed_delta=obs_delta,
                observed_pct=obs_pct,
                null_a_delta=float(null_a[:, 0].mean()), p_a_delta=float(p_delta_a),
                null_a_pct=float(null_a[:, 1].mean()), p_a_pct=float(p_pct_a),
                null_b_delta=float(null_b[:, 0].mean()), p_b_delta=float(p_delta_b),
                null_b_pct=float(null_b[:, 1].mean()), p_b_pct=float(p_pct_b))


def save_plot(results, output=OUT_PDF, title="M1"):
    ok = [r for r in results.values() if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["session"])
    if not ok:
        return
    x = np.arange(len(ok))
    fig, ax = plt.subplots(2, 1, figsize=(max(10, len(ok) * .32), 6.5), sharex=True,
                           constrained_layout=True)
    for a, observed, null_a, null_b, ylabel in [
        (ax[0], "observed_delta", "null_a_delta", "null_b_delta", "Mean ΔDSI"),
        (ax[1], "observed_pct", "null_a_pct", "null_b_pct", "% neurons above diagonal")]:
        a.plot(x, [r[observed] for r in ok], "o-", color="#b2182b", ms=4, label="Observed")
        a.plot(x, [r[null_a] for r in ok], "o-", color=".5", ms=3, label="A: LOO Poisson PSTH")
        a.plot(x, [r[null_b] for r in ok], "o-", color="#6a3d9a", ms=3, label="B: ISI permutation")
        a.set_ylabel(ylabel)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False)
    fig.suptitle(f"Indy RTT — {title} — surrogate A vs surrogate B")
    ax[1].set_xticks(x, [r["session"].replace("indy_", "") for r in ok], rotation=90, fontsize=6)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def download(url, path, expected_size):
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with open(path, "wb") as out:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            out.write(chunk)
            if Path(path).stat().st_size != expected_size:
                raise IOError("download size mismatch")
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(15 * (attempt + 1))


def main():
    metadata = requests.get(RECORD_API, timeout=60).json()
    assets = [f for f in metadata["files"] if f["key"].startswith("indy_")]
    assets.sort(key=lambda f: f["size"])  # early results and gentler disk use
    results = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    s1_results = json.loads(S1_CHECKPOINT.read_text()) if S1_CHECKPOINT.exists() else {}
    print(f"Indy assets={len(assets)}, M1 checkpointed={len(results)}, "
          f"S1 inspected={len(s1_results)}")
    for pos, asset in enumerate(assets, 1):
        name = asset["key"]
        if name in results and name in s1_results:
            continue
        print(f"[{pos}/{len(assets)}] {name} ({asset['size']/1e6:.1f} MB)", flush=True)
        try:
            with tempfile.NamedTemporaryFile(suffix=".mat", dir="/tmp") as tmp:
                download(asset["links"]["self"], tmp.name, asset["size"])
                t, cursor, target, groups, channel_counts = read_session(tmp.name)
                for region, checkpoint in (("M1", results), ("S1", s1_results)):
                    if name in checkpoint:
                        continue
                    if not groups[region]:
                        checkpoint[name] = dict(session=name.removesuffix(".mat"), region=region,
                                                channels=channel_counts[region], status="no_data")
                        print(f"  {region}: no labelled channels/units")
                        continue
                    row = analyse_arrays(t, cursor, target, groups[region], name, region,
                                         channel_counts[region])
                    row["status"] = "ok"
                    checkpoint[name] = row
                    print(f"  {region}: N={row['units']} reaches={row['reaches']} "
                          f"A: Δ={row['observed_delta']:.3f} vs {row['null_a_delta']:.3f} p={row['p_a_delta']:.3f}, "
                          f"%={row['observed_pct']:.1f} vs {row['null_a_pct']:.1f} p={row['p_a_pct']:.3f} | "
                          f"B: Δ p={row['p_b_delta']:.3f}, % p={row['p_b_pct']:.3f}")
        except Exception as exc:
            if name not in results:
                results[name] = dict(session=name.removesuffix(".mat"), region="M1",
                                     status="error", error=repr(exc))
            if name not in s1_results:
                s1_results[name] = dict(session=name.removesuffix(".mat"), region="S1",
                                        status="error", error=repr(exc))
            print(f"  ERROR: {exc!r}")
        CHECKPOINT.write_text(json.dumps(results, indent=2, sort_keys=True))
        S1_CHECKPOINT.write_text(json.dumps(s1_results, indent=2, sort_keys=True))
        save_plot(results)
        save_plot(s1_results, ROOT / "Fig0b_mcrtt_multisession_permutation_s1.pdf", "S1")

    def report(results, label):
        ok = [r for r in results.values() if r.get("status") == "ok"]
        if not ok:
            return
        da = np.array([r["observed_delta"] - r["null_a_delta"] for r in ok])
        db = np.array([r["observed_delta"] - r["null_b_delta"] for r in ok])
        pa = np.array([r["observed_pct"] - r["null_a_pct"] for r in ok])
        pb = np.array([r["observed_pct"] - r["null_b_pct"] for r in ok])
        wa_d = wilcoxon(da, alternative="greater").pvalue
        wb_d = wilcoxon(db, alternative="greater").pvalue
        wa_p = wilcoxon(pa, alternative="greater").pvalue
        wb_p = wilcoxon(pb, alternative="greater").pvalue
        print(f"\n{label}: sessions={len(ok)}")
        print(f"  A (Poisson): ΔDSI excess>0 in {(da>0).sum()}/{len(ok)}, median={np.median(da):.4f}, session p={wa_d:.4g}"
              f" | %above excess>0 in {(pa>0).sum()}/{len(ok)}, median={np.median(pa):.2f}, session p={wa_p:.4g}")
        print(f"  B (permut):  ΔDSI excess>0 in {(db>0).sum()}/{len(ok)}, median={np.median(db):.4f}, session p={wb_d:.4g}"
              f" | %above excess>0 in {(pb>0).sum()}/{len(ok)}, median={np.median(pb):.2f}, session p={wb_p:.4g}")

    report(results, "M1")
    report(s1_results, "S1")
    save_plot(results)
    save_plot(s1_results, ROOT / "Fig0b_mcrtt_multisession_permutation_s1.pdf", "S1")


if __name__ == "__main__":
    main()
