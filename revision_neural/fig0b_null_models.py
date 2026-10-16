"""Null-model controls for the burst-fraction result in Fig0b.

Surrogate A (rate-only): for every neuron and reach direction, estimate a
10-ms PSTH and draw independent inhomogeneous-Poisson spike trains from it.

Surrogate B (conditional ISI permutation): keep every trial's spike count
fixed and permute a neuron's observed within-window ISIs across all of its
trials/directions.  This preserves the pooled empirical ISI distribution and
the spike-count tuning while destroying direction-specific short-ISI
allocation.  Permuting ISIs only within their original trial is deliberately
not used: it leaves the number of ISIs <10 ms, hence burst fraction, unchanged.

Outputs: Fig0b_null_models.pdf and Fig0b_null_models_results.npz
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

WIN = 0.500
ISI_THRESHOLD = 0.010
PSTH_BIN = 0.010
N_SURROGATES = 200
SEED = 20260702


def dsi(x):
    lo, hi = x.min(axis=1), x.max(axis=1)
    return np.divide(hi - lo, hi + lo, out=np.zeros_like(lo), where=(hi + lo) > 0)


def summarize(total, burst, active_trials, good=None):
    """Match Fig0b exactly: its rate denominator counts active trials only."""
    denom = np.maximum(np.asarray(active_trials, float), 1)
    tonic_rate = (total - burst) / (denom * WIN)
    burst_fraction = np.divide(burst, total, out=np.zeros_like(burst), where=total > 0)
    if good is None:
        good = (active_trials >= 5).all(axis=1) & ((tonic_rate + burst_fraction).sum(axis=1) > .2)
    dr, db = dsi(tonic_rate[good]), dsi(burst_fraction[good])
    return {
        "good": good,
        "dsi_rate": dr,
        "dsi_burst": db,
        "delta": float(np.mean(db - dr)),
        "pct": float(100 * np.mean(db > dr)),
    }


def _maze_trials(nwb):
    trials = nwb.trials.to_dataframe().copy()
    angles = []
    for _, row in trials.iterrows():
        xy = row["target_pos"][int(row["active_target"])]
        angles.append(np.degrees(np.arctan2(float(xy[1]), float(xy[0]))))
    trials["direction"] = (((np.round(np.asarray(angles) / 45) * 45 + 180) % 360) - 180).astype(int)
    return trials, trials["go_cue_time"].to_numpy(float)


def _rtt_trials(nwb):
    trials = nwb.trials.to_dataframe()
    trials = trials[trials["split"] == "train"].copy()
    beh = nwb.processing["behavior"]
    target, cursor = beh["target_pos"], beh["cursor_pos"]
    t_start, rate = target.starting_time, target.rate
    td, cd = target.data[:], cursor.data[:]

    def sample(a, t):
        i = int(round((t - t_start) * rate))
        return a[max(0, min(i, len(a) - 1))]

    vectors = np.asarray([sample(td, r.start_time) - sample(cd, r.start_time)
                          for _, r in trials.iterrows()])
    distance = np.linalg.norm(vectors, axis=1)
    angle = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    trials["direction"] = (((np.round(angle / 45) * 45 + 180) % 360) - 180).astype(int)
    trials = trials[distance > 20].copy()
    return trials, trials["start_time"].to_numpy(float)


def load_dataset(filename, kind):
    with NWBHDF5IO(str(ROOT / filename), "r") as io:
        nwb = io.read()
        trials, starts = _maze_trials(nwb) if kind == "maze" else _rtt_trials(nwb)
        units = nwb.units.to_dataframe()
        spike_times = [np.asarray(units.iloc[u]["spike_times"]) for u in range(len(units))]

        directions = np.sort(trials["direction"].unique())
        direction_index = np.searchsorted(directions, trials["direction"].to_numpy())
        n_trials, n_units = len(starts), len(spike_times)
        n_bins = int(round(WIN / PSTH_BIN))
        counts = np.zeros((n_units, n_trials), dtype=np.int16)
        binned = np.zeros((n_units, n_trials, n_bins), dtype=np.int16)
        short = []

        for u, spikes in enumerate(spike_times):
            unit_short = []
            for t, t0 in enumerate(starts):
                w = spikes[(spikes >= t0) & (spikes < t0 + WIN)] - t0
                counts[u, t] = len(w)
                binned[u, t], _ = np.histogram(w, bins=n_bins, range=(0, WIN))
                unit_short.append(np.diff(w) < ISI_THRESHOLD)
            short.append(unit_short)

    trials_per_direction = np.bincount(direction_index, minlength=len(directions))
    return dict(name="MC_Maze" if kind == "maze" else "MC_RTT", counts=counts,
                binned=binned, short=short, direction_index=direction_index,
                directions=directions, trials_per_direction=trials_per_direction)


def aggregate_observed(data):
    n_units, _ = data["counts"].shape
    n_dir = len(data["directions"])
    total = np.zeros((n_units, n_dir), float)
    burst = np.zeros_like(total)
    active = np.zeros_like(total)
    for d in range(n_dir):
        ix = np.flatnonzero(data["direction_index"] == d)
        total[:, d] = data["counts"][:, ix].sum(axis=1)
        active[:, d] = (data["counts"][:, ix] > 0).sum(axis=1)
        for u in range(n_units):
            burst[u, d] = sum(np.count_nonzero(data["short"][u][t]) for t in ix)
    return total, burst, active


def surrogate_a(data, rng):
    """Direction-specific PSTH -> independent Poisson bin counts."""
    n_units, n_trials, n_bins = data["binned"].shape
    n_dir = len(data["directions"])
    total = np.zeros((n_units, n_dir), float)
    burst = np.zeros_like(total)
    active = np.zeros_like(total)
    for d in range(n_dir):
        ix = np.flatnonzero(data["direction_index"] == d)
        lam = data["binned"][:, ix].mean(axis=1)  # units x bins, per trial
        # trials x units x bins. Uniform jitter inside bins produces exact
        # continuous event times; grouping/sorting is vectorized over trials.
        draw = rng.poisson(lam[None, :, :], size=(len(ix), n_units, n_bins))
        total[:, d] = draw.sum(axis=(0, 2))
        active[:, d] = (draw.sum(axis=2) > 0).sum(axis=0)
        flat = draw.ravel()
        cells = np.repeat(np.arange(flat.size), flat)
        if len(cells) < 2:
            continue
        bins = cells % n_bins
        tmp = cells // n_bins
        units = tmp % n_units
        trials = tmp // n_units
        groups = trials * n_units + units
        times = (bins + rng.random(len(cells))) * PSTH_BIN
        order = np.lexsort((times, groups))
        groups, units, times = groups[order], units[order], times[order]
        is_short = (groups[1:] == groups[:-1]) & (np.diff(times) < ISI_THRESHOLD)
        burst[:, d] = np.bincount(units[1:][is_short], minlength=n_units)
    return total, burst, active


def surrogate_b(data, rng):
    """Shuffle pooled empirical ISIs among fixed trial transition slots."""
    counts = data["counts"]
    n_units, n_trials = counts.shape
    n_dir = len(data["directions"])
    total = np.zeros((n_units, n_dir), float)
    burst = np.zeros_like(total)
    active = np.zeros_like(total)
    for d in range(n_dir):
        ix = data["direction_index"] == d
        total[:, d] = counts[:, ix].sum(axis=1)
        active[:, d] = (counts[:, ix] > 0).sum(axis=1)
    for u in range(n_units):
        labels = np.concatenate(data["short"][u]) if n_trials else np.empty(0, bool)
        labels = rng.permutation(labels)
        # One direction label per within-trial transition; this vector has the
        # same order/length as concatenate(short[u]).
        slots = np.repeat(data["direction_index"], np.maximum(counts[u] - 1, 0))
        burst[u] = np.bincount(slots, weights=labels, minlength=n_dir)
    return total, burst, active


def run(data, n_surrogates=N_SURROGATES):
    rng = np.random.default_rng(SEED + (0 if data["name"] == "MC_Maze" else 1))
    total, burst, active = aggregate_observed(data)
    observed = summarize(total, burst, active)
    null_a = np.zeros((n_surrogates, 2))
    null_b = np.zeros((n_surrogates, 2))
    for i in range(n_surrogates):
        ta, ba, aa = surrogate_a(data, rng)
        tb, bb, ab = surrogate_b(data, rng)
        a = summarize(ta, ba, aa, observed["good"])
        b = summarize(tb, bb, ab, observed["good"])
        null_a[i] = a["delta"], a["pct"]
        null_b[i] = b["delta"], b["pct"]
        if (i + 1) % 50 == 0:
            print(f'  {data["name"]}: {i + 1}/{n_surrogates}', flush=True)
    return observed, null_a, null_b


def empirical_p(null, observed):
    return (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)


def main():
    datasets = [load_dataset("mc_maze_train.nwb", "maze"),
                load_dataset("mc_rtt_train.nwb", "rtt")]
    results = []
    for data in datasets:
        print(data["name"])
        results.append(run(data))

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5), constrained_layout=True)
    colors = ["#777777", "#6a3d9a"]
    for row, (data, (obs, na, nb)) in enumerate(zip(datasets, results)):
        for col, metric in enumerate((0, 1)):
            ax = axes[row, col]
            label = "mean ΔDSI" if metric == 0 else "% neurons above diagonal"
            for vals, color, name in ((na[:, metric], colors[0], "A: Poisson PSTH"),
                                      (nb[:, metric], colors[1], "B: ISI permutation")):
                ax.hist(vals, bins=28, density=True, histtype="step", lw=1.8,
                        color=color, label=name)
            observed_value = obs["delta"] if metric == 0 else obs["pct"]
            ax.axvline(observed_value, color="#b2182b", lw=2, label="Observed")
            ax.set_xlabel(label)
            ax.set_ylabel("Null density")
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_title(data["name"])
            if row == 0 and col == 1:
                ax.legend(frameon=False, fontsize=8)

    fig.savefig(ROOT / "Fig0b_null_models.pdf", dpi=300, bbox_inches="tight")
    payload = {}
    for data, (obs, na, nb) in zip(datasets, results):
        key = data["name"].lower()
        payload[f"{key}_observed_delta"] = obs["delta"]
        payload[f"{key}_observed_pct"] = obs["pct"]
        payload[f"{key}_null_a"] = na
        payload[f"{key}_null_b"] = nb
        print(f'\n{data["name"]}: N={obs["good"].sum()}  observed mean ΔDSI={obs["delta"]:.4f}, '
              f'{obs["pct"]:.1f}% above')
        print(f'  A: mean={na[:,0].mean():.4f}, p={empirical_p(na[:,0], obs["delta"]):.4g}; '
              f'%mean={na[:,1].mean():.1f}, p={empirical_p(na[:,1], obs["pct"]):.4g}')
        print(f'  B: mean={nb[:,0].mean():.4f}, p={empirical_p(nb[:,0], obs["delta"]):.4g}; '
              f'%mean={nb[:,1].mean():.1f}, p={empirical_p(nb[:,1], obs["pct"]):.4g}')
    np.savez(ROOT / "Fig0b_null_models_results.npz", **payload)
    print("\nSaved Fig0b_null_models.pdf and Fig0b_null_models_results.npz")


if __name__ == "__main__":
    main()
