"""Cross-dataset check: do MC_Maze and MC_RTT burst-fraction direction codes beat
the JITTER null (the honest fine-timing-destroyed rate control), the way monkey C
did? Earlier these were tested only against the (too-weak) Poisson-PSTH null.

If they also show a modest jitter-robust excess -> the coding signal is general
and the Poisson null was masking it. If not -> monkey C is a lab/sorting quirk.
Reuses the audit machinery from audit_monkeyC_nulls; only the loader differs.
"""
from pathlib import Path
import warnings
import numpy as np
from pynwb import NWBHDF5IO
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
import fig0b_null_models as fnm
from audit_monkeyC_nulls import observed, poisson_null, jitter_null, summ, N_SURR

ROOT = Path(__file__).resolve().parent


def load_mr(filename, kind):
    with NWBHDF5IO(str(ROOT / filename), "r") as io:
        nwb = io.read()
        trials, starts = fnm._maze_trials(nwb) if kind == "maze" else fnm._rtt_trials(nwb)
        u = nwb.units.to_dataframe()
        spikes = [np.sort(np.asarray(u.iloc[i]["spike_times"], float)) for i in range(len(u))]
        y = trials["direction"].to_numpy()
    return spikes, np.asarray(starts, float), np.asarray(y)


def run(name, filename, kind):
    spikes, starts, y = load_mr(filename, kind)
    directions = np.sort(np.unique(y))
    total, burst, active, binned, di, wins = observed(spikes, starts, y, directions)
    good = (active >= 5).all(1) & (burst.sum(1) >= 5)
    obs = summ(total, burst, active, good)
    rng = np.random.default_rng(9)
    pois = np.array([summ(*poisson_null(binned, di, directions, rng), good) for _ in range(N_SURR)])
    jit = np.array([summ(*jitter_null(wins, di, directions, rng), good) for _ in range(N_SURR)])
    p_pois_d = (1 + np.sum(pois[:, 0] >= obs[0])) / (N_SURR + 1)
    p_pois_p = (1 + np.sum(pois[:, 1] >= obs[1])) / (N_SURR + 1)
    p_jit_d = (1 + np.sum(jit[:, 0] >= obs[0])) / (N_SURR + 1)
    p_jit_p = (1 + np.sum(jit[:, 1] >= obs[1])) / (N_SURR + 1)
    print(f"\n{name}  (N={good.sum()} units, {len(directions)} dirs)")
    print(f"  observed:  ΔDSI={obs[0]:.3f}  %above={obs[1]:.1f}")
    print(f"  POISSON null: ΔDSI={pois[:,0].mean():.3f} (p={p_pois_d:.3g})  %above={pois[:,1].mean():.1f} (p={p_pois_p:.3g})")
    print(f"  JITTER  null: ΔDSI={jit[:,0].mean():.3f} (p={p_jit_d:.3g})  %above={jit[:,1].mean():.1f} (p={p_jit_p:.3g})  <-- honest bar")


if __name__ == "__main__":
    print("JITTER-null cross-check on curated datasets (vs monkey C's jitter-robust excess)")
    run("MC_Maze (000128)", "mc_maze_train.nwb", "maze")
    run("MC_RTT (000129)", "mc_rtt_train.nwb", "rtt")
