# Figure 1 — burst fraction is more direction-selective than firing rate

Burst fraction vs. tonic-rate direction selectivity in openly available macaque
motor-cortex recordings from the [DANDI Archive](https://dandiarchive.org).

| Dataset | Dandiset | Animal | Task |
|---------|----------|--------|------|
| MC\_Maze | [000128](https://dandiarchive.org/dandiset/000128) | Jenkins (Stanford) | delayed centre-out reaching |
| MC\_RTT  | [000129](https://dandiarchive.org/dandiset/000129) | Indy (UCSF)         | self-paced random-target reaching |

A spike is labelled a *burst spike* if the interval to the preceding spike of the
same unit is below 10 ms; all other spikes are tonic. For each neuron and reach
direction, in the 500 ms window after movement onset:

- **tonic rate** = tonic spikes / (trials × window)
- **burst fraction** = burst spikes / total spikes
- **DSI** (direction selectivity index) = (max − min) / (max + min) across directions

## Reproduce

Data files are **not** included; they are downloaded from DANDI into this folder
(several hundred MB for MC\_Maze). The analysis scripts run from this folder
regardless of where you invoke them.

```bash
python fig1_burst_fraction/download_data.py            # mc_maze_train.nwb, mc_rtt_train.nwb
python fig1_burst_fraction/burst_fraction_combined.py  # figure1_burst_fraction.pdf (= Figure 1)
```

**Result:** burst-fraction DSI exceeds tonic-rate DSI in 95% of MC\_Maze neurons
(p < 1e-29) and 90% of MC\_RTT neurons (p < 1e-16), Wilcoxon signed-rank; the
effect survives regressing tonic rate out of the burst fraction.

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `download_data.py`           | `mc_maze_train.nwb`, `mc_rtt_train.nwb` | Fetch both datasets from DANDI (cached; skips existing files). |
| `burst_fraction_combined.py` | `figure1_burst_fraction.pdf`, `figure1_population_structure.pdf` | **Fig. 1.** MC\_Maze + MC\_RTT combined figure, plus the population bilinear-structure panel. |
| `burst_fraction_mcmaze.py`   | `mc_maze_burst_fraction.pdf` | MC\_Maze analysis + null-model controls. |
| `burst_fraction_mcrtt.py`    | `mc_rtt_burst_fraction.pdf` | MC\_RTT analysis. |
| `burst_fraction_figure.py`   | `burst_fraction_summary.pdf` | Alternative summary layout. |

Needs `numpy`, `scipy`, `matplotlib`, `pynwb`, `h5py`, `dandi`.
