# Burst-fraction motor coding

Code to reproduce **Figure 1** of *Burst-selective motor coding reveals a bilinear
gating principle for adaptive control* (Capone, Falorsi, Ciardiello, Manneschi).

Figure 1 shows that in macaque motor cortex, **burst fraction encodes reach
direction far more selectively than tonic spike rate**, across two independent
datasets (MC\_Maze and MC\_RTT). A spike is labelled a *burst spike* if its
inter-spike interval to the preceding spike is below 10 ms; for each neuron we
compare the direction selectivity of the tonic firing rate against that of the
burst fraction over a 500 ms window after the go cue.

## Data

Both datasets are openly available from the [DANDI Archive](https://dandiarchive.org):

| Dataset | Dandiset | Animal | Task |
|---------|----------|--------|------|
| MC\_Maze | [000128](https://dandiarchive.org/dandiset/000128) | Jenkins (Stanford) | delayed centre-out reaching |
| MC\_RTT  | [000129](https://dandiarchive.org/dandiset/000129) | Indy (UCSF)         | self-paced random-target reaching |

Data files are **not** included in this repository; `download_data.py` fetches them
from DANDI into the working directory.

## Reproduce Figure 1

```bash
pip install -r requirements.txt
python download_data.py            # fetches mc_maze_train.nwb and mc_rtt_train.nwb
python burst_fraction_combined.py  # writes figure1_burst_fraction.pdf (= Figure 1)
```

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `burst_fraction_combined.py` | `figure1_burst_fraction.pdf`, `figure1_population_structure.pdf` | **Main.** MC\_Maze + MC\_RTT combined figure (Fig. 1), plus the population bilinear-structure panel. |
| `burst_fraction_mcmaze.py`   | `mc_maze_burst_fraction.pdf` | MC\_Maze analysis + controls (auto-downloads 000128). |
| `burst_fraction_mcrtt.py`    | `mc_rtt_burst_fraction.pdf` | MC\_RTT analysis. |
| `burst_fraction_figure.py`   | `burst_fraction_summary.pdf` | Alternative summary layout. |
| `download_data.py`           | `mc_maze_train.nwb`, `mc_rtt_train.nwb` | Fetch both datasets from DANDI. |

## Analysis summary

For each neuron and reach direction, over a 500 ms post-go-cue window:

- **tonic rate** = total spikes / (trials × window)
- **burst fraction** = burst spikes / total spikes, where a spike is a burst spike
  if the preceding inter-spike interval < 10 ms
- **DSI** (direction selectivity index) = (max − min) / (max + min) across directions

The main result: burst-fraction DSI exceeds tonic-rate DSI in 95% of MC\_Maze
neurons (p < 1e-29) and 90% of MC\_RTT neurons (p < 1e-16), Wilcoxon signed-rank.
The effect survives regressing tonic rate out of the burst fraction.

## License

MIT — see [LICENSE](LICENSE).
