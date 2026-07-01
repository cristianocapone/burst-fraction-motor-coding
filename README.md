# Burst-fraction motor coding

Code to reproduce **Figures 1 and 2** of *Burst-selective motor coding reveals a
bilinear gating principle for adaptive control* (Capone, Falorsi, Ciardiello,
Manneschi).

- **Figure 1** — in macaque motor cortex, **burst fraction encodes reach
  direction far more selectively than tonic spike rate**, across two independent
  datasets (MC\_Maze and MC\_RTT).
- **Figure 2** — a minimal **two-compartment burst-gating model** reproduces this
  coding signature and learns goal-directed reaching through context-selective
  bursting.

A spike is labelled a *burst spike* if its inter-spike interval to the preceding
spike is below 10 ms; for each neuron we compare the direction selectivity of the
tonic firing rate against that of the burst fraction.

## Install

```bash
pip install -r requirements.txt
```

Figure 1 needs `numpy/scipy/matplotlib/pynwb/h5py/dandi`.
Figure 2 additionally needs `gym`, `mujoco`, `imageio`, `tqdm`. Note that gym's
`Reacher-v4` requires `numpy < 2.0` (the scripts include a `np.bool8` shim, but a
numpy-1.x environment is the tested configuration for training and evaluation).

---

## Figure 1 — experimental data

Burst fraction vs. tonic rate direction selectivity in two openly available
datasets from the [DANDI Archive](https://dandiarchive.org):

| Dataset | Dandiset | Animal | Task |
|---------|----------|--------|------|
| MC\_Maze | [000128](https://dandiarchive.org/dandiset/000128) | Jenkins (Stanford) | delayed centre-out reaching |
| MC\_RTT  | [000129](https://dandiarchive.org/dandiset/000129) | Indy (UCSF)         | self-paced random-target reaching |

Data files are **not** included; `download_data.py` fetches them from DANDI.

```bash
python download_data.py            # fetches mc_maze_train.nwb and mc_rtt_train.nwb
python burst_fraction_combined.py  # writes figure1_burst_fraction.pdf (= Figure 1)
```

**Result:** burst-fraction DSI exceeds tonic-rate DSI in 95% of MC\_Maze neurons
(p < 1e-29) and 90% of MC\_RTT neurons (p < 1e-16), Wilcoxon signed-rank; the
effect survives regressing tonic rate out of the burst fraction.

---

## Figure 2 — two-compartment model

Each output unit is a probabilistic two-compartment neuron with a somatic voltage
`V_s` (state-dependent basal drive) and a dendritic voltage `V_d` (goal-selective
apical drive). A burst is emitted with probability

```
p_burst = < p_s(V_s) * p_d(V_d) >,   p_s = sigma((V_s-theta_s)/beta_s),   p_d = sigma((V_d-theta_d)/beta_d)
```

Because the two factors multiply, `p_burst` is high only when both compartments
are driven at the same time — a multiplicative gate `G_k(g) * Y_k(s)`. The motor
command is linear in `p_burst`, and weights are learned online with a three-factor
(reward-modulated Hebbian) rule.

The figure (3x3) shows: the neuron and its bilinear gate (A-C), reaching
performance (D-F), and that the model reproduces the cortical burst-fraction code
— ISI structure, sharp burst-fraction tuning, and DSI dominance over tonic rate
(G-I, cf. Fig. 1D-K).

### Reproduce the figure from the provided data

The trained checkpoint and precomputed data are included, so the figure builds
without retraining:

```bash
python burst_fraction_model.py     # writes figure2_model.pdf (= Figure 2)
```

### Regenerate the data from scratch (optional)

```bash
python train_reacher.py            # trains; writes reacher_checkpoint.pickle
                                   #  + reacher_learning_curve.npy
python eval_reacher.py             # writes reacher_eval_trajectories.pickle
python burst_fraction_model.py     # rebuilds Figure 2 from the fresh data
```

---

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `burst_fraction_combined.py` | `figure1_burst_fraction.pdf`, `figure1_population_structure.pdf` | **Fig. 1.** MC\_Maze + MC\_RTT combined figure, plus population bilinear-structure panel. |
| `burst_fraction_mcmaze.py`   | `mc_maze_burst_fraction.pdf` | MC\_Maze analysis + controls (auto-downloads 000128). |
| `burst_fraction_mcrtt.py`    | `mc_rtt_burst_fraction.pdf` | MC\_RTT analysis. |
| `burst_fraction_figure.py`   | `burst_fraction_summary.pdf` | Alternative summary layout. |
| `download_data.py`           | `mc_maze_train.nwb`, `mc_rtt_train.nwb` | Fetch both datasets from DANDI. |
| `burst_fraction_model.py`    | `figure2_model.pdf` | **Fig. 2.** Two-compartment model figure (builds from the provided data). |
| `train_reacher.py`           | `reacher_checkpoint.pickle`, `reacher_learning_curve.npy` | Train the burst-gating agent on the 4-goal reacher task. |
| `eval_reacher.py`            | `reacher_eval_trajectories.pickle`, `figure2_eval_trajectories.pdf` | Evaluate the checkpoint; best-of-N trajectories per goal. |

### Data files (included, for Fig. 2)

| File | Description |
|------|-------------|
| `reacher_checkpoint.pickle`        | Trained network weights. |
| `reacher_learning_curve.npy`       | 100-episode running-mean reward (panel D). |
| `reacher_eval_trajectories.pickle` | Best-of-20 evaluation fingertip trajectories per goal (panel F). |

### Model dependencies

`goal.py` (recurrent reservoir `GOAL_GRADED`), `optimizer.py` (Adam), `utils.py`.

## Analysis summary

For each neuron and reach direction:

- **tonic rate** = total spikes / (trials x window)
- **burst fraction** = burst spikes / total spikes (burst spike if preceding ISI < 10 ms)
- **DSI** (direction selectivity index) = (max - min) / (max + min) across directions

## License

MIT — see [LICENSE](LICENSE).
