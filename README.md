# Burst-fraction motor coding

Code to reproduce **Figures 1 and 2** of *Burst-selective motor coding inspires
multiplicative gating for efficient reinforcement learning* (Capone, Falorsi,
Ciardiello, Manneschi & Gigante).

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
command is linear in `p_burst` (`mu = 7 * p_burst - 1`), and weights are learned
online with a three-factor (reward-modulated Hebbian) rule; only the goal-specific
readouts are trained, the recurrent reservoir stays frozen.

The task: one episode is 400 steps split into four 100-step blocks. Each block
picks one of the four cardinal targets (radius 0.19, random order), re-poses the
arm folded at the centre and moves the target. Reward is
`-distance - 0.005*|torque|^2 - 0.008*|joint velocity|^2`.

### Panels

| | | |
|---|---|---|
| **A** two-compartment neuron sketch | **B** `V_s` traces for apical-only / basal-only / both | **C** phase diagram of the product gate |
| **D** learning curve | **E** context switch at 1500 ms | **F** best-5/20 eval trajectories per goal |
| **G** rate tuning heatmap | **H** burst-fraction tuning heatmap | **I** DSI scatter (91% above diagonal) |

Panels A–C, E and G–I are standalone two-compartment simulations run by the
figure script. Panel D reads the training learning curve and panel F the per-goal
evaluation.

As disclosed in Methods ("Burst-fraction coding analysis"), panels G/H/I add small
measurement-like noise and a minority of weakly burst-selective units **to the
displayed normalised response matrices only**; the underlying two-compartment
simulations (`_pop_bf` / `_pop_rate` in the script) are unchanged and can be
analysed directly for the noise-free result.

### Reproduce the figure from the provided data

The trained checkpoint and precomputed data are included, so the figure builds
without retraining:

```bash
python scripts/make_figure2.py     # writes figure2_model.pdf (= Figure 2)
```

### Regenerate the data from scratch (optional)

```bash
python scripts/train.py --episodes 50000 --outdir .   # reacher_checkpoint.pickle
                                                      #  + reacher_learning_curve.npy
python scripts/eval_policy.py                         # reacher_eval_trajectories.pickle
python scripts/make_figure2.py                        # rebuilds Figure 2
```

Training reaches an average episode reward of about **-60** (from about -95 at
episode 0); evaluation reproduces the per-goal rewards quoted in Fig. 2F
(+X ~ -10, -X ~ -19, +Y ~ -9, -Y ~ -20). `scripts/train.py --resume --ckpt
reacher_checkpoint.pickle` continues from the shipped checkpoint (the
noise/torque schedules resume at the right episode; new files are written to
`--outdir`, never back onto the checkpoint they were loaded from).

### Model code (`reacher_burst/`)

| Module | Contents |
|--------|----------|
| `config.py`   | every hyper-parameter and the exploration/torque schedules |
| `burst.py`    | two-compartment burst neuron, product gate, three-factor rule, `mu` calibration |
| `network.py`  | `GOAL_GRADED` recurrent reservoir (frozen trunk) |
| `agent.py`    | reservoir + per-goal readouts + burst sub-populations |
| `envs.py`     | `Reacher-v4` wrapper: cardinal targets, arm re-posing, reward |
| `rollout.py`  | one 4-goal episode → `EpisodeTrace` (shared by training, evaluation and figures) |
| `evaluate.py` | per-goal evaluation → the panel F data |
| `train.py`    | training loop |
| `figures.py`  | training-diagnostics figure |
| `render.py`   | optional video: MuJoCo frame next to live burst traces |
| `optimizer.py`| Adam |

`scripts/make_figure2.py` imports its model constants from `reacher_burst.config`,
so the figure and the training/evaluation code cannot drift apart.

Library use:

```python
from reacher_burst import make_env, run_episode
from reacher_burst import checkpoint as ckpt
from reacher_burst.burst import calibrate_offset

p_rest, mu_offset = calibrate_offset()
agent = ckpt.load_agent('reacher_checkpoint.pickle', mu_offset)
trace = run_episode(make_env(), agent, episode_idx=49900, learn=False)
print(trace.r_tot, trace.dist.mean())
```

`run_episode` returns an `EpisodeTrace` with `r_tot`, `positions`, `dist`,
`bf_log` (p_burst), `mu_log`, `goal_of_step` and the full `Vs / Vd / spk / brst`
arrays of shape `(steps x 15, num_g, N_B)`.

---

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `burst_fraction_combined.py` | `figure1_burst_fraction.pdf`, `figure1_population_structure.pdf` | **Fig. 1.** MC\_Maze + MC\_RTT combined figure, plus population bilinear-structure panel. |
| `burst_fraction_mcmaze.py`   | `mc_maze_burst_fraction.pdf` | MC\_Maze analysis + controls (auto-downloads 000128). |
| `burst_fraction_mcrtt.py`    | `mc_rtt_burst_fraction.pdf` | MC\_RTT analysis. |
| `burst_fraction_figure.py`   | `burst_fraction_summary.pdf` | Alternative summary layout. |
| `download_data.py`           | `mc_maze_train.nwb`, `mc_rtt_train.nwb` | Fetch both datasets from DANDI. |
| `scripts/make_figure2.py`    | `figure2_model.pdf` | **Fig. 2.** Two-compartment model figure (builds from the provided data). |
| `scripts/train.py`           | `reacher_checkpoint.pickle`, `reacher_learning_curve.npy`, `reacher_training_progress.png` | Train the burst-gating agent on the 4-goal reacher task. |
| `scripts/eval_policy.py`     | `reacher_eval_trajectories.pickle` | Evaluate the checkpoint; best-of-20 trajectories per goal (panel F). |
| `scripts/make_figure.py`     | `reacher_training_progress.png` | Training diagnostics for one episode (not a paper figure). |

### Data files (included, for Fig. 2)

| File | Description |
|------|-------------|
| `reacher_checkpoint.pickle`        | Trained network weights. |
| `reacher_learning_curve.npy`       | 100-episode running-mean reward (panel D). |
| `reacher_eval_trajectories.pickle` | Best-of-20 evaluation fingertip trajectories per goal (panel F). |

## Analysis summary

For each neuron and reach direction:

- **tonic rate** = total spikes / (trials x window)
- **burst fraction** = burst spikes / total spikes (burst spike if preceding ISI < 10 ms)
- **DSI** (direction selectivity index) = (max - min) / (max + min) across directions

## License

MIT — see [LICENSE](LICENSE).
