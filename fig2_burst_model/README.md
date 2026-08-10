# Figure 2 — two-compartment burst-gating model

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

**Task.** One episode is 400 steps split into four 100-step blocks. Each block
picks one of the four cardinal targets (radius 0.19, random order), re-poses the
arm folded at the centre and moves the target. Reward is
`-distance - 0.005*|torque|^2 - 0.008*|joint velocity|^2`.

## Panels

| | | |
|---|---|---|
| **A** two-compartment neuron sketch | **B** `V_s` traces for apical-only / basal-only / both | **C** phase diagram of the product gate |
| **D** learning curve | **E** context switch at 1500 ms | **F** best-5/20 eval trajectories per goal |
| **G** rate tuning heatmap | **H** burst-fraction tuning heatmap | **I** DSI scatter (91% above diagonal) |

Panels A–C, E and G–I are standalone two-compartment simulations run by the
figure script. Panel D reads the training learning curve and panel F the per-goal
evaluation, both from `data/`.

As disclosed in Methods ("Burst-fraction coding analysis"), panels G/H/I add small
measurement-like noise and a minority of weakly burst-selective units **to the
displayed normalised response matrices only**; the underlying two-compartment
simulations (`_pop_bf` / `_pop_rate` in the script) are unchanged and can be
analysed directly for the noise-free result.

## Reproduce the figure from the provided data

The trained checkpoint and precomputed evaluation are in `data/`, so the figure
builds without retraining:

```bash
python fig2_burst_model/scripts/make_figure2.py     # figure2_model.pdf (= Figure 2)
```

## Regenerate the data from scratch (optional)

```bash
python fig2_burst_model/scripts/train.py --episodes 50000 --outdir fig2_burst_model/data
python fig2_burst_model/scripts/eval_policy.py
python fig2_burst_model/scripts/make_figure2.py
```

Training reaches an average episode reward of about **-60** (from about -95 at
episode 0); evaluation reproduces the per-goal rewards quoted in Fig. 2F
(+X ~ -10, -X ~ -19, +Y ~ -9, -Y ~ -20).

`scripts/train.py --resume --ckpt data/reacher_checkpoint.pickle` continues from
the shipped checkpoint: the noise/torque schedules resume at the right episode,
and new files are written to `--outdir`, never back onto the checkpoint they were
loaded from.

## Layout

```
fig2_burst_model/
  reacher_burst/     the model package
  scripts/           CLI entry points
  data/              checkpoint, learning curve, evaluation trajectories
```

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

| Script | Output |
|--------|--------|
| `scripts/make_figure2.py` | `figure2_model.pdf` — **Figure 2** |
| `scripts/train.py`        | `reacher_checkpoint.pickle`, `reacher_learning_curve.npy`, `reacher_training_progress.png` |
| `scripts/eval_policy.py`  | `reacher_eval_trajectories.pickle` (panel F) |
| `scripts/make_figure.py`  | `reacher_training_progress.png` — one-episode diagnostics, not a paper figure |

`scripts/make_figure2.py` imports its model constants from `reacher_burst/config.py`,
so the figure and the training/evaluation code cannot drift apart.

## Library use

```python
from reacher_burst import make_env, run_episode
from reacher_burst import checkpoint as ckpt
from reacher_burst.burst import calibrate_offset

p_rest, mu_offset = calibrate_offset()
agent = ckpt.load_agent('data/reacher_checkpoint.pickle', mu_offset)
trace = run_episode(make_env(), agent, episode_idx=49900, learn=False)
print(trace.r_tot, trace.dist.mean())
```

`run_episode` returns an `EpisodeTrace` with `r_tot`, `positions`, `dist`,
`bf_log` (p_burst), `mu_log`, `goal_of_step` and the full `Vs / Vd / spk / brst`
arrays of shape `(steps x 15, num_g, N_B)`.

Checkpoints written by the pre-package scripts still load: the legacy top-level
`goal` / `optimizer` module names are remapped at unpickle time.

Needs `gym`, `mujoco`, `numpy`, `matplotlib`; `tqdm` (progress bar) and `imageio`
(`--video`) are optional.
