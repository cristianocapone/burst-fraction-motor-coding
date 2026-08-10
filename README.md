# Burst-fraction motor coding

Code to reproduce **Figures 1, 2 and 4E–H** of *Burst-selective motor coding
inspires multiplicative gating for efficient reinforcement learning* (Capone,
Falorsi, Ciardiello, Manneschi & Gigante).

- **Figure 1** — in macaque motor cortex, **burst fraction encodes reach
  direction more selectively than tonic spike rate**, against a timing-preserving
  jitter null.
- **Figure 2** — a minimal **two-compartment burst-gating model** reproduces this
  coding signature and learns goal-directed reaching through context-selective
  bursting.
- **Figure 4E–H** — sharing the same gate between actor and critic makes task
  transfer cheap: **G×Y reaches higher reward than successor-feature transfer
  while using 53.8× to 4095× less cumulative neural compute**.

A spike is labelled a *burst spike* if its inter-spike interval to the preceding
spike is below 10 ms; for each neuron we compare the direction selectivity of the
tonic firing rate against that of the burst fraction.

## Layout

One folder per figure, each self-contained and documented by its own README:

```
fig1_burst_fraction/   Fig. 1  — macaque motor-cortex analysis (data from DANDI)
fig2_burst_model/      Fig. 2  — two-compartment burst-gating model + reacher control
fig4_transfer/         Fig. 4E-H — task-transfer cost vs successor features
```

Scripts resolve their own paths, so they can be run from anywhere in the repo.

## Install

```bash
pip install -r requirements.txt
```

| Figure | Needs |
|--------|-------|
| 1 | `numpy`, `scipy`, `matplotlib`, `pynwb`, `h5py`, `dandi` |
| 2 | + `gym`, `mujoco` (`tqdm`, `imageio` optional) |
| 4E–H | + `torch` (CPU is enough) |

gym 0.26's `Reacher-v4` expects `numpy < 2.0` (the code includes a `np.bool8`
shim, but a numpy-1.x environment is the tested configuration).

---

## [Figure 1](fig1_burst_fraction) — burst fraction vs firing rate

Two openly available datasets from the [DANDI Archive](https://dandiarchive.org):
MC\_Maze ([000128](https://dandiarchive.org/dandiset/000128), monkey Jenkins) and
MC\_RTT ([000129](https://dandiarchive.org/dandiset/000129), monkey Indy). Data
files are **not** included; `download_data.py` fetches them.

```bash
python fig1_burst_fraction/download_data.py
python fig1_burst_fraction/burst_fraction_combined.py   # figure1_burst_fraction.pdf
```

Burst-fraction DSI exceeds tonic-rate DSI in 95% of MC\_Maze neurons (p < 1e-29)
and 90% of MC\_RTT neurons (p < 1e-16), Wilcoxon signed-rank.

---

## [Figure 2](fig2_burst_model) — two-compartment model

A burst is emitted with probability `p_burst = <p_s(V_s) * p_d(V_d)>`, the product
of a basal (state) and an apical (goal) factor — a multiplicative gate
`G_k(g) * Y_k(s)` at the single-cell level. The motor command is linear in
`p_burst` and the readouts are learned by a local three-factor rule.

The trained checkpoint and evaluation data are included, so the figure builds
without retraining:

```bash
python fig2_burst_model/scripts/make_figure2.py         # figure2_model.pdf
```

Retraining: `scripts/train.py` → `scripts/eval_policy.py` → `scripts/make_figure2.py`.
Training reaches ≈ −60 average episode reward and reproduces the per-goal rewards
in the Fig. 2F legend.

---

## [Figure 4E–H](fig4_transfer) — task-transfer cost

The MuJoCo Ant is pretrained on eight directions and evaluated on the eight
interleaved unseen directions (22.5° offsets, five seeds). Every method receives
the same descriptor `(cos theta, sin theta)`; the comparison isolates what each
one must *compute* to turn that descriptor into behaviour.

```bash
python fig4_transfer/make_figure4.py         # panels + merged table
python fig4_transfer/summarize_transfer.py   # paper-ready numbers and prose
```

Episode 5, mean ± SEM over five seeds (directions averaged within seed):

| | reward / step | cumulative GMAC | model time (s) | gradient steps |
|---|---|---|---|---|
| **G×Y**  | 4.102 ± 0.101 | 0.38 ± 0.01 | 1.85 ± 0.10 | 0 |
| SF-GPI   | 1.788 ± 0.140 | 20.43 ± 0.77 | 4.73 ± 0.18 | 0 |
| SF-actor | 1.611 ± 0.106 | 1553.71 ± 36.40 | 12.06 ± 0.30 | 2,990 ± 70 |

The published panels merge two benchmark runs (frozen-gate G×Y + the baselines);
`make_figure4.py` verifies both runs' metadata before merging. See the folder
README for why.

---

## Analysis summary

For each neuron and reach direction, in the 500 ms window after movement onset:

- **tonic rate** = tonic spikes / (trials × window)
- **burst fraction** = burst spikes / total spikes (burst spike if preceding ISI < 10 ms)
- **DSI** (direction selectivity index) = (max − min) / (max + min) across directions

## License

MIT — see [LICENSE](LICENSE).
