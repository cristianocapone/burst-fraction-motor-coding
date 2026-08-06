# Figure 4E–H — task-transfer cost against successor-feature baselines

Matched-information transfer benchmark on the MuJoCo `Ant-v4`. The agent is
pretrained on eight directions (45° apart) and evaluated on the eight interleaved
unseen directions (22.5° offsets), five episodes, five seeds, deterministic
evaluation, reward `r_t = vx cos(theta) + vy sin(theta)`.

All three methods receive the same task descriptor `(cos theta, sin theta)`:

| Method | Test-time computation |
|---|---|
| **G×Y** (`bilinear-rw` with `--g-lr 0`) | maps the descriptor once through a `Linear(2,8)` encoder (16 MACs) to infer the shared gate; gate and all network parameters then frozen — no critic evaluation, no replay, no gradient update |
| **SF-GPI** (`sf-gpi`) | one action from each of eight source actors, selected with the successor-feature value bank |
| **SF-actor** (`sf-actor`) | a 77,072-parameter MLP initialised from the nearest source policy, fine-tuned online through the frozen SF/GPI value envelope |

The successor-feature critics are genuine successor features: the vector Bellman
target is `phi_t + gamma * psi(s_{t+1}, a_{t+1})` with `phi = (vx, vy)`, not a
scalar value basis relabelled as successor features.

Linear-layer MACs are counted from executed operations, backward MACs estimated
as twice the corresponding forward MACs. `model_seconds` excludes MuJoCo
stepping; `total_seconds` includes it.

## Reproduce the figure from the provided data

```bash
python make_figure4.py        # merged table + panels in results/figure4/
python summarize_transfer.py  # paper-ready numbers table + prose
```

Output (episode 5, mean ± SEM over 5 seeds, directions averaged within seed):

| | reward / step | cumulative GMAC | model time (s) | gradient steps |
|---|---|---|---|---|
| G×Y      | 4.102 ± 0.101 | 0.38 ± 0.01 | 1.85 ± 0.10 | 0 |
| SF-GPI   | 1.788 ± 0.140 | 20.43 ± 0.77 | 4.73 ± 0.18 | 0 |
| SF-actor | 1.611 ± 0.106 | 1553.71 ± 36.40 | 12.06 ± 0.30 | 2,990 ± 70 |

→ SF-GPI needs **53.8×** and SF-actor **4095×** more cumulative neural compute
than G×Y, at 2.6× and 6.5× the model-only test-time processing.

### Why two runs are merged

The published panels combine two runs of `benchmark_transfer.py`, because the
G×Y condition reported in the paper is the frozen-gate one:

| Directory | Command | Supplies |
|---|---|---|
| `results/known_frozen_gy` | `--methods bilinear-rw --task-information known --g-lr 0 --deterministic` | the G×Y rows |
| `results/known` | `--methods sf-gpi sf-actor --task-information known --deterministic` | the SF-GPI / SF-actor rows |

The baselines never read the gate, so their rows are unaffected by `--g-lr`.
`make_figure4.py` checks both metadata files (`task_information == "known"`, and
`g_lr == 0` for the G×Y run) and refuses to merge mismatched runs. A single run
directory also plots on its own with `python benchmark_transfer.py plot
--output-dir results/known`, but that is not the published panel set — its G×Y
row is the online reward-weighted variant (1.956 ± 0.053 reward/step, 0.31 GMAC).

## Regenerate everything from scratch (optional, slow)

```bash
# 1. train the eight successor-feature source experts (hours on CPU)
python benchmark_transfer.py train-sf --episodes-per-task 2000 \
    --updates-per-episode 50

# 2. the two benchmark runs
python benchmark_transfer.py benchmark --methods bilinear-rw \
    --task-information known --g-lr 0 --deterministic \
    --output-dir results/known_frozen_gy
python benchmark_transfer.py benchmark --methods sf-gpi sf-actor \
    --task-information known --deterministic \
    --output-dir results/known

# 3. merge + plot
python make_figure4.py
```

The pretrained bilinear (G×Y) actor/critic is `checkpoints/pretrain_final.pth`;
it comes from the hierarchical SAC pretraining described in Methods §4.2 and is
not retrained here.

## Files

| Path | Contents |
|---|---|
| `benchmark_transfer.py` | SF expert training, the three transfer methods, MAC/latency instrumentation, single-run plotting |
| `make_figure4.py` | merges the two runs → `results/figure4/` + the four panels |
| `summarize_transfer.py` | paper-ready numbers table and prose |
| `checkpoints/sf_bank.pth` | eight successor-feature source experts |
| `checkpoints/pretrain_final.pth` | pretrained bilinear G×Y actor/critic |
| `results/known`, `results/known_frozen_gy` | the two benchmark runs (CSV + metadata) |

Needs `torch` in addition to the repo's other requirements. Runs on CPU.
