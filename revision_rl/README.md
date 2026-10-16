# Revision: reinforcement-learning analyses (Supplementary Text S5–S7)

Run every script from this folder. Requires `torch`, `numpy`, `gymnasium` with MuJoCo (`Ant-v4`), and the
pretrained checkpoints of `../fig4_transfer/checkpoints/` (eight-gate G×Y agent and successor-feature bank).

| Script | Analysis | Paper | Checkpoint |
|--------|----------|-------|------------|
| `benchmark_curl_field.py` | curl force field: adapting only the eight gate values vs successor-feature baselines; aftereffect | Results; Supplementary Text S5; figs. S9–S10 | `../fig4_transfer/checkpoints/` |
| `analyze_gate_primitives.py` | how the gate and the primitives form during training; gate interpolation | Supplementary Text S6; fig. S11 | `checkpoints/ant_8gate_training/` |
| `coldstart_g2_reproduce.py` | cold start from G = 0 with the two-gate agent (per-episode reward-weighted update) | Results; Supplementary Text S7; fig. S2C–E | `checkpoints/ant_2gate/` |
| `coldstart_noise_control.py` | cold start of the eight-gate agent and noise control | Supplementary Text S7 | `../fig4_transfer/checkpoints/` |
| `benchmark_sf_adaptation.py` | shared environment, agents and successor-feature baselines used by the scripts above | — | — |

```bash
python benchmark_curl_field.py --output-dir results_curl
python analyze_gate_primitives.py --out results_gate
python coldstart_g2_reproduce.py --per-episode --out results_coldstart_g2
python coldstart_noise_control.py --out results_coldstart
```

`checkpoints/ant_8gate_training/` holds the training checkpoints at episodes 500, 1000, 2000, 4000, 8000 and
15,000 of the eight-gate Ant agent; `checkpoints/ant_2gate/` the final two-gate agent (K = 2, gate without bias).
