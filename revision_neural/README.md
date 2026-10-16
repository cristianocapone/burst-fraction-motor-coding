# Revision: neural analyses (revised Fig. 1, Fig. 2G–I, Supplementary Text S1–S3 and S9)

Code and derived results for the analyses added in the revision:

- a **single-compartment null** fitted to every recorded unit (Poisson GLM with a free rate time course per
  direction and a spike-history filter), against which the burst selectivity is tested;
- the same test in **five monkeys** (C, M, T and J from DANDI 000688; Indy from Zenodo 583331) and
  separately for **M1 and PMd** units;
- a **direct fit of the gate** (a direction-specific burst term added to the GLM), its stability across
  halves of the trials, the **held-out burst probability at matched expected rate** (revised Fig. 1G) and a
  **multiplicative versus additive** comparison;
- the **two-compartment model** analysed exactly like the data (Fig. 2G–I, Supplementary Text S3 and S9).

All scripts resolve paths relative to this folder. Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`,
`pynwb`, `h5py` and `requests` (see the top-level `requirements.txt`).

## Rebuild the figures from the included results (minutes)

The derived results (JSON files in the sub-folders) are included, so the figures build without recomputing:

```bash
python download_dandi688.py C          # the 15 monkey C sessions used in Fig. 1C-E (about 4.3 GB)
python make_final_Fig1.py              # revised Fig. 1        -> final_Fig1.pdf
python final_Fig2.py                   # revised Fig. 2        -> final_Fig2.pdf
python figS12_areas.py                 # fig. S12              -> figS12_areas.pdf
python summarize_scn.py                # fig. S7 and Table S1  -> scn_summary.pdf / .md
```

## Data

| Data | Source | How |
|------|--------|-----|
| Monkeys C, M, T, J (centre-out) | DANDI 000688 (Perich & Miller) | `python download_dandi688.py C M T J` (C: the 15 largest sessions, as in the paper) |
| Monkey Indy, 37 M1 sessions | Zenodo 583331 (O'Doherty et al.) | streamed and discarded one session at a time by `--dataset indy` |
| MC_Maze, MC_RTT | DANDI 000128, 000129 | `python ../fig1_burst_fraction/download_data.py`, then link `mc_maze_train.nwb` and `mc_rtt_train.nwb` into this folder |

`dandi000688_unit_area.json` gives the cortical area (M1 or PMd) of every unit, read from the electrode-array
label in the data files. `dandi000688_assets.json` and `zenodo583331_meta.json` are cached file lists.

## Scripts and where they appear in the paper

| Script | Analysis | Paper | Results |
|--------|----------|-------|---------|
| `single_compartment_null.py` | ΔDSI against the fitted single-compartment null, per session; `--dataset C/M/T/J/indy/maze/rtt`, `--area M1/PMd`, `--good-mode total` (burst-independent unit selection, main text) | Results; Methods (iii); Supplementary Text S1–S2; Table S1; fig. S7 | `scn_glm_*.json`, `scn_glm_M/`, `scn_glm_TJ/`, `scn_area_total/` |
| `summarize_scn.py` | table and figure of the nulls; jitter-null calibration on synthetic neurons | Supplementary Text S1–S2; fig. S7 | — |
| `glm_threshold_robustness.py` | burst thresholds 5, 10, 15 ms | Supplementary Text S1; Fig. 1F | `thr_robust/` |
| `glm_gain_null.py` | null with trial-to-trial gain | Supplementary Text S1 | — |
| `isolation_glm.py` | excess by unit isolation (waveform SNR) | Supplementary Text S1 | — |
| `matched_count_selectivity.py`, `matched_count_glm_null.py` | bursts vs the same number of tonic spikes (equal noise) | Fig. 1D–F; Supplementary Text S1 | `matchednull/`, `matchednull_model/` |
| `matched_count_glm_null_area.py` | the same, restricted to one cortical area (`M1`/`PMd`) | Fig. 1F, Fig. 2I (M1 band); Supplementary Text S1 | `matchednull_M1/` |
| `model_vs_glm_null.py` | two-compartment populations scored like the data; apical-drive sweep (`--i-api`, `--api-cv`, `--n-trials`) | Fig. 2G–I; Supplementary Text S3, S9; Table S10 | `model_vs_glm_null.json`, `model_sweep_m1/` |
| `gated_glm2.py` | direct fit of a direction-specific burst term, likelihood-ratio test against a common burst term (7 df), null with trial gain; `--model-api` for model populations; `--subsample-trials`, `--thin` for the power controls | Results; Supplementary Text S9; Table S10; fig. S12B | `gated2/`, `gated2_MTJ/`, `gated2_sub40/`, `gated2_thin/` |
| `gated_glm4.py` | burst term fitted on odd and on even trials (stability); null with direction-specific trial gain | Supplementary Text S9; fig. S12C | `gated4/` |
| `burst_prob_cv.py` | held-out burst probability at matched expected rate, in each unit's own gate direction vs the opposite one | Fig. 1G; Supplementary Text S9; fig. S12D | `bpmcv/` |
| `gate_ms.py` | multiplicative (`m`) vs additive (`s`) gate, per spike | Supplementary Text S9 | `gatems/` |
| `burst_prob_matched.py`, `summarize_bpm.py` | as `burst_prob_cv.py` but aligned to the rate-preferred direction (exploratory, not in the paper) | — | `bpm/` |
| `summarize_gated4.py`, `summarize_gatems.py` | summaries of the stability and m/s fits | Supplementary Text S9 | — |
| `make_final_Fig1.py`, `final_Fig2.py`, `figS12_areas.py`, `make_new_fig1F_2GHI.py` | figures | Figs. 1, 2; fig. S12 | — |

Helpers: `audit_monkeyC_nulls.py` (loader, jitter null, DSI), `glm_newton.py` (Newton/IRLS fit of the GLMs),
`matched_panels.py` (plotting), `fig0b_null_models.py`, `jitter_audit_maze_rtt.py` and
`mcrtt_multisession_permutation.py` (MC_Maze/MC_RTT and Indy readers), `run_pool.py` (runs a job list in
parallel), `download_dandi688.py`.

## Recompute (hours to days)

Each script processes one session per call and saves after every unit, so interrupted runs resume. Examples:

```bash
python single_compartment_null.py --dataset C --models glm --good-mode total --check scn_glm_C.json
python single_compartment_null.py --dataset M --session 20140929 --area M1 --models glm --good-mode total \
       --check scn_area_total/M_20140929_M1.json
python gated_glm2.py   --dataset C --session 20150309 --out gated2/C_20150309.json
python gated_glm4.py   --dataset M --session 20140929 --out gated4/M_20140929.json
python burst_prob_cv.py --dataset C --session 20150309 --out bpmcv/C_20150309.json
python gate_ms.py      --dataset C --session 20150309 --out gatems/C_20150309.json
python gated_glm2.py   --model-api 1.0 --model-api-cv 1 --model-trials 110 --out gated2/m110_api1.0_cv1.json
python model_vs_glm_null.py --graded --hazard-scale 0.02 --basal-depth 0.5 --trial-cv 0.3 \
       --i-api 1.0 --api-cv 1 --n-trials 110 --seed 1 --out model_sweep_m1/api1.0_cv1_s1.json
```

A job list (one `name command` per line) can be run in parallel with `python run_pool.py jobs.txt 4 logs/`.
The GLM fits take about 0.5 s each; a monkey C session takes 5–20 minutes per analysis on one core.
