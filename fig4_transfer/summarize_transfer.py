#!/usr/bin/env python3
"""Paper-ready numbers table for the known-direction transfer benchmark.

Runs on any results directory, including the merged one written by make_figure4.py:

    python summarize_transfer.py --output-dir results/figure4
"""
import argparse, csv, math
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["bilinear-rw", "sf-gpi", "sf-actor"]
LABELS = {"bilinear-rw": "G×Y", "sf-gpi": "SF-GPI", "sf-actor": "SF + actor PI"}

FIELDS = [
    ("reward_per_step", "Reward / step", 1.0),
    ("env_steps", "Environment steps", 1.0),
    ("cumulative_macs", "Cumulative GMAC", 1e-9),
    ("model_seconds", "Model time (s)", 1.0),
    ("total_seconds", "End-to-end time (s)", 1.0),
    ("mean_action_ms", "Policy inference (ms)", 1.0),
    ("gradient_steps", "Gradient updates", 1.0),
    ("adapted_parameters", "Task-specific / adapted values", 1.0),
    ("resident_parameter_mb", "Model parameters (MiB)", 1.0),
    ("peak_rss_mb", "Peak RSS (MiB)", 1.0),
]

def seed_summary(rows, method, field, episode=5):
    vals = []
    seeds = sorted({int(r["seed"]) for r in rows})
    for seed in seeds:
        selected = [float(r[field]) for r in rows if r["method"] == method
                    and int(r["episode"]) == episode and int(r["seed"]) == seed]
        vals.append(np.mean(selected))
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals)))

def fmt(mean, sem, field):
    if field in {"env_steps", "gradient_steps", "adapted_parameters"}:
        return f"{mean:,.0f} ± {sem:,.0f}"
    if field in {"cumulative_macs", "model_seconds", "total_seconds", "resident_parameter_mb", "peak_rss_mb"}:
        return f"{mean:.2f} ± {sem:.2f}"
    return f"{mean:.3f} ± {sem:.3f}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results" / "figure4"),
                   help="results directory (default: the merged one from make_figure4.py)")
    p.add_argument("--input", default=None,
                   help="results CSV (default: <output-dir>/adaptation_results.csv)")
    args = p.parse_args()
    csv_path = Path(args.input) if args.input else Path(args.output_dir) / "adaptation_results.csv"
    rows = list(csv.DictReader(open(csv_path)))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(dict)
    for field, _, scale in FIELDS:
        for method in METHODS:
            mean, sem = seed_summary(rows, method, field)
            stats[method][field] = (mean * scale, sem * scale)
    for method in METHODS:
        m_time, s_time = stats[method]["model_seconds"]
        m_steps, _ = stats[method]["env_steps"]
        stats[method]["processing_ms"] = (1000*m_time/m_steps, 1000*s_time/m_steps)
        m_macs, s_macs = stats[method]["cumulative_macs"]
        stats[method]["mmac_per_step"] = (1000*m_macs/m_steps, 1000*s_macs/m_steps)

    table_fields = FIELDS + [
        ("processing_ms", "Inference + adaptation (ms/step)", 1),
        ("mmac_per_step", "Compute (MMAC/step)", 1),
    ]
    with (out / "known_adaptation_numbers.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Metric"] + [LABELS[m] for m in METHODS])
        for field, label, _ in table_fields:
            w.writerow([label] + [fmt(*stats[m][field], field) for m in METHODS])

    gy, gpi, actor = stats["bilinear-rw"], stats["sf-gpi"], stats["sf-actor"]
    ratio = lambda other, field: other[field][0] / gy[field][0]
    caption = (
        "**Figure X | Known-direction transfer exposes the computational advantage of shared "
        "bilinear gating.** The MuJoCo Ant was pretrained on eight directions and evaluated on "
        "eight interleaved unseen directions (22.5° offsets; five seeds). The known descriptor "
        "$(\\cos\\theta,\\sin\\theta)$ was supplied to each method. G×Y mapped it once to the "
        "eight-dimensional shared gate (16 MACs), which was then frozen (blue dashed reference); SF-GPI "
        "evaluated actions proposed by its policy bank; SF-actor updated an independently "
        "parameterized MLP by policy-improvement backpropagation through the frozen SF value envelope. "
        "**A,** Reward versus environment interaction. **B,** Reward versus cumulative neural "
        "compute. **C,** Reward versus model-only test-time computation. **D,** Policy-inference "
        "latency (light) and total model processing per environment step, including adaptation "
        "(solid). Lines show means and shading denotes SEM across five seeds after averaging "
        "the eight directions within each seed."
    )
    results = f"""## Methods

The MuJoCo Ant-v4 agent was trained on eight directions separated by 45° and tested on the eight intermediate directions. The reward was $r_t=v_x\\cos\\theta+v_y\\sin\\theta$. Each test used five episodes, five random seeds, and deterministic policy evaluation. In the known-task condition, all methods received $(\\cos\\theta,\\sin\\theta)$. G×Y used a single `Linear(2,8)` encoder (16 MACs) to infer the shared gate once; the gate and all network parameters were then frozen, so no critic evaluation, replay, or gradient update was required for adaptation. SF-GPI generated one action from each of eight source actors and selected among them with the successor-feature value bank. SF-actor initialized a conventional 77,072-parameter MLP from the nearest source policy and performed online gradient updates through the frozen SF/GPI value envelope. Linear-layer MACs were counted from executed operations; backward MACs were estimated as twice the corresponding forward MACs. Model time excluded MuJoCo stepping. Statistics are mean ± SEM across five seeds after averaging directions within seed.

## Results

Known-direction encoding restored effective zero-shot G×Y transfer. At episode 5, G×Y achieved {fmt(*gy['reward_per_step'], 'reward_per_step')} reward per step, compared with {fmt(*gpi['reward_per_step'], 'reward_per_step')} for SF-GPI and {fmt(*actor['reward_per_step'], 'reward_per_step')} for SF-actor. G×Y consumed {fmt(*gy['cumulative_macs'], 'cumulative_macs')} GMAC, versus {fmt(*gpi['cumulative_macs'], 'cumulative_macs')} for SF-GPI and {fmt(*actor['cumulative_macs'], 'cumulative_macs')} for SF-actor. Thus SF-GPI required {ratio(gpi,'cumulative_macs'):.1f}× and SF-actor {ratio(actor,'cumulative_macs'):.0f}× more cumulative neural compute. Model-only test-time processing was {fmt(*gy['model_seconds'], 'model_seconds')} s for G×Y, {fmt(*gpi['model_seconds'], 'model_seconds')} s for SF-GPI, and {fmt(*actor['model_seconds'], 'model_seconds')} s for SF-actor, corresponding to {ratio(gpi,'model_seconds'):.1f}× and {ratio(actor,'model_seconds'):.1f}× overheads relative to G×Y. G×Y inferred eight gate coefficients once and used no gradient updates; SF-actor adapted 77,072 parameters using {fmt(*actor['gradient_steps'], 'gradient_steps')} gradient steps. Episode termination and fall rate should be reported alongside reward and compute.

## Figure caption

{caption}
"""
    (out / "known_adaptation_paper_text.md").write_text(results)
    print(out / "known_adaptation_paper_text.md")
    print(out / "known_adaptation_numbers.csv")

if __name__ == "__main__": main()
