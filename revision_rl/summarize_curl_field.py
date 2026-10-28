"""Summarise the curl-field adaptation benchmark (benchmark_curl_field.py).

Reads full_gain1.0/curl_field_results.csv (perturbed runs, all methods) and
ref_gain0/curl_field_results.csv (unperturbed reference) and writes
curl_field_summary.md / .pdf. For gy-es only the exploit episodes (odd episode
numbers: the current best gate is executed) enter the learning curve; trial
episodes are exploration.
"""
from pathlib import Path
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "results_curl_field"
LABEL = {"gy-frozen": "G×Y, gate frozen", "gy-es": "G×Y, gate search (8 params)",
         "gy-rw-base": "G×Y, per-step RW rule", "sf-gpi": "SF-GPI", "sf-actor": "SF + actor fine-tuning (77k params)"}
PLOT_LABEL = {"gy-frozen": "G-shared, gate frozen", "gy-es": "G-shared, gate search (8 params)",
              "gy-rw-base": "G-shared, per-step rule", "sf-gpi": "SF-GPI", "sf-actor": "SF + actor (77k params)"}
STYLE = {'font.size': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
         'axes.labelsize': 7, 'axes.titlesize': 7, 'axes.linewidth': 0.6, 'axes.titlepad': 4,
         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
         'xtick.minor.width': 0.4, 'xtick.minor.size': 1.5,
         'lines.linewidth': 1.0, 'lines.markersize': 3, 'pdf.fonttype': 42, 'font.family': 'sans-serif'}
COLOR = {"gy-frozen": "0.45", "gy-es": "#1a6fbe", "gy-rw-base": "#7fb3e6", "sf-gpi": "#2ca02c", "sf-actor": "#cc2222"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("gain", "g_lr", "theta_deg", "reward_per_step", "cumulative_reward", "heading_error_deg",
                  "frac_on_target", "mean_speed", "cumulative_macs", "model_seconds", "g_norm"):
            r[k] = float(r[k])
        for k in ("seed", "episode", "env_steps", "steps", "gradient_steps", "adapted_parameters"):
            r[k] = int(r[k])
        r["field_on"] = r["field_on"] == "True"; r["terminated"] = r["terminated"] == "True"
    return rows


def curve(rows, method, key, field_on):
    """mean and SEM over (direction, seed) runs of `key` per episode."""
    per_run = defaultdict(dict)
    for r in rows:
        if r["method"] != method or r["field_on"] != field_on:
            continue
        if method == "gy-es" and field_on and r["episode"] % 2 == 0:
            continue                       # trial episodes
        per_run[(r["theta_deg"], r["seed"])][r["episode"]] = r[key]
    episodes = sorted({e for d in per_run.values() for e in d})
    M = np.array([[d.get(e, np.nan) for e in episodes] for d in per_run.values()])
    return np.array(episodes), np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0)), M


def main():
    import sys
    run_dir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "full_gain1.0")   # e.g. es80_sigma0.7
    full = load(run_dir / "curl_field_results.csv")
    ref_path = ROOT / "ref_gain0/curl_field_results.csv"
    ref = load(ref_path) if ref_path.exists() else []
    methods = [m for m in LABEL if any(r["method"] == m for r in full)]
    n_runs = len({(r["theta_deg"], r["seed"]) for r in full})
    ref_reward = np.nanmean([r["reward_per_step"] for r in ref if r["method"] == "gy-frozen"]) if ref else np.nan
    ref_heading = np.nanmean([r["heading_error_deg"] for r in ref if r["method"] == "gy-frozen"]) if ref else np.nan

    n_field = max(r["episode"] for r in full if r["field_on"])
    lines = [f"# Curl-field adaptation benchmark (MuJoCo Ant, gain 1.0, known descriptor) -- {run_dir.name}", "",
             f"Runs: {n_runs} (direction x seed). Field on for {n_field} episodes, off for 5 (washout). "
             f"Unperturbed reference (gain 0, frozen gate): reward/step {ref_reward:.2f}, heading {ref_heading:+.0f} deg.", "",
             "| method | reward/step ep. 1 | reward/step last 5 field ep. | deficit recovered | heading ep. 1 | heading last 5 | washout ep. 1 reward | washout heading (aftereffect) | GMAC at end of field | gradient steps | adapted params |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    frozen_end = None
    for m in methods:
        ep, mu, se, M = curve(full, m, "reward_per_step", True)
        eph, muh, seh, _ = curve(full, m, "heading_error_deg", True)
        epw, muw, sew, _ = curve(full, m, "reward_per_step", False)
        _, muwh, sewh, _ = curve(full, m, "heading_error_deg", False)
        last5 = np.nanmean(M[:, -5:], 1)
        macs = np.array([r["cumulative_macs"] for r in full if r["method"] == m and r["episode"] == n_field]) / 1e9
        grads = np.array([r["gradient_steps"] for r in full if r["method"] == m and r["episode"] == n_field])
        params = 0 if m == "gy-frozen" else [r["adapted_parameters"] for r in full if r["method"] == m][0]
        summary[m] = dict(ep=ep, mu=mu, se=se, eph=eph, muh=muh, seh=seh, epw=epw, muw=muw, sew=sew, muwh=muwh,
                          last5=np.nanmean(last5), start=mu[0], macs=np.nanmean(macs), grads=np.nanmean(grads), params=params)
        if m == "gy-frozen":
            frozen_end = np.nanmean(last5)
    for m in methods:
        s = summary[m]
        rec = (s["last5"] - frozen_end) / (ref_reward - frozen_end) if frozen_end is not None and np.isfinite(ref_reward) else np.nan
        lines.append(f"| {LABEL[m]} | {s['start']:.2f} | {s['last5']:.2f} | {100*rec:.0f}% | {s['muh'][0]:+.0f} | {np.nanmean(s['muh'][-5:]):+.0f} | "
                     f"{s['muw'][0]:.2f} | {s['muwh'][0]:+.0f} | {s['macs']:.2f} | {s['grads']:.0f} | {s['params']} |")
    lines += ["", "Deficit recovered = (method's reward in the last 5 field episodes - frozen gate) / (unperturbed reference - frozen gate). "
              "Aftereffect = heading in the first washout episode; the frozen gate's washout heading is the unperturbed bias of the pretrained policy.",
              "Per-step reward = torso velocity along the target direction (m/s)."]
    (run_dir / "curl_field_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(6.5, 2.5))
    gs = fig.add_gridspec(1, 4, left=0.07, right=0.985, bottom=0.22, top=0.72, wspace=0.62)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    handles = []
    ax = axes[0]
    for m in methods:
        s = summary[m]; h, = ax.plot(s["ep"], s["mu"], color=COLOR[m], lw=1.0, label=PLOT_LABEL[m]); handles.append(h)
        ax.fill_between(s["ep"], s["mu"] - s["se"], s["mu"] + s["se"], color=COLOR[m], alpha=0.2, lw=0)
    if np.isfinite(ref_reward):
        handles.append(ax.axhline(ref_reward, color="k", ls=":", lw=0.8, label="no field (frozen gate)"))
    # ceiling of any gate change with frozen primitives: best descriptor offset in the aiming-off scan
    import glob, re
    scan = {}
    for f in glob.glob(str(ROOT / "aimscan/off*/curl_field_results.csv")):
        rows_s = [r for r in load(f) if r["field_on"]]
        scan[int(re.search(r"off(-?\d+)", f).group(1))] = np.mean([r["reward_per_step"] for r in rows_s])
    if scan:
        best_off = max(scan, key=scan.get)
        handles.append(ax.axhline(scan[best_off], color="#1a6fbe", ls="--", lw=0.8,
                                  label=f"re-aiming ceiling ({best_off:+d} deg)"))
        lines.append(f"Re-aiming ceiling from the aiming-off scan: best descriptor offset {best_off:+d} deg gives reward/step {scan[best_off]:.2f} under the field.")
        (run_dir / "curl_field_summary.md").write_text("\n".join(lines))
    ax.set_xlabel("episode (field on)"); ax.set_ylabel("reward/step (m/s)")
    ax.set_title("adaptation"); ax.set_ylim(0.8, 5.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    for m in methods:
        s = summary[m]; ax.plot(s["eph"], s["muh"], color=COLOR[m], lw=1.0)
        ax.fill_between(s["eph"], s["muh"] - s["seh"], s["muh"] + s["seh"], color=COLOR[m], alpha=0.2, lw=0)
    if np.isfinite(ref_heading):
        ax.axhline(ref_heading, color="k", ls=":", lw=0.8)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("episode (field on)"); ax.set_ylabel("heading error (deg)")
    ax.set_title("heading, field on"); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    x = np.arange(len(methods))
    ax.bar(x, [summary[m]["muwh"][0] for m in methods], color=[COLOR[m] for m in methods], width=0.7)
    if np.isfinite(ref_heading):
        ax.axhline(ref_heading, color="k", ls=":", lw=0.8)
    ax.axhline(0, color="0.8", lw=0.6)
    SHORT = {"gy-frozen": "frozen", "gy-es": "search", "gy-rw-base": "per-step", "sf-gpi": "SF-GPI", "sf-actor": "SF+actor"}
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in methods], rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("washout heading (deg)"); ax.set_title("aftereffect")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[3]
    for m in methods:
        s = summary[m]
        ax.plot(max(s["macs"], 1e-3), s["last5"], "o", color=COLOR[m], ms=4)
    ax.set_xscale("log"); ax.margins(x=0.12); ax.set_xlabel("compute (GMAC)"); ax.set_ylabel("reward/step, last 5 ep.")
    ax.set_title("reward vs compute")
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handlelength=1.5, columnspacing=0.8, borderaxespad=0.2)
    for a, l in zip(axes, "ABCD"):
        b = a.get_position()
        fig.text(b.x0 - 0.055, b.y1 + 0.035, l, fontsize=10, fontweight="bold", ha="left", va="bottom")
    fig.savefig(run_dir / "curl_field_summary.pdf")
    print(f"saved {run_dir.name}/curl_field_summary.md / .pdf")


if __name__ == "__main__":
    main()
