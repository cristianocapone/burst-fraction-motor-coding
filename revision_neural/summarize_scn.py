"""Aggregate single_compartment_null.py checkpoints into one table and figure.

Reads scn_*.json in this folder (monkey C GLM / gamma+calibration, MC_Maze,
MC_RTT, Indy if present) and writes
  scn_summary.md   per-session table + session-level tests
  scn_summary.pdf  A: observed vs jitter vs GLM null per session (sorted)
                   B: jitter-test calibration on synthetic non-bursting neurons
Internal working figure; nothing here touches the manuscript.
"""
from pathlib import Path
import glob
import json

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
# print style (same as matched_panels.STYLE): all text 7 pt, panel letters 10 pt bold
STYLE = {'font.size': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
         'axes.labelsize': 7, 'axes.titlesize': 7, 'axes.linewidth': 0.6,
         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
         'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
         'lines.linewidth': 1.0, 'lines.markersize': 3, 'pdf.fonttype': 42, 'font.family': 'sans-serif'}


def panel_letter(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def load_all():
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "scn_*.json"))):
        for name, r in json.load(open(f)).items():
            if r.get("good_mode", "paper") != "paper":
                name = f"{name} [{r['good_mode']} mask]"   # keep alternative unit masks separate
            rows.setdefault(name, {}).update(r)      # gamma and glm runs of one session merge
    return rows


def dataset_of(name):
    if name.startswith("C_CO"):
        return "monkey C"
    if name.startswith("indy_"):
        return "Indy"
    return name


def main():
    rows = load_all()
    glm = {k: r for k, r in rows.items() if "glm_d" in r}
    lines = ["# Single-compartment GLM null vs observed burst-fraction dDSI", "",
             "| session | N units | observed | jitter null | p (jitter) | GLM null | p (GLM) | bf obs / GLM |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(glm.items()):
        lines.append(f"| {k} | {r['n_units']} | {r['obs_d']:.3f} | {r['jit_d']:.3f} | {r['p_jit_d']:.3g} | "
                     f"{r['glm_d']:.3f} | {r['p_glm_d']:.3g} | {r['obs_bf']:.3f} / {r['glm_bf']:.3f} |")
    lines.append("")
    for ds in sorted({dataset_of(k) for k in glm}) + ["all"]:
        sel = [r for k, r in glm.items() if "[" not in k and (ds == "all" or dataset_of(k) == ds)]
        if len(sel) < 2:
            continue
        o = np.array([r["obs_d"] for r in sel])
        j = np.array([r["jit_d"] for r in sel]); g = np.array([r["glm_d"] for r in sel])
        sig_j = sum(r["p_jit_d"] < .05 for r in sel); sig_g = sum(r["p_glm_d"] < .05 for r in sel)
        pj = wilcoxon(o - j, alternative="greater").pvalue if len(sel) >= 5 else np.nan
        pg = wilcoxon(o - g, alternative="greater").pvalue if len(sel) >= 5 else np.nan
        lines.append(f"**{ds}** ({len(sel)} sessions): observed {o.mean():.3f}; jitter null {j.mean():.3f} "
                     f"(above in {np.sum(o > j)}/{len(sel)}, per-session p<.05 in {sig_j}/{len(sel)}, Wilcoxon p={pj:.3g}); "
                     f"GLM null {g.mean():.3f} (above in {np.sum(o > g)}/{len(sel)}, per-session p<.05 in {sig_g}/{len(sel)}, "
                     f"Wilcoxon p={pg:.3g}); mean excess over GLM {np.mean(o - g):+.3f} vs over jitter {np.mean(o - j):+.3f}")
        lines.append("")

    calib = {k: r["calib"] for k, r in rows.items() if "calib" in r}
    if calib:
        lines += ["## Jitter-test calibration on synthetic non-bursting renewal neurons", "",
                  "| session | kappa | synthetic dDSI | its jitter null | median p |", "|---|---|---|---|---|"]
        for k, c in sorted(calib.items()):
            for kap, v in sorted(c.items(), key=lambda kv: float(kv[0])):
                lines.append(f"| {k} | {kap} | {v['syn_d']:.3f} | {v['jit_d']:.3f} | {v['p_median']:.3g} |")
        lines.append("")
    (ROOT / "scn_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- figure ----
    plt.rcParams.update(STYLE)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.5), constrained_layout=True)
    if glm:
        keys = sorted([k for k in glm if "[" not in k], key=lambda k: glm[k]["obs_d"])
        o = np.array([glm[k]["obs_d"] for k in keys]); j = np.array([glm[k]["jit_d"] for k in keys])
        g = np.array([glm[k]["glm_d"] for k in keys]); x = np.arange(len(keys))
        axA.vlines(x, g, o, color="0.85", lw=0.8, zorder=0)
        axA.plot(x, j, "o", color="#e08a1e", ms=3, label="jitter null (paper)")
        axA.plot(x, g, "s", color="#1f77b4", ms=3, label="single-compartment GLM null")
        axA.plot(x, o, "o", color="#333", ms=3, label="observed")
        for i, k in enumerate(keys):
            if glm[k]["p_glm_d"] >= .05:
                axA.plot(i, o[i], "o", mfc="none", mec="#d62728", ms=6.5, mew=0.9)
        axA.set_xlabel(f"session (sorted), n={len(keys)}; red ring: n.s. vs GLM null")
        axA.set_ylabel("dDSI = DSI(burst fraction) - DSI(tonic rate)")
        axA.legend(frameon=False)
        axA.spines[["top", "right"]].set_visible(False)
        axA.set_title("observed vs nulls", loc="left")
        panel_letter(axA, "A")
    if calib:
        kappas = sorted({float(k) for c in calib.values() for k in c})
        for name, c in sorted(calib.items()):
            ks = [float(k) for k in sorted(c, key=float)]
            excess = [c[str(k) if str(k) in c else f"{k:.1f}"]["syn_d"] - c[str(k) if str(k) in c else f"{k:.1f}"]["jit_d"]
                      for k in ks]
            axB.plot(ks, excess, "-o", ms=3, lw=1, alpha=0.7, label=name if len(calib) <= 6 else None)
        axB.axhline(0, color="0.5", lw=0.8)
        axB.set_xscale("log"); axB.set_xticks(kappas); axB.set_xticklabels([f"{k:g}" for k in kappas]); axB.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axB.set_xlabel("ISI regularity κ (1 = Poisson)")
        axB.set_ylabel("excess over own jitter null")
        axB.set_title("jitter test on non-bursting neurons", loc="left")
        if len(calib) <= 6:
            axB.legend(frameon=False)
        axB.spines[["top", "right"]].set_visible(False)
        panel_letter(axB, "B")
    fig.savefig(ROOT / "scn_summary.pdf")
    print("saved scn_summary.md / scn_summary.pdf")


if __name__ == "__main__":
    main()
