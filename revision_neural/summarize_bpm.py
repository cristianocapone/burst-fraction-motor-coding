"""burst_prob_matched.py results (bpm/): P(burst | expected rate rho, direction class), pooled over units.
Figure: A: P(followed within 10 ms) vs rho, preferred vs opposite direction, data (solid) vs null (dashed).
        B: ratio preferred / opposite vs rho, data vs null (flat > 1: multiplicative gate).
Usage: python summarize_bpm.py [pattern]   (default bpm/C_*.json)"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matched_panels as MP
plt.rcParams.update(MP.STYLE)
ROOT = Path(__file__).resolve().parent
pattern = sys.argv[1] if len(sys.argv) > 1 else "bpm/C_*.json"
EDGES = np.logspace(np.log10(0.02), np.log10(5.0), 13); MID = np.sqrt(EDGES[:-1] * EDGES[1:])

units, sess = [], {}
for f in sorted(glob.glob(str(ROOT / pattern))):
    for name, us in json.load(open(f)).items():
        sess[name] = list(us.values()); units += list(us.values())
print(f"{len(sess)} sessions, {len(units)} units")
N = {g: sum(np.array(u[g]["n"]) for u in units) for g in ("data", "null")}
K = {g: sum(np.array(u[g]["k"]) for u in units) for g in ("data", "null")}
P = {g: K[g] / np.maximum(N[g], 1) for g in N}
ok = (N["data"][0] >= 500) & (N["data"][2] >= 500)
for g in ("data", "null"):
    print(g, "P pref:", np.round(P[g][0][ok], 3), "\n     P opp: ", np.round(P[g][2][ok], 3), "\n     ratio: ", np.round(P[g][0][ok] / np.maximum(P[g][2][ok], 1e-9), 2))
# per-session ratio (log), data minus null, for a session-level test
from scipy import stats
def lr(v, g):
    n = sum(np.array(u[g]["n"]) for u in v); k = sum(np.array(u[g]["k"]) for u in v)
    p = (k + 0.5) / (n + 1)
    w = (n[0] >= 100) & (n[2] >= 100)
    return float(np.mean(np.log(p[0][w] / p[2][w]))) if w.any() else np.nan
d = np.array([lr(v, "data") for v in sess.values()]); nl = np.array([lr(v, "null") for v in sess.values()])
m = ~np.isnan(d) & ~np.isnan(nl)
print(f"mean log-ratio pref/opp over rho bins, per session: data {d[m].mean():+.3f} null {nl[m].mean():+.3f}, above {np.sum(d[m]>nl[m])}/{m.sum()}, Wilcoxon p={stats.wilcoxon(d[m]-nl[m], alternative='greater').pvalue:.1g}")

BLUE, GREY, RED = "#2c6fbb", "0.5", "#d62728"
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw=dict(left=0.09, right=0.98, bottom=0.2, top=0.9, wspace=0.35))
ax = axs[0]
for g, ls, lab in (("data", "-", "data"), ("null", "--", "no gate (simulated)")):
    ax.plot(MID[ok], P[g][0][ok], ls, color=BLUE, lw=1.2, label=f"preferred, {lab}")
    ax.plot(MID[ok], P[g][2][ok], ls, color=RED, lw=1.2, label=f"opposite, {lab}")
ax.set_xscale("log"); ax.set_xlabel(r"expected spikes in the next 10 ms, $\rho$ (additive model)"); ax.set_ylabel("P(another spike within 10 ms)")
ax.legend(frameon=False, loc="upper left", handlelength=1.8); ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.18, 1.04, "A", transform=ax.transAxes, fontsize=10, fontweight="bold")
ax = axs[1]
for g, ls, col in (("data", "-", BLUE), ("null", "--", GREY)):
    ax.plot(MID[ok], P[g][0][ok] / np.maximum(P[g][2][ok], 1e-9), ls, color=col, lw=1.2, marker="o", ms=2.5, label=g if g == "data" else "no gate (simulated)")
ax.axhline(1, color="0.8", lw=0.6); ax.set_xscale("log"); ax.set_xlabel(r"$\rho$"); ax.set_ylabel("ratio preferred / opposite")
ax.legend(frameon=False, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.2, 1.04, "B", transform=ax.transAxes, fontsize=10, fontweight="bold")
tag = Path(pattern).name.split("_")[0]
fig.savefig(ROOT / f"bpm_{tag}.pdf"); fig.savefig(ROOT / f"bpm_{tag}.png", dpi=220)
print("saved", f"bpm_{tag}.pdf")
