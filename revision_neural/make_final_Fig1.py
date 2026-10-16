"""FINAL revised Figure 1 (D-F at matched noise; see matched_panels.py).
Original docstring: Build the main-text neural Figure 1 (empirical anchor), 3 cols x 2 rows:
A task sketch | B hypothesis sketch | C temporal traces with bursts highlighted
D tonic-rate heatmap | E burst-fraction heatmap | F population DSI scatter
Real panels C-F from cached data; A,B vector schematics. Output fig1_neural.pdf
Null-model / generality / fragmentation controls (formerly F,G,H) now live in
Fig. S1 -- see make_figS1_burst_nulls.py.
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matched_panels as MP
plt.rcParams.update(MP.STYLE)
from matplotlib.patches import Circle, FancyBboxPatch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
from jitter_audit_maze_rtt import load_mr
from audit_monkeyC_nulls import observed, load as load_mc, BIN, NB
WIN, ISI = 0.5, 0.010
BURST_C, TONIC_C = "#d62728", "0.55"
RED, BLUE = "#c0392b", "#2c6fbb"
REFRAC = 0.002   # 2 ms dead time enforced in the Poisson-refractory null


def poisson_refractory_null(binned, di, directions, rng, refractory=REFRAC):
    """Poisson-PSTH null with a hard refractory period: draws spike counts per
    10-ms PSTH bin as in the plain Poisson null, but the within-bin spike times
    are then thinned so no two spikes of the same simulated unit/trial fall
    closer than `refractory`. This removes the biologically implausible
    near-zero-ISI events that make the plain Poisson null unreliable."""
    n, _, nb = binned.shape; nd = len(directions)
    total = np.zeros((n, nd)); burst = np.zeros((n, nd)); active = np.zeros((n, nd))
    for d in range(nd):
        ix = np.flatnonzero(di == d); lam = binned[:, ix].mean(1)
        draw = rng.poisson(lam[None, :, :], size=(len(ix), n, nb))
        flat = draw.ravel(); cells = np.repeat(np.arange(flat.size), flat)
        if len(cells) < 2: continue
        bn = cells % nb; tmp = cells // nb; un = tmp % n; trn = tmp // n
        g = trn * n + un; tms = (bn + rng.random(len(cells))) * BIN
        o = np.lexsort((tms, g)); g, un, tms, trn = g[o], un[o], tms[o], trn[o]
        keep = np.ones(len(g), bool)
        last_t, last_g = -np.inf, -1
        for i in range(len(g)):
            if g[i] != last_g:
                last_g = g[i]; last_t = tms[i]
            elif tms[i] - last_t < refractory:
                keep[i] = False
            else:
                last_t = tms[i]
        g, un, tms, trn = g[keep], un[keep], tms[keep], trn[keep]
        total[:, d] = np.bincount(un, minlength=n)
        if len(g):
            pairs = np.unique(np.stack([un, trn], 1), axis=0)
            active[:, d] = np.bincount(pairs[:, 0], minlength=n)
            sh = (g[1:] == g[:-1]) & (np.diff(tms) < ISI)
            burst[:, d] = np.bincount(un[1:][sh], minlength=n)
    return total, burst, active


# ---------- data: monkey C session CO-20150309 (largest observed-vs-Poisson-refractory-null gap: obs 0.530 vs Poisson-null 0.291, vs jitter-null 0.379) ----------
DATASET_NAME = "Monkey C"
print("loading monkey C session CO-20150309...")
spikes, starts, y = load_mc(str(ROOT / "dandi000688_C_cache" / "sub-C_ses-CO-20150309_behavior+ecephys.nwb"))
directions = np.sort(np.unique(y))
total, burst, active, binned, di, wins = observed(spikes, starts, y, directions)
denom = np.maximum(active, 1)
rate = (total - burst) / (denom * WIN)
frac = np.divide(burst, total, out=np.zeros_like(burst), where=total > 0)
def dsi(x):
    lo, hi = x.min(1), x.max(1)
    return np.divide(hi - lo, hi + lo, out=np.zeros_like(lo), where=(hi + lo) > 0)
dsr, dsf = dsi(rate), dsi(frac)
good = (active >= 5).all(1) & (burst.sum(1) >= 30) & (total.sum(1) > 200)
# example neuron: VISIBLE bursts (peak burst frac >= 0.12), non-saturated DSI,
# clear burst>rate selectivity -> good for the traces panel
# flatter rate + sharper burst -> honest, clear contrast; visible bursts
cand = good & (frac.max(1) >= 0.12) & (dsf > 0.6) & (dsf < 0.9) & (dsr < 0.35)
if cand.sum() == 0:
    cand = good & (frac.max(1) >= 0.10) & (dsf > 0.5) & (dsr < 0.45)
score = np.where(cand, (dsf - dsr), -9)
ex = int(np.argmax(score))
print(f"example neuron {ex}: DSI_frac={dsf[ex]:.2f} DSI_rate={dsr[ex]:.2f} peak_bf={frac[ex].max():.2f}")

# ---------- figure ----------
fig = plt.figure(figsize=(6.5, 5.0))
gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.55, left=0.075, right=0.985, top=0.90, bottom=0.10, height_ratios=[1, 1.05])

def lab(ax, L, dx=-0.12):
    ax.text(dx, 1.13, L, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")

# ===== A task sketch =====
axA = fig.add_subplot(gs[0, 0]); axA.set_aspect("equal"); axA.axis("off")
axA.set_xlim(-1.3, 1.3); axA.set_ylim(-1.3, 1.3)
for k in range(8):
    a = k * np.pi / 4
    axA.add_patch(Circle((np.cos(a), np.sin(a)), 0.16, fill=False, ec="0.5", lw=1.5))
axA.add_patch(Circle((0, 0), 0.13, fc="0.8", ec="0.4"))
tgt = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
axA.add_patch(Circle(tgt, 0.16, fc=RED, ec=RED, alpha=0.85))
tt = np.linspace(0, 1, 40)
curve = np.outer(1 - tt, [0, 0]) + np.outer(tt, tgt) + np.outer(np.sin(tt*np.pi)*0.12, [-1, 1])
axA.plot(curve[:, 0], curve[:, 1], color=BLUE, lw=2)
axA.annotate("", xy=curve[-3], xytext=curve[-8], arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
axA.set_title("Reaching task", fontsize=11)
lab(axA, "A", -0.05)

# ===== B hypothesis sketch =====
axB = fig.add_subplot(gs[0, 1]); axB.axis("off"); axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_title("Hypothesis", fontsize=11)
# two directions, same rate, different burst fraction
def mini_train(x0, y0, w, bursts, seed):
    r = np.random.default_rng(seed); n = 14
    xs = np.sort(r.uniform(0, w, n))
    isb = np.zeros(n, bool)
    for c in bursts:  # cluster indices -> burst
        isb[c] = True
    for i, xi in enumerate(xs):
        axB.plot([x0+xi, x0+xi], [y0, y0+0.07], color=BURST_C if isb[i] else TONIC_C,
                 lw=1.6 if isb[i] else 1.0)
axB.text(0.5, 0.98, "same firing rate,\ndifferent burst fraction", ha="center", va="top", fontsize=8.5)
axB.text(0.02, 0.72, "dir 1", fontsize=8); mini_train(0.22, 0.70, 0.55, [3,4,9,10,11], 1)
axB.text(0.02, 0.52, "dir 2", fontsize=8); mini_train(0.22, 0.50, 0.55, [6], 2)
axB.add_patch(FancyBboxPatch((0.05, 0.10), 0.9, 0.28, boxstyle="round,pad=0.02",
                             fc="0.96", ec="0.6"))
axB.text(0.5, 0.28, "rate channel  ≈ state", ha="center", fontsize=8.5, color=BLUE)
axB.text(0.5, 0.19, "burst channel  → goal ?", ha="center", fontsize=9, color=BURST_C, fontweight="bold")
axB.text(0.5, 0.05, "is burst fraction more direction-\nselective than firing rate?", ha="center", va="top", fontsize=8, style="italic")
lab(axB, "B", -0.05)

# ===== C temporal traces with bursts highlighted =====
axC = fig.add_subplot(gs[0, 2]); axC.set_title("Example neuron", fontsize=11, pad=14)
pref = int(np.argmax(frac[ex])); anti = int(np.argmin(frac[ex])); orth = int(np.argsort(frac[ex])[len(directions)//2])
sel = [(pref, "preferred"), (orth, "orthogonal"), (anti, "anti-pref")]
lane_h = 1.0; ntr = 12
ytop = 0
yticks, ylabels = [], []
for li, (d, name) in enumerate(sel):
    trials = np.flatnonzero(di == d)[:ntr]
    base = li * (ntr + 3)
    for r, t in enumerate(trials):
        w = np.sort(wins[ex][t])
        if len(w) == 0: continue
        isb = np.zeros(len(w), bool)
        if len(w) > 1:
            sh = np.diff(w) < ISI; isb[1:] |= sh; isb[:-1] |= sh
        yy = base + r
        axC.vlines(w[~isb]*1000, yy+0.1, yy+0.9, color=TONIC_C, lw=0.7)
        axC.vlines(w[isb]*1000, yy+0.1, yy+0.9, color=BURST_C, lw=1.1)
    yticks.append(base + ntr/2); ylabels.append(f"{name}\nbf={frac[ex][d]:.2f}")
axC.set_yticks(yticks); axC.set_yticklabels(ylabels, fontsize=8)
axC.set_xlabel("time (ms)", fontsize=9); axC.set_xlim(0, WIN*1000)
axC.set_ylim(-1, 3*(ntr+3)); axC.invert_yaxis()
for li in range(1, 3):                       # dividers between the 3 direction groups
    axC.axhline(li*(ntr+3) - 1.5, color="0.65", lw=0.8)
for s in ["top", "right", "left"]: axC.spines[s].set_visible(False)
axC.plot([], [], color=BURST_C, lw=1.4, label="burst (ISI<10ms)")
axC.plot([], [], color=TONIC_C, lw=0.9, label="tonic")
axC.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0, borderaxespad=0.1)
lab(axC, "C", -0.14)


# ===== D-F: burst spikes vs the same number of tonic spikes (held-out trials) =====
rng_m = np.random.default_rng(0)
good_m = (active >= 5).all(1) & (total.sum(1) >= 100)          # burst-independent selection
R = MP.matrices(wins, di, len(directions), good_m, rng_m)
dlabs = [f"{int(round(d))}" for d in directions]
row_ex = None
if ex in R["units"]:
    row_ex = int(np.flatnonzero(R["units"][R["order"]] == ex)[0])
print(f"D-F: {len(R['units'])} units; example neuron row {row_ex}")

gsb = gs[1, :].subgridspec(1, 7, width_ratios=[0.9, 0.9, 0.25, 0.8, 0.72, 0.36, 0.8], wspace=0.75)
axD = fig.add_subplot(gsb[0])
MP.heatmap(axD, fig, R["T"], R["order"], dlabs, "norm. tonic", mark_row=row_ex)
axD.set_title("tonic, same no. of spikes", fontsize=10)
lab(axD, "D", -0.30)

axE = fig.add_subplot(gsb[1])
MP.heatmap(axE, fig, R["B"], R["order"], dlabs, "norm. burst", ylabel=False, mark_row=row_ex)
axE.set_title("burst spikes", fontsize=10)
lab(axE, "E", -0.18)

axF = fig.add_subplot(gsb[3])
MP.aligned_curves(axF, R["Ta"], R["Ba"], len(directions))
lab(axF, "F", -0.36)
tt, tb = MP.data_session_depths("obs")
nt, nb = MP.data_session_depths("null")
axF2 = fig.add_subplot(gsb[4])
ex_ddsi = MP.data_session_ddsi_excess()                 # dDSI, all units, 15 sessions
ex_depth = (tb - tt) - (nb - nt)                        # equal-noise tuning depth, all units
import glob as _glob, json as _json
_m1 = {}
for _f in sorted(_glob.glob(str(ROOT / "scn_area_total" / "C_*_M1.json"))):
    _m1.update({k: r for k, r in _json.loads(Path(_f).read_text()).items() if "glm_d" in r})
ex_ddsi_m1 = np.array([_m1[k]["obs_d"] - _m1[k]["glm_d"] for k in sorted(_m1)])          # dDSI, M1 units
_tb1, _tt1, _nb1, _nt1 = [], [], [], []
for _f in sorted((ROOT / "matchednull_M1").glob("*.json")):
    for _n, _r in sorted(_json.loads(Path(_f).read_text()).items()):
        _tb1.append(_r["obs"]["cv_b"]); _tt1.append(_r["obs"]["cv_t"]); _nb1.append(_r["null"]["cv_b"]); _nt1.append(_r["null"]["cv_t"])
ex_depth_m1 = (np.array(_tb1) - np.array(_tt1)) - (np.array(_nb1) - np.array(_nt1))       # depth, M1 units
M1C = "#1f4e8c"
_rng = np.random.default_rng(0)
for _i, (_v, _c) in enumerate(zip([ex_ddsi, ex_depth, ex_ddsi_m1, ex_depth_m1], [BLUE, BLUE, M1C, M1C])):
    axF2.scatter(_i + _rng.uniform(-0.15, 0.15, len(_v)), _v, s=7, color=_c, lw=0, zorder=3)
    axF2.plot([_i - 0.28, _i + 0.28], [_v.mean()] * 2, color="k", lw=1.0, zorder=4)
axF2.axhline(0, color="0.6", lw=0.6, ls="--", zorder=1)
axF2.set_xticks(range(4)); axF2.set_xticklabels(["$\\Delta$DSI", "depth", "$\\Delta$DSI", "depth"], rotation=90)
axF2.set_xlim(-0.6, 3.6)
axF2.set_ylabel("excess over null")
axF2.set_yticks([0, 0.05, 0.1]); axF2.set_ylim(-0.06, 0.105)
axF2.yaxis.tick_right(); axF2.yaxis.set_label_position("right"); axF2.spines[["left"]].set_visible(False); axF2.spines[["right"]].set_visible(True)
axF2.text(0.5, 1.03, "all", transform=axF2.get_xaxis_transform(), ha="center", va="bottom", color=BLUE)
axF2.text(2.5, 1.03, "M1", transform=axF2.get_xaxis_transform(), ha="center", va="bottom", color=M1C)
axF2.axvline(1.5, color="0.85", lw=0.6, zorder=0)
print(f"excess dDSI all: {ex_ddsi.mean():+.3f} ({np.sum(ex_ddsi > 0)}/{len(ex_ddsi)}) | depth all: {ex_depth.mean():+.3f} ({np.sum(ex_depth > 0)}/{len(ex_depth)}) | dDSI M1: {ex_ddsi_m1.mean():+.3f} ({np.sum(ex_ddsi_m1 > 0)}/{len(ex_ddsi_m1)}) | depth M1: {ex_depth_m1.mean():+.3f} ({np.sum(ex_depth_m1 > 0)}/{len(ex_depth_m1)})")

# ===== G: burst probability at matched expected rate, in each unit's own gate direction estimated on the other half of the trials (burst_prob_cv.py) =====
axG = fig.add_subplot(gsb[6])
import glob as _glob, json as _json
_area = _json.loads((ROOT / "dandi000688_unit_area.json").read_text())
_ED = np.logspace(np.log10(0.02), np.log10(5.0), 13); _MID = np.sqrt(_ED[:-1] * _ED[1:])
def _units(sub):
    out = []
    for f in sorted(_glob.glob(str(ROOT / "bpmcv" / f"{sub}_*.json"))):
        for name, us in _json.loads(Path(f).read_text()).items():
            out += [u for k, u in us.items() if _area[name][int(k)] == "M1"]
    return out
_null = []
for sub, col, labx in (("C", BLUE, "monkey C"), ("M", "#1b9e77", "monkey M")):
    us = _units(sub); _null += us
    N = sum(np.array(u["data"]["n"]) for u in us); K = sum(np.array(u["data"]["k"]) for u in us)
    ok = (N[0] >= 500) & (N[2] >= 500)
    axG.plot(_MID[ok], (K[0] / N[0])[ok] / (K[2] / N[2])[ok], "o-", color=col, ms=2.5, lw=1.0, label=labx.replace("monkey ", ""))
N = sum(np.array(u["null"]["n"]) for u in _null); K = sum(np.array(u["null"]["k"]) for u in _null)
ok = (N[0] >= 500) & (N[2] >= 500)
axG.plot(_MID[ok], (K[0] / N[0])[ok] / (K[2] / N[2])[ok], "o-", color="0.55", ms=2.0, lw=0.8, label="no gate")
axG.axhline(1, color="0.6", lw=0.6, ls=":")
axG.set_xscale("log"); axG.set_xlim(0.025, 1.0); axG.set_ylim(0.9, 1.75)
axG.set_xticks([0.03, 0.1, 0.3]); axG.set_xticklabels(["0.03", "0.1", "0.3"]); axG.minorticks_off()
axG.set_xlabel("expected spikes\nin next 10 ms"); axG.set_ylabel("burst prob., gate dir. / opp.")
axG.set_title("M1, held-out", fontsize=7)
axG.legend(frameon=False, loc="upper right", handlelength=1.0, borderaxespad=0.1, handletextpad=0.4)
axG.spines[["top", "right"]].set_visible(False)
lab(axG, "G", -0.42)
ng = len(_null)
print(f"G: {ng} M1 units (C + M)")
from scipy.stats import wilcoxon
adv_d, adv_n = tb - tt, nb - nt
print(f"advantage data {adv_d.mean():+.3f} vs 1-comp {adv_n.mean():+.3f}; data>null {np.sum(adv_d > adv_n)}/{len(adv_d)}, one-sided p={wilcoxon(adv_d - adv_n, alternative="greater").pvalue:.2g}")
print(f"sessions: CV depth burst {tb.mean():.3f} vs tonic {tt.mean():.3f}, burst>tonic {np.sum(tb > tt)}/{len(tb)}")

MP.restyle(fig, letters=set("ABCDEFG"))
fig.savefig(ROOT / "final_Fig1.pdf", dpi=300)
fig.savefig(ROOT / "final_Fig1.png", dpi=200)
print("saved final_Fig1")
