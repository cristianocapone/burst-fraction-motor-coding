"""fig. S12: burst gating by animal and cortical area (M1 vs PMd).
A: dDSI excess over the single-compartment (GLM) null, per session.
B: units with a direction-specific burst term (7-df test, p<0.01), data vs null.
C: stability of the gate across halves of the trials (correlation of g_d fitted on odd vs even trials), data minus null, per session.
D: held-out burst probability at matched expected rate, own gate direction / opposite (M1 vs PMd, data vs null).
Data: scn_area/, scn_glm_indy.json, gated2/, gated2_MTJ/, bpm/. Area labels: dandi000688_unit_area.json."""
import glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import matched_panels as MP
plt.rcParams.update(MP.STYLE)
ROOT = Path(__file__).resolve().parent
AREA = json.loads((ROOT / "dandi000688_unit_area.json").read_text())
BLUE, TEAL, PURPLE, ORANGE, GREY = "#2c6fbb", "#1b9e77", "#7b3294", "#e08214", "0.55"
ang = 2 * np.pi * np.arange(8) / 8

GROUPS = [  # label, animal prefix, area, colour
    ("C", "C", "M1", BLUE), ("Mihili", "M", "M1", TEAL), ("Jaco", "J", "M1", ORANGE), ("Indy", "I", "M1", PURPLE),
    ("C", "C", "PMd", BLUE), ("Mihili", "M", "PMd", TEAL), ("MrT", "T", "PMd", "0.35")]
XPOS = [0, 1, 2, 3, 4.6, 5.6, 6.6]


def load(pattern):
    out = {}
    for f in sorted(glob.glob(str(ROOT / pattern))):
        for name, units in json.loads(Path(f).read_text()).items():
            out.setdefault(name, {}).update(units)
    return out


def area_of(name, k):
    return "M1" if name.startswith("indy") else AREA[name][int(k)]


# ---------- A: dDSI excess per session
scn = {}
for f in glob.glob(str(ROOT / "scn_area_total/*.json")):
    scn.update({k: r for k, r in json.loads(Path(f).read_text()).items() if "glm_d" in r})
indy = json.loads((ROOT / "scn_glm_indy.json").read_text())
def excess(sub, area):
    if sub == "I":
        return [(r["obs_d"] - r["glm_d"], k.split("_")[1] < "20160900") for k, r in indy.items()]
    return [(r["obs_d"] - r["glm_d"], False) for k, r in scn.items() if k.startswith(sub + "_") and k.endswith("_" + area) and r["n_units"] >= 10]

# ---------- B, C: gate fit
G = {**load("gated2/C_*.json"), **load("gated2_MTJ/*.json"), **load("gated2/I_*.json")}
def gate_sessions(sub, area):
    pre = "indy" if sub == "I" else sub + "_"
    out = []
    for name, units in sorted(G.items()):
        if not name.startswith(pre) or (sub == "I" and name.split("_")[1] < "20160900"):
            continue
        v = [u for k, u in units.items() if area_of(name, k) == area]
        if len(v) >= 10:
            out.append(v)
    return out
def prof(u, grp):
    """g_pref - g_opp, preferred = direction of highest rate (as in fig. S12 and Fig. 1G)."""
    g = np.array(u[grp]["g"]); pref = int(np.argmax(u["tonic"]))
    return g[pref] - g[(pref + 4) % 8]

fig = plt.figure(figsize=(6.5, 5.0))
gs = fig.add_gridspec(2, 2, left=0.1, right=0.985, bottom=0.08, top=0.95, wspace=0.3, hspace=0.55, width_ratios=[1.15, 1])

def xaxis(ax):
    ax.set_xticks(XPOS); ax.set_xticklabels([g[0] for g in GROUPS], rotation=0)
    ax.set_xlim(-0.6, 7.2)
    for lab, x0, x1 in (("M1", 0, 3), ("PMd", 4.6, 6.6)):
        ax.annotate("", xy=(x0 - 0.3, -0.16), xytext=(x1 + 0.3, -0.16), xycoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
        ax.text((x0 + x1) / 2, -0.19, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

rng = np.random.default_rng(0)
ax = fig.add_subplot(gs[0, 0])
for x, (lab, sub, area, col) in zip(XPOS, GROUPS):
    e = excess(sub, area)
    if not e:
        continue
    vals = np.array([v for v, _ in e]); early = np.array([b for _, b in e])
    xs = x + rng.uniform(-0.18, 0.18, len(vals))
    ax.scatter(xs[~early], vals[~early], s=6, color=col, lw=0, zorder=3)
    if early.any():
        ax.scatter(xs[early], vals[early], s=6, facecolor="none", edgecolor=col, lw=0.5, zorder=3)
    main = vals[~early] if sub == "I" else vals
    ax.plot([x - 0.28, x + 0.28], [np.mean(main)] * 2, color="k", lw=1.0, zorder=4)
    p = stats.wilcoxon(main, alternative="greater").pvalue if len(main) >= 5 else np.nan
    ax.text(x, 1.0, f"{np.sum(main > 0)}/{len(main)}" + ("\n*" if p < 0.01 else ""), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5)
ax.axhline(0, color=GREY, lw=0.6)
ax.set_ylabel(r"$\Delta$DSI, data $-$ single-" "\ncompartment null")
ax.set_ylim(-0.08, 0.16)
xaxis(ax)
ax.text(-0.15, 1.05, "A", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

ax = fig.add_subplot(gs[0, 1])
for x, (lab, sub, area, col) in zip(XPOS, GROUPS):
    ss = gate_sessions(sub, area)
    if not ss:
        continue
    allu = [u for v in ss for u in v]
    d = 100 * np.mean([u["data"]["p7"] < 0.01 for u in allu]); n = 100 * np.mean([u["null"]["p7"] < 0.01 for u in allu])
    ax.bar(x - 0.17, d, 0.32, color=col); ax.bar(x + 0.17, n, 0.32, color=GREY)
    fd = np.array([np.mean([u["data"]["p7"] < 0.01 for u in v]) for v in ss]); fn = np.array([np.mean([u["null"]["p7"] < 0.01 for u in v]) for v in ss])
    p = stats.wilcoxon(fd - fn, alternative="greater").pvalue if len(ss) >= 5 and np.any(fd != fn) else np.nan
    ax.text(x - 0.17, d + 0.6, f"{d:.0f}%" + ("*" if p < 0.01 else ""), ha="center", va="bottom", fontsize=6.5)
ax.set_ylabel("units with a direction-specific\nburst term (%)"); ax.set_ylim(0, 36)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="0.25", label="data"), Patch(color=GREY, label="no gate (simulated)")], frameon=False, loc="upper right")
xaxis(ax)
ax.text(-0.15, 1.05, "B", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

ax = fig.add_subplot(gs[1, 0])
G4 = {**load("gated4/C_*.json"), **load("gated4/M_*.json"), **load("gated4/J_*.json"), **load("gated4/I_*.json")}
def hcorr(u, grp):
    a, b = np.array(u[grp]["g_odd"]), np.array(u[grp]["g_even"]); a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def g4_sessions(sub, area):
    pre = "indy" if sub == "I" else sub + "_"
    out = []
    for name, units in sorted(G4.items()):
        if not name.startswith(pre) or (sub == "I" and name.split("_")[1] < "20160900"):
            continue
        v = [u for k, u in units.items() if area_of(name, k) == area]
        if len(v) >= 10:
            out.append(v)
    return out
for x, (lab, sub, area, col) in zip(XPOS, GROUPS):
    ss = g4_sessions(sub, area)
    if not ss:
        continue
    dd = np.array([np.mean([hcorr(u, "data") - hcorr(u, "null") for u in v]) for v in ss])
    ax.scatter(x + rng.uniform(-0.18, 0.18, len(dd)), dd, s=6, color=col, lw=0, zorder=3)
    ax.plot([x - 0.28, x + 0.28], [dd.mean()] * 2, color="k", lw=1.0, zorder=4)
    p = stats.wilcoxon(dd, alternative="greater").pvalue if len(dd) >= 5 else np.nan
    ax.text(x, 1.0, f"{np.sum(dd > 0)}/{len(dd)}" + ("\n*" if p < 0.01 else ""), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5)
ax.axhline(0, color=GREY, lw=0.6)
ax.set_ylabel("gate stability across trial halves\n(correlation, data $-$ no gate)")
ax.set_ylim(-0.2, 0.42)
xaxis(ax)
ax.text(-0.15, 1.05, "C", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

# ---------- D: held-out matched-rate ratio, own gate direction (burst_prob_cv.py)
EDGES = np.logspace(np.log10(0.02), np.log10(5.0), 13); MID = np.sqrt(EDGES[:-1] * EDGES[1:])
B = {**load("bpmcv/C_*.json"), **load("bpmcv/M_*.json"), **load("bpmcv/J_*.json"), **load("bpmcv/T_*.json")}
ax = fig.add_subplot(gs[1, 1])
for area, col in (("M1", BLUE), ("PMd", "0.3")):
    us = [u for name, units in B.items() for k, u in units.items() if area_of(name, k) == area]
    Nd = sum(np.array(u["data"]["n"]) for u in us); Nn = sum(np.array(u["null"]["n"]) for u in us)
    ok = (Nd[0] >= 1000) & (Nd[2] >= 1000) & (Nn[0] >= 1000) & (Nn[2] >= 1000)
    for grp, lsg, lab in (("data", "-", f"{area}, data ({len(us)} units)"), ("null", "--", f"{area}, no gate")):
        N = sum(np.array(u[grp]["n"]) for u in us); K = sum(np.array(u[grp]["k"]) for u in us)
        P = K / np.maximum(N, 1)
        ax.plot(MID[ok], P[0][ok] / P[2][ok], lsg, color=col, lw=1.1, marker="o" if grp == "data" else None, ms=2.5, label=lab)
ax.axhline(1, color="0.8", lw=0.6); ax.set_xscale("log")
ax.set_xlabel(r"expected spikes in the next 10 ms")
ax.set_ylabel("held-out burst probability,\ngate direction / opposite")
ax.set_ylim(0.85, 1.5)
ax.legend(frameon=False, loc="upper right", ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.17, 1.05, "D", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

out = ROOT / "figS12_areas"
fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=220)
print("saved", out)
