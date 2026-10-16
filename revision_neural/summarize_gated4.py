"""gated_glm4.py results: (1) 7-df test against a null with direction-specific trial gain;
(2) split-half consistency of the gate direction (phase of the first Fourier component of g_d,
odd vs even trials) for units with a significant gate, data vs null."""
import glob, json
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parent
ang = 2 * np.pi * np.arange(8) / 8


def load(pattern):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "gated4" / pattern))):
        for name, units in json.loads(Path(f).read_text()).items():
            out.setdefault(name, {}).update(units)
    return {k: list(v.values()) for k, v in out.items() if len(v) >= 10}


def phase(g):
    g = np.array(g); return np.angle(np.sum((g - g.mean()) * np.exp(1j * ang)))


def dphi(u, grp):
    return abs(np.angle(np.exp(1j * (phase(u[grp]["g_odd"]) - phase(u[grp]["g_even"]))))) * 180 / np.pi


def corr(u, grp):
    a, b = np.array(u[grp]["g_odd"]), np.array(u[grp]["g_even"])
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def row(label, sess):
    allu = [u for v in sess for u in v]
    f = lambda g: 100 * np.mean([u[g]["p7"] < 0.01 for u in allu])
    fd = np.array([100 * np.mean([u["data"]["p7"] < 0.01 for u in v]) for v in sess]); fn = np.array([100 * np.mean([u["null"]["p7"] < 0.01 for u in v]) for v in sess])
    p = stats.wilcoxon(fd - fn, alternative="greater").pvalue if len(sess) >= 5 else float("nan")
    out = f"{label:13s} sess={len(sess):2d} units={len(allu):4d} | p7<0.01 data {f('data'):.1f}% null(dir gain) {f('null'):.1f}% (sess {np.sum(fd>fn)}/{len(fd)}, p={p:.1g})"
    for grp in ("data", "null"):
        sig = [u for u in allu if u[grp]["p7"] < 0.01]
        d = np.array([dphi(u, grp) for u in sig]); c = np.array([corr(u, grp) for u in sig])
        out += f" | {grp} sig n={len(sig)}: split-half |dphi| median {np.median(d) if len(d) else float('nan'):.0f} deg, <45deg {100*np.mean(d<45) if len(d) else float('nan'):.0f}%, profile corr {c.mean() if len(c) else float('nan'):+.2f}"
    da = np.array([corr(u, "data") for u in allu]); na = np.array([corr(u, "null") for u in allu])
    out += f" | all units profile corr data {da.mean():+.3f} null {na.mean():+.3f} (p={stats.wilcoxon(da-na).pvalue:.1g})"
    print(out)


C, I = load("C_*.json"), load("I_*.json")
row("monkey C", list(C.values()))
row("Indy Sep16+", [v for k, v in I.items() if k.split("_")[1] >= "20160900"])
