"""Summary of gate_ms.py (gatems/): multiplicative (m) vs autonomous (s) apical gate.
Per group: median m and s (data, null), fraction of units with LR_M / LR_S > 6.63 (p<0.01, 1 df),
per-session sums of log-likelihood difference M - S, and per-session Wilcoxon tests."""
import glob, json
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parent
THR = 6.63


def load(pattern):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "gatems" / pattern))):
        for name, units in json.loads(Path(f).read_text()).items():
            out.setdefault(name, {}).update(units)
    return {k: list(v.values()) for k, v in out.items() if len(v) >= 10}


def row(label, sess):
    allu = [u for v in sess for u in v]
    ok = lambda u: u["data"]["n_burst"] >= 10 and u["null"]["n_burst"] >= 10
    allu = [u for u in allu if ok(u)]
    sess = [[u for u in v if ok(u)] for v in sess]
    f = lambda g, var: 100 * np.mean([u[g][var]["lr"] > THR for u in allu])
    med = lambda g, var, key: np.median([u[g][var][key] for u in allu])
    per = lambda fn: np.array([fn([u for u in v]) for v in sess if len(v) >= 5])
    fm = per(lambda v: 100 * np.mean([u["data"]["M"]["lr"] > THR for u in v]) - 100 * np.mean([u["null"]["M"]["lr"] > THR for u in v]))
    fs = per(lambda v: 100 * np.mean([u["data"]["S"]["lr"] > THR for u in v]) - 100 * np.mean([u["null"]["S"]["lr"] > THR for u in v]))
    dll = per(lambda v: np.sum([u["data"]["dll_M_minus_S"] for u in v]) - np.sum([u["null"]["dll_M_minus_S"] for u in v]))
    mm = per(lambda v: np.median([u["data"]["M"]["m"] for u in v]) - np.median([u["null"]["M"]["m"] for u in v]))
    p = lambda x: stats.wilcoxon(x, alternative="greater").pvalue if len(x) >= 5 else float("nan")
    print(f"{label:16s} sess={len(sess):2d} units={len(allu):4d} | LR_M>thr data {f('data','M'):4.1f}% null {f('null','M'):4.1f}% (sess {np.sum(fm>0)}/{len(fm)}, p={p(fm):.1g}) "
          f"| LR_S>thr data {f('data','S'):4.1f}% null {f('null','S'):4.1f}% (sess {np.sum(fs>0)}/{len(fs)}, p={p(fs):.1g}) "
          f"| median m data {med('data','M','m'):+.3f} null {med('null','M','m'):+.3f} (sess p={p(mm):.1g}) | median s data {med('data','S','s'):.4f} null {med('null','S','s'):.4f} "
          f"| sum LL(M)-LL(S), data-null: mean {dll.mean():+.1f}, sess {np.sum(dll>0)}/{len(dll)} (p={p(dll):.1g}) | c median {med('data','base','c'):.2f}")


C, I = load("C_*.json"), load("I_*.json")
row("monkey C", list(C.values()))
row("Indy Sep16+", [v for k, v in I.items() if k.split("_")[1] >= "20160900"])
row("Indy Apr-Jun16", [v for k, v in I.items() if k.split("_")[1] < "20160900"])
for nm, pat in [("model additive", "m110_api0.json"), ("model gating unif", "m110_api0.5.json"), ("model gating spread", "m110_api0.5_cv1.json")]:
    d = load(pat)
    if d:
        row(nm, list(d.values()))
