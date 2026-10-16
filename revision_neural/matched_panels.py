"""Shared panels for the final Fig. 1D-F (data) and Fig. 2G-I (model).

Burst spikes vs an equal number of tonic spikes (matched noise):
  - heatmaps: rows = units sorted by preferred direction (estimated on half of
    the trials), columns = absolute direction, values from the other half,
    normalised per unit (coolwarm, as the simulation figure);
  - summary: mean tuning aligned to the preferred direction (held-out trials),
    with an inset of the cross-validated tuning depth per session / population.
Style: all text 7 pt, panel letters 10 pt bold.
"""
from pathlib import Path
import json
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import audit_monkeyC_nulls as A
import matched_count_selectivity as MC

ROOT = Path(__file__).resolve().parent
BURST_C, TONIC_C = "#d62728", "0.35"
STYLE = {'font.size': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
         'axes.labelsize': 7, 'axes.titlesize': 7, 'axes.linewidth': 0.6,
         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
         'lines.linewidth': 1.0, 'pdf.fonttype': 42, 'font.family': 'sans-serif'}


def matrices(wins, di, n_dir, good, rng, n_draw=50, min_burst=20, extra_unit=None):
    """Per-unit tuning of burst spikes and of matched tonic spikes on held-out trials.
    Returns dict with raw (absolute-direction) and aligned matrices, preferred
    direction per unit, unit ids."""
    B, T, P, U = [], [], [], []
    for u in np.flatnonzero(good):
        b_tr, t_tr = MC.spike_sets(wins[u], A.ISI)
        if b_tr.sum() < min_burst:
            continue
        tm = np.mean([MC.matched_counts(b_tr, t_tr, rng) for _ in range(n_draw)], 0)
        perm = rng.permutation(len(di)); h1, h2 = perm[::2], perm[1::2]
        pref = MC.tuning(b_tr + t_tr, di, n_dir, h1).argmax()      # from all spikes, half 1
        B.append(MC.tuning(b_tr, di, n_dir, h2)); T.append(MC.tuning(tm, di, n_dir, h2)); P.append(pref); U.append(u)
    B, T, P = np.array(B), np.array(T), np.array(P)
    norm = lambda M: M / np.maximum(M.max(1, keepdims=True), 1e-9)
    align = lambda M: np.array([np.roll(m, -p + n_dir // 2) for m, p in zip(M, P)])
    order = np.lexsort((-(norm(B).max(1) - norm(B).mean(1)), P))
    return dict(T=norm(T), B=norm(B), Ta=norm(align(T)), Ba=norm(align(B)), P=P, units=np.array(U), order=order)


def heatmap(ax, fig, M, order, dir_labels, cb_label, ylabel=True, mark_row=None):
    im = ax.imshow(M[order], aspect="auto", cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, ticks=[0, 0.5, 1])
    cb.set_label(cb_label, fontsize=10); cb.ax.tick_params(labelsize=7, width=0.6, length=2)
    cb.outline.set_linewidth(0.6)
    n = M.shape[1]
    ax.set_xticks(range(n)); ax.set_xticklabels(dir_labels, rotation=90)
    ax.set_xlabel("reach direction (deg)")
    if ylabel:
        ax.set_ylabel("unit (sorted)")
    ax.set_yticks([0, len(order) - 1]); ax.set_yticklabels(["1", str(len(order))])
    if mark_row is not None:
        ax.axhline(mark_row, color="k", lw=0.8, ls="--")
    return im


def aligned_curves(ax, Ta, Ba, n_dir, step=45):
    xt = (np.arange(n_dir) - n_dir // 2) * step
    for M, col, lab in ((Ta, TONIC_C, "tonic (matched)"), (Ba, BURST_C, "burst")):
        m = M.mean(0); s = M.std(0) / np.sqrt(len(M))
        ax.fill_between(xt, m - s, m + s, color=col, alpha=0.2, lw=0)
        ax.plot(xt, m, "o-", color=col, ms=2.5, label=lab)
    ax.set_ylim(0, 1.05); ax.set_xlim(xt[0] - 10, xt[-1] + 10)
    ax.set_xticks(xt[::2]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("dir. $-$ pref. (deg)"); ax.set_ylabel("norm. count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)


def depth_inset(ax, tonic, burst, xlabels=("tonic", "burst"), title=None, groups=None):
    """Paired points (one line per session/population): cross-validated depth."""
    ins = ax.inset_axes([0.66, 0.56, 0.34, 0.44])
    if groups is None:
        groups = [(0, tonic, burst, "0.6")]
    for off, t, b, c in groups:
        for ti, bi in zip(t, b):
            ins.plot([off, off + 1], [ti, bi], "-", color=c, lw=0.5, alpha=0.7)
        ins.scatter(np.full(len(t), off), t, s=4, color=TONIC_C, zorder=3, lw=0)
        ins.scatter(np.full(len(b), off + 1), b, s=4, color=BURST_C, zorder=3, lw=0)
    ticks = sorted({g[0] for g in groups} | {g[0] + 1 for g in groups})
    ins.set_xticks(ticks); ins.set_xticklabels(list(xlabels), fontsize=7)
    ins.set_xlim(min(ticks) - 0.4, max(ticks) + 0.4)
    ins.set_ylim(0, 0.5); ins.set_yticks([0, 0.25, 0.5])
    ins.tick_params(labelsize=7, length=2, width=0.5)
    ins.set_ylabel("tuning depth (CV)", fontsize=7, labelpad=1)
    if title:
        ins.set_title(title, fontsize=7, pad=2)
    ins.spines[["top", "right"]].set_visible(False)
    for s in ins.spines.values():
        s.set_linewidth(0.5)
    return ins


def paired_depth(ax, tonic, burst, title=None, groups=None, xlabels=("tonic", "burst")):
    """Paired points (one line per session or model population): cross-validated tuning depth."""
    if groups is None:
        groups = [(0, tonic, burst)]
    for off, t, b in groups:
        for ti, bi in zip(t, b):
            ax.plot([off, off + 1], [ti, bi], "-", color="0.7", lw=0.5)
        ax.scatter(np.full(len(t), off), t, s=6, color=TONIC_C, zorder=3, lw=0)
        ax.scatter(np.full(len(b), off + 1), b, s=6, color=BURST_C, zorder=3, lw=0)
    ticks = [g[0] for g in groups] + [g[0] + 1 for g in groups]
    ax.set_xticks(sorted(ticks)); ax.set_xticklabels(list(xlabels), rotation=90)
    ax.set_xlim(min(ticks) - 0.5, max(ticks) + 0.5)
    ax.set_ylim(0, 0.6); ax.set_yticks([0, 0.25, 0.5])
    ax.set_ylabel("tuning depth")
    if title:
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)


def data_session_depths(which="obs"):
    """Cross-validated depth (matched tonic, burst) per monkey C session.
    which="obs": data; which="null": fitted single-compartment (GLM) neuron, same units."""
    tb, tt = [], []
    for f in sorted((ROOT / "matchednull").glob("*.json")):
        for name, r in sorted(json.loads(f.read_text()).items()):
            tb.append(r[which]["cv_b"]); tt.append(r[which]["cv_t"])
    return np.array(tt), np.array(tb)


def excess_values():
    """Excess of the burst advantage (burst - matched tonic depth) over the fitted
    single-compartment null: per model population (gate off / on) and per data session."""
    adv = lambda r: (r["obs"]["cv_b"] - r["obs"]["cv_t"]) - (r["null"]["cv_b"] - r["null"]["cv_t"])
    out = {"off": [], "on": [], "data": []}
    for f in sorted((ROOT / "matchednull_model").glob("*.json")):
        for name, r in json.loads(f.read_text()).items():
            out["off" if "api0.0" in name else "on"].append(adv(r))
    for f in sorted((ROOT / "matchednull").glob("*.json")):
        for name, r in json.loads(f.read_text()).items():
            out["data"].append(adv(r))
    return {k: np.array(v) for k, v in out.items()}


def excess_strip(ax, ex, rng=None):
    """Dots per population/session with mean bar: excess over the single-compartment null."""
    rng = rng or np.random.default_rng(0)
    cols = {"off": "0.55", "on": BURST_C, "data": "#2c6fbb"}
    labs = {"off": "gate off", "on": "gate on", "data": "data"}
    for i, k in enumerate(("off", "on", "data")):
        v = ex[k]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=7, color=cols[k], lw=0, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="k", lw=1.0, zorder=4)
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([labs[k] for k in ("off", "on", "data")], rotation=90)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("excess over 1-comp.")
    ax.set_yticks([-0.05, 0, 0.05, 0.1]); ax.set_yticklabels(["-0.05", "0", "0.05", "0.10"])
    ax.spines[["top", "right"]].set_visible(False)


def excess_scan_values():
    """Excess over the 1-comp. null per model population, grouped by apical drive; and per data session."""
    adv = lambda r: (r["obs"]["cv_b"] - r["obs"]["cv_t"]) - (r["null"]["cv_b"] - r["null"]["cv_t"])
    model = {}
    for f in sorted((ROOT / "matchednull_model").glob("*.json")):
        for name, r in json.loads(f.read_text()).items():
            api = float(re.search(r"api([\d.]+)_", name).group(1))
            model.setdefault(api, []).append(adv(r))
    data = [adv(r) for f in sorted((ROOT / "matchednull").glob("*.json")) for r in json.loads(f.read_text()).values()]
    data_m1 = [adv(r) for f in sorted((ROOT / "matchednull_M1").glob("*.json")) for r in json.loads(f.read_text()).values()]
    return {k: np.array(v) for k, v in sorted(model.items())}, np.array(data), np.array(data_m1)


def excess_scan(ax, model, data, data_m1=None):
    """Model excess against apical drive (dots: populations; line: mean) with the data as bands (mean +- SEM):
    all monkey C units (light) and, if given, M1 units only (dark)."""
    x = np.array(list(model.keys()))
    m = np.array([v.mean() for v in model.values()])
    dm, ds = data.mean(), data.std(ddof=1) / np.sqrt(len(data))
    ax.axhspan(dm - ds, dm + ds, color="#2c6fbb", alpha=0.15, lw=0)
    ax.axhline(dm, color="#2c6fbb", lw=0.7, alpha=0.6)
    ax.text(x[0] - 0.02, dm + ds + 0.002, "data (all)", color="#2c6fbb", ha="left", va="bottom", alpha=0.9)
    if data_m1 is not None and len(data_m1):
        m1, s1 = data_m1.mean(), data_m1.std(ddof=1) / np.sqrt(len(data_m1))
        ax.axhspan(m1 - s1, m1 + s1, color="#1f4e8c", alpha=0.3, lw=0)
        ax.axhline(m1, color="#1f4e8c", lw=0.9)
        ax.text(x[0] - 0.02, m1 + s1 + 0.002, "data (M1)", color="#1f4e8c", ha="left", va="bottom")
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=1)
    for xi, v in model.items():
        ax.scatter(np.full(len(v), xi), v, s=6, color="0.6", lw=0, zorder=3)
    ax.plot(x, m, "o-", color=BURST_C, ms=3, lw=1.0, zorder=4, label="model")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlim(x[0] - 0.06, x[-1] + 0.06)
    ax.set_xlabel("apical drive"); ax.set_ylabel("excess over 1-comp.")
    ax.set_yticks([-0.05, 0, 0.05, 0.1]); ax.set_yticklabels(["-0.05", "0", "0.05", "0.10"])
    ax.spines[["top", "right"]].set_visible(False)


def model_seed_depths():
    """Cross-validated depth per model population, from matched_model.log."""
    out = {}
    for line in (ROOT / "matched_model.log").read_text().splitlines():
        m = re.search(r"api=([\d.]+) seed=\d+.*CV depth burst ([\d.]+) vs tonic ([\d.]+)", line)
        if m:
            out.setdefault(float(m.group(1)), []).append((float(m.group(3)), float(m.group(2))))
    return {k: np.array(v) for k, v in out.items()}


def restyle(fig, letters=None, keep_large=None):
    """Force print sizes: panel letters 10 pt bold, all other text 7 pt (normal weight for titles)."""
    for ax in fig.axes:
        ax.tick_params(labelsize=7, width=0.6, length=2.5)
        ax.title.set_fontweight("normal")
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    for t in fig.findobj(matplotlib.text.Text):
        txt = t.get_text().strip()
        if letters and txt in letters and t.get_fontweight() in ("bold", 700) and t.get_fontsize() >= 12:
            t.set_fontsize(10); t.set_fontweight("bold")
        elif txt:
            t.set_fontsize(7)


def gated_profile(pattern="C_*.json", only_sig=True, p_thr=0.01):
    """Fitted gate g_d (gated_glm.py) aligned to each unit's rate-preferred direction,
    minus its mean over directions; mean and SEM over units (significant ones by default)."""
    import glob as _glob
    prof = []
    for f in sorted(_glob.glob(str(ROOT / "gated" / pattern))):
        for units in json.loads(Path(f).read_text()).values():
            for u in units.values():
                if only_sig and u["p"] >= p_thr:
                    continue
                g = np.array(u["g"]); pref = int(np.argmax(u["tonic"])); n = len(g)
                prof.append(np.roll(g - g.mean(), n // 2 - pref))
    P = np.array(prof)
    return (np.arange(P.shape[1]) - P.shape[1] // 2) * 45, P.mean(0), P.std(0) / np.sqrt(len(P)), len(P)


def gated2_profile(pattern="C_*.json", key="p7", p_thr=0.01, grp="data", only_sig=True):
    """As gated_profile, from gated_glm2.py results (gated2/): units whose burst term is
    direction-specific (likelihood ratio against a common burst term, 7 df)."""
    import glob as _glob
    units = {}
    for f in sorted(_glob.glob(str(ROOT / "gated2" / pattern))):
        for name, us in json.loads(Path(f).read_text()).items():
            for uid, u in us.items():
                units[(name, uid)] = u
    prof = []
    for u in units.values():
        if only_sig and u[grp][key] >= p_thr:
            continue
        g = np.array(u[grp]["g"]); pref = int(np.argmax(u["tonic"])); n = len(g)
        prof.append(np.roll(g - g.mean(), n // 2 - pref))
    P = np.array(prof)
    return (np.arange(P.shape[1]) - P.shape[1] // 2) * 45, P.mean(0), P.std(0) / np.sqrt(len(P)), len(P)


def data_session_ddsi_excess():
    """Per-session excess of dDSI over the fitted single-compartment null (10-ms threshold,
    burst-independent unit selection; glm_threshold_robustness.py results in thr_robust/)."""
    import glob as _glob
    out = {}
    for f in sorted(_glob.glob(str(ROOT / "thr_robust" / "C_*.json"))):
        for name, r in json.loads(Path(f).read_text()).items():
            out[name] = r["obs_10"] - r["glm_10"]
    return np.array([out[k] for k in sorted(out)])


def excess_columns(ax, cols, labels, color="#2c6fbb"):
    """Per-session excess over the single-compartment null, one column per measure:
    dots (sessions), bar (mean), zero line."""
    rng = np.random.default_rng(0)
    for i, v in enumerate(cols):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7, color=color, lw=0, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color="k", lw=1.0, zorder=4)
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylabel("excess over null")
    ax.spines[["top", "right"]].set_visible(False)
