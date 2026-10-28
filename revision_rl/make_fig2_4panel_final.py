"""
make_fig2_4panel_final.py

Final-style version of the main-text Ant zero-shot + transfer-benchmark figure
(Fig. 4 in the text, \\label{fig2}; the manuscript currently includes
Fig2_2row_adaptation.pdf).

Derived from make_fig2_4panel.py.  The original script cropped rasters of the
Inkscape-assembled Fig2.pdf and of adaptation_compute.png; this version
re-draws every panel natively at print size (6.5 in wide, 7 pt text, 10 pt
panel letters):

  A  task illustration: the same raster illustration embedded in Fig2.pdf; its
     tiny embedded text (title, legend, 'Clockwise Bias') is removed and
     replaced by 7 pt Matplotlib text/legend.
  B,C,D  trajectories: the plotted (x, y) trajectories are recovered exactly
     from the vector source trajectories_comparison.pdf (Matplotlib output of
     code_2/SACant8Dir_NONLINEARG-1.ipynb; the trajectory arrays were kept in
     memory only and never saved).  Mean trajectories are recomputed from the
     recovered single trajectories and checked against the plotted means.
  E-H  transfer benchmark: re-plotted from
     results_adaptation_known_corrected/adaptation_results.csv with the same
     statistics as benchmark_sf_adaptation.py::plot_results (directions averaged
     within seed, mean +/- SEM across the five seeds).  F uses a log x axis so
     that all three methods are visible.

Outputs (project root): Fig4_final.pdf, Fig4_final.png
"""
from pathlib import Path
import re
import subprocess
import tempfile
import zlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


CODE2 = Path(__file__).resolve().parent
ROOT = CODE2
SOURCE_PDF = ROOT / "Fig2.pdf"                       # panel A illustration of the submitted figure (not used by make_fig4_final.py)
TRAJ_PDF = CODE2 / "data" / "trajectories_comparison.pdf"
RESULTS = CODE2 / "data" / "results_adaptation_known_corrected"
OUT_PDF = ROOT / "Fig4_final.pdf"
OUT_PNG = ROOT / "Fig4_final.png"

FIG_W, FIG_H = 6.5, 4.62

STYLE = {
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.title_fontsize": 7, "font.family": "sans-serif",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 2.0, "axes.titlepad": 3.0,
    "lines.linewidth": 1.0, "lines.markersize": 3,
    "legend.frameon": False, "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4, "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
    "legend.columnspacing": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
plt.rcParams.update(STYLE)

METHODS = ["bilinear-rw", "sf-gpi", "sf-actor"]
COLORS = {"bilinear-rw": "#1a6fbe", "sf-gpi": "#2ca02c", "sf-actor": "#cc2222"}
SHORT = {"bilinear-rw": "G-shared", "sf-gpi": "SF-GPI", "sf-actor": "SF-actor"}
LONG = {"bilinear-rw": "G-shared (gate set once from the goal, frozen)",
        "sf-gpi": "SF-GPI (8-policy bank)",
        "sf-actor": "SF-actor (MLP fine-tuned)"}
ANGLE_LABELS = {-135: "−3π/4", -90: "−π/2", -45: "−π/4", 0: "0",
                45: "π/4", 90: "π/2", 135: "3π/4", 180: "π"}


# ─────────────────────────────────────────────────────────────────────────────
# Minimal interpreter for single-page Matplotlib PDFs (recovers plotted data)
# ─────────────────────────────────────────────────────────────────────────────
_TOK = re.compile(rb"\((?:\\.|[^\\)])*\)|\[|\]|/[^\s/\[\]()<>]+|"
                  rb"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[A-Za-z'\"*]+")


def _pdf_objects(data):
    heads = list(re.finditer(rb"(?m)^(\d+) 0 obj", data))
    starts = [m.start() for m in heads] + [len(data)]
    objs = {}
    for i, m in enumerate(heads):
        body = data[m.end():starts[i + 1]]
        k, si = body.find(b"endobj"), body.find(b"stream")
        if si != -1 and (k == -1 or si < k):
            st = si + 6 + (2 if body[si + 6:si + 8] == b"\r\n" else 1)
            raw = body[st:body.rfind(b"endstream")]
            if b"FlateDecode" in body[:si]:
                raw = zlib.decompressobj().decompress(raw)
            objs[int(m.group(1))] = (body[:si], raw)
        else:
            objs[int(m.group(1))] = (body[:k], None)
    return objs


def _mul(a, b):
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5])


def parse_mpl_pdf(path):
    """Return the stroked/filled paths (page coordinates) of a Matplotlib PDF."""
    objs = _pdf_objects(Path(path).read_bytes())
    page = next(d for d, r in objs.values() if r is None and b"/Type /Page " in d)
    content = objs[int(re.search(rb"/Contents (\d+) 0 R", page).group(1))][1]
    alphas = {}
    for d, _ in objs.values():
        for m in re.finditer(rb"/(A\d+) << /Type /ExtGState /CA ([\d.]+) /ca ([\d.]+)", d):
            alphas[m.group(1).decode()] = (float(m.group(2)), float(m.group(3)))
    st = dict(ctm=(1, 0, 0, 1, 0, 0), RG=(0, 0, 0), rg=(0, 0, 0), w=1.0, CA=1.0, ca=1.0, dash=())
    stack, paths, cur, sub, args = [], [], [], [], []
    app = lambda m, x, y: (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
    for t in _TOK.findall(content):
        if t[:1].isdigit() or t[:1] in b"-+.":
            args.append(float(t)); continue
        if t[:1] == b"/":
            args.append(t[1:].decode()); continue
        if t[:1] in b"([]":
            args.append(t); continue
        op, a, args = t.decode(), args, []
        if op == "q": stack.append(dict(st))
        elif op == "Q": st = stack.pop()
        elif op == "cm": st["ctm"] = _mul(tuple(a[-6:]), st["ctm"])
        elif op == "w": st["w"] = a[-1]
        elif op == "RG": st["RG"] = tuple(a[-3:])
        elif op == "rg": st["rg"] = tuple(a[-3:])
        elif op == "G": st["RG"] = (a[-1],) * 3
        elif op == "g": st["rg"] = (a[-1],) * 3
        elif op == "gs": st["CA"], st["ca"] = alphas.get(a[-1], (1.0, 1.0))
        elif op == "d": st["dash"] = tuple(v for v in a if isinstance(v, float))[:-1]
        elif op == "m":
            if sub: cur.append(sub)
            sub = [app(st["ctm"], a[-2], a[-1])]
        elif op in ("l", "c"): sub.append(app(st["ctm"], a[-2], a[-1]))
        elif op == "h":
            if sub: sub.append(sub[0])
        elif op == "re":
            x, y, w, h = a[-4:]
            if sub: cur.append(sub)
            sub = [app(st["ctm"], *p) for p in [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]
        elif op in ("S", "s", "f", "f*", "B", "B*", "b", "n"):
            if sub: cur.append(sub)
            if op != "n" and cur:
                paths.append(dict(op=op, subs=[np.array(s) for s in cur], stroke=st["RG"],
                                  fill=st["rg"], w=st["w"], CA=st["CA"], ca=st["ca"], dash=st["dash"]))
            cur, sub = [], []
    return paths


def lin_map(page_coords, values):
    k, b = np.polyfit(np.asarray(page_coords, float), np.asarray(values, float), 1)
    return lambda p: k * np.asarray(p) + b


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_trajectories():
    """Single and mean trajectories of the 'Trained' and 'Zero-shot' panels."""
    paths = parse_mpl_pdf(TRAJ_PDF)
    # axis calibration from the tick marks (-400 ... 400) of each panel
    ticks_y = [63.22, 124.02, 184.83, 245.63, 306.44]
    vals = [-400, -200, 0, 200, 400]
    panels = {"trained": (73.54, 316.76, 52.0, 352.0), "zeroshot": (429.58, 672.8, 408.0, 708.0),
              "comparison": (785.62, 1028.84, 764.0, 1064.0)}
    fy = lin_map(ticks_y, vals)
    out = {}
    for key, (t0, t1, x0, x1) in panels.items():
        fx = lin_map(np.linspace(t0, t1, 5), vals)
        singles, means = {}, {}
        for p in paths:
            s = p["subs"][0]
            if p["op"] != "S" or len(p["subs"]) != 1 or len(s) != 100 or not (x0 <= s[0, 0] <= x1):
                continue
            col = tuple(np.round(p["stroke"], 4))
            xy = np.c_[fx(s[:, 0]), fy(s[:, 1])]
            if abs(p["w"] - 0.8) < 1e-6:
                singles.setdefault(col, []).append(xy)
            elif abs(p["w"] - 2.2) < 1e-6:
                means.setdefault((col, bool(p["dash"])), xy)
        out[key] = (singles, means)
    return out


def direction_of(xy):
    ang = np.degrees(np.arctan2(xy[-1, 1], xy[-1, 0]))
    return ang


def load_benchmark():
    data = np.genfromtxt(RESULTS / "adaptation_results.csv", delimiter=",", names=True,
                         dtype=None, encoding="utf8")
    res = {}
    for method in METHODS:
        d = data[data["method"] == method]
        episodes, seeds = np.unique(d["episode"]), np.unique(d["seed"])

        def seed_curve(key):
            return np.asarray([[d[key][(d["episode"] == ep) & (d["seed"] == s)].mean()
                                for ep in episodes] for s in seeds])
        sc = seed_curve("reward_per_step")
        r = dict(env_steps=seed_curve("env_steps").mean(0),
                 gmac=seed_curve("cumulative_macs").mean(0) / 1e9,
                 model_s=seed_curve("model_seconds").mean(0),
                 y=sc.mean(0), e=sc.std(0, ddof=1) / np.sqrt(len(seeds)))
        # G×Y: frozen gate, so episode-to-episode changes are evaluation noise; plotted per episode
        # (as for the baselines) so that the value at episode 5 matches the text (4.10).
        final = d[d["episode"] == d["episode"].max()]
        r["inference_ms"] = final["mean_action_ms"].mean()
        r["processing_ms"] = (1000 * final["model_seconds"] / final["env_steps"]).mean()
        res[method] = r
    return res


def load_illustration():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdfimages", "-png", str(SOURCE_PDF), f"{td}/im"], check=True)
        ims = [Image.open(p).convert("RGB") for p in sorted(Path(td).glob("im-*.png"))]
        im = next(i for i in ims if i.size == (1024, 1052))
        arr = np.asarray(im).copy()
    arr[343:375, 574:750] = 255          # remove embedded 'Clockwise Bias' label
    return arr[240:945, 35:990]          # drop embedded title + legend box


# ─────────────────────────────────────────────────────────────────────────────
def add_ax(fig, x, y, w, h, **kw):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def panel_letter(fig, x, y, letter):
    fig.text(x / FIG_W, y / FIG_H, letter, fontsize=10, fontweight="bold", ha="left", va="top")


def outward(ax, d=3):
    ax.spines["left"].set_position(("outward", d))
    ax.spines["bottom"].set_position(("outward", d))


def arrow_end(ax, xy, color, ls="-"):
    ax.annotate("", xy=xy[-1], xytext=xy[-4],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=6,
                                shrinkA=0, shrinkB=0), zorder=4)


def main():
    traj = load_trajectories()
    bench = load_benchmark()
    illus = load_illustration()

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---------- A: task illustration ----------
    top_y, sq = 2.79, 1.30
    axA = add_ax(fig, 0.0, 2.47, 1.80, 1.80 * illus.shape[0] / illus.shape[1])
    axA.imshow(illus, interpolation="lanczos")
    axA.axis("off")
    axA.text(548, 118, "clockwise\nbias", ha="left", va="center", linespacing=1.0)
    axA.legend(handles=[Line2D([], [], color="0.35", lw=0.9, ls=(0, (3, 1.5)), label="trained reference"),
                        Line2D([], [], color="0.35", lw=0.9, label="zero-shot (biased)")],
               loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handlelength=1.8)
    panel_letter(fig, 0.02, FIG_H - 0.02, "A")

    # ---------- B, C, D: trajectories ----------
    lim = (-470, 470)
    singles_tr, means_tr = traj["trained"]
    singles_zs, means_zs = traj["zeroshot"]
    # angle (deg) of each colour from its mean trained trajectory
    angle_of = {}
    for col, runs in singles_tr.items():
        m = np.mean(runs, axis=0)
        angle_of[col] = int(45 * np.round(direction_of(m) / 45)) if direction_of(m) > -157.5 else 180
        if angle_of[col] == -180:
            angle_of[col] = 180
    order = sorted(angle_of, key=lambda c: angle_of[c])
    # consistency check: recomputed means vs plotted means
    err = max(np.abs(np.mean(singles_tr[c], 0) - means_tr[(c, False)]).max() for c in order)
    print(f"trajectory recovery: {sum(len(v) for v in singles_tr.values())} trained + "
          f"{sum(len(v) for v in singles_zs.values())} zero-shot runs; "
          f"max |recomputed mean - plotted mean| = {err:.3f} position units")

    def traj_panel(ax, singles, title, show_ylabel):
        for col in order:
            for xy in singles[col]:
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.35, alpha=0.15, zorder=1)
        for col in order:
            m = np.mean(singles[col], axis=0)
            ax.plot(m[:, 0], m[:, 1], color=col, lw=1.2, zorder=3, solid_capstyle="butt")
            arrow_end(ax, m, col)
        ax.set_title(title)

    def style_square(ax, show_ylabel):
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xticks([-400, 0, 400]); ax.set_yticks([-400, 0, 400])
        ax.set_xlabel("x position")
        if show_ylabel:
            ax.set_ylabel("y position", labelpad=1)
        else:
            ax.set_yticklabels([])
        outward(ax)

    xB, xC, xD = 2.20, 3.62, 5.04
    axB = add_ax(fig, xB, top_y, sq, sq)
    traj_panel(axB, singles_tr, r"Trained ($\theta=k\pi/4$)", True)
    style_square(axB, True)
    panel_letter(fig, 1.84, FIG_H - 0.02, "B")

    axC = add_ax(fig, xC, top_y, sq, sq)
    traj_panel(axC, singles_zs, r"Zero-shot ($\theta+\pi/8$)", False)
    style_square(axC, False)
    panel_letter(fig, xC - 0.10, FIG_H - 0.02, "C")

    axD = add_ax(fig, xD, top_y, sq, sq)
    for col in order:
        m_tr = np.mean(singles_tr[col], axis=0)
        m_zs = np.mean(singles_zs[col], axis=0)
        # same convention as panel A: trained = dashed, zero-shot = solid
        axD.plot(m_tr[:, 0], m_tr[:, 1], color=col, lw=1.2, ls=(0, (3, 1.5)), zorder=3)
        arrow_end(axD, m_tr, col)
        axD.plot(m_zs[:, 0], m_zs[:, 1], color=col, lw=1.2, zorder=3)
        arrow_end(axD, m_zs, col)
    axD.set_title("Mean trajectories")
    axD.legend(handles=[Line2D([], [], color="0.35", lw=1.2, ls=(0, (3, 1.5)), label="trained"),
                        Line2D([], [], color="0.35", lw=1.2, label="zero-shot")],
               loc="lower left", bbox_to_anchor=(-0.02, -0.02), handlelength=1.8)
    style_square(axD, False)
    panel_letter(fig, xD - 0.10, FIG_H - 0.02, "D")

    fig.legend(handles=[Line2D([], [], color=c, lw=1.4, label=ANGLE_LABELS[angle_of[c]]) for c in order],
               loc="upper center", ncol=8, bbox_to_anchor=((xB + xD + sq) / 2 / FIG_W, (top_y - 0.36) / FIG_H),
               title=r"Trained direction $\theta$ (zero-shot runs use $\theta+\pi/8$, same colour)",
               handlelength=1.1, columnspacing=0.7, handletextpad=0.3)

    # ---------- E-H: known-direction transfer benchmark ----------
    by, bh, bw = 0.63, 1.24, 1.12
    xs = [0.40, 1.83, 3.26, 4.95]
    axE, axF, axG, axH = [add_ax(fig, x, by, bw, bh) for x in xs]
    axH.set_position([xs[3] / FIG_W, by / FIG_H, 1.50 / FIG_W, bh / FIG_H])
    for ax, letter, x, dx in zip((axE, axF, axG, axH), "EFGH", xs, (0.40, 0.30, 0.30, 0.42)):
        panel_letter(fig, x - dx, by + bh + 0.27, letter)

    for m in METHODS:
        r, c = bench[m], COLORS[m]
        ls = (0, (3, 1.5)) if m == "bilinear-rw" else "-"
        for ax, key in ((axE, "env_steps"), (axF, "gmac"), (axG, "model_s")):
            x = r[key]
            ax.fill_between(x, r["y"] - r["e"], r["y"] + r["e"], color=c, alpha=0.18, lw=0)
            ax.plot(x, r["y"], ls=ls, color=c, lw=1.1, marker="o", ms=2.5, mew=0,
                    label=LONG[m] if ax is axE else None)
    for ax in (axE, axF, axG):
        ax.set_ylim(1.0, 4.5); ax.set_yticks([1, 2, 3, 4])
    axE.set_ylabel("Reward / step")
    axE.set_xlabel("Environment steps"); axE.set_title("Sample efficiency")
    axE.set_xlim(500, 5200); axE.set_xticks([1000, 3000, 5000])
    fig.legend(handles=axE.get_legend_handles_labels()[0], loc="lower left", ncol=3,
               bbox_to_anchor=(0.30 / FIG_W, 0.0), handlelength=2.2, columnspacing=1.6)
    axF.set_xscale("log"); axF.set_xlim(0.05, 4000)
    axF.set_xticks([0.1, 10, 1000], ["0.1", "10", "1000"])
    axF.set_xlabel("Cumulative compute (GMAC)"); axF.set_title("Compute efficiency")
    axG.set_xlim(0, 13); axG.set_xticks([0, 4, 8, 12])
    axG.set_xlabel("Model time (s)"); axG.set_title("Test-time computation")

    x = np.arange(len(METHODS)); w = 0.36
    inf = [bench[m]["inference_ms"] for m in METHODS]
    proc = [bench[m]["processing_ms"] for m in METHODS]
    axH.bar(x - w / 2, inf, w, color=[COLORS[m] for m in METHODS], alpha=0.4, edgecolor="black", lw=0.4)
    axH.bar(x + w / 2, proc, w, color=[COLORS[m] for m in METHODS], edgecolor="black", lw=0.4)
    axH.set_xticks(x, [SHORT[m] for m in METHODS])
    axH.tick_params(axis="x", length=0)
    axH.set_ylim(0, 5.0); axH.set_yticks([0, 1, 2, 3, 4])
    axH.spines["left"].set_bounds(0, 4)
    axH.set_ylabel("Time per env. step (ms)")
    axH.set_title("Latency per step")
    axH.legend(handles=[Patch(facecolor="0.55", alpha=0.4, edgecolor="black", lw=0.4, label="inference only"),
                        Patch(facecolor="0.35", edgecolor="black", lw=0.4, label="inference + adaptation")],
               loc="upper left", bbox_to_anchor=(-0.03, 1.04), handlelength=1.0, handleheight=0.8)
    for ax in (axE, axF, axG, axH):
        outward(ax)

    fig.savefig(OUT_PDF, dpi=600)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PDF.name} + {OUT_PNG.name}")


if __name__ == "__main__":
    main()
