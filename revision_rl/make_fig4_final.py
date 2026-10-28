"""
make_fig4_final.py

Fig. 4 with (i) a redrawn panel A (top-down task schematic in the
colours of B-D) and (ii) a new bottom row with the curl force-field experiment
(Supplementary Text S5):

  I  schematic: force perpendicular to velocity bends the path; protocol
  J  heading error while the field is on
  K  aftereffect: heading in the first episode after the field is removed
  L  reward in the last five field episodes vs compute spent adapting

Panels B-H are drawn exactly as in make_fig2_4panel_final.py (same data, same
code), shifted up by the height of the new row.
Data for I-L: results_curl_field/full_gain1.0 and ref_gain0 (same files as
results_curl_field/summarize_curl_field.py).

Outputs (project root): Fig4_final.pdf, Fig4_final.png (--new-traj: B-D from trajectories_gshared.npz, Fig4_final_newtraj.*)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle, FancyArrowPatch, FancyBboxPatch

CODE2 = Path(__file__).resolve().parent
ROOT = CODE2
sys.path.insert(0, str(CODE2))
import make_fig2_4panel_final as F
import summarize_curl_field as SC

NEW_TRAJ = "--new-traj" in sys.argv   # B-D from the G-shared agent of E-L (trajectories_gshared.npz)
OUT_PDF = ROOT / ("Fig4_final_newtraj.pdf" if NEW_TRAJ else "Fig4_final.pdf")
OUT_PNG = ROOT / ("Fig4_final_newtraj.png" if NEW_TRAJ else "Fig4_final.png")

FIG_W, H_OLD, Y_OFF = 6.5, 4.62, 2.30
FIG_H = H_OLD + Y_OFF
plt.rcParams.update(F.STYLE)

ANT = "#d9a066"; ANT_EDGE = "#9c6b33"
CURL_METHODS = ["gy-frozen", "gy-es", "sf-gpi", "sf-actor"]
CURL_COLOR = {"gy-frozen": "0.55", "gy-es": F.COLORS["bilinear-rw"],
              "sf-gpi": F.COLORS["sf-gpi"], "sf-actor": F.COLORS["sf-actor"]}
CURL_LABEL = {"gy-frozen": "G-shared, gate frozen", "gy-es": "G-shared, gate search (8 values)",
              "sf-gpi": "SF-GPI", "sf-actor": "SF-actor (77k parameters)"}
CURL_SHORT = {"gy-frozen": "frozen", "gy-es": "search", "sf-gpi": "SF-GPI", "sf-actor": "SF-actor"}


def add_ax(fig, x, y, w, h, **kw):
    """Axes in inches; y measured in the old figure is shifted up by the new row."""
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def letter(fig, x, y, s):
    fig.text(x / FIG_W, y / FIG_H, s, fontsize=10, fontweight="bold", ha="left", va="top")


def draw_ant(ax, x0, y0, r, heading=0.0, z=5):
    """Top-down Ant: round torso, four two-segment legs."""
    for a in np.radians([45, 135, 225, 315]) + heading:
        hip = np.array([x0, y0]) + 0.8 * r * np.array([np.cos(a), np.sin(a)])
        knee = hip + 1.1 * r * np.array([np.cos(a), np.sin(a)])
        foot = knee + 1.0 * r * np.array([np.cos(a + 0.55), np.sin(a + 0.55)])
        for lw, col in ((3.4, ANT_EDGE), (2.6, ANT)):
            ax.plot(*np.c_[hip, knee, foot], color=col, lw=lw, solid_capstyle="round",
                    solid_joinstyle="round", zorder=z)
    ax.add_patch(Circle((x0, y0), r, facecolor=ANT, edgecolor=ANT_EDGE, lw=0.6, zorder=z + 1))


def mean_path(paths):
    """Mean x-y path; runs that end early (fall) are held at their last position."""
    n = max(len(q) for q in paths)
    P = np.stack([np.vstack([q, np.repeat(q[-1:], n - len(q), 0)]) for q in paths])
    return P.mean(0)


def load_curl():
    rd = ROOT / "results_curl_field"
    full = SC.load(rd / "full_gain1.0/curl_field_results.csv")
    ref = SC.load(rd / "ref_gain0/curl_field_results.csv")
    n_field = max(r["episode"] for r in full if r["field_on"])
    out = {"ref_heading": np.nanmean([r["heading_error_deg"] for r in ref if r["method"] == "gy-frozen"]),
           "ref_reward": np.nanmean([r["reward_per_step"] for r in ref if r["method"] == "gy-frozen"])}
    for m in CURL_METHODS:
        eph, muh, seh, _ = SC.curve(full, m, "heading_error_deg", True)
        _, _, _, M = SC.curve(full, m, "reward_per_step", True)
        _, muwh, sewh, _ = SC.curve(full, m, "heading_error_deg", False)
        macs = np.array([r["cumulative_macs"] for r in full if r["method"] == m and r["episode"] == n_field]) / 1e9
        out[m] = dict(eph=eph, muh=muh, seh=seh, after=muwh[0], after_se=sewh[0],
                      last5=np.nanmean(np.nanmean(M[:, -5:], 1)),
                      last5_se=np.nanstd(np.nanmean(M[:, -5:], 1)) / np.sqrt(M.shape[0]),
                      gmac=np.nanmean(macs))
    return out


def main():
    traj = F.load_trajectories()
    bench = F.load_benchmark()
    curl = load_curl()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # colours of the eight trained directions, as in B-D
    singles_tr, _ = traj["trained"]
    singles_zs, _ = traj["zeroshot"]
    angle_of = {}
    for col, runs in singles_tr.items():
        d = F.direction_of(np.mean(runs, axis=0))
        a = int(45 * np.round(d / 45)) if d > -157.5 else 180
        angle_of[col] = 180 if a == -180 else a
    order = sorted(angle_of, key=lambda c: angle_of[c])
    if NEW_TRAJ:
        T = np.load(CODE2 / "trajectories_gshared.npz", allow_pickle=True)
        col_of = {a % 360: c for c, a in angle_of.items()}
        singles_tr = {col_of[int(round(float(k.split("_")[1])))]: list(T[k]) for k in T.files if k.startswith("trained")}
        singles_zs = {col_of[int(round(float(k.split("_")[1]) - 22.5)) % 360]: list(T[k]) for k in T.files if k.startswith("new")}

    # ---------- A: task schematic (top-down) ----------
    top_y, sq = 2.79 + Y_OFF, 1.30
    axA = add_ax(fig, 0.12, top_y - 0.02, 1.42, 1.42)
    axA.set_xlim(-1.32, 1.32); axA.set_ylim(-1.32, 1.32); axA.set_aspect("equal"); axA.axis("off")
    for col in order:
        th = np.radians(angle_of[col])
        axA.plot([0, 0.98 * np.cos(th)], [0, 0.98 * np.sin(th)], color=col, lw=0.6, alpha=0.6, zorder=1)
        axA.plot(np.cos(th), np.sin(th), "o", ms=5.2, color=col, mec="none", zorder=3)
        tz = th + np.pi / 8
        axA.plot(np.cos(tz), np.sin(tz), "o", ms=4.6, mfc="white", mec=col, mew=0.9, zorder=3)
    draw_ant(axA, 0, 0, 0.13)
    axA.set_title("Walk toward a goal", pad=2)
    axA.legend(handles=[Line2D([], [], ls="none", marker="o", ms=5, color="0.35", mec="none", label="trained goal"),
                        Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white", mec="0.35", mew=0.9,
                               label=r"new goal ($\theta+\pi/8$)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=1, handletextpad=0.2)
    letter(fig, 0.02, FIG_H - 0.02, "A")

    # ---------- B, C, D: trajectories (unchanged) ----------
    lim = (-330, 330) if NEW_TRAJ else (-470, 470)
    ticks = [-300, 0, 300] if NEW_TRAJ else [-400, 0, 400]

    def traj_panel(ax, singles, title):
        for col in order:
            for xy in singles[col]:
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.35, alpha=0.15, zorder=1)
        for col in order:
            m = mean_path(singles[col])
            ax.plot(m[:, 0], m[:, 1], color=col, lw=1.2, zorder=3, solid_capstyle="butt")
            F.arrow_end(ax, m, col)
        ax.set_title(title)

    def style_square(ax, show_ylabel):
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xlabel("x position")
        if show_ylabel:
            ax.set_ylabel("y position", labelpad=1)
        else:
            ax.set_yticklabels([])
        F.outward(ax)

    xB, xC, xD = 2.20, 3.62, 5.04
    axB = add_ax(fig, xB, top_y, sq, sq)
    traj_panel(axB, singles_tr, r"Trained ($\theta=k\pi/4$)")
    style_square(axB, True)
    letter(fig, 1.84, FIG_H - 0.02, "B")
    axC = add_ax(fig, xC, top_y, sq, sq)
    traj_panel(axC, singles_zs, r"Zero-shot ($\theta+\pi/8$)")
    style_square(axC, False)
    letter(fig, xC - 0.10, FIG_H - 0.02, "C")
    axD = add_ax(fig, xD, top_y, sq, sq)
    for col in order:
        m_tr, m_zs = mean_path(singles_tr[col]), mean_path(singles_zs[col])
        axD.plot(m_tr[:, 0], m_tr[:, 1], color=col, lw=1.2, ls=(0, (3, 1.5)), zorder=3)
        F.arrow_end(axD, m_tr, col)
        axD.plot(m_zs[:, 0], m_zs[:, 1], color=col, lw=1.2, zorder=3)
        F.arrow_end(axD, m_zs, col)
    axD.set_title("Mean trajectories")
    axD.legend(handles=[Line2D([], [], color="0.35", lw=1.2, ls=(0, (3, 1.5)), label="trained"),
                        Line2D([], [], color="0.35", lw=1.2, label="zero-shot")],
               loc="lower left", bbox_to_anchor=(-0.02, -0.02), handlelength=1.8)
    style_square(axD, False)
    letter(fig, xD - 0.10, FIG_H - 0.02, "D")
    fig.legend(handles=[Line2D([], [], color=c, lw=1.4, label=F.ANGLE_LABELS[angle_of[c]]) for c in order],
               loc="upper center", ncol=8, bbox_to_anchor=((xB + xD + sq) / 2 / FIG_W, (top_y - 0.36) / FIG_H),
               title=r"Trained direction $\theta$ (zero-shot runs use $\theta+\pi/8$, same colour)",
               handlelength=1.1, columnspacing=0.7, handletextpad=0.3)

    # ---------- E-H: transfer benchmark (unchanged) ----------
    by, bh, bw = 0.63 + Y_OFF, 1.24, 1.12
    xs = [0.40, 1.83, 3.26, 4.95]
    axE, axF, axG, axH = [add_ax(fig, x, by, bw, bh) for x in xs]
    axH.set_position([xs[3] / FIG_W, by / FIG_H, 1.50 / FIG_W, bh / FIG_H])
    for x, s, dx in zip(xs, "EFGH", (0.40, 0.30, 0.30, 0.42)):
        letter(fig, x - dx, by + bh + 0.27, s)
    for m in F.METHODS:
        r, c = bench[m], F.COLORS[m]
        ls = (0, (3, 1.5)) if m == "bilinear-rw" else "-"
        for ax, key in ((axE, "env_steps"), (axF, "gmac"), (axG, "model_s")):
            ax.fill_between(r[key], r["y"] - r["e"], r["y"] + r["e"], color=c, alpha=0.18, lw=0)
            ax.plot(r[key], r["y"], ls=ls, color=c, lw=1.1, marker="o", ms=2.5, mew=0,
                    label=F.LONG[m] if ax is axE else None)
    for ax in (axE, axF, axG):
        ax.set_ylim(1.0, 4.5); ax.set_yticks([1, 2, 3, 4])
    axE.set_ylabel("Reward / step")
    axE.set_xlabel("Environment steps"); axE.set_title("Sample efficiency")
    axE.set_xlim(500, 5200); axE.set_xticks([1000, 3000, 5000])
    fig.legend(handles=axE.get_legend_handles_labels()[0], loc="lower left", ncol=3,
               bbox_to_anchor=(0.30 / FIG_W, Y_OFF / FIG_H), handlelength=2.2, columnspacing=1.6)
    axF.set_xscale("log"); axF.set_xlim(0.05, 4000)
    axF.set_xticks([0.1, 10, 1000], ["0.1", "10", "1000"])
    axF.set_xlabel("Cumulative compute (GMAC)"); axF.set_title("Compute efficiency")
    axG.set_xlim(0, 13); axG.set_xticks([0, 4, 8, 12])
    axG.set_xlabel("Model time (s)"); axG.set_title("Test-time computation")
    x = np.arange(len(F.METHODS)); w = 0.36
    axH.bar(x - w / 2, [bench[m]["inference_ms"] for m in F.METHODS], w,
            color=[F.COLORS[m] for m in F.METHODS], alpha=0.4, edgecolor="black", lw=0.4)
    axH.bar(x + w / 2, [bench[m]["processing_ms"] for m in F.METHODS], w,
            color=[F.COLORS[m] for m in F.METHODS], edgecolor="black", lw=0.4)
    axH.set_xticks(x, [F.SHORT[m] for m in F.METHODS])
    axH.tick_params(axis="x", length=0)
    axH.set_ylim(0, 5.0); axH.set_yticks([0, 1, 2, 3, 4]); axH.spines["left"].set_bounds(0, 4)
    axH.set_ylabel("Time per env. step (ms)"); axH.set_title("Latency per step")
    axH.legend(handles=[Patch(facecolor="0.55", alpha=0.4, edgecolor="black", lw=0.4, label="inference only"),
                        Patch(facecolor="0.35", edgecolor="black", lw=0.4, label="inference + adaptation")],
               loc="upper left", bbox_to_anchor=(-0.03, 1.04), handlelength=1.0, handleheight=0.8)
    for ax in (axE, axF, axG, axH):
        F.outward(ax)

    # ---------- I-L: curl force field ----------
    ry, rh = 0.66, 1.05
    # separator line between benchmark and force-field rows
    fig.add_artist(Line2D([0.02, 0.98], [(Y_OFF - 0.05) / FIG_H] * 2, color="0.8", lw=0.5))

    # I: schematic
    axI = add_ax(fig, 0.05, ry - 0.16, 1.55, rh + 0.44)
    axI.set_xlim(0, 1.55); axI.set_ylim(0, 1.48); axI.axis("off")
    axI.text(0.78, 1.30, "Curl force field", ha="center", va="bottom")
    axI.annotate("", xy=(1.45, 0.56), xytext=(0.10, 0.56),
                 arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.8, ls=(0, (3, 2)), mutation_scale=7))
    axI.text(1.45, 0.47, "goal", ha="right", va="top", color="0.35")
    t = np.linspace(0, 1, 60)
    px, py = 0.10 + 1.05 * t, 0.56 + 0.62 * t ** 1.8
    axI.plot(px, py, color="#cc2222", lw=1.3)
    i0 = 34
    p0 = np.array([px[i0], py[i0]]); v = np.array([px[i0 + 1] - px[i0 - 1], py[i0 + 1] - py[i0 - 1]])
    v = v / np.linalg.norm(v)
    draw_ant(axI, *p0, 0.045, heading=np.arctan2(v[1], v[0]), z=6)
    axI.annotate("", xy=p0 + 0.30 * v, xytext=p0, arrowprops=dict(arrowstyle="-|>", color="#1a6fbe", lw=1.0, mutation_scale=7), zorder=7)
    nrm = np.array([-v[1], v[0]])
    axI.annotate("", xy=p0 + 0.28 * nrm, xytext=p0, arrowprops=dict(arrowstyle="-|>", color="#cc2222", lw=1.0, mutation_scale=7), zorder=7)
    axI.text(*(p0 + 0.34 * v + np.array([0.02, -0.04])), r"$\mathbf{v}$", color="#1a6fbe", ha="left", va="center")
    axI.text(*(p0 + 0.31 * nrm + np.array([-0.02, 0.03])), r"$\mathbf{F}\perp\mathbf{v}$", color="#cc2222", ha="right", va="bottom")
    # protocol strip
    for x0, wdt, lab, fc in ((0.10, 0.95, "field on (40 ep.)", "#f4c7c3"), (1.08, 0.37, "off (5)", "#d9e7f5")):
        axI.add_patch(FancyBboxPatch((x0, 0.08), wdt, 0.18, boxstyle="round,pad=0,rounding_size=0.03",
                                     facecolor=fc, edgecolor="0.4", lw=0.5))
        axI.text(x0 + wdt / 2, 0.17, lab, ha="center", va="center")
    letter(fig, 0.02, ry + rh + 0.33, "I")

    # J: heading error
    xj, xk, xl, pw = 2.12, 3.78, 5.28, 1.10
    axJ = add_ax(fig, xj, ry, pw, rh)
    for m in CURL_METHODS:
        s = curl[m]
        axJ.fill_between(s["eph"], s["muh"] - s["seh"], s["muh"] + s["seh"], color=CURL_COLOR[m], alpha=0.2, lw=0)
        axJ.plot(s["eph"], s["muh"], color=CURL_COLOR[m], lw=1.0, label=CURL_LABEL[m])
    axJ.axhline(0, color="0.8", lw=0.6, zorder=0)
    axJ.set_xlabel("Episode (field on)"); axJ.set_ylabel("Heading error (deg)")
    axJ.set_title("Heading"); axJ.set_xlim(0, 41); axJ.set_ylim(-10, 35); axJ.set_yticks([0, 15, 30])
    letter(fig, xj - 0.42, ry + rh + 0.33, "J")

    # K: aftereffect
    axK = add_ax(fig, xk, ry, pw - 0.15, rh)
    xx = np.arange(len(CURL_METHODS))
    axK.bar(xx, [curl[m]["after"] for m in CURL_METHODS], 0.7, yerr=[curl[m]["after_se"] for m in CURL_METHODS],
            color=[CURL_COLOR[m] for m in CURL_METHODS], error_kw=dict(lw=0.6, capsize=0))
    axK.axhline(curl["ref_heading"], color="k", ls=":", lw=0.8)
    axK.text(len(CURL_METHODS) - 0.45, curl["ref_heading"] - 1.2, "no field", ha="right", va="top", fontsize=6.5)
    axK.axhline(0, color="0.8", lw=0.6, zorder=0)
    axK.set_xticks([]); axK.spines["bottom"].set_visible(False)
    axK.set_ylabel("Heading, field off (deg)"); axK.set_title("Aftereffect")
    letter(fig, xk - 0.42, ry + rh + 0.33, "K")

    # L: reward vs compute
    axL = add_ax(fig, xl, ry, pw, rh)
    for m in CURL_METHODS:
        s = curl[m]
        axL.errorbar(max(s["gmac"], 1e-3), s["last5"], yerr=s["last5_se"], fmt="o", ms=4, color=CURL_COLOR[m],
                     mec="none", elinewidth=0.6)
    axL.set_xscale("log"); axL.set_xlim(0.5, 1e5)
    axL.set_xticks([1, 100, 10000], ["1", "100", "10$^4$"])
    axL.set_ylim(1.0, 2.7); axL.set_yticks([1.0, 1.5, 2.0, 2.5])
    g, a = curl["gy-es"], curl["sf-actor"]
    axL.annotate("", xy=(a["gmac"] * 0.75, g["last5"]), xytext=(g["gmac"] * 1.5, g["last5"]),
                 arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.6, mutation_scale=5))
    axL.text(np.sqrt(a["gmac"] * g["gmac"]), g["last5"] + 0.06, f"{a['gmac'] / g['gmac']:,.0f}$\\times$",
             ha="center", va="bottom", color="0.3")
    axL.set_xlabel("Compute (GMAC)"); axL.set_ylabel("Reward / step")
    axL.set_title("Reward vs compute")
    letter(fig, xl - 0.42, ry + rh + 0.33, "L")
    for ax in (axJ, axK, axL):
        F.outward(ax)
    fig.legend(handles=axJ.get_legend_handles_labels()[0], loc="lower left", ncol=4,
               bbox_to_anchor=(0.30 / FIG_W, 0.0), handlelength=1.6, columnspacing=1.2)

    fig.savefig(OUT_PDF, dpi=600)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PDF.name}; ratio SF-actor/search GMAC = {a['gmac'] / g['gmac']:.0f}; "
          f"aftereffects: " + ", ".join(f"{m} {curl[m]['after']:+.1f}" for m in CURL_METHODS))


if __name__ == "__main__":
    main()
