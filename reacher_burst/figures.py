"""
The summary figure: learning curve, burst-neuron voltages, behaviour.

Panels (2×3), identical in content to the original ``master_plot``:
  (0,0) learning curve — avg reward per 100 episodes
  (0,1) somatic  V_s of the display joint, one lane per goal, spikes drawn on top
  (0,2) apical   V_d of the display joint, burst onsets drawn on top
  (1,0) fingertip trajectory, coloured by the active goal block
  (1,1) V-based p_burst per goal (left axis) and the resulting mu (right axis)
  (1,2) homeostatic basal bias δb, joints × goals
"""
import numpy as np

from . import config as C


def _add_spike_excursions(V_arr, spk_arr, threshold, disp_h):
    """Draw a cosmetic spike: clamp V to ``threshold + disp_h`` at spike times."""
    V_disp = V_arr.copy()
    for t in np.where(spk_arr > 0)[0]:
        V_disp[t] = threshold + disp_h
    return V_disp


def make_figure(trace, r_avg, delta_b, mu_offset, iteration,
                out_path='rewards_reacher_burst_vprob.png',
                rewards=None, num_g=C.NUM_G, dpi=120, fignum=1):
    """Render the summary figure for one episode; returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    nbs = C.N_BURST_STEPS
    n_bt = trace.ep_len * nbs
    t_bt = np.arange(n_bt) * C.DT_B
    n_show = min(C.BURST_TRACE_MS, n_bt)
    t_show = t_bt[-n_show:]
    Vs_win = trace.Vs[:n_bt][-n_show:]
    Vd_win = trace.Vd[:n_bt][-n_show:]
    spk_win = trace.spk[:n_bt][-n_show:]
    brst_win = trace.brst[:n_bt][-n_show:]

    if plt.fignum_exists(fignum):        # avoid matplotlib's "ignoring args" warning
        plt.close(fignum)
    fig = plt.figure(fignum, figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── learning curve ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    if r_avg is not None and len(r_avg):
        ax.plot(r_avg, color='k', lw=1.4)
    ax.set_xlabel('Episode (×100)')
    ax.set_ylabel('Avg reward')
    ax.set_title('Learning curve', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── somatic voltage, one lane per goal ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for ki in range(num_g):
        lane = ki * (C.V_THRESH_B + C.SPIKE_DISP_H + 0.5)
        Vs_one = _add_spike_excursions(Vs_win[:, ki, 0], spk_win[:, ki, 0],
                                       C.V_THRESH_B, C.SPIKE_DISP_H)
        ax.plot(t_show, Vs_one + lane, color=C.GOAL_COLORS[ki],
                lw=0.9, alpha=0.9, label=f'g{ki}')
        ax.axhline(lane + C.V_THRESH_B, color=C.GOAL_COLORS[ki],
                   lw=0.4, ls='--', alpha=0.35)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('$V_s$')
    ax.set_title(f'$V_s$ — joint {C.BF_DISP_JOINT}, last {C.BURST_TRACE_MS} ms',
                 fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── apical voltage + burst onsets ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for ki in range(num_g):
        lane = ki * (C.I_API_SCALE + 0.5)
        Vd_one = _add_spike_excursions(Vd_win[:, ki, 0], brst_win[:, ki, 0],
                                       C.V_BURST_B, C.SPIKE_DISP_H * 0.5)
        ax.plot(t_show, Vd_one + lane, color=C.GOAL_COLORS[ki],
                lw=0.9, alpha=0.9, label=f'g{ki}')
        ax.axhline(lane + C.V_BURST_B, color=C.GOAL_COLORS[ki],
                   lw=0.4, ls='--', alpha=0.35)
        ax.axhline(lane + C.I_API_SCALE, color=C.GOAL_COLORS[ki],
                   lw=0.4, ls=':', alpha=0.2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('$V_d$')
    ax.set_title(f'$V_d$ — joint {C.BF_DISP_JOINT}, last {C.BURST_TRACE_MS} ms\n'
                 f'dashed=burst thr, dotted=I_API eq.', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── fingertip trajectory ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    POS = np.asarray(trace.positions)
    for seg in range(num_g):
        sl = slice(seg * C.GOAL_BLOCK, min((seg + 1) * C.GOAL_BLOCK, len(POS)))
        gi = int(trace.permutation[seg])
        ax.plot(POS[sl, 0], POS[sl, 1], color=C.GOAL_COLORS[gi],
                lw=1.2, alpha=0.85, label=f'g{gi}')
        if sl.start < len(POS):
            ax.plot(POS[sl.start, 0], POS[sl.start, 1], 'o',
                    color=C.GOAL_COLORS[gi], ms=5)
        ax.plot(*C.TARGETS[gi], marker='x', color=C.GOAL_COLORS[gi],
                ms=8, mew=2, ls='none')
    ax.set_xlabel('fingertip X')
    ax.set_ylabel('fingertip Y')
    ax.set_title('Fingertip trajectory (× = target)', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── p_burst and mu ───────────────────────────────────────────────────────
    ax_fr = fig.add_subplot(gs[1, 1])
    bf_arr = np.asarray(trace.bf_log)
    for ki in range(num_g):
        ax_fr.plot(bf_arr[:, ki], color=C.GOAL_COLORS[ki],
                   label=f'g{ki}', lw=0.9, alpha=0.85)
    ax_fr.axhline(C.P_BURST_TARGET, color='k', lw=0.8, ls='--', alpha=0.6,
                  label=f'p_rest≈{C.P_BURST_TARGET:.2f}')
    ax_fr.set_xlabel('Env step')
    ax_fr.set_ylabel('p_burst (V)')
    ax_fr.set_title(f'V-based p_burst + mu — joint {C.BF_DISP_JOINT}',
                    fontweight='bold')
    ax_fr.set_ylim(-0.05, 0.5)
    ax_fr.legend(fontsize=7, framealpha=0.7, ncol=2, loc='upper left')
    ax_fr.spines['top'].set_visible(False)
    ax_mu = ax_fr.twinx()
    for ki in range(num_g):
        ax_mu.plot(C.P_BURST_SCALE * bf_arr[:, ki] - mu_offset,
                   color=C.GOAL_COLORS[ki], lw=1.2, alpha=0.5, ls='--')
    ax_mu.axhline(0, color='k', lw=0.6, ls=':', alpha=0.5)
    ax_mu.set_ylabel('mu (dashed)', color='gray')
    ax_mu.tick_params(axis='y', labelcolor='gray')
    ax_mu.set_ylim(-1.05, 1.05)

    # ── homeostatic basal bias ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(np.asarray(delta_b).T, aspect='auto', cmap='RdBu_r',
                   vmin=-C.DB_CLIP, vmax=C.DB_CLIP)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    ax.set_xlabel('Joint')
    ax.set_ylabel('Goal')
    ax.set_yticks(range(num_g))
    ax.set_title('Homeostatic δb', fontweight='bold')

    if rewards is not None and len(rewards):
        head = f'avg reward (last 100): {np.mean(rewards[-100:]):.1f}'
    else:
        head = f'episode reward: {trace.r_tot:.1f}'
    fig.suptitle(f'[Reacher] Episode {iteration}  |  {head}'
                 f'  [I_API={C.I_API_SCALE}  scale={C.P_BURST_SCALE}'
                 f'  win={C.BURST_TRACE_MS}ms]',
                 fontsize=11, fontweight='bold')

    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    return fig
