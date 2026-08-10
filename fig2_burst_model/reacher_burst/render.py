"""
Optional video rendering: MuJoCo frame side-by-side with live burst traces.

Requires ``imageio`` (frame writing) and ``ffmpeg`` (encoding); both are only
imported/invoked when a video is actually requested.
"""
import gc
import os
import shutil
import subprocess
import tempfile

import numpy as np

from . import config as C
from .figures import _add_spike_excursions


def make_frame(gym_frame, roll_Vs, roll_Vd, roll_spk, roll_brst,
               k_act, mu_vec, p_burst_mat, roll_mu, mu_offset,
               num_g=C.NUM_G, out_dim=C.OUT_DIM, n_b=C.N_B):
    """Compose one video frame: env render | per-goal burst panel + action trace."""
    import matplotlib.pyplot as plt

    H, W = gym_frame.shape[:2]
    dpi = 100
    PW = int(W * 1.8)
    t_ms = np.arange(roll_Vs.shape[0]) * C.DT_B
    t_mu = np.arange(roll_mu.shape[0]) * C.DT_B
    n_show = min(4, n_b)
    lane_gap = C.V_THRESH_B + C.SPIKE_DISP_H + 0.6

    fig = plt.figure(figsize=(PW / dpi, H / dpi), dpi=dpi, facecolor='k')
    gs = fig.add_gridspec(num_g + 1, 1,
                          height_ratios=[0.72 / num_g] * num_g + [0.28],
                          hspace=0.07, left=0.04, right=0.97,
                          top=0.97, bottom=0.06)

    for ki in range(num_g):
        ax = fig.add_subplot(gs[ki])
        for ni in range(n_show):
            Vs_one = _add_spike_excursions(roll_Vs[:, ki, ni], roll_spk[:, ki, ni],
                                           C.V_THRESH_B, C.SPIKE_DISP_H)
            ax.plot(t_ms, Vs_one + ni * lane_gap, color=C.GOAL_COLORS[ki],
                    lw=0.8, alpha=0.9 - ni * 0.15)
        Vd_one = _add_spike_excursions(roll_Vd[:, ki, 0], roll_brst[:, ki, 0],
                                       C.V_BURST_B, C.SPIKE_DISP_H * 0.35)
        ax.plot(t_ms, Vd_one - 0.5, color='white', lw=0.5, alpha=0.3, ls='--')
        ax.set_facecolor('k')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(-0.7, n_show * lane_gap)
        lw_b = 2.5 if ki == k_act else 0.5     # highlight the active goal
        for sp in ax.spines.values():
            sp.set_edgecolor(C.GOAL_COLORS[ki])
            sp.set_linewidth(lw_b)
        pb = p_burst_mat[C.BF_DISP_JOINT, ki]
        ax.text(0.005, 0.95,
                f'g{ki}  p(V)={pb:.3f}  μ={C.P_BURST_SCALE * pb - mu_offset:.2f}',
                transform=ax.transAxes, color=C.GOAL_COLORS[ki],
                fontsize=7, va='top')

    ax_mu = fig.add_subplot(gs[num_g])
    ax_mu.set_facecolor('k')
    cmap = plt.get_cmap('tab10')
    for ji in range(out_dim):
        ax_mu.plot(t_mu, roll_mu[:, ji], color=cmap(ji / out_dim),
                   lw=0.75, alpha=0.85, label=f'j{ji}')
    ax_mu.axhline(0, color='white', lw=0.5, alpha=0.5)
    ax_mu.axhline(1, color='white', lw=0.3, alpha=0.2, ls=':')
    ax_mu.axhline(-1, color='white', lw=0.3, alpha=0.2, ls=':')
    for ji in range(out_dim):
        ax_mu.scatter([t_mu[-1]], [mu_vec[ji]], color=cmap(ji / out_dim), s=12, zorder=5)
    mu_max = max(np.abs(roll_mu).max(), 0.15)
    ax_mu.set_ylim(-mu_max * 1.15, mu_max * 1.15)
    ax_mu.set_xlim(t_mu[0], t_mu[-1])
    ax_mu.tick_params(colors='white', labelsize=5)
    ax_mu.set_ylabel('μ', color='white', fontsize=7)
    ax_mu.set_xlabel('ms', color='white', fontsize=6)
    ax_mu.set_title(f'action trajectory  (goal {k_act} active)',
                    color='white', fontsize=7, pad=2)
    for sp in ax_mu.spines.values():
        sp.set_edgecolor('dimgray')
        sp.set_linewidth(0.5)
    ax_mu.legend(ncol=out_dim, loc='upper left', fontsize=4.5,
                 facecolor='k', labelcolor='white', framealpha=0.4,
                 handlelength=1, columnspacing=0.5, borderpad=0.2)

    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    gc.collect()

    PH, PW2 = panel.shape[:2]
    if PH != H:
        panel = panel[:H] if PH > H else np.vstack(
            [panel, np.zeros((H - PH, PW2, 3), np.uint8)])
    return np.hstack([gym_frame, panel])


def ffmpeg_bin():
    return (shutil.which('ffmpeg')
            or ('/opt/homebrew/bin/ffmpeg'
                if os.path.exists('/opt/homebrew/bin/ffmpeg') else None))


def encode(frame_dir, path, fps=30):
    """Encode ``frame_%05d.png`` in ``frame_dir`` into an mp4 at ``path``."""
    ff = ffmpeg_bin()
    if ff is None:
        raise RuntimeError('ffmpeg not found; cannot encode video')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    return subprocess.run([ff, '-y', '-r', str(fps),
                           '-i', os.path.join(frame_dir, 'frame_%05d.png'),
                           '-vcodec', 'libx264', '-crf', '23',
                           '-pix_fmt', 'yuv420p', path],
                          capture_output=True)


def save_video(frames, path, fps=30):
    """Encode an in-memory list of RGB frames to mp4."""
    import imageio
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, f in enumerate(frames):
            imageio.imwrite(os.path.join(tmpdir, f'frame_{i:05d}.png'), f)
        encode(tmpdir, path, fps=fps)


class VideoRecorder:
    """Step callback that writes one composed frame per env step.

    Usage::

        rec = VideoRecorder(env, agent.mu_offset)
        trace = run_episode(env, agent, ep, step_cb=rec)
        rec.close('movies/ep.mp4')
    """

    def __init__(self, env, mu_offset, roll_len=800, num_g=C.NUM_G,
                 out_dim=C.OUT_DIM, n_b=C.N_B):
        import imageio                      # fail early if missing
        self._imageio = imageio
        self.env = env
        self.mu_offset = mu_offset
        self.num_g, self.out_dim, self.n_b = num_g, out_dim, n_b
        self.tmp = tempfile.TemporaryDirectory()
        self.idx = 0
        self.L = roll_len
        self.roll_Vs = np.zeros((roll_len, num_g, n_b))
        self.roll_Vd = np.zeros((roll_len, num_g, n_b))
        self.roll_spk = np.zeros((roll_len, num_g, n_b), dtype=bool)
        self.roll_brst = np.zeros((roll_len, num_g, n_b), dtype=bool)
        self.roll_mu = np.zeros((roll_len, out_dim))

    def __call__(self, k, k_act, mu, p_burst_pop, traces):
        Vs_tr, Vd_tr, spk_tr, brst_tr = traces
        n = min(C.N_BURST_STEPS, self.L)
        for buf, src in ((self.roll_Vs, Vs_tr), (self.roll_Vd, Vd_tr),
                         (self.roll_spk, spk_tr), (self.roll_brst, brst_tr)):
            buf[:] = np.roll(buf, -n, axis=0)
            buf[-n:] = src[:n]
        self.roll_mu[:] = np.roll(self.roll_mu, -n, axis=0)
        self.roll_mu[-n:] = mu[np.newaxis, :]

        frame = make_frame(self.env.render(), self.roll_Vs, self.roll_Vd,
                           self.roll_spk, self.roll_brst, k_act, mu,
                           p_burst_pop, self.roll_mu, self.mu_offset,
                           self.num_g, self.out_dim, self.n_b)
        self._imageio.imwrite(f'{self.tmp.name}/frame_{self.idx:05d}.png', frame)
        self.idx += 1

    def close(self, path, fps=30):
        if self.idx > 0:
            encode(self.tmp.name, path, fps=fps)
        self.tmp.cleanup()
        return path if self.idx > 0 else None
