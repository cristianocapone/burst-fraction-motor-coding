"""Control for the cold-start adaptation result (fig. S2C-E, "~77% of zero-shot
by episode 10").

In the original runs (Fig4_8dir_G_estimation.ipynb, rw_persistent) the gate is
set to G = 0 and updated online with the reward-weighted rule, but the policy
is stochastic and its noise amplitude comes from log_std_layer(obs), whose input
includes the goal descriptor (cos th, sin th). The goal therefore reaches the
policy through the exploration noise. Here we repeat the experiment with the
same model, rule and hyperparameters, in three conditions:

  rw-orig      descriptor present in the noise head (as in the paper)
  rw-noctx     descriptor replaced by (0, 0) everywhere: the agent receives
               no information about the goal except the reward
  pretrained   reference: gate from the descriptor (zero-shot), stochastic policy

Same protocol as the notebook: 16 directions, 10 trials x 10 episodes of up to
1000 steps; G reset to 0 at the start of each trial and carried over episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_sf_adaptation import make_env, load_bilinear, set_seed

ETA, G_MAX, NUM_G = 0.0125, 15.0, 8
ANGLES = np.sort(np.concatenate([np.arange(8) * np.pi / 4, np.arange(8) * np.pi / 4 + np.pi / 8]))
ANGLES = (ANGLES + np.pi) % (2 * np.pi) - np.pi


def run(actor, critic, theta, mode, n_trials, n_ep, ep_len, seed):
    env = make_env(seed)
    ctx_goal = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    ctx_in = ctx_goal if mode != "rw-noctx" else np.zeros(2, dtype=np.float32)
    curves = np.zeros((n_trials, n_ep))
    for trial in range(n_trials):
        g = torch.zeros((1, NUM_G))
        for ep in range(n_ep):
            state, _ = env.reset(seed=seed * 100_000 + trial * 100 + ep)
            r_tot, steps = 0.0, 0
            for _ in range(ep_len):
                obs = torch.as_tensor(np.r_[state, ctx_in], dtype=torch.float32)[None]
                with torch.no_grad():
                    if mode == "pretrained":
                        a, _ = actor.sample(obs, None, deterministic=False)
                    else:
                        a, _ = actor.sample(obs, g, deterministic=False)
                    psi = critic.features(torch.as_tensor(state, dtype=torch.float32)[None], a)
                nxt, _, term, trunc, _ = env.step(a.squeeze(0).numpy())
                r = float(nxt[13] * np.cos(theta) + nxt[14] * np.sin(theta))
                if mode != "pretrained":
                    with torch.no_grad():
                        g = g + ETA * r * psi
                        n = g.norm()
                        if n > G_MAX:
                            g = g / n * G_MAX
                r_tot += r; steps += 1
                state = nxt
                if term or trunc:
                    break
            curves[trial, ep] = r_tot / max(steps, 1)
    env.close()
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../fig4_transfer/checkpoints/pretrain_final.pth")
    ap.add_argument("--modes", nargs="+", default=["rw-orig", "rw-noctx", "pretrained"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ep-len", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    actor, critic = load_bilinear(args.checkpoint, torch.device("cpu"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / "coldstart_noise_control.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for mode in args.modes:
        res.setdefault(mode, {})
        for i, th in enumerate(ANGLES):
            key = f"{np.degrees(th):.1f}"
            if key in res[mode]:
                continue
            set_seed(i)
            c = run(actor, critic, th, mode, args.trials, args.episodes, args.ep_len, seed=i)
            res[mode][key] = c.tolist()
            res_path.write_text(json.dumps(res))
            print(f"{mode:10s} theta={np.degrees(th):7.1f}  reward/step ep1 {c[:, 0].mean():.2f}  ep10 {c[:, -1].mean():.2f}", flush=True)
    # summary
    ref = np.mean([np.mean(v) for v in res.get("pretrained", {}).values()]) if res.get("pretrained") else np.nan
    for mode in args.modes:
        M = np.array(list(res[mode].values()))      # (angles, trials, episodes)
        ep = M.mean(axis=(0, 1))
        print(f"{mode:10s}: reward/step by episode " + " ".join(f"{x:.2f}" for x in ep) +
              (f" | episode 10 = {100 * ep[-1] / ref:.0f}% of pretrained ({ref:.2f})" if np.isfinite(ref) and mode != "pretrained" else ""))


if __name__ == "__main__":
    main()
