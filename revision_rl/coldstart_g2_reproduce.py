"""Clean reproduction of the cold-start result of fig. S2C-E ("~77% of zero-shot
by episode 10"), whose original generating code is not available.

Model: models_8dir_g2_nobias/pretrain_final.pth (K = 2 gates, bias-free gate
layer, gate shared by actor and critics; architecture copied from
eval_single_episode_8dir_g2_nobias.ipynb).

Protocol (from the cached all_methods_results.npy: ETA = 0.002, G_MAX = 5,
10 trials x 10 episodes, 16 directions):
  G = 0 at the start of each trial, kept across the trial's episodes;
  after every step  G <- G + ETA * r_t * psi(s_t, a_t),  psi_k = critic feature k;
  |G| clipped to G_MAX.  Stochastic policy, as in training.
Reference: the zero-shot gate G = W_g c from the descriptor.
Optional --no-ctx: replace the descriptor by (0,0) in the noise head too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K, LOG_STD_MIN, LOG_STD_MAX = 2, -2, 2
ANGLES = np.sort(np.concatenate([np.arange(8) * np.pi / 4, np.arange(8) * np.pi / 4 + np.pi / 8]))
ANGLES = (ANGLES + np.pi) % (2 * np.pi) - np.pi


class Actor(nn.Module):
    def __init__(self, obs_dim=29, act_dim=8, hidden=256, k=K):
        super().__init__()
        self.l1 = nn.Linear(obs_dim - 2, hidden); self.l2 = nn.Linear(hidden, hidden)
        self.g_layer = nn.Linear(2, k, bias=False)
        self.y_layers = nn.ModuleList([nn.Linear(hidden, act_dim) for _ in range(k)])
        self.log_std_layer = nn.Linear(obs_dim, act_dim)

    def sample(self, obs, g=None):
        x = F.relu(self.l2(F.relu(self.l1(obs[:, :-2]))))
        if g is None:
            g = self.g_layer(obs[:, -2:])
        mean = sum(g[:, i:i + 1] * Y(x) for i, Y in enumerate(self.y_layers))
        std = torch.clamp(self.log_std_layer(obs), LOG_STD_MIN, LOG_STD_MAX).exp()
        return torch.tanh(torch.distributions.Normal(mean, std).sample())


class Critic(nn.Module):
    def __init__(self, obs_dim=29, act_dim=8, k=K):
        super().__init__()
        self.l1_ag = nn.Linear(act_dim + obs_dim - 2, 256); self.l2_ag = nn.Linear(256, 256)
        self.g_layer = nn.Linear(2, k, bias=False)
        self.y_layers = nn.ModuleList([nn.Linear(256, 1) for _ in range(k)])

    def features(self, state, act):
        x = F.relu(self.l2_ag(F.relu(self.l1_ag(torch.cat([act, state], -1)))))
        return torch.cat([Y(x) for Y in self.y_layers], -1)          # (1, K)


def run(actor, critic, theta, mode, n_trials, n_ep, ep_len, eta, g_max, seed, no_ctx, per_episode=False):
    env = gym.make("Ant-v4")
    ctx_goal = np.array([np.cos(theta), np.sin(theta)], np.float32)
    ctx_in = np.zeros(2, np.float32) if no_ctx else ctx_goal
    curves, gtr = np.zeros((n_trials, n_ep)), np.zeros((n_trials, n_ep + 1, K))
    for trial in range(n_trials):
        g = torch.zeros((1, K))
        for ep in range(n_ep):
            gtr[trial, ep] = g.numpy()[0]
            state, _ = env.reset(seed=seed * 100_000 + trial * 100 + ep)
            r_tot, steps = 0.0, 0
            dg = torch.zeros((1, K))
            for _ in range(ep_len):
                obs = torch.as_tensor(np.r_[state, ctx_in if mode == "rw" else ctx_goal], dtype=torch.float32)[None]
                with torch.no_grad():
                    a = actor.sample(obs, g if mode == "rw" else None)
                nxt, _, term, trunc, _ = env.step(a.squeeze(0).numpy())
                r = float(nxt[13] * np.cos(theta) + nxt[14] * np.sin(theta))
                if mode == "rw":
                    with torch.no_grad():
                        psi = critic.features(torch.as_tensor(state, dtype=torch.float32)[None], a)
                        if per_episode:          # accumulate; G stays fixed during the episode
                            dg = dg + eta * r * psi
                        else:                    # update and bound after every step
                            g = g + eta * r * psi
                            n = g.norm()
                            if n > g_max:
                                g = g / n * g_max
                r_tot += r; steps += 1; state = nxt
                if term or trunc:
                    break
            curves[trial, ep] = r_tot / max(steps, 1)
            if mode == "rw" and per_episode:     # apply the summed update, then project onto |G| <= g_max
                g = g + dg
                n = g.norm()
                if n > g_max:
                    g = g / n * g_max
        gtr[trial, n_ep] = g.numpy()[0]
    env.close()
    return curves, gtr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ant_2gate/pretrain_final.pth")
    ap.add_argument("--mode", choices=["rw", "pretrained"], default="rw")
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--eta", type=float, default=0.002)
    ap.add_argument("--g-max", type=float, default=5.0)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ep-len", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-episode", action="store_true", help="apply the summed update once per episode (G fixed within an episode)")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor, critic = Actor(), Critic()
    actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic1"])
    actor.eval(); critic.eval()
    torch.manual_seed(0)
    res = {}
    for i, th in enumerate(ANGLES):
        np.random.seed(i); torch.manual_seed(i)
        c, gt = run(actor, critic, th, args.mode, args.trials, args.episodes, args.ep_len, args.eta, args.g_max, i, args.no_ctx, args.per_episode)
        res[f"{np.degrees(th):.1f}"] = dict(curves=c.tolist(), gtrace=gt.tolist())
        Path(args.out).write_text(json.dumps(dict(args=vars(args), results=res)))
        print(f"{args.mode}{'-noctx' if args.no_ctx else ''} theta={np.degrees(th):7.1f}  ep1 {c[:, 0].mean():.2f}  ep5 {c[:, 4].mean():.2f}  ep10 {c[:, -1].mean():.2f}  |G| end {np.linalg.norm(gt[:, -1], axis=1).mean():.2f}", flush=True)
    M = np.array([v["curves"] for v in res.values()])
    print("by episode:", " ".join(f"{x:.2f}" for x in M.mean(axis=(0, 1))))


if __name__ == "__main__":
    main()
