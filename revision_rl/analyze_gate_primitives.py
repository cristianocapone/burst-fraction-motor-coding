"""Reviewer 1, comment 6: how do primitives and gates form during learning, and
how does the agent interpolate behaviour by changing the gate G?

Uses the training checkpoints of the G x Y Ant agent (models_8dir_g_est/ckpt_*.pth,
same architecture as the benchmark of Fig. 2E-H):
  action mean = tanh( sum_k G_k(c) * Y_k(s) ),  G(c) = W_g c + b,  c = (cos th, sin th)

Analyses
  1. Gate formation: G(th) for th on the circle at each checkpoint; norm, tuning
     of each component, similarity between gates of different directions, and
     alignment between the actor's gate and the critic's gate (separate layers
     in this model).
  2. Primitive formation: the movement produced by each primitive alone
     (G = +/- g0 * e_k): direction and speed of the walker, per checkpoint.
  3. Interpolation (final checkpoint): G(a) = (1-a) G(th1) + a G(th2) for adjacent
     training directions; executed heading as a function of a.
  4. Steering by single components (final checkpoint): heading change when one
     gate value is increased.

Outputs (in --out): gate_primitives_results.json and figures, plus a markdown summary.
Run: env -u PYTHONPATH .venv_benchmark/bin/python analyze_gate_primitives.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_sf_adaptation import make_env, BilinearActor, BilinearCritic, reward_weights, set_seed

CKPTS = [499, 999, 1999, 3999, 7999, 14999]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a, c = BilinearActor(), BilinearCritic()
    a.load_state_dict(ck["actor"]); c.load_state_dict(ck["critic1"])
    a.eval(); c.eval()
    return a, c, ck.get("rewards", [])


def gate(layer, th):
    ctx = torch.as_tensor(np.stack([np.cos(th), np.sin(th)], -1), dtype=torch.float32)
    with torch.no_grad():
        return layer(ctx).numpy()


def rollout(actor, g, ctx_theta, seed, steps):
    """Run the policy with a fixed gate; return mean planar velocity and steps survived."""
    set_seed(seed)
    env = make_env(seed)
    state, _ = env.reset(seed=seed)
    ctx = reward_weights(ctx_theta)
    gt = torch.as_tensor(g, dtype=torch.float32)[None]
    vs = []
    for _ in range(steps):
        obs = torch.as_tensor(np.r_[state, ctx], dtype=torch.float32)[None]
        with torch.no_grad():
            a, _ = actor.sample(obs, gt, deterministic=True)
        state, _, term, trunc, _ = env.step(a.squeeze(0).numpy())
        vs.append(state[13:15])
        if term or trunc:
            break
    env.close()
    vs = np.asarray(vs)
    return vs[len(vs) // 5:].mean(0) if len(vs) > 10 else np.zeros(2), len(vs)


def heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def circ_diff(a, b):
    return float((a - b + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/ant_8gate_training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thetas = np.radians(np.arange(0, 360, 5))
    train = np.radians(np.arange(0, 360, 45))
    res = {"ckpts": CKPTS, "theta_deg": np.degrees(thetas).tolist()}

    # ---------- 1 + 2: gates and primitives across training ----------
    per = []
    for ep in CKPTS:
        actor, critic, rewards = load(Path(args.ckpt_dir) / f"ckpt_{ep}.pth")
        Ga = gate(actor.g_layer, thetas)                # (72, 8)
        Gc = gate(critic.g_layer, thetas)
        Gt = gate(actor.g_layer, train)                 # (8 dirs, 8)
        norm = np.linalg.norm(Ga, axis=1).mean()
        # similarity between gates of the 8 training directions
        Gn = Gt / np.linalg.norm(Gt, axis=1, keepdims=True)
        sim = Gn @ Gn.T
        # actor-critic gate alignment across directions (correlation of centred gate trajectories)
        A = Ga - Ga.mean(0); C = Gc - Gc.mean(0)
        align = float(np.sum(A * C) / (np.linalg.norm(A) * np.linalg.norm(C) + 1e-9))
        # preferred direction and depth of each gate component
        pref = [heading(np.array([np.sum(Ga[:, k] * np.cos(thetas)), np.sum(Ga[:, k] * np.sin(thetas))])) for k in range(8)]
        depth = (Ga.max(0) - Ga.min(0)).tolist()
        # primitives: movement produced by +/- g0 e_k (g0 = typical gate norm)
        g0 = float(norm)
        prims = []
        for k in range(8):
            for sign in (+1, -1):
                g = np.zeros(8); g[k] = sign * g0
                vel = [rollout(actor, g, 0.0, s, args.steps) for s in range(args.seeds)]
                v = np.mean([x[0] for x in vel], 0); surv = np.mean([x[1] for x in vel])
                prims.append(dict(k=k, sign=sign, vx=float(v[0]), vy=float(v[1]), speed=float(np.linalg.norm(v)),
                                  heading=heading(v), steps=float(surv)))
        # executed heading/speed for the 8 trained directions with their own gate
        execd = []
        for th, g in zip(train, Gt):
            vel = [rollout(actor, g, th, s, args.steps)[0] for s in range(args.seeds)]
            v = np.mean(vel, 0)
            execd.append(dict(target=float(np.degrees(th)), heading=heading(v), speed=float(np.linalg.norm(v)),
                              along=float(v @ np.array([np.cos(th), np.sin(th)]))))
        r = np.asarray(rewards, float)
        per.append(dict(ckpt=ep, gate_norm=g0, gate_sim=sim.tolist(), actor_critic_alignment=align, gate_pref=pref,
                        gate_depth=depth, gate_curves=Ga.tolist(), primitives=prims, executed=execd,
                        reward_recent=float(np.mean(r[-200:])) if len(r) else float("nan")))
        mean_along = np.mean([e["along"] for e in execd])
        err = np.mean([abs(circ_diff(e["heading"], e["target"])) for e in execd])
        moving = sum(p["speed"] > 0.5 for p in prims)
        print(f"ckpt {ep:6d}: |G|={g0:.2f} actor-critic gate alignment={align:+.2f} | trained dirs: velocity toward target {mean_along:.2f}, "
              f"|heading error| {err:.0f} deg | primitives moving (>0.5 m/s): {moving}/16", flush=True)
    res["per_ckpt"] = per
    res["learning_curve"] = [float(x) for x in load(Path(args.ckpt_dir) / f"ckpt_{CKPTS[-1]}.pth")[2]]

    # ---------- 3: interpolation at the final checkpoint ----------
    actor, critic, _ = load(Path(args.ckpt_dir) / f"ckpt_{CKPTS[-1]}.pth")
    alphas = np.linspace(0, 1, 9)
    interp = []
    for i in range(8):
        th1, th2 = train[i], train[(i + 2) % 8]           # 90 deg apart
        g1, g2 = gate(actor.g_layer, np.array([th1]))[0], gate(actor.g_layer, np.array([th2]))[0]
        for a in alphas:
            g = (1 - a) * g1 + a * g2
            ctx_th = np.arctan2((1 - a) * np.sin(th1) + a * np.sin(th2), (1 - a) * np.cos(th1) + a * np.cos(th2))
            vel = [rollout(actor, g, ctx_th, s, args.steps)[0] for s in range(args.seeds)]
            v = np.mean(vel, 0)
            interp.append(dict(start=float(np.degrees(th1)), alpha=float(a), heading_rel=circ_diff(heading(v), np.degrees(th1)),
                               speed=float(np.linalg.norm(v))))
        print(f"interp {np.degrees(th1):5.0f}->{np.degrees(th2):5.0f}: rel. heading " +
              " ".join(f"{x['heading_rel']:+.0f}" for x in interp[-len(alphas):]), flush=True)
    res["interpolation"] = interp

    # ---------- 4: steering by single gate components ----------
    steer = []
    for i, th in enumerate(train):
        g = gate(actor.g_layer, np.array([th]))[0]
        base = np.mean([rollout(actor, g, th, s, args.steps)[0] for s in range(args.seeds)], 0)
        for k in range(8):
            gp = g.copy(); gp[k] += 0.25 * np.linalg.norm(g)
            v = np.mean([rollout(actor, gp, th, s, args.steps)[0] for s in range(args.seeds)], 0)
            steer.append(dict(target=float(np.degrees(th)), k=k, d_heading=circ_diff(heading(v), heading(base)),
                              d_speed=float(np.linalg.norm(v) - np.linalg.norm(base))))
    res["steering"] = steer
    (out / "gate_primitives_results.json").write_text(json.dumps(res))
    print("saved", out / "gate_primitives_results.json")


if __name__ == "__main__":
    main()
