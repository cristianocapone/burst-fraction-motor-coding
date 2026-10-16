"""Adaptation to a velocity-dependent (curl) force field with the pretrained
Ant agents of benchmark_sf_adaptation.py.

Motor-adaptation paradigm (Shadmehr & Mussa-Ivaldi 1994): a force
F = B * (-v_y, v_x, 0) is applied to the torso, i.e. perpendicular to the
planar velocity, which rotates the heading of an unadapted walker. The task
(reach direction theta, known descriptor) is unchanged; only the dynamics
change. Adaptation must therefore re-weight the learned primitives rather than
select a new goal, which is the case Reviewer 1 asked about (task-parameter
change rather than a rotation of goals).

Conditions (all with the known descriptor (cos theta, sin theta)):
  gy-frozen   G x Y with the gate set once by the descriptor and frozen
  gy-rw       G x Y with online reward-weighted adaptation of the 8 gate
              coefficients (Delta G = eta * r * psi), primitives frozen
  sf-gpi      successor-feature GPI over the 8 source actors (no adaptation
              possible with a known descriptor)
  sf-actor    MLP actor initialised from the nearest source policy and
              fine-tuned online through the frozen SF value envelope
Schedule: field ON for the first --field-episodes episodes, then OFF for
--washout-episodes (aftereffect / washout). --gain 0 gives the unperturbed
reference.

Per episode we record reward per step, the signed heading error between the
mean velocity and the target direction (positive = rotated in the direction
the field pushes, i.e. counter-clockwise), the fraction of steps spent moving
within 15 degrees of the target, termination (fall), cumulative neural
compute, gradient steps and adapted parameters, as in the paper's benchmark.

Run with the benchmark venv and a clean environment, e.g.
  env -u PYTHONPATH .venv_benchmark/bin/python benchmark_curl_field.py --gain 0.5 --output-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from benchmark_sf_adaptation import (
    make_env, load_bilinear, load_sf_bank, BilinearRW, SFGPI, SFActorImprovement,
    reward_features, reward_weights, MacCounter, set_seed, peak_rss_mb, parameter_mb,
    NOVEL_ANGLES,
)

TORSO_BODY = 1  # Ant-v4 body index of the torso (checked: name 'torso', mass 0.327 kg)


class BilinearRWBaseline(BilinearRW):
    """Reward-weighted gate update with a running reward baseline,
    Delta G = eta * (r - rbar) * psi. From a warm start the baseline-free rule
    of the paper mostly rescales the current gate (rewards are positive); the
    centred rule follows the sign of the improvement instead."""
    name = "gy-rw"

    def __init__(self, *a, baseline_tau=0.01, **kw):
        super().__init__(*a, **kw)
        self.rbar, self.tau, self.n = 0.0, baseline_tau, 0

    def observe(self, state, action, reward, next_state, done):
        if self.args.g_lr == 0:
            return
        self.n += 1
        self.rbar += (reward - self.rbar) * (1.0 / self.n if self.n < 1.0 / self.tau else self.tau)
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        at = torch.as_tensor(action, dtype=torch.float32, device=self.device)[None]
        with torch.no_grad():
            psi = self.critic.features(st, at)
            self.g.add_(self.args.g_lr * (reward - self.rbar) * psi)
            norm = self.g.norm()
            if norm > self.args.g_max:
                self.g.mul_(self.args.g_max / norm)


class CurlField:
    """Applies F = gain * (-v_y, v_x, 0) to the torso before every control step.
    Duck-typed wrapper (works with both gym and gymnasium environments)."""

    def __init__(self, env, gain: float):
        self.env, self.gain, self.on = env, float(gain), False

    def step(self, action):
        d = self.env.unwrapped.data
        if self.on and self.gain != 0.0:
            vx, vy = float(d.qvel[0]), float(d.qvel[1])
            d.xfrc_applied[TORSO_BODY, 0] = -self.gain * vy
            d.xfrc_applied[TORSO_BODY, 1] = self.gain * vx
        else:
            d.xfrc_applied[TORSO_BODY, :3] = 0.0
        return self.env.step(action)

    def reset(self, **kw):
        self.env.unwrapped.data.xfrc_applied[TORSO_BODY, :3] = 0.0
        return self.env.reset(**kw)

    def close(self):
        return self.env.close()


@dataclass
class EpisodeStats:
    method: str
    gain: float
    g_lr: float
    theta_deg: float
    seed: int
    episode: int
    field_on: bool
    env_steps: int
    steps: int
    terminated: bool
    reward_per_step: float
    cumulative_reward: float
    heading_error_deg: float
    frac_on_target: float
    mean_speed: float
    cumulative_macs: float
    model_seconds: float
    gradient_steps: int
    adapted_parameters: int
    g_norm: float


class BilinearES(BilinearRW):
    """Episodic reward-driven search over the 8 gate coefficients, primitives
    frozen: a (1+1) evolution strategy with the 1/5 success rule. Each episode
    runs a perturbed gate G_best + sigma*xi; it replaces G_best if the episode
    reward per step improves. No gradients, no critic: the only adapted
    quantities are the eight numbers of the gate, updated once per episode."""
    name = "gy-es"

    def __init__(self, *a, sigma0=0.7, margin=0.02, **kw):
        super().__init__(*a, **kw)
        self.args = argparse.Namespace(**vars(self.args)); self.args.g_lr = 0.0   # no within-episode update
        self.g_best, self.best_r = self.g.clone(), None
        self.sigma, self.margin, self.g_try = sigma0, margin, None
        self.exploit = True                  # episodes alternate: evaluate current best / try a perturbation
        self.generator = torch.Generator(device="cpu").manual_seed(int(1e6 * float(self.theta)) % (2**31))

    def start_episode(self):
        if self.exploit:
            self.g_try = self.g_best.clone()
        else:
            xi = torch.randn(self.g_best.shape, generator=self.generator).to(self.g_best.device)
            self.g_try = self.g_best + self.sigma * xi
            norm = self.g_try.norm()
            if norm > self.args.g_max:
                self.g_try.mul_(self.args.g_max / norm)
        self.g = self.g_try

    def end_episode(self, reward_per_step):
        if self.exploit:
            # running estimate of the current best gate's value (evaluations are noisy)
            self.best_r = reward_per_step if self.best_r is None else 0.5 * (self.best_r + reward_per_step)
        else:
            if reward_per_step > self.best_r + self.margin:
                self.best_r, self.g_best = reward_per_step, self.g_try.clone()
                self.sigma = min(self.sigma * 1.5, 2.0)
            else:
                self.sigma = max(self.sigma * 0.85, 0.1)
        self.exploit = not self.exploit
        self.g = self.g_best


def signed_angle_deg(target, v):
    """Angle from target to v, counter-clockwise positive, in degrees."""
    cross = target[0] * v[1] - target[1] * v[0]
    dot = target[0] * v[0] + target[1] * v[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def run_condition(factory, name, theta, seed, args, device):
    set_seed(seed)
    env = CurlField(make_env(seed), args.gain)
    # --aim-offset-deg: the gate is set for a descriptor rotated by this angle
    # ("aiming off"), while the reward is still measured toward the true theta.
    # Used to scan whether any re-aiming of the frozen primitives can recover
    # the reward lost under the field (oracle for what a gate change can do).
    method = factory(theta + np.radians(getattr(args, "aim_offset_deg", 0.0)))
    method.theta = theta + np.radians(getattr(args, "aim_offset_deg", 0.0))
    counter = MacCounter(method.models)
    method.counter = counter
    target = reward_weights(theta)
    rows, env_steps, cumulative_reward = [], 0, 0.0
    model_seconds = getattr(method, "setup_seconds", 0.0)
    n_episodes = args.field_episodes + args.washout_episodes
    try:
        for episode in range(1, n_episodes + 1):
            env.on = episode <= args.field_episodes
            state, _ = env.reset(seed=seed * 10_000 + episode)
            if hasattr(method, "start_episode") and env.on:
                method.start_episode()
            ep_reward, steps, vsum, on_target, terminated = 0.0, 0, np.zeros(2), 0, False
            for _ in range(args.episode_length):
                t0 = time.perf_counter()
                action = method.act_and_adapt(state)
                t_act = time.perf_counter() - t0
                next_state, _, term, trunc, _ = env.step(action)
                phi = reward_features(next_state)
                reward = float(phi @ target)
                done = term or trunc
                t0 = time.perf_counter()
                method.observe(state, action, reward, next_state, done)
                model_seconds += time.perf_counter() - t0 + t_act
                state = next_state
                ep_reward += reward; cumulative_reward += reward
                vsum += phi
                if np.linalg.norm(phi) > 0.2 and abs(signed_angle_deg(target, phi)) < 15:
                    on_target += 1
                steps += 1; env_steps += 1
                if done:
                    terminated = bool(term)
                    break
            if hasattr(method, "end_episode") and env.on:
                method.end_episode(ep_reward / max(steps, 1))
            vmean = vsum / max(steps, 1)
            g = getattr(method, "g", None)
            rows.append(EpisodeStats(
                name + (f"@{args.aim_offset_deg:+.0f}" if getattr(args, "aim_offset_deg", 0.0) else ""),
                args.gain, args.g_lr if name.startswith("gy-rw") else 0.0, float(np.degrees(theta)), seed, episode,
                bool(env.on), env_steps, steps, terminated, ep_reward / max(steps, 1), cumulative_reward,
                signed_angle_deg(target, vmean), on_target / max(steps, 1), float(np.linalg.norm(vmean)),
                counter.macs + getattr(method, "backward_macs", 0) + getattr(method, "setup_macs", 0),
                model_seconds, method.gradient_steps, method.adapted_parameters,
                float(g.norm()) if g is not None else float("nan")))
    finally:
        counter.close(); env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--sf-checkpoint", default="../fig4_transfer/checkpoints/sf_bank.pth")
    p.add_argument("--bilinear-checkpoint", default="../fig4_transfer/checkpoints/pretrain_final.pth")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--methods", nargs="+", default=["gy-frozen", "gy-rw-base", "sf-gpi", "sf-actor"],
                   choices=["gy-frozen", "gy-rw", "gy-rw-base", "gy-es", "sf-gpi", "sf-actor"])
    p.add_argument("--es-sigma", type=float, default=0.7, help="initial perturbation scale of the gate search (gy-es)")
    p.add_argument("--baseline-tau", type=float, default=0.01, help="running-mean rate of the reward baseline (gy-rw-base)")
    p.add_argument("--aim-offset-deg", type=float, default=0.0, help="rotate the descriptor given to the gate (aiming-off scan)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--test-angles", nargs="+", type=float, default=[float(np.degrees(a)) for a in NOVEL_ANGLES])
    p.add_argument("--gain", type=float, default=0.5, help="curl-field gain B in N s/m (0 = no field)")
    p.add_argument("--field-episodes", type=int, default=8)
    p.add_argument("--washout-episodes", type=int, default=4)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--g-lr", type=float, default=1e-4)
    p.add_argument("--g-max", type=float, default=10.0)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--actor-buffer", type=int, default=100_000)
    p.add_argument("--actor-batch", type=int, default=256)
    p.add_argument("--actor-updates", type=int, default=1)
    p.add_argument("--entropy-alpha", type=float, default=0.2)
    p.add_argument("--reward-ridge", type=float, default=1e-3)
    p.add_argument("--deterministic", action="store_true", default=True)
    args = p.parse_args()
    args.task_information = "known"     # descriptor known to every method, as in the paper's benchmark
    args.gamma = 0.99

    device = torch.device(args.device)
    sf_actors, sf_critics = load_sf_bank(args.sf_checkpoint, device)
    actor, critic = load_bilinear(args.bilinear_checkpoint, device)

    def frozen_args():
        a = argparse.Namespace(**vars(args)); a.g_lr = 0.0; return a

    def named(obj, name):
        obj.name = name; return obj

    factories = {
        "gy-frozen": lambda th: named(BilinearRW(actor, critic, th, frozen_args(), device), "gy-frozen"),
        "gy-rw": lambda th: named(BilinearRW(actor, critic, th, args, device), "gy-rw"),
        "gy-rw-base": lambda th: named(BilinearRWBaseline(actor, critic, th, args, device, baseline_tau=args.baseline_tau), "gy-rw-base"),
        "gy-es": lambda th: named(BilinearES(actor, critic, th, args, device, sigma0=args.es_sigma), "gy-es"),
        "sf-gpi": lambda th: SFGPI(sf_actors, sf_critics, th, args, device),
        "sf-actor": lambda th: SFActorImprovement(sf_actors, sf_critics, th, args, device),
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "curl_field_results.csv"
    rows = []
    for theta_deg in args.test_angles:
        theta = np.radians(theta_deg)
        for seed in args.seeds:
            for name in args.methods:
                t0 = time.time()
                rows.extend(run_condition(factories[name], name, theta, seed, args, device))
                last = [r for r in rows if r.method.startswith(name) and r.seed == seed and r.theta_deg == theta_deg]
                on = [r for r in last if r.field_on]; off = [r for r in last if not r.field_on]
                print(f"{name:9s} gain={args.gain} theta={theta_deg:6.1f} seed={seed} | field: reward/step "
                      f"{on[0].reward_per_step:.2f}->{on[-1].reward_per_step:.2f}, heading {on[0].heading_error_deg:+.0f}->"
                      f"{on[-1].heading_error_deg:+.0f} deg | washout: reward {off[0].reward_per_step:.2f}, heading "
                      f"{off[0].heading_error_deg:+.0f} deg | {time.time()-t0:.0f}s", flush=True)
                with csv_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                    w.writeheader(); w.writerows(asdict(r) for r in rows)
    meta = vars(args).copy()
    meta.update(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
