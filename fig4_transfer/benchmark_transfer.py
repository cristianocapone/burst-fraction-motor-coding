#!/usr/bin/env python3
"""Matched-information test-time transfer benchmark on the MuJoCo Ant (Fig. 4E-H).

Pretrained on eight directions (45 deg apart), evaluated on the eight interleaved
unseen directions (22.5 deg offsets), five seeds, deterministic evaluation, reward
r_t = vx cos(theta) + vy sin(theta).  In the known-task condition every method
receives the descriptor (cos theta, sin theta).

Methods
-------
bilinear-rw
    Existing shared-G actor/critic.  All weights are frozen and only the
    eight-dimensional G vector is updated with G <- G + eta * r * psi.
sf-gpi
    Canonical successor features + generalized policy improvement.  A bank of
    independently parametrized policies proposes actions; all successor-
    feature heads score every proposal for the new reward vector.
sf-actor
    An independently parametrized (non-bilinear) actor is initialized from the
    closest source policy and improved by gradient ascent through the frozen
    GPI value envelope.

The script records reward against environment steps and cumulative measured
linear-layer MACs, model-only wall time, action latency, gradient steps,
trainable parameters, and peak RSS.  MuJoCo stepping is deliberately excluded
from ``model_seconds`` and included in ``total_seconds``.

The two runs behind Figure 4E-H (see make_figure4.py, which merges them):

  # G x Y: descriptor mapped once through Linear(2,8), gate then frozen (--g-lr 0)
  python benchmark_transfer.py benchmark --methods bilinear-rw \
      --task-information known --g-lr 0 --deterministic \
      --output-dir results/known_frozen_gy

  # successor-feature baselines
  python benchmark_transfer.py benchmark --methods sf-gpi sf-actor \
      --task-information known --deterministic \
      --output-dir results/known

Other commands:

  python benchmark_transfer.py train-sf --episodes-per-task 2000   # SF source experts
  python benchmark_transfer.py plot --output-dir results/known     # single-run figure
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import resource
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent

STATE_DIM = 27
ACT_DIM = 8
REWARD_DIM = 2
NUM_TASKS = 8
HIDDEN = 256
LOG_STD_MIN, LOG_STD_MAX = -2.0, 2.0
TRAIN_ANGLES = np.arange(NUM_TASKS, dtype=np.float32) * (2 * np.pi / NUM_TASKS)
NOVEL_ANGLES = TRAIN_ANGLES + np.pi / NUM_TASKS


def reward_weights(theta: float) -> np.ndarray:
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)


def reward_features(next_obs: np.ndarray) -> np.ndarray:
    """Features used by the paper: planar torso velocity (vx, vy)."""
    return np.asarray(next_obs[13:15], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int):
    env = gym.make("Ant-v4")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class MacCounter:
    """Counts multiply-accumulates executed by nn.Linear layers."""

    def __init__(self, modules: Iterable[nn.Module]):
        self.macs = 0
        self._handles = []
        seen = set()
        for root in modules:
            for layer in root.modules():
                if isinstance(layer, nn.Linear) and id(layer) not in seen:
                    seen.add(id(layer))
                    self._handles.append(layer.register_forward_hook(self._hook))

    def _hook(self, layer, inputs, output):
        n = int(np.prod(output.shape[:-1])) if output.ndim > 1 else 1
        self.macs += n * layer.in_features * layer.out_features

    def close(self):
        for handle in self._handles:
            handle.remove()


class GaussianActor(nn.Module):
    """Conventional actor: no bilinear or multiplicative task pathway."""

    def __init__(self, state_dim=STATE_DIM, act_dim=ACT_DIM, hidden=HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def distribution(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, state, deterministic=False):
        mean, std = self.distribution(state)
        raw = mean if deterministic else torch.distributions.Normal(mean, std).rsample()
        action = torch.tanh(raw)
        logp = torch.distributions.Normal(mean, std).log_prob(raw)
        logp -= torch.log(1 - action.square() + 1e-6)
        return action, logp.sum(-1)


class ScalarCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, act_dim=ACT_DIM, hidden=HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(state_dim + act_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = F.relu(self.l1(torch.cat([state, action], -1)))
        x = F.relu(self.l2(x))
        return self.out(x).squeeze(-1)


class SuccessorFeatureCritic(nn.Module):
    """Vector critic trained with psi = phi + gamma E_pi[psi']."""

    def __init__(self, state_dim=STATE_DIM, act_dim=ACT_DIM, hidden=HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(state_dim + act_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, REWARD_DIM)

    def forward(self, state, action):
        x = F.relu(self.l1(torch.cat([state, action], -1)))
        x = F.relu(self.l2(x))
        return self.out(x)


class BilinearActor(nn.Module):
    """Exact architecture used by models_8dir_g_est/pretrain_final.pth."""

    def __init__(self, state_dim=STATE_DIM, act_dim=ACT_DIM, k=NUM_TASKS):
        super().__init__()
        self.l1 = nn.Linear(state_dim, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.g_layer = nn.Linear(REWARD_DIM, k)
        self.y_layers = nn.ModuleList([nn.Linear(HIDDEN, act_dim) for _ in range(k)])
        self.log_std_layer = nn.Linear(state_dim + REWARD_DIM, act_dim)

    def sample(self, obs_ctx, g_override=None, deterministic=False):
        state, ctx = obs_ctx[:, :-2], obs_ctx[:, -2:]
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        g = self.g_layer(ctx) if g_override is None else g_override
        primitives = torch.stack([head(x) for head in self.y_layers], 1)
        mean = (g.unsqueeze(-1) * primitives).sum(1)
        std = torch.clamp(self.log_std_layer(obs_ctx), LOG_STD_MIN, LOG_STD_MAX).exp()
        raw = mean if deterministic else torch.distributions.Normal(mean, std).rsample()
        action = torch.tanh(raw)
        return action, primitives


class BilinearCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, act_dim=ACT_DIM, k=NUM_TASKS):
        super().__init__()
        self.l1_ag = nn.Linear(state_dim + act_dim, HIDDEN)
        self.l2_ag = nn.Linear(HIDDEN, HIDDEN)
        self.g_layer = nn.Linear(REWARD_DIM, k)
        self.y_layers = nn.ModuleList([nn.Linear(HIDDEN, 1) for _ in range(k)])

    def features(self, state, action):
        x = F.relu(self.l1_ag(torch.cat([action, state], -1)))
        x = F.relu(self.l2_ag(x))
        return torch.cat([head(x) for head in self.y_layers], -1)


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, phi, next_state, done):
        self.data.append((state, action, phi, next_state, done))

    def sample(self, batch, device):
        rows = random.sample(self.data, batch)
        arrays = map(np.stack, zip(*rows))
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays)


def soft_update(target, source, tau=0.005):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(p, alpha=tau)


def train_one_sf_expert(args, task_index, device):
    """Train SAC policy and a genuine SF critic for one source direction."""
    theta = float(TRAIN_ANGLES[task_index])
    w = torch.as_tensor(reward_weights(theta), device=device)
    env = make_env(args.seed + task_index)
    actor = GaussianActor().to(device)
    q1, q2 = ScalarCritic().to(device), ScalarCritic().to(device)
    tq1, tq2 = copy.deepcopy(q1), copy.deepcopy(q2)
    sf = SuccessorFeatureCritic().to(device)
    tsf = copy.deepcopy(sf)
    opts = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in (actor, q1, q2, sf)]
    buffer = ReplayBuffer(args.buffer_size)
    alpha, gamma = args.entropy_alpha, args.gamma
    returns = []

    def update():
        state, action, phi, next_state, done = buffer.sample(args.batch_size, device)
        with torch.no_grad():
            next_action, next_logp = actor.sample(next_state)
            target_q = phi @ w + gamma * (1 - done) * (
                torch.minimum(tq1(next_state, next_action), tq2(next_state, next_action))
                - alpha * next_logp
            )
            target_sf = phi + gamma * (1 - done).unsqueeze(-1) * tsf(next_state, next_action)
        losses = [F.mse_loss(q1(state, action), target_q),
                  F.mse_loss(q2(state, action), target_q),
                  F.mse_loss(sf(state, action), target_sf)]
        for opt, loss in zip(opts[1:], losses):
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        new_action, logp = actor.sample(state)
        actor_loss = (alpha * logp - torch.minimum(q1(state, new_action), q2(state, new_action))).mean()
        opts[0].zero_grad(set_to_none=True); actor_loss.backward(); opts[0].step()
        soft_update(tq1, q1); soft_update(tq2, q2); soft_update(tsf, sf)

    for episode in range(args.episodes_per_task):
        state, _ = env.reset(seed=args.seed + task_index * 100_000 + episode)
        total = 0.0
        for _ in range(args.episode_length):
            if len(buffer.data) < args.warmup_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor.sample(st)[0].squeeze(0).cpu().numpy()
            next_state, _, terminated, truncated, _ = env.step(action)
            phi = reward_features(next_state)
            reward = float(phi @ reward_weights(theta))
            done = terminated or truncated
            buffer.add(state, action, phi, next_state, float(done))
            state, total = next_state, total + reward
            if done:
                break
        returns.append(total)
        if len(buffer.data) >= args.batch_size:
            for _ in range(args.updates_per_episode):
                update()
        if (episode + 1) % args.log_every == 0:
            print(f"expert={task_index} episode={episode+1} return100={np.mean(returns[-100:]):.2f}", flush=True)
    env.close()
    return {"actor": actor.state_dict(), "sf": sf.state_dict(), "returns": returns}


def train_sf_bank(args):
    device = torch.device(args.device)
    set_seed(args.seed)
    checkpoint = {"format": 1, "train_angles": TRAIN_ANGLES, "experts": []}
    for task_index in range(NUM_TASKS):
        checkpoint["experts"].append(train_one_sf_expert(args, task_index, device))
        Path(args.sf_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.sf_checkpoint)
    print(f"saved {args.sf_checkpoint}")


def load_sf_bank(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    actors, critics = [], []
    if len(ck["experts"]) != NUM_TASKS:
        raise ValueError(f"SF bank is incomplete: {len(ck['experts'])}/{NUM_TASKS} experts")
    for item in ck["experts"]:
        actor, critic = GaussianActor().to(device), SuccessorFeatureCritic().to(device)
        actor.load_state_dict(item["actor"]); critic.load_state_dict(item["sf"])
        actor.eval(); critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
        actors.append(actor); critics.append(critic)
    return actors, critics


def load_bilinear(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    actor, critic = BilinearActor().to(device), BilinearCritic().to(device)
    actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic1"])
    actor.eval(); critic.eval()
    for model in (actor, critic):
        for p in model.parameters(): p.requires_grad_(False)
    return actor, critic


@dataclass
class MethodStats:
    method: str
    theta_deg: float
    seed: int
    episode: int
    env_steps: int
    episode_reward: float
    reward_per_step: float
    cumulative_reward: float
    cumulative_macs: int
    model_seconds: float
    total_seconds: float
    mean_action_ms: float
    gradient_steps: int
    adapted_parameters: int
    resident_parameter_mb: float
    peak_rss_mb: float


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def parameter_mb(models):
    seen, nbytes = set(), 0
    for model in models:
        for p in model.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                nbytes += p.numel() * p.element_size()
    return nbytes / (1024 * 1024)


class BilinearRW:
    name = "bilinear-rw"
    def __init__(self, actor, critic, theta, args, device):
        self.actor, self.critic, self.theta = actor, critic, theta
        self.args, self.device = args, device
        self.setup_macs, self.setup_seconds = 0, 0.0
        ctx = torch.as_tensor(reward_weights(theta), device=device)[None]
        if args.task_information == "known":
            setup_start = time.perf_counter()
            self.g = actor.g_layer(ctx).detach().clone()
            self.setup_seconds = time.perf_counter() - setup_start
            # One descriptor-to-gate projection: R^2 -> R^8. The cosine/sine
            # descriptor itself is supplied by the task, as it is in Eq. 9.
            self.setup_macs = REWARD_DIM * NUM_TASKS
        else:
            self.g = torch.zeros(1, NUM_TASKS, device=device)
        self.models = [actor, critic]
        self.gradient_steps = 0
        self.adapted_parameters = NUM_TASKS

    def act_and_adapt(self, state, transition=None):
        ctx = reward_weights(self.theta)
        obs = torch.as_tensor(np.r_[state, ctx], dtype=torch.float32, device=self.device)[None]
        with torch.no_grad(): action, _ = self.actor.sample(obs, self.g, self.args.deterministic)
        return action.squeeze(0).cpu().numpy()

    def observe(self, state, action, reward, next_state, done):
        if self.args.g_lr == 0:
            return
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        at = torch.as_tensor(action, dtype=torch.float32, device=self.device)[None]
        with torch.no_grad():
            psi = self.critic.features(st, at)
            self.g.add_(self.args.g_lr * reward * psi)
            norm = self.g.norm()
            if norm > self.args.g_max: self.g.mul_(self.args.g_max / norm)


class SFGPI:
    name = "sf-gpi"
    def __init__(self, actors, critics, theta, args, device):
        self.actors, self.critics, self.theta = actors, critics, theta
        self.args, self.device = args, device
        self.w = (torch.as_tensor(reward_weights(theta), device=device) if args.task_information == "known"
                  else torch.zeros(REWARD_DIM, device=device))
        self.ridge_a = args.reward_ridge * torch.eye(REWARD_DIM, device=device)
        self.ridge_b = torch.zeros(REWARD_DIM, device=device)
        self.models = actors + critics
        self.gradient_steps = 0
        self.adapted_parameters = REWARD_DIM

    def act_and_adapt(self, state, transition=None):
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        with torch.no_grad():
            candidates = torch.cat([a.sample(st, self.args.deterministic)[0] for a in self.actors], 0)
            states = st.expand(NUM_TASKS, -1)
            values = torch.stack([(sf(states, candidates) @ self.w) for sf in self.critics], 0)
            index = torch.max(values, 0).values.argmax()
        return candidates[index].cpu().numpy()

    def observe(self, state, action, reward, next_state, done):
        if self.args.task_information == "reward-only":
            phi = torch.as_tensor(reward_features(next_state), device=self.device)
            self.ridge_a.add_(torch.outer(phi, phi))
            self.ridge_b.add_(phi * reward)
            self.w = torch.linalg.solve(self.ridge_a, self.ridge_b)


class SFActorImprovement:
    name = "sf-actor"
    def __init__(self, actors, critics, theta, args, device):
        self.theta, self.args, self.device = theta, args, device
        closest = (int(np.argmax(np.cos(TRAIN_ANGLES - theta)))
                   if args.task_information == "known" else 0)
        self.actor = copy.deepcopy(actors[closest]).train()
        self.critics = critics
        self.w = (torch.as_tensor(reward_weights(theta), device=device) if args.task_information == "known"
                  else torch.zeros(REWARD_DIM, device=device))
        self.ridge_a = args.reward_ridge * torch.eye(REWARD_DIM, device=device)
        self.ridge_b = torch.zeros(REWARD_DIM, device=device)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.states = deque(maxlen=args.actor_buffer)
        self.models = [self.actor] + critics
        self.gradient_steps = 0
        self.backward_macs = 0
        self.adapted_parameters = sum(p.numel() for p in self.actor.parameters())

    def act_and_adapt(self, state, transition=None):
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        with torch.no_grad(): action = self.actor.sample(st, self.args.deterministic)[0]
        return action.squeeze(0).cpu().numpy()

    def observe(self, state, action, reward, next_state, done):
        if self.args.task_information == "reward-only":
            phi = torch.as_tensor(reward_features(next_state), device=self.device)
            self.ridge_a.add_(torch.outer(phi, phi))
            self.ridge_b.add_(phi * reward)
            self.w = torch.linalg.solve(self.ridge_a, self.ridge_b)
        self.states.append(np.asarray(state, dtype=np.float32))
        if len(self.states) < self.args.actor_batch:
            return
        for _ in range(self.args.actor_updates):
            macs_before = self.counter.macs
            idx = np.random.randint(len(self.states), size=self.args.actor_batch)
            states = torch.as_tensor(np.stack([self.states[i] for i in idx]), device=self.device)
            actions, logp = self.actor.sample(states)
            # Differentiable GPI envelope; SF parameters stay frozen, gradients
            # pass through their action inputs into the independent actor.
            values = torch.stack([sf(states, actions) @ self.w for sf in self.critics], 0)
            objective = torch.max(values, 0).values - self.args.entropy_alpha * logp
            forward_macs = self.counter.macs - macs_before
            self.opt.zero_grad(set_to_none=True)
            (-objective.mean()).backward()
            self.opt.step()
            # For dense layers, reverse-mode differentiation costs about twice
            # the forward MACs (input and weight gradients). Keep this explicit
            # rather than silently reporting forward-only compute.
            self.backward_macs += 2 * forward_macs
            self.gradient_steps += 1


def run_method(factory, theta, seed, args, device):
    set_seed(seed)
    env = make_env(seed)
    method = factory(theta)
    counter = MacCounter(method.models)
    method.counter = counter
    rows, env_steps, cumulative_reward = [], 0, 0.0
    model_seconds = getattr(method, "setup_seconds", 0.0)
    total_start = time.perf_counter()
    action_latencies = []
    try:
        for episode in range(1, args.test_episodes + 1):
            state, _ = env.reset(seed=seed * 10_000 + episode)
            episode_reward, steps = 0.0, 0
            for _ in range(args.episode_length):
                t0 = time.perf_counter()
                action = method.act_and_adapt(state)
                action_latencies.append(time.perf_counter() - t0)
                next_state, _, terminated, truncated, _ = env.step(action)
                phi = reward_features(next_state)
                reward = float(phi @ reward_weights(theta))
                done = terminated or truncated
                t0 = time.perf_counter()
                method.observe(state, action, reward, next_state, done)
                model_seconds += time.perf_counter() - t0 + action_latencies[-1]
                state = next_state
                episode_reward += reward; cumulative_reward += reward
                steps += 1; env_steps += 1
                if done: break
            rows.append(MethodStats(
                method.name, float(np.degrees(theta)), seed, episode, env_steps,
                episode_reward, episode_reward / max(steps, 1), cumulative_reward,
                counter.macs + getattr(method, "backward_macs", 0)
                + getattr(method, "setup_macs", 0),
                model_seconds, time.perf_counter() - total_start,
                1000 * float(np.mean(action_latencies)), method.gradient_steps,
                method.adapted_parameters, parameter_mb(method.models), peak_rss_mb()))
    finally:
        counter.close(); env.close()
    return rows


def benchmark(args):
    device = torch.device(args.device)
    sf_actors, sf_critics = load_sf_bank(args.sf_checkpoint, device)
    bilinear_actor, bilinear_critic = load_bilinear(args.bilinear_checkpoint, device)
    factories = {
        "bilinear-rw": lambda theta: BilinearRW(bilinear_actor, bilinear_critic, theta, args, device),
        "sf-gpi": lambda theta: SFGPI(sf_actors, sf_critics, theta, args, device),
        "sf-actor": lambda theta: SFActorImprovement(sf_actors, sf_critics, theta, args, device),
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = []
    csv_path = output / "adaptation_results.csv"

    def save_progress():
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader(); writer.writerows(asdict(r) for r in rows)
        # Regenerate the paper-facing figure after every atomic simulation.
        plot_args = argparse.Namespace(output_dir=args.output_dir)
        plot_results(plot_args)

    for theta in args.test_angles:
        theta = np.radians(theta)
        for seed in args.seeds:
            for name in args.methods:
                print(f"method={name} theta={np.degrees(theta):.1f} seed={seed}", flush=True)
                rows.extend(run_method(factories[name], theta, seed, args, device))
                save_progress()
    metadata = vars(args).copy()
    metadata.update(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"saved {csv_path}")


def plot_results(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(args.output_dir) / "adaptation_results.csv"
    metadata_path = Path(args.output_dir) / "metadata.json"
    zero_shot_gy = False
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        zero_shot_gy = (metadata.get("task_information") == "known"
                        and float(metadata.get("g_lr", 1.0)) == 0.0)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")
    preferred = ["bilinear-rw", "sf-gpi", "sf-actor"]
    present = set(np.atleast_1d(data["method"]).tolist())
    methods = [method for method in preferred if method in present]
    colors = {"bilinear-rw": "#1a6fbe", "sf-gpi": "#2ca02c", "sf-actor": "#cc2222"}
    short = {"bilinear-rw": "G×Y", "sf-gpi": "SF-GPI", "sf-actor": "SF + actor PI"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    panel_labels = "ABCD"
    zero_shot_reference = None
    for method in methods:
        d = data[data["method"] == method]
        episodes = np.unique(d["episode"])
        seeds = np.unique(d["seed"])
        # Directions are repeated conditions, not independent replicates:
        # average directions within seed, then compute SEM across seeds (n=5).
        seed_curve = lambda key: np.asarray([
            [d[key][(d["episode"] == ep) & (d["seed"] == seed)].mean()
             for ep in episodes] for seed in seeds])
        mean = lambda key: seed_curve(key).mean(0)
        sem = lambda key: (seed_curve(key).std(0, ddof=1) / np.sqrt(len(seeds))
                           if len(seeds) > 1 else np.zeros(len(episodes)))
        y, e = mean("reward_per_step"), sem("reward_per_step")
        line_style = "--" if method == "bilinear-rw" and zero_shot_gy else "-"
        if method == "bilinear-rw" and zero_shot_gy:
            # No learning occurs after the one-shot encoder pass. Display the
            # aggregate measured zero-shot level as a reference, rather than
            # suggesting that reset-to-reset noise is an adaptation curve.
            seed_levels = seed_curve("reward_per_step").mean(1)
            y[:] = seed_levels.mean()
            e[:] = (seed_levels.std(ddof=1) / np.sqrt(len(seed_levels))
                    if len(seed_levels) > 1 else 0.0)
            zero_shot_reference = (float(y[0]), float(e[0]))
        axes[0].plot(mean("env_steps"), y, line_style, label=short[method], color=colors[method])
        axes[0].fill_between(mean("env_steps"), y-e, y+e, color=colors[method], alpha=.15)
        if not (method == "bilinear-rw" and zero_shot_gy):
            axes[1].plot(mean("cumulative_macs") / 1e9, y, line_style, color=colors[method])
            axes[2].plot(mean("model_seconds"), y, line_style, color=colors[method])
    if zero_shot_reference is not None:
        level, error = zero_shot_reference
        # G×Y has no learning trajectory in these panels. Span the complete
        # comparison axis so its zero-shot reference does not collapse into a
        # short segment near x=0 when SF methods set a much larger x-range.
        for ax in axes[1:3]:
            ax.axhline(level, color=colors["bilinear-rw"], linestyle="--", linewidth=1.5)
            xmin, xmax = ax.get_xlim()
            ax.fill_between([xmin, xmax], level-error, level+error,
                            color=colors["bilinear-rw"], alpha=.15)
            ax.set_xlim(xmin, xmax)
    final = data[data["episode"] == data["episode"].max()]
    x = np.arange(len(methods))
    width = .36
    inference = np.asarray([final["mean_action_ms"][final["method"] == m].mean() for m in methods])
    processing = np.asarray([
        (1000 * final["model_seconds"][final["method"] == m] /
         final["env_steps"][final["method"] == m]).mean() for m in methods])
    axes[3].bar(x - width/2, inference, width, color=[colors[m] for m in methods],
                alpha=.45, edgecolor="black", linewidth=.5, label="Policy inference")
    axes[3].bar(x + width/2, processing, width, color=[colors[m] for m in methods],
                edgecolor="black", linewidth=.5, label="Inference + adaptation")
    axes[0].set(xlabel="Environment steps", ylabel="Reward / step", title="Sample efficiency")
    axes[1].set(xlabel="Cumulative GMAC", ylabel="Reward / step", title="Compute efficiency")
    axes[2].set(xlabel="Model time (s)", ylabel="Reward / step", title="Test-time adaptation")
    axes[3].set(xlabel="Method", ylabel="Time per environment step (ms)",
                title="Inference and adaptation latency")
    axes[3].set_xticks(x, [short[m] for m in methods], rotation=20)
    axes[3].legend(frameon=False, fontsize=7)
    axes[0].legend(frameon=False, fontsize=8)
    for label, ax in zip(panel_labels, axes):
        ax.text(-.14, 1.06, label, transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="left", va="top")
        ax.title.set_fontweight("bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(Path(args.output_dir) / f"adaptation_compute.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures in {args.output_dir}")


def common_parser(parser):
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--entropy-alpha", type=float, default=.2)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-sf", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common_parser(train)
    train.add_argument("--sf-checkpoint", default=str(HERE / "checkpoints" / "sf_bank.pth"))
    train.add_argument("--episodes-per-task", type=int, default=2000)
    train.add_argument("--warmup-steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--buffer-size", type=int, default=1_000_000)
    train.add_argument("--updates-per-episode", type=int, default=50)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--log-every", type=int, default=100)

    bench = sub.add_parser("benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common_parser(bench)
    bench.add_argument("--sf-checkpoint", default=str(HERE / "checkpoints" / "sf_bank.pth"))
    bench.add_argument("--bilinear-checkpoint", default=str(HERE / "checkpoints" / "pretrain_final.pth"))
    bench.add_argument("--output-dir", default=str(HERE / "results" / "run"))
    bench.add_argument("--methods", nargs="+", choices=["bilinear-rw", "sf-gpi", "sf-actor"],
                       default=["bilinear-rw", "sf-gpi", "sf-actor"])
    bench.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    bench.add_argument("--test-angles", nargs="+", type=float,
                       default=[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
    bench.add_argument("--test-episodes", type=int, default=5)
    bench.add_argument("--g-lr", type=float, default=1e-4)
    bench.add_argument("--g-max", type=float, default=10.0)
    bench.add_argument("--actor-lr", type=float, default=3e-4)
    bench.add_argument("--actor-buffer", type=int, default=100_000)
    bench.add_argument("--actor-batch", type=int, default=256)
    bench.add_argument("--actor-updates", type=int, default=1)
    bench.add_argument("--task-information", choices=["known", "reward-only"], default="reward-only",
                       help="whether theta is supplied or inferred from scalar reward")
    bench.add_argument("--reward-ridge", type=float, default=1e-3)
    bench.add_argument("--deterministic", action="store_true")

    plot = sub.add_parser("plot", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    plot.add_argument("--output-dir", default=str(HERE / "results" / "run"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train-sf": train_sf_bank(args)
    elif args.command == "benchmark": benchmark(args)
    else: plot_results(args)


if __name__ == "__main__":
    main()
