"""Evaluación con política greedy (sin exploración) sobre episodios completos (3 vidas).

`evaluate_policy` corre `episodes` juegos en paralelo (uno por entorno de un DummyVecEnv)
y devuelve el puntaje REAL del juego (sin clipping) de cada episodio.
"""
from __future__ import annotations

import numpy as np
import torch

from .agent import select_actions
from .env import FrameStacker, make_env_fn
from .networks import QNetwork
from .vec_env import DummyVecEnv


def make_eval_vec_env(env_config: dict, episodes: int) -> DummyVecEnv:
    fns = [make_env_fn(env_config, episodic_life=False, clip_reward=False) for _ in range(episodes)]
    return DummyVecEnv(fns)


@torch.no_grad()
def evaluate_policy(
    net: QNetwork,
    vec_env: DummyVecEnv,
    device: torch.device,
    epsilon: float = 0.0,
    seed: int | None = None,
    frame_stack: int = 4,
    max_steps: int = 30_000,
) -> tuple[list[float], list[int]]:
    """Devuelve (puntajes, longitudes) de un episodio completo por entorno del vec_env."""
    was_training = net.training
    net.eval()
    net.set_noise(False)
    rng = np.random.default_rng(seed)
    n = vec_env.num_envs
    obs = vec_env.reset(seed=seed)
    h, w = obs.shape[1:]
    stacker = FrameStacker(n, frame_stack, h, w)
    stacker.reset(obs)
    scores = [None] * n
    lengths = [None] * n
    finished = np.zeros(n, dtype=bool)
    for _ in range(max_steps):
        actions = select_actions(net, stacker.obs, epsilon, device, rng)
        obs, _, dones, infos = vec_env.step(actions)
        stacker.push(obs, dones)
        for i, info in enumerate(infos):
            if not finished[i] and "game_score" in info:
                scores[i] = float(info["game_score"])
                lengths[i] = int(info["game_length"])
                finished[i] = True
        if finished.all():
            break
    net.set_noise(True)
    if was_training:
        net.train()
    scores = [s if s is not None else 0.0 for s in scores]
    lengths = [l if l is not None else max_steps for l in lengths]
    return scores, lengths
