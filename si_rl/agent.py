"""Selección de acciones y función de pérdida (DQN / Double DQN / C51) para el agente."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .networks import QNetwork


@torch.no_grad()
def select_actions(
    net: QNetwork,
    obs: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """Política epsilon-greedy (con noisy nets se usa epsilon=0 y el ruido explora)."""
    n = obs.shape[0]
    if epsilon >= 1.0:
        return rng.integers(net.num_actions, size=n)
    if net.noisy:
        net.reset_noise()
    x = torch.as_tensor(obs, device=device)
    greedy = net.q_values(x).argmax(dim=1).cpu().numpy()
    if epsilon > 0.0:
        rand = rng.random(n) < epsilon
        greedy[rand] = rng.integers(net.num_actions, size=int(rand.sum()))
    return greedy


def compute_loss(
    online: QNetwork,
    target: QNetwork,
    batch: dict,
    device: torch.device,
    double: bool = True,
    loss_fn: str = "huber",
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """Devuelve (pérdida escalar ponderada por IS weights, prioridades nuevas, métricas)."""
    obs = torch.as_tensor(batch["obs"], device=device)
    next_obs = torch.as_tensor(batch["next_obs"], device=device)
    actions = torch.as_tensor(batch["actions"], device=device)
    returns = torch.as_tensor(batch["returns"], device=device)
    discounts = torch.as_tensor(batch["discounts"], device=device)
    weights = torch.as_tensor(batch["weights"], device=device)

    if online.noisy:
        online.reset_noise()
        target.reset_noise()

    if online.distributional:
        return _c51_loss(online, target, obs, next_obs, actions, returns, discounts, weights, double)

    q = online(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        if double:
            next_actions = online(next_obs).argmax(dim=1, keepdim=True)
            q_next = target(next_obs).gather(1, next_actions).squeeze(1)
        else:
            q_next = target(next_obs).max(dim=1).values
        y = returns + discounts * q_next
    td = q - y
    if loss_fn == "huber":
        per_sample = F.smooth_l1_loss(q, y, reduction="none")
    else:
        per_sample = 0.5 * td.pow(2)
    loss = (weights * per_sample).mean()
    prios = td.detach().abs().float().cpu().numpy()
    stats = dict(loss=float(loss.item()), q_mean=float(q.mean().item()), td_abs=float(prios.mean()))
    return loss, prios, stats


def _c51_loss(online, target, obs, next_obs, actions, returns, discounts, weights, double):
    support = online.support                       # (atoms,)
    atoms = support.numel()
    v_min, v_max = float(support[0]), float(support[-1])
    delta_z = (v_max - v_min) / (atoms - 1)

    log_p = online(obs)                                                  # (B, A, atoms)
    log_p_a = log_p.gather(1, actions[:, None, None].expand(-1, 1, atoms)).squeeze(1)  # (B, atoms)

    with torch.no_grad():
        p_next_target = target(next_obs).exp()                           # (B, A, atoms)
        if double:
            q_next_online = (online(next_obs).exp() * support).sum(2)   # (B, A)
            a_star = q_next_online.argmax(1)
        else:
            a_star = (p_next_target * support).sum(2).argmax(1)
        p_next = p_next_target[torch.arange(len(a_star), device=obs.device), a_star]  # (B, atoms)

        # Proyección de Bellman: Tz = R + gamma^n * z  (discounts = 0 si terminal)
        tz = (returns[:, None] + discounts[:, None] * support[None, :]).clamp(v_min, v_max)
        b = (tz - v_min) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()
        lower[(upper > 0) & (lower == upper)] -= 1
        upper[(lower < atoms - 1) & (lower == upper)] += 1
        m = torch.zeros_like(p_next)
        m.scatter_add_(1, lower, p_next * (upper.float() - b))
        m.scatter_add_(1, upper, p_next * (b - lower.float()))

    per_sample = -(m * log_p_a).sum(1)                                   # cross-entropy
    loss = (weights * per_sample).mean()
    prios = per_sample.detach().float().cpu().numpy()
    q_mean = float((log_p_a.exp() * support).sum(1).mean().item())
    stats = dict(loss=float(loss.item()), q_mean=q_mean, td_abs=float(prios.mean()))
    return loss, prios, stats
