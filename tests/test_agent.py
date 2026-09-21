"""Pruebas de la pérdida C51 (proyección de Bellman) y de la pérdida DQN/Double.

Ejecutar:  python tests/test_agent.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from si_rl.agent import compute_loss  # noqa: E402
from si_rl.networks import build_network  # noqa: E402


def reference_projection(p_next, returns, discounts, support):
    """Proyección categórica ingenua (bucle) de Bellemare et al. 2017."""
    B, atoms = p_next.shape
    v_min, v_max = support[0], support[-1]
    dz = (v_max - v_min) / (atoms - 1)
    m = np.zeros((B, atoms))
    for i in range(B):
        for j in range(atoms):
            tz = min(max(returns[i] + discounts[i] * support[j], v_min), v_max)
            b = (tz - v_min) / dz
            l, u = int(np.floor(b)), int(np.ceil(b))
            if l == u:
                m[i, l] += p_next[i, j]
            else:
                m[i, l] += p_next[i, j] * (u - b)
                m[i, u] += p_next[i, j] * (b - l)
    return m


def test_c51_projection_matches_reference():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = dict(arch="nature", dueling=True, noisy=False, c51=True, atoms=51, v_min=-10.0, v_max=10.0)
    online = build_network(6, cfg)
    target = build_network(6, cfg)
    target.load_state_dict(online.state_dict())
    B = 16
    batch = dict(
        obs=rng.integers(0, 256, (B, 4, 84, 84), dtype=np.uint8),
        next_obs=rng.integers(0, 256, (B, 4, 84, 84), dtype=np.uint8),
        actions=rng.integers(0, 6, B),
        returns=rng.choice([-2.0, -1.0, 0.0, 1.0, 2.5, 9.9, 12.0], B).astype(np.float32),
        discounts=np.where(rng.random(B) < 0.3, 0.0, 0.99 ** 3).astype(np.float32),
        weights=np.ones(B, dtype=np.float32),
    )
    loss, prios, stats = compute_loss(online, target, batch, torch.device("cpu"), double=True)
    # recomputar objetivo con la referencia
    with torch.no_grad():
        support = online.support.numpy()
        obs, nobs = torch.as_tensor(batch["obs"]), torch.as_tensor(batch["next_obs"])
        a_star = online.q_values(nobs).argmax(1)
        p_next = target(nobs).exp()[torch.arange(B), a_star].numpy()
        m_ref = reference_projection(p_next, batch["returns"], batch["discounts"], support)
        log_p = online(obs)[torch.arange(B), torch.as_tensor(batch["actions"])].numpy()
        ref_loss = -(m_ref * log_p).sum(1)
    assert np.allclose(m_ref.sum(1), 1.0, atol=1e-5)
    assert np.allclose(prios, ref_loss, atol=1e-4), (prios[:4], ref_loss[:4])
    assert abs(loss.item() - ref_loss.mean()) < 1e-4
    print("test_c51_projection_matches_reference OK")


def test_dqn_loss_reference():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    for double in (False, True):
        cfg = dict(arch="nature", dueling=False, noisy=False, c51=False)
        online = build_network(6, cfg)
        target = build_network(6, cfg)  # pesos distintos a propósito
        B = 8
        batch = dict(
            obs=rng.integers(0, 256, (B, 4, 84, 84), dtype=np.uint8),
            next_obs=rng.integers(0, 256, (B, 4, 84, 84), dtype=np.uint8),
            actions=rng.integers(0, 6, B),
            returns=rng.normal(size=B).astype(np.float32),
            discounts=np.where(rng.random(B) < 0.3, 0.0, 0.99 ** 3).astype(np.float32),
            weights=rng.random(B).astype(np.float32),
        )
        loss, prios, _ = compute_loss(online, target, batch, torch.device("cpu"), double=double, loss_fn="huber")
        with torch.no_grad():
            obs, nobs = torch.as_tensor(batch["obs"]), torch.as_tensor(batch["next_obs"])
            q = online(obs)[torch.arange(B), torch.as_tensor(batch["actions"])]
            if double:
                qn = target(nobs)[torch.arange(B), online(nobs).argmax(1)]
            else:
                qn = target(nobs).max(1).values
            y = torch.as_tensor(batch["returns"]) + torch.as_tensor(batch["discounts"]) * qn
            td = (q - y).abs()
            huber = torch.where(td < 1, 0.5 * td ** 2, td - 0.5)
            ref = (torch.as_tensor(batch["weights"]) * huber).mean()
        assert np.allclose(prios, td.numpy(), atol=1e-5)
        assert abs(float(loss) - float(ref)) < 1e-5, (float(loss), float(ref))
        loss.backward()
        assert all(p.grad is not None for p in online.parameters())
    print("test_dqn_loss_reference OK")


def test_noisy_eval_deterministic():
    cfg = dict(arch="nature", dueling=True, noisy=True, c51=False)
    net = build_network(6, cfg)
    x = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    net.set_noise(False)
    q1 = net.q_values(x)
    net.reset_noise()
    q2 = net.q_values(x)
    assert torch.allclose(q1, q2)
    net.set_noise(True)
    net.reset_noise()
    q3 = net.q_values(x)
    assert not torch.allclose(q1, q3)
    print("test_noisy_eval_deterministic OK")


if __name__ == "__main__":
    test_c51_projection_matches_reference()
    test_dqn_loss_reference()
    test_noisy_eval_deterministic()
    print("TODAS LAS PRUEBAS PASARON")
