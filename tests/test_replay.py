"""Pruebas del replay buffer contra una implementación de referencia ingenua.

Ejecutar:  python -m pytest tests/ -q     (o simplemente  python tests/test_replay.py)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from si_rl.env import FrameStacker  # noqa: E402
from si_rl.replay import ReplayBuffer, SumTree  # noqa: E402


def simulate(num_envs, steps, n_step, gamma, S, cap_total, prioritized, seed, p_done=0.15, H=3, W=3):
    """Simula streams aleatorios, guarda todo en el buffer y en una referencia (listas)."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(cap_total, num_envs, (H, W), S, n_step, gamma, prioritized=prioritized, seed=seed)
    stacker = FrameStacker(num_envs, S, H, W)
    # referencia: por env, listas de (frame, stack, a, r, done, start)
    ref = [dict(frames=[], stacks=[], a=[], r=[], done=[], start=[]) for _ in range(num_envs)]
    obs = rng.integers(1, 256, size=(num_envs, H, W), dtype=np.uint8)
    stacker.reset(obs)
    starts = np.ones(num_envs, dtype=bool)
    for t in range(steps):
        a = rng.integers(6, size=num_envs)
        r = rng.choice([-1.0, 0.0, 1.0], size=num_envs).astype(np.float32)
        d = rng.random(num_envs) < p_done
        for e in range(num_envs):
            ref[e]["frames"].append(obs[e].copy())
            ref[e]["stacks"].append(stacker.obs[e].copy())
            ref[e]["a"].append(int(a[e]))
            ref[e]["r"].append(float(r[e]))
            ref[e]["done"].append(bool(d[e]))
            ref[e]["start"].append(bool(starts[e]))
        buf.add(obs, a, r, d, starts)
        obs = rng.integers(1, 256, size=(num_envs, H, W), dtype=np.uint8)
        stacker.push(obs, d)
        starts = d.copy()
    return buf, ref, stacker


def reference_transition(ref_e, t, n, gamma):
    R, disc, done_any = 0.0, gamma ** n, False
    for k in range(n):
        R += (gamma ** k) * ref_e["r"][t + k]
        if ref_e["done"][t + k]:
            done_any = True
            break
    if done_any:
        disc = 0.0
    next_stack = ref_e["stacks"][t + n]
    return ref_e["stacks"][t], ref_e["a"][t], np.float32(R), next_stack, np.float32(disc)


def check_buffer(buf, ref, steps):
    """Compara TODAS las posiciones válidas del buffer con la referencia."""
    cap, n, S = buf.cap, buf.n, buf.S
    checked = 0
    for e in range(buf.N):
        for idx in range(cap):
            if not buf._is_valid(np.array([idx]))[0]:
                continue
            # tiempo absoluto de la posición idx
            if buf.full:
                age = (buf.pos - 1 - idx) % cap
                t = steps - 1 - age
            else:
                t = idx
            assert 0 <= t < steps
            got = buf.get_transitions(np.array([e]), np.array([idx]))
            obs, a, R, nxt, disc = reference_transition(ref[e], t, n, buf.gamma)
            assert np.array_equal(got["obs"][0], obs), f"obs distinta e={e} idx={idx} t={t}"
            assert got["actions"][0] == a
            assert abs(got["returns"][0] - R) < 1e-5, (got["returns"][0], R)
            assert got["discounts"][0] == disc, (got["discounts"][0], disc)
            if disc > 0:
                assert np.array_equal(got["next_obs"][0], nxt), f"next_obs distinta e={e} idx={idx} t={t}"
            checked += 1
    return checked


def test_transitions_match_reference():
    total_checked = 0
    for seed, (num_envs, cap_total, n_step, S) in enumerate([(2, 40, 3, 4), (3, 60, 1, 4), (1, 30, 5, 4), (4, 80, 3, 2)]):
        for steps in (7, 12, 19, 25, 37, 60):
            buf, ref, _ = simulate(num_envs, steps, n_step, 0.99, S, cap_total, False, seed)
            n_valid = buf.num_valid_per_env
            checked = check_buffer(buf, ref, steps)
            assert checked == n_valid * num_envs, (checked, n_valid, num_envs)
            total_checked += checked
    print(f"test_transitions_match_reference OK ({total_checked} transiciones verificadas)")


def test_uniform_sampler_range():
    for steps in (10, 30, 100):
        buf, ref, _ = simulate(2, steps, 3, 0.99, 4, 40, False, seed=steps)
        e, i = buf._sample_uniform(2000)
        assert buf._is_valid(i).all()
        # cubre todas las posiciones válidas
        assert len(set(i.tolist())) == buf.num_valid_per_env
    print("test_uniform_sampler_range OK")


def test_per_priorities_zero_on_invalid():
    for steps in (5, 10, 23, 40, 77):
        buf, ref, _ = simulate(2, steps, 3, 0.99, 4, 40, True, seed=steps)
        for e in range(buf.N):
            for idx in range(buf.cap):
                p = buf.tree.get(np.array([e * buf.cap + idx]))[0]
                if buf._is_valid(np.array([idx]))[0]:
                    assert p > 0, (steps, e, idx)
                else:
                    assert p == 0, (steps, e, idx, p)
        if buf.num_valid_per_env > 0:
            batch = buf.sample(64, beta=0.4)
            assert buf._is_valid(batch["flat_idx"] % buf.cap).all()
            assert batch["weights"].max() <= 1.0 + 1e-6
            buf.update_priorities(batch["flat_idx"], np.random.rand(64))
            # tras actualizar prioridades, las posiciones válidas siguen > 0 y las inválidas 0
            for e in range(buf.N):
                for idx in range(buf.cap):
                    p = buf.tree.get(np.array([e * buf.cap + idx]))[0]
                    assert (p > 0) == bool(buf._is_valid(np.array([idx]))[0])
    print("test_per_priorities_zero_on_invalid OK")


def test_sumtree():
    rng = np.random.default_rng(0)
    tree = SumTree(37)
    p = rng.random(37)
    p[[3, 10, 36]] = 0.0
    tree.set(np.arange(37), p)
    assert abs(tree.total - p.sum()) < 1e-9
    vals = rng.random(5000) * tree.total
    found = tree.find(vals)
    cum = np.cumsum(p)
    expected = np.searchsorted(cum, vals, side="right")
    assert np.array_equal(found, expected), (found[:10], expected[:10])
    # actualización con índices duplicados
    tree.set(np.array([5, 5, 7]), np.array([1.0, 2.0, 3.0]))
    p[5], p[7] = 2.0, 3.0
    assert abs(tree.total - p.sum()) < 1e-9
    print("test_sumtree OK")


if __name__ == "__main__":
    test_sumtree()
    test_transitions_match_reference()
    test_uniform_sampler_range()
    test_per_priorities_zero_on_invalid()
    print("TODAS LAS PRUEBAS PASARON")
