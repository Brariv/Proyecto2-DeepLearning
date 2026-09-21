"""Replay buffer eficiente en memoria con retornos n-step y Prioritized Experience Replay.

Idea (como en Dopamine): en lugar de guardar la observación apilada (4x84x84) y la
siguiente (otras 4x84x84) por transición (~56 KB), se guarda UN solo frame de 84x84
(~7 KB) por paso y las observaciones apiladas se reconstruyen al muestrear leyendo los
4 frames consecutivos del mismo entorno.  Con N entornos en paralelo cada uno tiene su
propio "stream" circular, todos comparten el mismo cursor de escritura.

Por posición i (de un entorno) se guarda:
    frames[i]  -> observación (frame) con la que se eligió actions[i]
    actions[i], rewards[i] (ya con clipping), dones[i] (terminó el episodio tras la acción)
    starts[i]  -> frames[i] es el primer frame de un episodio (obs de reset)

Reconstrucción del stack en i: [frames[i-3], frames[i-2], frames[i-1], frames[i]] donde
los frames anteriores a un `start` se ponen en cero (misma regla que FrameStacker).
La observación siguiente de la transición n-step es el stack en i+n.

Posiciones NO muestreables: las n más nuevas (aún no existe su futuro) y, con el buffer
lleno, las 3 más viejas (su pasado ya fue sobreescrito).  Con PER sus prioridades se
mantienen en 0 para que el sum-tree nunca las elija.
"""
from __future__ import annotations

import math

import numpy as np


class SumTree:
    """Árbol de sumas vectorizado (búsqueda por prefijo en O(log n) para un batch entero)."""

    def __init__(self, size: int):
        self.size = size
        self.depth = max(1, math.ceil(math.log2(size)))
        self.capacity = 1 << self.depth
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    def set(self, idx: np.ndarray, values: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        nodes = idx + self.capacity
        self.tree[nodes] = values
        nodes = np.unique(nodes // 2)
        while True:
            self.tree[nodes] = self.tree[2 * nodes] + self.tree[2 * nodes + 1]
            if nodes[0] == 1:
                break
            nodes = np.unique(nodes // 2)

    def get(self, idx: np.ndarray) -> np.ndarray:
        return self.tree[np.asarray(idx, dtype=np.int64) + self.capacity]

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def find(self, values: np.ndarray) -> np.ndarray:
        """Para cada v devuelve la hoja i tal que sum(p[:i]) <= v < sum(p[:i+1])."""
        values = np.array(values, dtype=np.float64)
        idx = np.ones(len(values), dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * idx
            left_sum = self.tree[left]
            go_left = values < left_sum
            idx = np.where(go_left, left, left + 1)
            values = np.where(go_left, values, values - left_sum)
        return np.minimum(idx - self.capacity, self.size - 1)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_shape: tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        n_step: int = 3,
        gamma: float = 0.99,
        prioritized: bool = False,
        alpha: float = 0.5,
        priority_eps: float = 1e-6,
        seed: int = 0,
    ):
        self.N = num_envs
        self.cap = capacity // num_envs
        self.S = frame_stack
        self.n = n_step
        self.gamma = gamma
        self.gamma_n = gamma ** n_step
        self.prioritized = prioritized
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.rng = np.random.default_rng(seed)
        assert self.cap > self.n + self.S + 1, "capacidad demasiado pequeña"

        H, W = obs_shape
        self.frames = np.zeros((num_envs, self.cap, H, W), dtype=np.uint8)
        self.actions = np.zeros((num_envs, self.cap), dtype=np.int64)
        self.rewards = np.zeros((num_envs, self.cap), dtype=np.float32)
        self.dones = np.zeros((num_envs, self.cap), dtype=bool)
        self.starts = np.zeros((num_envs, self.cap), dtype=bool)
        self.pos = 0
        self.full = False
        self._gammas = (gamma ** np.arange(n_step)).astype(np.float32)
        self._env_offsets = np.arange(num_envs, dtype=np.int64) * self.cap

        if prioritized:
            self.tree = SumTree(num_envs * self.cap)
            self.max_priority = 1.0

    # ------------------------------------------------------------------ escritura
    def __len__(self) -> int:
        return (self.cap if self.full else self.pos) * self.N

    @property
    def num_valid_per_env(self) -> int:
        if self.full:
            return self.cap - self.n - (self.S - 1)
        return max(0, self.pos - self.n)

    def add(self, frames, actions, rewards, dones, starts) -> None:
        p = self.pos
        self.frames[:, p] = frames
        self.actions[:, p] = actions
        self.rewards[:, p] = rewards
        self.dones[:, p] = dones
        self.starts[:, p] = starts

        if self.prioritized:
            if self.full or p >= self.n:
                # la posición p-n acaba de volverse muestreable (ya tiene n pasos de futuro)
                flat = self._env_offsets + (p - self.n) % self.cap
                self.tree.set(flat, np.full(self.N, self.max_priority))
            if self.full:
                # la posición p+S-1 pasa a ser una de las S-1 más viejas: su pasado se perdió
                flat = self._env_offsets + (p + self.S - 1) % self.cap
                self.tree.set(flat, np.zeros(self.N))

        self.pos = (p + 1) % self.cap
        if self.pos == 0 and not self.full:
            self.full = True
            if self.prioritized:
                for k in range(self.S - 1):
                    self.tree.set(self._env_offsets + k, np.zeros(self.N))

    # ------------------------------------------------------------------ lectura
    def _stack_at(self, env_idx: np.ndarray, idx: np.ndarray) -> np.ndarray:
        offs = np.arange(-(self.S - 1), 1)
        pos = (idx[:, None] + offs[None, :]) % self.cap            # (B, S)
        frames = self.frames[env_idx[:, None], pos]               # (B, S, H, W)
        st = self.starts[env_idx[:, None], pos]                   # (B, S)
        valid = np.ones_like(st)
        seen = np.zeros(len(idx), dtype=bool)
        for k in range(self.S - 1, -1, -1):                       # del más nuevo al más viejo
            valid[:, k] = ~seen
            seen |= st[:, k]
        if not valid.all():
            frames[~valid] = 0  # frames de un episodio anterior -> ceros (regla de FrameStacker)
        return frames

    def _is_valid(self, idx: np.ndarray) -> np.ndarray:
        d = (self.pos - 1 - idx) % self.cap  # antigüedad: 0 = más nuevo
        ok = d >= self.n
        if self.full:
            ok &= d <= self.cap - self.S
        else:
            ok &= idx < self.pos
        return ok

    def _sample_uniform(self, batch_size: int):
        env_idx = self.rng.integers(self.N, size=batch_size)
        u = self.rng.integers(self.num_valid_per_env, size=batch_size)
        if self.full:
            idx = (self.pos + self.S - 1 + u) % self.cap
        else:
            idx = u
        return env_idx, idx

    def sample(self, batch_size: int, beta: float = 0.4) -> dict:
        assert self.num_valid_per_env > 0, "el buffer aún no tiene transiciones muestreables"
        if self.prioritized:
            total = self.tree.total
            segment = total / batch_size
            vals = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
            flat = self.tree.find(vals)
            env_idx, idx = flat // self.cap, flat % self.cap
            bad = ~self._is_valid(idx) | (self.tree.get(flat) <= 0)
            if bad.any():  # redondeo numérico: reemplazar por muestras uniformes válidas
                e2, i2 = self._sample_uniform(int(bad.sum()))
                env_idx[bad], idx[bad] = e2, i2
                flat = env_idx * self.cap + idx
            probs = self.tree.get(flat) / total
            n_valid = self.num_valid_per_env * self.N
            weights = (n_valid * probs) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        else:
            env_idx, idx = self._sample_uniform(batch_size)
            flat = env_idx * self.cap + idx
            weights = np.ones(batch_size, dtype=np.float32)

        batch = self.get_transitions(env_idx, idx)
        batch["weights"] = weights
        batch["flat_idx"] = flat
        return batch

    def get_transitions(self, env_idx: np.ndarray, idx: np.ndarray) -> dict:
        """Transiciones n-step (obs, acción, retorno, next_obs, descuento) para índices dados."""
        env_idx = np.asarray(env_idx, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        obs = self._stack_at(env_idx, idx)
        next_obs = self._stack_at(env_idx, (idx + self.n) % self.cap)

        offs = np.arange(self.n)
        pos = (idx[:, None] + offs[None, :]) % self.cap             # (B, n)
        r = self.rewards[env_idx[:, None], pos]                      # (B, n)
        d = self.dones[env_idx[:, None], pos]                        # (B, n)
        alive = np.ones_like(r)
        if self.n > 1:
            alive[:, 1:] = np.cumprod(1.0 - d[:, :-1], axis=1)
        returns = (r * alive * self._gammas[None, :]).sum(axis=1).astype(np.float32)
        done_any = d.any(axis=1)
        discounts = (self.gamma_n * (~done_any)).astype(np.float32)

        return dict(
            obs=obs,
            actions=self.actions[env_idx, idx],
            returns=returns,
            next_obs=next_obs,
            discounts=discounts,   # gamma^n si no hubo terminal en la ventana, 0 si lo hubo
        )

    def update_priorities(self, flat_idx: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.prioritized:
            return
        p = (np.abs(td_errors) + self.priority_eps) ** self.alpha
        self.tree.set(flat_idx, p)
        self.max_priority = max(self.max_priority, float(p.max()))
