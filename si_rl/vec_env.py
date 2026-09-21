"""Entornos vectorizados mínimos (implementación propia, sin dependencias externas).

Semántica de `step(actions)` -> (obs, rewards, dones, infos):
    * `dones[i]` = terminated or truncated del entorno i.
    * Si `dones[i]` es True, el entorno i ya fue reseteado y `obs[i]` es la PRIMERA
      observación del nuevo episodio (el agente no necesita la observación final:
      en un transition terminal no se hace bootstrapping).
    * `infos[i]` conserva las claves de interés del paso terminal (game_score, ...).

`step_async` / `step_wait` permiten solapar la simulación de los entornos con la
actualización de la red (los subprocesos avanzan mientras la GPU entrena).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Callable, Sequence

import numpy as np

INFO_KEYS = ("game_score", "game_length")


def _filter_info(info: dict) -> dict:
    return {k: info[k] for k in INFO_KEYS if k in info}


class DummyVecEnv:
    """Todos los entornos en el mismo proceso (útil para evaluación y debugging)."""

    def __init__(self, env_fns: Sequence[Callable]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._actions = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=None if seed is None else seed + i)
            obs.append(o)
        return np.stack(obs)

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        obs, rews, dones, infos = [], [], [], []
        for env, a in zip(self.envs, self._actions):
            o, r, term, trunc, info = env.step(int(a))
            done = term or trunc
            info = _filter_info(info)
            if done:
                o, _ = env.reset()
            obs.append(o)
            rews.append(r)
            dones.append(done)
            infos.append(info)
        return np.stack(obs), np.asarray(rews, dtype=np.float32), np.asarray(dones, dtype=bool), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        for env in self.envs:
            env.close()


def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = None
    try:
        env = env_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                o, r, term, trunc, info = env.step(int(data))
                done = term or trunc
                info = _filter_info(info)
                if done:
                    o, _ = env.reset()
                remote.send((o, r, done, info))
            elif cmd == "reset":
                o, _ = env.reset(seed=data)
                remote.send(o)
            elif cmd == "spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError, BrokenPipeError, ConnectionResetError):
        # Ctrl+C (en Windows llega a todos los procesos de la consola) o el proceso principal terminó
        pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            remote.close()
        except OSError:
            pass


class SubprocVecEnv:
    """Un subproceso por entorno, comunicación por pipes (estilo baselines, simplificado)."""

    def __init__(self, env_fns: Sequence[Callable], start_method: str | None = None):
        self.num_envs = len(env_fns)
        if start_method is None:
            # Linux: fork (rápido). Windows solo tiene spawn y en macOS fork no es seguro.
            # SI_RL_START_METHOD permite forzarlo (p. ej. para probar spawn en Linux).
            start_method = os.environ.get("SI_RL_START_METHOD") or (
                "fork" if sys.platform.startswith("linux") else "spawn")
        ctx = mp.get_context(start_method)
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.procs = []
        for wr, r, fn in zip(work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(wr, r, fn), daemon=True)
            p.start()
            self.procs.append(p)
            wr.close()
        self.remotes[0].send(("spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.waiting = False
        self.closed = False

    def reset(self, seed: int | None = None) -> np.ndarray:
        for i, r in enumerate(self.remotes):
            r.send(("reset", None if seed is None else seed + i))
        return np.stack([r.recv() for r in self.remotes])

    def step_async(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        self.waiting = True

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.asarray(rews, dtype=np.float32), np.asarray(dones, dtype=bool), list(infos)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.waiting:
            # vaciar respuestas pendientes sin bloquear (algunas ya pudieron leerse antes de una interrupción)
            for r in self.remotes:
                try:
                    if r.poll(1.0):
                        r.recv()
                except (EOFError, OSError):
                    pass
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (EOFError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def make_vec_env(env_fns: Sequence[Callable], subproc: bool = True):
    if subproc and len(env_fns) > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
