"""Creación del entorno ALE/SpaceInvaders-v5 y wrappers de preprocesamiento.

Pipeline (de adentro hacia afuera):

    gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.25)
      -> VideoRecorder      (opcional, graba TODOS los frames crudos a 60 fps)
      -> AtariPreprocessing (frame_skip=4 + max-pool de los 2 últimos frames,
                             escala de grises, resize 84x84, no-op reset opcional)
      -> ScoreTracker       (acumula el puntaje REAL del juego, sin clipping, con
                             las 3 vidas; lo publica en info["game_score"] al terminar)
      -> EpisodicLife       (solo entrenamiento: perder una vida = terminal para el
                             aprendizaje, pero el juego continúa)
      -> ClipReward         (solo entrenamiento: recompensa -> sign(r))

El apilado de 4 frames NO se hace con un wrapper: lo hace `FrameStacker` del lado
del agente (y el replay buffer lo reconstruye igual), para ahorrar memoria.

IMPORTANTE: el entorno se crea con frameskip=1 porque `AtariPreprocessing` hace el
frame-skip de 4 con max-pooling (necesario en Space Invaders: los disparos parpadean
en frames alternos). La dinámica del juego es la misma que ALE/SpaceInvaders-v5 por
defecto (4 frames por acción, sticky actions p=0.25, 6 acciones, 108000 frames máx).
"""
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable

import gymnasium as gym
import numpy as np

import ale_py

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"

# Configuración por defecto del preprocesamiento. Se guarda dentro del checkpoint
# para que la evaluación use EXACTAMENTE el mismo pipeline que el entrenamiento.
DEFAULT_ENV_CONFIG = dict(
    env_id=ENV_ID,
    frame_skip=4,
    screen_size=84,
    frame_stack=4,
    noop_max=0,                      # v5 ya es estocástico por sticky actions
    repeat_action_probability=0.25,  # valor por defecto de la versión v5
    full_action_space=False,         # 6 acciones: NOOP FIRE RIGHT LEFT RIGHTFIRE LEFTFIRE
    max_episode_frames=108_000,
)


class ScoreTracker(gym.Wrapper):
    """Acumula el puntaje real del juego (recompensa sin clipping, todas las vidas).

    Al terminar el juego (terminated o truncated del entorno real) agrega a `info`:
        game_score  -> puntaje total del episodio completo
        game_length -> pasos de agente del episodio completo
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.score = 0.0
        self.length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.score = 0.0
        self.length = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.score += float(reward)
        self.length += 1
        if terminated or truncated:
            info = dict(info)
            info["game_score"] = self.score
            info["game_length"] = self.length
        return obs, reward, terminated, truncated, info


class EpisodicLife(gym.Wrapper):
    """Perder una vida se señala como `terminated` (corta el bootstrapping del valor),
    pero el juego NO se reinicia hasta que se pierdan todas las vidas.
    Truco estándar de DeepMind (Mnih et al. 2015) reimplementado desde cero.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op para avanzar desde el estado de "vida perdida".
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipReward(gym.RewardWrapper):
    """Recompensa -> {-1, 0, +1} (Mnih et al. 2015). Solo para entrenamiento."""

    def reward(self, reward):
        return float(np.sign(reward))


class VideoRecorder(gym.Wrapper):
    """Graba cada frame crudo (210x160 RGB) del entorno base y escribe un .mp4 al cerrar.

    Se coloca DEBAJO de AtariPreprocessing para capturar los 4 frames de cada paso del
    agente (video fluido a 60 fps, la velocidad real del Atari 2600).
    Requiere render_mode="rgb_array" en el entorno base.
    """

    def __init__(self, env: gym.Env, path: str, fps: int = 60):
        super().__init__(env)
        self.path = path
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self._closed = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.append(self.env.render())
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(self.env.render())
        return out

    def save(self) -> str:
        if not self.frames:
            return self.path
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(
                self.path, fps=self.fps, codec="libx264", quality=8,
                pixelformat="yuv420p", macro_block_size=1,
            ) as writer:
                for f in self.frames:
                    writer.append_data(f)
        except Exception as exc:  # fallback a OpenCV (mp4v)
            print(f"[VideoRecorder] imageio/ffmpeg no disponible ({exc}); usando OpenCV.")
            import cv2

            h, w = self.frames[0].shape[:2]
            vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for f in self.frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
        print(f"[VideoRecorder] video guardado en {self.path} ({len(self.frames)} frames, {self.fps} fps)")
        return self.path

    def close(self):
        if not self._closed:
            self._closed = True
            self.save()
        return self.env.close()


def crear_entorno(
    env_id: str = ENV_ID,
    *,
    frame_skip: int = 4,
    screen_size: int = 84,
    noop_max: int = 0,
    repeat_action_probability: float = 0.25,
    full_action_space: bool = False,
    max_episode_frames: int = 108_000,
    episodic_life: bool = False,
    clip_reward: bool = False,
    render_mode: str | None = None,
    video_path: str | None = None,
    video_fps: int = 60,
    frame_stack: int = 4,  # aceptado por compatibilidad con env_config (el stack lo hace el agente)
    **_ignored: Any,
) -> gym.Env:
    """Crea el entorno con el pipeline de preprocesamiento completo.

    Para ENTRENAR: episodic_life=True, clip_reward=True.
    Para EVALUAR (puntaje real, 3 vidas): episodic_life=False, clip_reward=False.
    """
    from gymnasium.wrappers import AtariPreprocessing

    if video_path is not None:
        render_mode = "rgb_array"
    env = gym.make(
        env_id,
        frameskip=1,  # el skip lo hace AtariPreprocessing (con max-pool)
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        max_num_frames_per_episode=max_episode_frames,
        render_mode=render_mode,
    )
    if video_path is not None:
        env = VideoRecorder(env, video_path, fps=video_fps)
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = ScoreTracker(env)
    if episodic_life:
        env = EpisodicLife(env)
    if clip_reward:
        env = ClipReward(env)
    return env


def make_env_fn(env_config: dict, **kwargs) -> Callable[[], gym.Env]:
    """Devuelve una función sin argumentos que crea el entorno (picklable, para subprocesos)."""
    cfg = dict(DEFAULT_ENV_CONFIG)
    cfg.update(env_config)
    cfg.update(kwargs)
    return partial(crear_entorno, **cfg)


class FrameStacker:
    """Mantiene el stack de los últimos `k` frames para N entornos (uint8, N x k x H x W).

    Al inicio de un episodio (o tras perder una vida en entrenamiento) el stack se
    rellena con ceros: exactamente la misma regla que usa el replay buffer para
    reconstruir observaciones, de modo que la red ve lo mismo online y en el batch.
    """

    def __init__(self, num_envs: int, k: int, height: int, width: int):
        self.k = k
        self.stack = np.zeros((num_envs, k, height, width), dtype=np.uint8)

    def reset(self, frames: np.ndarray) -> np.ndarray:
        self.stack[:] = 0
        self.stack[:, -1] = frames
        return self.stack

    def push(self, frames: np.ndarray, starts: np.ndarray | None = None) -> np.ndarray:
        """Agrega el frame nuevo. `starts[i]=True` => `frames[i]` es el primer frame de un episodio."""
        if starts is not None and np.any(starts):
            self.stack[starts] = 0
        self.stack[:, :-1] = self.stack[:, 1:]
        self.stack[:, -1] = frames
        return self.stack

    @property
    def obs(self) -> np.ndarray:
        return self.stack
