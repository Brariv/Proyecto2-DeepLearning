#!/usr/bin/env python
"""Evaluación del agente entrenado + generación de video (día de la competencia).

Funciones (mismos nombres que el Laboratorio 5):
    crear_entorno(...)            -> entorno ALE/SpaceInvaders-v5 con el preprocesamiento del checkpoint
    cargar_agente(checkpoint)     -> política greedy (callable) a partir de los pesos guardados
    ejecutar_episodio(env, pol)   -> (puntaje real, pasos) de UN juego completo (3 vidas)
    generar_video_agente(pol, path) -> graba un episodio completo en .mp4 y devuelve el puntaje

Uso rápido (competencia: 5 episodios greedy + video):
    python evaluate.py --checkpoint runs/<run>/best.pt --episodes 5 --video videos/agente.mp4

Grabar TODOS los episodios de la evaluación (un .mp4 por episodio, con el puntaje en el nombre):
    python evaluate.py --checkpoint modelo_final/best.pt --episodes 5 --video-dir videos/evaluacion

Evaluar todos los checkpoints de una corrida para elegir el mejor:
    python evaluate.py --checkpoints-dir runs/<run> --episodes 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from si_rl.env import DEFAULT_ENV_CONFIG, FrameStacker
from si_rl.env import crear_entorno as _crear_entorno
from si_rl.networks import build_network
from si_rl.utils import get_device


def crear_entorno(env_config: dict | None = None, render_mode: str | None = None,
                  video_path: str | None = None, video_fps: int = 60):
    """Entorno de EVALUACIÓN: juego completo (3 vidas), recompensa sin clipping."""
    cfg = dict(DEFAULT_ENV_CONFIG)
    if env_config:
        cfg.update(env_config)
    return _crear_entorno(**cfg, episodic_life=False, clip_reward=False,
                          render_mode=render_mode, video_path=video_path, video_fps=video_fps)


class PoliticaGreedy:
    """Política greedy a partir de un checkpoint. Llamable: politica(frame) -> acción."""

    def __init__(self, checkpoint_path: str, device: str = "auto", epsilon: float = 0.0):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        self.env_config = ckpt.get("env_config", dict(DEFAULT_ENV_CONFIG))
        self.step = int(ckpt.get("step", 0))
        self.device = get_device(device)
        self.net = build_network(int(ckpt["num_actions"]), self.config).to(self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.net.set_noise(False)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(0)
        k = self.env_config.get("frame_stack", 4)
        s = self.env_config.get("screen_size", 84)
        self.stacker = FrameStacker(1, k, s, s)
        self.num_actions = int(ckpt["num_actions"])

    def reset(self, frame: np.ndarray) -> None:
        self.stacker.reset(frame[None])

    @torch.no_grad()
    def __call__(self, frame: np.ndarray, first: bool = False) -> int:
        if first:
            self.stacker.reset(frame[None])
        else:
            self.stacker.push(frame[None])
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        x = torch.as_tensor(self.stacker.obs, device=self.device)
        return int(self.net.q_values(x).argmax(dim=1).item())


def cargar_agente(checkpoint_path: str, device: str = "auto", epsilon: float = 0.0) -> PoliticaGreedy:
    return PoliticaGreedy(checkpoint_path, device=device, epsilon=epsilon)


def ejecutar_episodio(env, politica: PoliticaGreedy, seed: int | None = None, max_steps: int = 30_000):
    """Juega UN episodio completo (hasta perder las 3 vidas). Devuelve (puntaje real, pasos)."""
    frame, _ = env.reset(seed=seed)
    action = politica(frame, first=True)
    score, steps = 0.0, 0
    for _ in range(max_steps):
        frame, reward, terminated, truncated, info = env.step(action)
        score += float(reward)
        steps += 1
        if terminated or truncated:
            score = float(info.get("game_score", score))
            break
        action = politica(frame)
    return score, steps


def generar_video_agente(politica: PoliticaGreedy, video_path: str, seed: int | None = None, fps: int = 60):
    """Graba un episodio completo del agente en `video_path` (.mp4) y devuelve (puntaje, pasos)."""
    env = crear_entorno(politica.env_config, video_path=video_path, video_fps=fps)
    try:
        score, steps = ejecutar_episodio(env, politica, seed=seed)
    finally:
        env.close()  # escribe el video
    return score, steps


def evaluar(checkpoint: str, episodes: int, seed: int, device: str, epsilon: float, verbose: bool = True,
            video_dir: str | None = None, video_fps: int = 60):
    """Juega `episodes` juegos completos (semillas seed, seed+1, ...). Devuelve (puntajes, pasos, videos).

    Con video_dir, cada episodio se graba en su propio .mp4 mientras se juega: los videos son
    exactamente los episodios evaluados, y el nombre incluye el número de episodio, la semilla y el puntaje.
    """
    politica = cargar_agente(checkpoint, device=device, epsilon=epsilon)
    env = None if video_dir else crear_entorno(politica.env_config)
    scores, lengths, videos = [], [], []
    t0 = time.time()
    try:
        for i in range(episodes):
            ep_seed = seed + i
            if video_dir:
                tmp = os.path.join(video_dir, f"episodio_{i+1}_grabando.mp4")
                env_ep = crear_entorno(politica.env_config, video_path=tmp, video_fps=video_fps)
                try:
                    s, l = ejecutar_episodio(env_ep, politica, seed=ep_seed)
                finally:
                    env_ep.close()  # cierra el archivo de video
                final = os.path.join(video_dir, f"episodio_{i+1}_seed{ep_seed}_{int(s)}pts.mp4")
                os.replace(tmp, final)
                videos.append(final)
            else:
                s, l = ejecutar_episodio(env, politica, seed=ep_seed)
            scores.append(s)
            lengths.append(l)
            if verbose:
                extra = f"  -> {videos[-1]}" if video_dir else ""
                print(f"  episodio {i+1}/{episodes}: puntaje={s:.0f} pasos={l}{extra}", flush=True)
    finally:
        if env is not None:
            env.close()
    if verbose and scores:
        mejor = int(np.argmax(scores))
        print(f"  media={np.mean(scores):.1f}  max={np.max(scores):.0f} (episodio {mejor+1}, seed {seed+mejor})  "
              f"min={np.min(scores):.0f}  std={np.std(scores):.1f}  ({time.time()-t0:.0f}s)")
    return scores, lengths, videos


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="ruta a best.pt / last.pt / ckpt_*.pt")
    p.add_argument("--checkpoints-dir", help="evaluar todos los .pt de esta carpeta y resumir")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--epsilon", type=float, default=0.0, help="epsilon de evaluación (0 = greedy puro)")
    p.add_argument("--video", default=None, help="ruta del .mp4 a generar (un episodio completo)")
    p.add_argument("--video-fps", type=int, default=60)
    p.add_argument("--video-seed", type=int, default=None, help="semilla del episodio del video (default: seed)")
    p.add_argument("--video-dir", default=None,
                   help="grabar TODOS los episodios evaluados en esta carpeta (un .mp4 por episodio)")
    p.add_argument("--json", default=None, help="guardar resultados en este .json")
    args = p.parse_args()

    if args.checkpoints_dir:
        paths = sorted(glob.glob(os.path.join(args.checkpoints_dir, "*.pt")))
        assert paths, f"no hay checkpoints en {args.checkpoints_dir}"
        print(f"{len(paths)} checkpoints x {args.episodes} episodios. Ctrl+C detiene y muestra el ranking parcial.")
        results = []
        try:
            for k, path in enumerate(paths, 1):
                print(f"\n== [{k}/{len(paths)}] {path}", flush=True)
                scores, _, _ = evaluar(path, args.episodes, args.seed, args.device, args.epsilon, verbose=True)
                results.append(dict(checkpoint=path, mean=float(np.mean(scores)), max=float(np.max(scores)),
                                    min=float(np.min(scores)), std=float(np.std(scores)), scores=scores))
        except KeyboardInterrupt:
            print(f"\n[evaluate] interrumpido: ranking con los {len(results)} checkpoints ya evaluados.")
        results.sort(key=lambda r: r["mean"], reverse=True)
        print("\n== Ranking por media ==")
        for r in results:
            print(f"  {r['mean']:8.1f}  (max {r['max']:6.0f})  {r['checkpoint']}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)
        return

    assert args.checkpoint, "indica --checkpoint o --checkpoints-dir"
    print(f"== Evaluando {args.checkpoint} ({args.episodes} episodios, epsilon={args.epsilon})")
    scores, lengths, videos = evaluar(args.checkpoint, args.episodes, args.seed, args.device, args.epsilon,
                                      video_dir=args.video_dir, video_fps=args.video_fps)
    out = dict(checkpoint=args.checkpoint, episodes=args.episodes, seed=args.seed, epsilon=args.epsilon,
               scores=scores, lengths=lengths, mean=float(np.mean(scores)), max=float(np.max(scores)))
    if videos:
        out["videos"] = videos
    if args.video:
        politica = cargar_agente(args.checkpoint, device=args.device, epsilon=args.epsilon)
        vseed = args.seed if args.video_seed is None else args.video_seed
        print(f"== Generando video en {args.video} (seed={vseed}) ...")
        vscore, vsteps = generar_video_agente(politica, args.video, seed=vseed, fps=args.video_fps)
        print(f"   puntaje del episodio grabado: {vscore:.0f} ({vsteps} pasos)")
        out["video"] = dict(path=args.video, score=vscore, steps=vsteps, seed=vseed)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    print("\nRESUMEN:", json.dumps({k: out[k] for k in ("scores", "mean", "max")}))


main()
