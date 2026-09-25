#!/usr/bin/env python
"""Video del progreso del entrenamiento: un clip por checkpoint, todos en un solo .mp4.

Carga varios checkpoints de una corrida, juega un episodio con cada uno (recortado a
--max-steps para que el video no se alargue) y escribe todos los clips seguidos en un
mismo archivo, con un rótulo que muestra el paso de entrenamiento y el puntaje acumulado.

Ejemplos:
    # checkpoints elegidos a mano
    python video_progreso.py --run runs/it6_rainbow_fast --steps 500000 1000000 2500000 4500000 7500000

    # los N checkpoints repartidos a lo largo de la corrida
    python video_progreso.py --run runs/it6_rainbow_fast --auto 5

Los cuadros se escriben al archivo conforme se generan, así que la memoria no crece.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np
import torch

from evaluate import cargar_agente, crear_entorno


def rotular(frame: np.ndarray, texto: str, escala: int = 3) -> np.ndarray:
    """Amplía el frame del Atari y le dibuja una franja con el texto."""
    grande = cv2.resize(frame, (frame.shape[1] * escala, frame.shape[0] * escala),
                        interpolation=cv2.INTER_NEAREST)
    alto_franja = 34
    lienzo = np.zeros((grande.shape[0] + alto_franja, grande.shape[1], 3), dtype=np.uint8)
    lienzo[alto_franja:] = grande
    cv2.putText(lienzo, texto, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return lienzo


def clip_de_checkpoint(writer, checkpoint: str, etiqueta: str, device: str, max_steps: int,
                       seed: int, congelar: int) -> tuple[float, int]:
    """Juega un episodio con ese checkpoint y escribe sus cuadros en el writer abierto."""
    politica = cargar_agente(checkpoint, device=device)
    env = crear_entorno(politica.env_config, render_mode="rgb_array")
    frame, _ = env.reset(seed=seed)
    accion = politica(frame, first=True)
    score, pasos, ultimo = 0.0, 0, None
    while pasos < max_steps:
        frame, reward, terminated, truncated, info = env.step(accion)
        score += float(reward)
        pasos += 1
        ultimo = rotular(env.render(), f"{etiqueta}  |  puntaje {int(score)}")
        writer.append_data(ultimo)
        if terminated or truncated:
            break
        accion = politica(frame)
    env.close()
    for _ in range(congelar):  # dejar el último cuadro fijo un momento
        writer.append_data(ultimo)
    return score, pasos


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="carpeta de la corrida (runs/<nombre>)")
    p.add_argument("--steps", type=int, nargs="*", default=None,
                   help="pasos de los checkpoints a incluir, p. ej. 500000 2500000 7500000")
    p.add_argument("--auto", type=int, default=None, help="usar N checkpoints repartidos en la corrida")
    p.add_argument("--out", default="videos/progreso_entrenamiento.mp4")
    p.add_argument("--max-steps", type=int, default=600, help="pasos máximos por clip (600 = 40 s de video)")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0, help="misma semilla en todos los clips (partidas comparables)")
    p.add_argument("--device", default="auto")
    p.add_argument("--freeze", type=int, default=45, help="cuadros congelados al final de cada clip")
    args = p.parse_args()

    disponibles = {}
    for ruta in sorted(glob.glob(os.path.join(args.run, "ckpt_*.pt"))):
        m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(ruta))
        if m:
            disponibles[int(m.group(1))] = ruta
    assert disponibles, f"no hay checkpoints ckpt_*.pt en {args.run}"

    if args.steps:
        elegidos = []
        for s in args.steps:
            cercano = min(disponibles, key=lambda k: abs(k - s))
            elegidos.append(cercano)
    elif args.auto:
        claves = sorted(disponibles)
        idx = np.linspace(0, len(claves) - 1, args.auto).round().astype(int)
        elegidos = [claves[i] for i in dict.fromkeys(idx)]
    else:
        elegidos = sorted(disponibles)
    elegidos = sorted(dict.fromkeys(elegidos))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import imageio.v2 as imageio

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                               pixelformat="yuv420p", macro_block_size=1)
    print(f"== {len(elegidos)} clips -> {args.out}")
    try:
        for paso in elegidos:
            etiqueta = f"{paso/1e6:.1f}M pasos" if paso >= 1e6 else f"{paso/1e3:.0f}k pasos"
            score, pasos = clip_de_checkpoint(writer, disponibles[paso], etiqueta, args.device,
                                              args.max_steps, args.seed, args.freeze)
            print(f"   {etiqueta:>12}: puntaje {score:.0f} en {pasos} pasos", flush=True)
    finally:
        writer.close()
    dur = None
    try:
        dur = imageio.get_reader(args.out).get_meta_data().get("duration")
    except Exception:
        pass
    print(f"Video guardado en {args.out}" + (f" ({dur:.0f} s)" if dur else ""))


if __name__ == "__main__":
    main()
