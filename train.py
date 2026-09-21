#!/usr/bin/env python
"""Entrenamiento de un agente DQN / Rainbow para ALE/SpaceInvaders-v5.

Ejemplos:
    python train.py --preset dqn      --total-steps 5000000
    python train.py --preset rainbow  --run-name it5_rainbow
    python train.py --resume runs/it6_rainbow_fast/last.pt --run-name it6_rainbow_fast --total-steps 20000000
    python train.py --help

El código de entrenamiento está en si_rl/trainer.py. Este archivo es solo el punto de entrada y
no importa nada pesado a propósito: en Windows y macOS los entornos corren en subprocesos creados
con "spawn", que vuelven a ejecutar este archivo al arrancar. Si aquí se importara PyTorch, cada
uno de los N subprocesos cargaría PyTorch y CUDA, lo que en Windows agota la memoria virtual
(OSError: [WinError 1455] El archivo de paginación es demasiado pequeño) y hace lento el arranque.
"""

if __name__ == "__main__":
    from si_rl.trainer import main

    main()
