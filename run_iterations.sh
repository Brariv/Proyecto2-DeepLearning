#!/usr/bin/env bash
# Plan de iteraciones sugerido (ejecutar una por una; cada una deja su carpeta en runs/).
# Ajusta --total-steps según el tiempo disponible. Ctrl+C guarda last.pt y se puede reanudar con --resume.
set -e

# Iteraciones cortas de comparación (2M pasos = 8M frames c/u)
python train.py --preset dqn       --run-name it1_dqn       --total-steps 2_000_000
python train.py --preset ddqn      --run-name it2_ddqn      --total-steps 2_000_000
python train.py --preset dueling   --run-name it3_dueling   --total-steps 2_000_000
python train.py --preset per_nstep --run-name it4_per_nstep --total-steps 2_000_000
python train.py --preset rainbow   --run-name it5_rainbow   --total-steps 2_000_000

# Candidato final: Rainbow eficiente (IMPALA CNN). Cuantos más pasos, mejor (10M+).
python train.py --preset rainbow_fast --run-name it6_rainbow_fast --total-steps 10_000_000

# Comparar y graficar todo para el informe
python plot_results.py runs/it* --out figs

# Elegir el mejor checkpoint del candidato final (10 episodios por checkpoint)
python evaluate.py --checkpoints-dir runs/it6_rainbow_fast --episodes 10 --json figs/ranking_ckpts.json

# Competencia: 5 episodios greedy + video
python evaluate.py --checkpoint runs/it6_rainbow_fast/best.pt --episodes 5 --video videos/agente_final.mp4
