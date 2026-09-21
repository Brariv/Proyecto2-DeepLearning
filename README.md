# Proyecto 2 – Agente de RL para `ALE/SpaceInvaders-v5` (CC3092 Deep Learning y Sistemas Inteligentes)

Agente de Aprendizaje por Refuerzo de la familia **DQN → Rainbow** implementado desde cero en PyTorch
(sin Stable-Baselines ni otras librerías de RL), con todo lo necesario para:

* entrenar con distintas configuraciones (una por iteración del informe) y **reanudar** si se corta la sesión,
* registrar automáticamente cada corrida (CSV + TensorBoard) y **graficar** las curvas para el informe,
* **evaluar** el agente con política greedy sobre episodios completos (3 vidas, puntaje real del juego) y
* **generar el video** `.mp4` del agente jugando, con exactamente el mismo preprocesamiento del entrenamiento.

Componentes implementados (todos activables por bandera): DQN, Double DQN, Dueling, Prioritized Experience
Replay, retornos *n-step*, NoisyNets, C51 (distribucional), Nature CNN e IMPALA CNN (con spectral norm),
mixed precision (AMP) en GPU.

---

## 1. Estructura

```
proyecto2/
├── train.py              entrenamiento (presets + overrides por CLI, checkpoints, resume, eval periódica)
├── evaluate.py           evaluación greedy (5 episodios) + video .mp4  [crear_entorno / ejecutar_episodio / generar_video_agente]
├── plot_results.py       curvas y tabla resumen de todas las iteraciones (para el informe)
├── run_iterations.sh     secuencia sugerida de corridas
├── si_rl/
│   ├── env.py            crear_entorno(): wrappers de preprocesamiento (implementación propia + AtariPreprocessing)
│   ├── vec_env.py        entornos vectorizados por subprocesos (implementación propia)
│   ├── replay.py         replay buffer eficiente en memoria: n-step + PER con sum-tree vectorizado
│   ├── networks.py       NatureCNN, ImpalaCNN, NoisyLinear, cabeza dueling / C51
│   ├── agent.py          selección de acciones y pérdida (Huber DQN/Double, cross-entropy C51)
│   ├── presets.py        hiperparámetros de cada iteración
│   └── utils.py          dispositivo (cuda/mps/cpu), semillas, loggers
├── tests/test_replay.py  pruebas del replay buffer contra una implementación de referencia
├── colab/entrenar_colab.ipynb   cuaderno para Colab/Kaggle (checkpoints en Drive)
├── requirements.txt
└── modelo_final/         (crear al final) best.pt del agente entregado
```

## 2. Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # (opcional)
pip install -r requirements.txt
python tests/test_replay.py                            # debe imprimir "TODAS LAS PRUEBAS PASARON"
```

* **Mac (Apple Silicon)**: PyTorch usa la GPU vía `mps` automáticamente (`--device auto`). `ale-py` trae
  wheels para macOS arm64. Usa el preset `rainbow_fast_small` (16 entornos, batch 128, IMPALA ×1). Si alguna
  operación no está implementada en MPS, exporta `PYTORCH_ENABLE_MPS_FALLBACK=1` antes de correr; como último
  recurso `--device cpu`. Conecta el cargador: el Air limita la frecuencia por temperatura.
* **Google Colab / Kaggle (GPU T4/P100)**: abre `colab/entrenar_colab.ipynb`. Colab tiene 2 vCPUs, así que la
  simulación de los entornos es el cuello de botella; aun así `rainbow_fast` corre a ~600-900 pasos/s.
* **PC de escritorio con GPU NVIDIA (Windows/Linux)**: instala PyTorch con CUDA desde pytorch.org y el resto
  con `pip install -r requirements.txt`. Preset recomendado: `rainbow_fast` (usa AMP automáticamente en CUDA).
  Pon `--num-envs` ≈ núcleos de CPU y mantén `batch_size × train_count / num_envs ≈ 8` (p. ej. 8 núcleos:
  `--num-envs 16 --batch-size 128 --train-count 1`; 16 núcleos: el preset tal cual). En Windows los subprocesos
  arrancan con `spawn` (tarda unos segundos más); WSL2 también funciona.

Requisitos: Python ≥ 3.10, `gymnasium ≥ 1.1`, `ale-py ≥ 0.11` (probado con gymnasium 1.3.0 y ale-py 0.12.1).

## 3. Uso rápido

```bash
# entrenar (el preset define todos los hiperparámetros; cualquiera se puede sobreescribir)
python train.py --preset rainbow_fast --run-name it6_rainbow_fast --total-steps 10_000_000

# reanudar una corrida interrumpida (Ctrl+C también guarda last.pt)
python train.py --resume runs/it6_rainbow_fast/last.pt --run-name it6_rainbow_fast --total-steps 10_000_000

# curvas en vivo
tensorboard --logdir runs

# gráficas + tabla resumen de todas las iteraciones (informe, sección 2.3)
python plot_results.py runs/it* --out figs

# COMPETENCIA: 5 episodios con política greedy + video de un episodio completo
python evaluate.py --checkpoint modelo_final/best.pt --episodes 5 --video videos/agente_final.mp4

# evaluar TODOS los checkpoints de una corrida (10 episodios c/u) y elegir el mejor
python evaluate.py --checkpoints-dir runs/it6_rainbow_fast --episodes 10
```

Cada corrida deja en `runs/<run-name>/`: `config.json`, `episodes.csv` (puntaje real de cada juego durante el
entrenamiento), `train.csv` (pérdida, Q medio, epsilon, pasos/s), `eval.csv` (evaluación greedy periódica),
`best.pt` (mejor evaluación), `last.pt` y `ckpt_<step>.pt` (cada `--save-every` pasos).

`train.py --help` lista todas las banderas. Unidades: 1 paso de agente = 4 frames del juego.

### Cargar los pesos del modelo final desde Python

```python
from evaluate import cargar_agente, crear_entorno, ejecutar_episodio, generar_video_agente

politica = cargar_agente("modelo_final/best.pt")        # red + preprocesamiento guardados en el checkpoint
env = crear_entorno(politica.env_config)                 # ALE/SpaceInvaders-v5, juego completo, sin clipping
puntaje, pasos = ejecutar_episodio(env, politica, seed=0)
puntaje_video, _ = generar_video_agente(politica, "videos/agente.mp4", seed=0)
```

El checkpoint contiene los pesos de la red, la configuración de la red (`config`) y la del preprocesamiento
(`env_config`), por lo que la evaluación reconstruye **exactamente** el pipeline usado al entrenar.

## 4. Entorno y preprocesamiento (informe §2.1)

| Aspecto | Decisión | Justificación |
|---|---|---|
| Entorno | `ALE/SpaceInvaders-v5` con `repeat_action_probability=0.25` (sticky actions), `full_action_space=False` (6 acciones), 108 000 frames máx. | Configuración por defecto de v5 y la que usa la evaluación del curso. Las 12 acciones extra del espacio completo son redundantes en este juego. |
| Frame-skip | El entorno se crea con `frameskip=1` y `AtariPreprocessing(frame_skip=4)` repite cada acción 4 frames haciendo **max-pool de los 2 últimos**. | Misma dinámica que v5 (una decisión cada 4 frames) pero sin perder objetos que parpadean: en Space Invaders los disparos se dibujan en frames alternos. Es el uso recomendado en la documentación de Gymnasium. |
| Observación | Escala de grises, `84x84` (INTER_AREA), 4 frames apilados → tensor `uint8 (4,84,84)`, normalizado /255 dentro de la red. | Estándar DeepMind: reduce 210x160x3x4 = 403 KB a 28 KB por estado y da información de movimiento. |
| Vidas | Entrenamiento: perder una vida marca `terminated` para el aprendizaje (**EpisodicLife**) pero el juego continúa hasta perder las 3. Evaluación: episodio completo. | Sin la señal de vida perdida el agente no la "castiga"; reiniciar el juego completo desperdiciaría las etapas avanzadas. |
| Recompensa | Entrenamiento: `sign(r)` ∈ {-1,0,1}. Métrica registrada siempre: puntaje real acumulado del juego completo (`game_score`). | Los puntos por alien van de 5 a 30 (+200 la nave nodriza): el clipping estabiliza la escala de Q. La señal es densa pero muy variable. |
| No-op reset | 0 | Los sticky actions ya introducen estocasticidad (Machado et al. 2018). |
| Truncamiento | 108 000 frames (27 000 pasos) → se trata como terminal. | Prácticamente nunca ocurre en Space Invaders. |

## 5. Algoritmo y arquitectura (informe §2.2)

**Familia DQN.** Q-learning con red convolucional, replay buffer y target network. Componentes:

* **Double DQN**: la acción del target se elige con la red online y se evalúa con la target (menos sobreestimación).
* **Dueling**: dos cabezas, V(s) y A(s,a), Q = V + A − mean(A).
* **PER** (α = 0.5, β: 0.4→1): muestreo proporcional a |δ| con sum-tree; pesos de importancia normalizados.
* **n-step (n = 3)**: objetivo R_t^(3) + γ³ max Q; propaga la recompensa más rápido.
* **NoisyNets** (σ₀ = 0.5): ruido gaussiano factorizado en las capas densas → exploración aprendida, sin ε.
* **C51** (51 átomos, [-10, 10]): opcional; distribución del retorno y pérdida de entropía cruzada.
* **Encoders**: *Nature CNN* (3 conv + 512 densas, 1.7M parámetros) o *IMPALA CNN large ×2* (3 bloques
  conv+maxpool+2 residuales, spectral norm, 5.1M parámetros).

Entrenamiento: N entornos en paralelo (subprocesos) con `step_async` para solapar simulación y gradiente;
Adam; pérdida Huber; recorte de gradiente (norma 10); target network sincronizada cada 8 000 pasos;
**relación de repetición** (replay ratio) 8 muestras por transición en todos los presets
(`batch_size × train_count / num_envs = 8`).

### Presets (`si_rl/presets.py`)

| preset | algoritmo | red | envs | batch | lr | exploración | buffer |
|---|---|---|---|---|---|---|---|
| `dqn` | DQN | Nature | 8 | 32 | 1e-4 | ε 1→0.01 en 1M pasos | 500k |
| `ddqn` | + Double | Nature | 8 | 32 | 1e-4 | ε-greedy | 500k |
| `dueling` | + Dueling | Nature | 8 | 32 | 1e-4 | ε-greedy | 500k |
| `per_nstep` | + PER + 3-step | Nature | 8 | 32 | 1e-4 | ε-greedy | 500k |
| `rainbow` | + NoisyNets | Nature | 8 | 32 | 6.25e-5 | noisy | 500k |
| `rainbow_c51` | + C51 | Nature | 8 | 32 | 6.25e-5 | noisy | 500k |
| `rainbow_fast` | Rainbow sin C51 | IMPALA ×2 + SN | 32 | 256 | 2.5e-4 | noisy | 500k |
| `rainbow_fast_small` | igual, para laptop | IMPALA ×1 | 16 | 128 | 2.5e-4 | noisy | 300k |

`rainbow_fast` sigue a Schmidt & Schmied (2021), que con 10M frames obtienen resultados comparables a Rainbow con
200M: encoder IMPALA, batch grande, muchos entornos, lr alto, AMP, y **sin** distribucional (no aporta en
presupuestos cortos). Sus componentes más importantes en la ablación fueron PER y n-step, luego Noisy y Dueling.

## 6. Plan de iteraciones (informe §2.3)

`run_iterations.sh` ejecuta la secuencia: `dqn → ddqn → dueling → per_nstep → rainbow` con 2M pasos cada una
(≈ 1–2 h en GPU cada una) y después el candidato final `rainbow_fast` con 10M pasos o más. Con `plot_results.py`
obtienes la tabla (`figs/resumen.md`) y las curvas (`curvas_entrenamiento.png`, `curvas_evaluacion.png`,
`curvas_perdida.png`, `<run>_detalle.png`) que pide la rúbrica: recompensa promedio en entrenamiento y en
evaluación greedy por iteración, curvas de recompensa/pérdida, y evidencia de inestabilidad (Q medio, pérdida).

Referencias de puntaje (puntaje del juego, agente greedy, episodios completos): agente aleatorio ≈ 150;
DQN a 10M pasos ≈ 1 000–1 500 (CleanRL: 1 441); C51 ≈ 2 000; Rainbow 200M frames ≈ 6 000–12 000.

### Consejos para maximizar el puntaje de la competencia

1. **Pasos**: el factor más importante. Entrena `rainbow_fast` el mayor tiempo posible (10–20M pasos) reanudando
   entre sesiones de Colab/Kaggle; guarda `runs/` en Drive.
2. **Elige el checkpoint por evaluación, no el último**: `evaluate.py --checkpoints-dir ... --episodes 10`
   evalúa cada `ckpt_*.pt`; el puntaje de DQN oscila entre checkpoints. La competencia toma el **máximo de 5
   episodios**, así que también mira la columna `max`.
3. **Buffer**: si la máquina tiene ≥ 12 GB libres usa `--buffer-size 1_000_000` (7 GB de frames).
4. **Entornos**: `--num-envs` ≈ número de núcleos (Mac M3: 8; Colab: 32 funciona aunque solo hay 2 vCPUs).
5. Al presentar, deja listo: `python evaluate.py --checkpoint modelo_final/best.pt --episodes 5 --video videos/final.mp4`.

### Tiempos aproximados

| máquina | preset | pasos/s | 10M pasos |
|---|---|---|---|
| Colab T4 (2 vCPU) | `rainbow_fast` | 600–900 | 3–5 h |
| Colab T4 | `dqn` / `rainbow` (Nature) | 800–1 200 | 2.5–3.5 h |
| MacBook Air M3 (mps) | `rainbow_fast_small` | 300–500 | 6–9 h |
| CPU 2 núcleos | cualquiera | 40–60 | no recomendado |

Memoria RAM: `buffer_size × 7 KB` (500k → 3.5 GB; 1M → 7 GB) + ~1 GB.

## 7. Video

`evaluate.py --video ruta.mp4` graba **todos** los frames crudos (210×160 RGB) del episodio a 60 fps (velocidad
real del Atari) mientras el agente actúa con el mismo preprocesamiento del entrenamiento; usa `imageio-ffmpeg`
(H.264) o, si no está disponible, OpenCV. Con `--video-seed` puedes elegir la semilla del episodio grabado y
`--seed` la de los 5 episodios de evaluación (los resultados son reproducibles con la misma semilla).

## 8. Referencias

* Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
* van Hasselt, Guez & Silver (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
* Wang et al. (2016). Dueling Network Architectures for Deep RL. *ICML*.
* Schaul et al. (2016). Prioritized Experience Replay. *ICLR*.
* Fortunato et al. (2018). Noisy Networks for Exploration. *ICLR*.
* Bellemare, Dabney & Munos (2017). A Distributional Perspective on RL (C51). *ICML*.
* Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL. *AAAI*.
* Espeholt et al. (2018). IMPALA: Scalable Distributed Deep-RL. *ICML*.
* Schmidt & Schmied (2021). Fast and Data-Efficient Training of Rainbow. arXiv:2111.10247.
* Machado et al. (2018). Revisiting the Arcade Learning Environment. *JAIR*.
* Huang et al. (2022). CleanRL. *JMLR* — benchmarks de referencia.
