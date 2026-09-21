"""Configuraciones (presets) por iteración de desarrollo.

Cada preset es un diccionario de overrides sobre `BASE`.  Cualquier valor se puede
sobreescribir desde la línea de comandos, por ejemplo:

    python train.py --preset rainbow --total-steps 5_000_000 --num-envs 16

Unidades: todos los "steps" son pasos del agente (1 paso = 4 frames del juego).
"""
from __future__ import annotations

BASE = dict(
    # entorno
    num_envs=8,
    subproc=True,
    noop_max=0,
    repeat_action_probability=0.25,
    full_action_space=False,
    frame_skip=4,
    screen_size=84,
    frame_stack=4,
    # presupuesto
    total_steps=10_000_000,
    learning_starts=50_000,
    # replay
    buffer_size=500_000,
    batch_size=32,
    train_count=2,           # actualizaciones por paso vectorizado (8 envs * 1/4 = 2)
    n_step=1,
    gamma=0.99,
    per=False,
    per_alpha=0.5,
    per_beta0=0.4,
    per_beta_steps=None,     # None -> hasta total_steps
    # algoritmo
    double=False,
    dueling=False,
    noisy=False,
    noisy_sigma0=0.5,
    c51=False,
    atoms=51,
    v_min=-10.0,
    v_max=10.0,
    arch="nature",
    model_size=2,
    spectral_norm="all",
    hidden=512,
    # optimización
    lr=1e-4,
    adam_eps=1.5e-4,
    max_grad_norm=10.0,
    loss_fn="huber",
    target_update=8_000,
    amp=False,
    # exploración
    eps_start=1.0,
    eps_end=0.01,
    eps_decay_steps=1_000_000,
    # evaluación / logging
    eval_every=250_000,
    eval_episodes=5,
    eval_eps=0.0,
    save_every=500_000,
    log_every=10_000,
    seed=1,
)

PRESETS: dict[str, dict] = {
    # Iteración 1: DQN clásico (Mnih et al. 2015) con Adam.
    "dqn": dict(),
    # Iteración 2: Double DQN (van Hasselt et al. 2016): reduce la sobreestimación de Q.
    "ddqn": dict(double=True),
    # Iteración 3: + Dueling (Wang et al. 2016): separa V(s) y A(s,a).
    "dueling": dict(double=True, dueling=True),
    # Iteración 4: + Prioritized Experience Replay + retornos de 3 pasos.
    "per_nstep": dict(double=True, dueling=True, per=True, n_step=3),
    # Iteración 5: Rainbow sin distribucional (Double + Dueling + PER + n-step + NoisyNets).
    "rainbow": dict(
        double=True, dueling=True, per=True, n_step=3, noisy=True,
        eps_start=0.0, eps_end=0.0, eps_decay_steps=1, lr=6.25e-5,
    ),
    # Iteración 5b: Rainbow completo (con C51 distribucional).
    "rainbow_c51": dict(
        double=True, dueling=True, per=True, n_step=3, noisy=True, c51=True,
        eps_start=0.0, eps_end=0.0, eps_decay_steps=1, lr=6.25e-5,
    ),
    # Iteración 6 (candidato final): Rainbow eficiente (Schmidt & Schmied 2021):
    # IMPALA CNN large x2 + spectral norm, batch 256, 32 entornos, lr 2.5e-4, AMP.
    "rainbow_fast": dict(
        double=True, dueling=True, per=True, n_step=3, noisy=True,
        eps_start=0.0, eps_end=0.0, eps_decay_steps=1,
        arch="impala", model_size=2, spectral_norm="all",
        num_envs=32, batch_size=256, train_count=1,
        lr=2.5e-4, adam_eps=0.005 / 256, per_beta0=0.45,
        learning_starts=100_000, target_update=8_000, amp=True,
    ),
    # Variante del candidato final para máquinas con pocos núcleos / poca RAM (p. ej. laptop).
    "rainbow_fast_small": dict(
        double=True, dueling=True, per=True, n_step=3, noisy=True,
        eps_start=0.0, eps_end=0.0, eps_decay_steps=1,
        arch="impala", model_size=1, spectral_norm="all",
        num_envs=16, batch_size=128, train_count=1,
        lr=2.5e-4, adam_eps=0.005 / 128, per_beta0=0.45,
        learning_starts=50_000, target_update=8_000, buffer_size=300_000, amp=True,
    ),
}


def get_config(preset: str) -> dict:
    if preset not in PRESETS:
        raise KeyError(f"preset desconocido '{preset}'. Opciones: {list(PRESETS)}")
    cfg = dict(BASE)
    cfg.update(PRESETS[preset])
    cfg["preset"] = preset
    return cfg
