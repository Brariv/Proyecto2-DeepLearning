"""Entrenamiento de un agente DQN / Rainbow para ALE/SpaceInvaders-v5 (se ejecuta con `python train.py`).

Ejemplos:
    python train.py --preset dqn      --total-steps 5_000_000
    python train.py --preset rainbow  --run-name it5_rainbow
    python train.py --preset rainbow_fast --resume runs/it6/last.pt      # continuar
    python train.py --preset rainbow --num-envs 16 --batch-size 64 --train-count 1

Todos los hiperparámetros de `si_rl/presets.py` se pueden sobreescribir con
`--nombre-del-parametro valor` (booleanos: `--double` / `--no-double`).

Salida (carpeta runs/<run-name>/):
    config.json     configuración completa usada
    episodes.csv    puntaje real de cada juego completo durante el entrenamiento
    train.csv       pérdida, Q medio, epsilon, SPS ... cada `log_every` pasos
    eval.csv        evaluación greedy periódica (media, máx, puntajes)
    last.pt / best.pt / ckpt_<step>.pt   checkpoints (pesos + config + env_config)
    tb/             logs de TensorBoard (opcional)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import time
from collections import deque

import numpy as np
import torch

from .agent import compute_loss, select_actions
from .env import DEFAULT_ENV_CONFIG, FrameStacker, make_env_fn
from .evaluation import evaluate_policy, make_eval_vec_env
from .networks import build_network
from .presets import BASE, PRESETS, get_config
from .replay import ReplayBuffer
from .utils import CSVLogger, TBLogger, Timer, fmt_time, get_device, linear_schedule, save_json, set_seed
from .vec_env import make_vec_env

ENV_KEYS = tuple(DEFAULT_ENV_CONFIG.keys())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="rainbow", choices=list(PRESETS), help="configuración base")
    p.add_argument("--run-name", default=None, help="nombre de la corrida (default: <preset>_s<seed>_<fecha>)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--resume", default=None, help="checkpoint .pt desde el cual continuar")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--refill-steps", type=int, default=50_000,
                   help="al reanudar: pasos para rellenar el replay buffer antes de volver a entrenar")
    for k, v in BASE.items():
        name = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(name, dest=k, action=argparse.BooleanOptionalAction, default=None)
        elif v is None:
            p.add_argument(name, dest=k, type=int, default=None)
        else:
            p.add_argument(name, dest=k, type=type(v), default=None)
    return p.parse_args()


def build_config(args, saved: dict | None = None) -> dict:
    """Config = preset (o la guardada en el checkpoint al reanudar) + overrides explícitos de la CLI."""
    if saved is not None:
        cfg = dict(BASE)
        cfg.update(saved)
        if args.preset != saved.get("preset", args.preset):
            print(f"[train] aviso: --preset {args.preset} ignorado; se usa la config del checkpoint "
                  f"({saved.get('preset')}). Solo se aplican los overrides explícitos.")
    else:
        cfg = get_config(args.preset)
    for k in BASE:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if cfg["per_beta_steps"] is None:
        cfg["per_beta_steps"] = cfg["total_steps"]
    return cfg


def env_config_from(cfg: dict) -> dict:
    ec = dict(DEFAULT_ENV_CONFIG)
    for k in ENV_KEYS:
        if k in cfg:
            ec[k] = cfg[k]
    return ec


def save_checkpoint(path, online, target, optimizer, scaler, step, num_updates, best_eval, cfg, env_config,
                    num_actions, episodes_seen, wall, light: bool = False):
    """light=True guarda solo los pesos de la red online (suficiente para evaluar; ~4x más pequeño)."""
    torch.save(
        dict(
            model=online.state_dict(),
            target=None if light else target.state_dict(),
            optimizer=None if light else optimizer.state_dict(),
            scaler=None if (light or scaler is None) else scaler.state_dict(),
            step=step,
            num_updates=num_updates,
            best_eval=best_eval,
            config=cfg,
            env_config=env_config,
            num_actions=num_actions,
            episodes_seen=episodes_seen,
            wall_time=wall,
        ),
        path,
    )


def main():
    args = parse_args()
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = build_config(args, saved=resume_ckpt["config"])
        # la arquitectura y el preprocesamiento deben ser EXACTAMENTE los del checkpoint
        saved = resume_ckpt["config"]
        for k in ("arch", "model_size", "spectral_norm", "hidden", "dueling", "noisy", "noisy_sigma0",
                  "c51", "atoms", "v_min", "v_max", "frame_stack"):
            cfg[k] = saved.get(k, cfg[k])
        env_config = resume_ckpt.get("env_config", env_config_from(cfg))
    else:
        cfg = build_config(args)
        env_config = env_config_from(cfg)

    run_name = args.run_name or f"{cfg['preset']}_s{cfg['seed']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_json(dict(cfg, env_config=env_config, run_name=run_name), os.path.join(run_dir, "config.json"))

    set_seed(cfg["seed"])
    device = get_device(args.device)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[train] run_dir={run_dir}  device={device}  amp={use_amp}")
    print(f"[train] preset={cfg['preset']} double={cfg['double']} dueling={cfg['dueling']} noisy={cfg['noisy']} "
          f"per={cfg['per']} n_step={cfg['n_step']} c51={cfg['c51']} arch={cfg['arch']}")

    # ----------------------------------------------------------------- entornos
    N = cfg["num_envs"]
    env_fns = [make_env_fn(env_config, episodic_life=True, clip_reward=True) for _ in range(N)]
    envs = make_vec_env(env_fns, subproc=cfg["subproc"])
    num_actions = envs.action_space.n
    H, W = envs.observation_space.shape
    S = cfg["frame_stack"]
    eval_envs = make_eval_vec_env(env_config, cfg["eval_episodes"]) if cfg["eval_every"] > 0 else None

    # ----------------------------------------------------------------- redes
    online = build_network(num_actions, cfg).to(device)
    target = build_network(num_actions, cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except TypeError:  # torch < 2.3
            scaler = torch.cuda.amp.GradScaler()
    amp_context = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if use_amp else contextlib.nullcontext
    n_params = sum(p.numel() for p in online.parameters())
    replay_ratio = cfg["batch_size"] * cfg["train_count"] / N
    print(f"[train] red={cfg['arch']} parámetros={n_params/1e6:.2f}M acciones={num_actions}")
    print(f"[train] envs={N} batch={cfg['batch_size']} train_count={cfg['train_count']} -> "
          f"replay ratio = {replay_ratio:.1f} muestras por transición (referencia Rainbow: 8)")
    if not 4 <= replay_ratio <= 16:
        print("[train] AVISO: replay ratio fuera del rango típico (4-16); ajusta --batch-size/--train-count/--num-envs")

    step, num_updates, best_eval, episodes_seen, wall_prev = 0, 0, -float("inf"), 0, 0.0
    train_from = cfg["learning_starts"]
    if resume_ckpt is not None:
        online.load_state_dict(resume_ckpt["model"])
        if resume_ckpt.get("target") is not None:
            target.load_state_dict(resume_ckpt["target"])
        else:  # checkpoint "ligero" (ckpt_*.pt): sin target ni optimizador
            target.load_state_dict(online.state_dict())
            print("[train] aviso: checkpoint ligero (sin optimizador); Adam se reinicia.")
        if resume_ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scaler is not None and resume_ckpt.get("scaler"):
            scaler.load_state_dict(resume_ckpt["scaler"])
        step = int(resume_ckpt["step"])
        num_updates = int(resume_ckpt.get("num_updates", 0))
        best_eval = float(resume_ckpt.get("best_eval", -float("inf")))
        episodes_seen = int(resume_ckpt.get("episodes_seen", 0))
        wall_prev = float(resume_ckpt.get("wall_time", 0.0))
        train_from = step + args.refill_steps
        print(f"[train] reanudando desde step={step:,} (best_eval={best_eval}); "
              f"rellenando buffer hasta step={train_from:,}")

    # ----------------------------------------------------------------- replay buffer
    buffer = ReplayBuffer(
        capacity=cfg["buffer_size"], num_envs=N, obs_shape=(H, W), frame_stack=S,
        n_step=cfg["n_step"], gamma=cfg["gamma"], prioritized=cfg["per"], alpha=cfg["per_alpha"],
        seed=cfg["seed"],
    )
    print(f"[train] replay buffer: {buffer.cap * N:,} frames ({buffer.cap * N * H * W / 1e9:.2f} GB)")

    # ----------------------------------------------------------------- logging
    ep_log = CSVLogger(os.path.join(run_dir, "episodes.csv"), append=resume_ckpt is not None)
    tr_log = CSVLogger(os.path.join(run_dir, "train.csv"), append=resume_ckpt is not None,
                       fieldnames=["step", "updates", "sps", "eps", "beta", "score_avg100", "buffer", "time",
                                   "loss", "q_mean", "td_abs", "grad_norm"])
    ev_log = CSVLogger(os.path.join(run_dir, "eval.csv"), append=resume_ckpt is not None)
    tb = TBLogger(os.path.join(run_dir, "tb"), enabled=not args.no_tensorboard)
    recent_scores = deque(maxlen=100)
    recent_stats = deque(maxlen=200)
    timer = Timer()
    last_log_step, last_log_time = step, time.time()

    # ----------------------------------------------------------------- loop principal
    rng = np.random.default_rng(cfg["seed"])
    obs = envs.reset(seed=cfg["seed"])
    stacker = FrameStacker(N, S, H, W)
    stacker.reset(obs)
    starts = np.ones(N, dtype=bool)
    total = cfg["total_steps"]
    bucket = lambda s, every: s // every if every > 0 else 0

    last_eval_step = -1

    def do_eval(step_now: int, tag: str = ""):
        nonlocal best_eval, last_eval_step
        last_eval_step = step_now
        t0 = time.time()
        scores, lengths = evaluate_policy(
            online, eval_envs, device, epsilon=cfg["eval_eps"], seed=cfg["seed"] + 10_000 + step_now,
            frame_stack=S,
        )
        mean, mx = float(np.mean(scores)), float(np.max(scores))
        row = dict(step=step_now, mean=mean, std=float(np.std(scores)), min=float(np.min(scores)), max=mx,
                   mean_length=float(np.mean(lengths)), scores=json.dumps(scores), time=wall_prev + timer.elapsed())
        ev_log.log(row)
        tb.scalar("eval/mean_score", mean, step_now)
        tb.scalar("eval/max_score", mx, step_now)
        improved = mean > best_eval
        if improved:
            best_eval = mean
            save_checkpoint(os.path.join(run_dir, "best.pt"), online, target, optimizer, scaler, step_now,
                            num_updates, best_eval, cfg, env_config, num_actions, episodes_seen,
                            wall_prev + timer.elapsed())
        print(f"[eval{tag}] step={step_now:,} media={mean:.1f} max={mx:.0f} puntajes={[int(s) for s in scores]} "
              f"{'** nuevo best **' if improved else ''} ({time.time()-t0:.0f}s)")

    def _on_sigterm(signum, frame):  # Colab/Kaggle/SLURM matan con SIGTERM: guardar last.pt igual que con Ctrl+C
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        while step < total:
            # --- actuar
            if step < cfg["learning_starts"]:
                eps = 1.0
            else:
                eps = linear_schedule(cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"],
                                      step - cfg["learning_starts"])
            actions = select_actions(online, stacker.obs, eps, device, rng)
            envs.step_async(actions)

            # --- entrenar mientras los entornos avanzan
            if step >= train_from and buffer.num_valid_per_env > 0:
                beta = linear_schedule(cfg["per_beta0"], 1.0, cfg["per_beta_steps"], step)
                for _ in range(cfg["train_count"]):
                    batch = buffer.sample(cfg["batch_size"], beta)
                    with amp_context():
                        loss, prios, stats = compute_loss(online, target, batch, device,
                                                          double=cfg["double"], loss_fn=cfg["loss_fn"])
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        gn = torch.nn.utils.clip_grad_norm_(online.parameters(), cfg["max_grad_norm"])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        gn = torch.nn.utils.clip_grad_norm_(online.parameters(), cfg["max_grad_norm"])
                        optimizer.step()
                    buffer.update_priorities(batch["flat_idx"], prios)
                    stats["grad_norm"] = float(gn)
                    recent_stats.append(stats)
                    num_updates += 1

            # --- recibir resultado de los entornos y almacenar
            next_obs, rewards, dones, infos = envs.step_wait()
            buffer.add(obs, actions, rewards, dones, starts)
            stacker.push(next_obs, dones)
            obs, starts = next_obs, dones.copy()
            prev_step = step
            step += N

            for info in infos:
                if "game_score" in info:
                    episodes_seen += 1
                    recent_scores.append(info["game_score"])
                    ep_log.log(dict(step=step, episode=episodes_seen, score=info["game_score"],
                                    length=info["game_length"], time=wall_prev + timer.elapsed()))
                    tb.scalar("train/episode_score", info["game_score"], step)

            # --- target network
            if bucket(step, cfg["target_update"]) != bucket(prev_step, cfg["target_update"]):
                target.load_state_dict(online.state_dict())

            # --- logging
            if bucket(step, cfg["log_every"]) != bucket(prev_step, cfg["log_every"]):
                now = time.time()
                sps = (step - last_log_step) / max(now - last_log_time, 1e-6)
                last_log_step, last_log_time = step, now
                mean_stats = {k: float(np.mean([s[k] for s in recent_stats])) for k in recent_stats[0]} if recent_stats else {}
                avg100 = float(np.mean(recent_scores)) if recent_scores else float("nan")
                row = dict(step=step, updates=num_updates, sps=sps, eps=eps,
                           beta=linear_schedule(cfg["per_beta0"], 1.0, cfg["per_beta_steps"], step) if cfg["per"] else 0.0,
                           score_avg100=avg100, buffer=len(buffer), time=wall_prev + timer.elapsed(), **mean_stats)
                tr_log.log(row)
                for k, v in mean_stats.items():
                    tb.scalar(f"train/{k}", v, step)
                tb.scalar("train/sps", sps, step)
                tb.scalar("train/epsilon", eps, step)
                tb.scalar("train/score_avg100", avg100, step)
                eta = (total - step) / max(sps, 1e-6)
                print(f"[train] step={step:,}/{total:,} sps={sps:.0f} eps={eps:.3f} ep={episodes_seen} "
                      f"score100={avg100:.1f} loss={mean_stats.get('loss', float('nan')):.4f} "
                      f"q={mean_stats.get('q_mean', float('nan')):.2f} t={fmt_time(timer.elapsed())} eta={fmt_time(eta)}")

            # --- evaluación
            if eval_envs is not None and bucket(step, cfg["eval_every"]) != bucket(prev_step, cfg["eval_every"]) \
                    and step >= cfg["learning_starts"]:
                do_eval(step)

            # --- checkpoints
            if bucket(step, cfg["save_every"]) != bucket(prev_step, cfg["save_every"]):
                wall = wall_prev + timer.elapsed()
                save_checkpoint(os.path.join(run_dir, "last.pt"), online, target, optimizer, scaler, step,
                                num_updates, best_eval, cfg, env_config, num_actions, episodes_seen, wall)
                save_checkpoint(os.path.join(run_dir, f"ckpt_{step:09d}.pt"), online, target, optimizer, scaler,
                                step, num_updates, best_eval, cfg, env_config, num_actions, episodes_seen, wall,
                                light=True)

    except KeyboardInterrupt:
        print("\n[train] interrumpido por el usuario; guardando last.pt ...")
    except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
        # En Windows Ctrl+C llega también a los subprocesos de los entornos; si alguno se cierra
        # antes que el proceso principal, la comunicación falla con este tipo de error.
        print(f"\n[train] un subproceso de entorno se cerró ({type(exc).__name__}); guardando last.pt ...")
    finally:
        wall = wall_prev + timer.elapsed()
        save_checkpoint(os.path.join(run_dir, "last.pt"), online, target, optimizer, scaler, step, num_updates,
                        best_eval, cfg, env_config, num_actions, episodes_seen, wall)
        if eval_envs is not None and step >= cfg["learning_starts"] and step >= total and last_eval_step != step:
            do_eval(step, tag="-final")
        for lg in (ep_log, tr_log, ev_log, tb):
            lg.close()
        envs.close()
        if eval_envs is not None:
            eval_envs.close()
        print(f"[train] fin. step={step:,} updates={num_updates:,} episodios={episodes_seen} "
              f"best_eval={best_eval:.1f} tiempo total={fmt_time(wall)}  -> {run_dir}")

