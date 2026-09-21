#!/usr/bin/env python
"""Gráficas y tabla resumen de las iteraciones (para el informe, sección 2.3).

    python plot_results.py runs/it1_dqn runs/it2_ddqn runs/it5_rainbow --out informe/figs
    python plot_results.py runs/*            # todas las corridas

Genera en --out:
    curvas_entrenamiento.png  puntaje por episodio (media móvil) vs pasos, todas las corridas
    curvas_evaluacion.png     puntaje medio de evaluación greedy vs pasos
    curvas_perdida.png        pérdida y Q medio vs pasos
    <run>_detalle.png         figura individual por corrida (episodios + eval + pérdida + Q)
    resumen.md / resumen.csv  tabla resumen por corrida (para la tabla de iteraciones)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_run(run_dir: str) -> dict:
    d = dict(name=os.path.basename(os.path.normpath(run_dir)), dir=run_dir)
    for f in ("episodes", "train", "eval"):
        path = os.path.join(run_dir, f"{f}.csv")
        d[f] = pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 0 else None
    cfg_path = os.path.join(run_dir, "config.json")
    d["config"] = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    return d


def rolling(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(1, window // 5)).mean()


def describe_config(cfg: dict) -> str:
    parts = [cfg.get("preset", "?"), cfg.get("arch", "nature")]
    for k in ("double", "dueling", "noisy", "per", "c51"):
        if cfg.get(k):
            parts.append(k)
    if cfg.get("n_step", 1) > 1:
        parts.append(f"n={cfg['n_step']}")
    return " + ".join(parts)


def summarize(run: dict) -> dict:
    cfg, ep, ev, tr = run["config"], run["episodes"], run["eval"], run["train"]
    row = dict(
        corrida=run["name"],
        config=describe_config(cfg),
        pasos=int(tr["step"].iloc[-1]) if tr is not None and len(tr) else (int(ep["step"].iloc[-1]) if ep is not None and len(ep) else 0),
        episodios=int(len(ep)) if ep is not None else 0,
        train_media_ult100=float(ep["score"].tail(100).mean()) if ep is not None and len(ep) else np.nan,
        train_max=float(ep["score"].max()) if ep is not None and len(ep) else np.nan,
        eval_mejor_media=float(ev["mean"].max()) if ev is not None and len(ev) else np.nan,
        eval_ultima_media=float(ev["mean"].iloc[-1]) if ev is not None and len(ev) else np.nan,
        eval_max=float(ev["max"].max()) if ev is not None and len(ev) else np.nan,
        horas=float((tr["time"].iloc[-1] if tr is not None and len(tr) else 0) / 3600),
        lr=cfg.get("lr"), batch=cfg.get("batch_size"), envs=cfg.get("num_envs"), buffer=cfg.get("buffer_size"),
    )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="carpetas de corridas (runs/<nombre>)")
    p.add_argument("--out", default="figs")
    p.add_argument("--window", type=int, default=50, help="ventana de la media móvil (episodios)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # PowerShell/cmd no expanden comodines (runs/it*), así que se expanden aquí
    dirs = []
    for patron in args.runs:
        for d in sorted(glob.glob(patron)) or [patron]:
            if os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    runs = [load_run(r) for r in dirs]
    runs = [r for r in runs if r["episodes"] is not None or r["eval"] is not None]
    assert runs, "no se encontraron corridas con datos"

    # --- comparación: entrenamiento
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        if r["episodes"] is not None and len(r["episodes"]):
            ep = r["episodes"]
            ax.plot(ep["step"], rolling(ep["score"], args.window), label=r["name"])
    ax.set_xlabel("pasos de agente (1 paso = 4 frames)")
    ax.set_ylabel(f"puntaje por episodio (media móvil {args.window})")
    ax.set_title("Entrenamiento: puntaje real por juego completo")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curvas_entrenamiento.png"), dpi=150)
    plt.close(fig)

    # --- comparación: evaluación greedy
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        if r["eval"] is not None and len(r["eval"]):
            ev = r["eval"]
            ax.plot(ev["step"], ev["mean"], marker="o", ms=3, label=r["name"])
    ax.set_xlabel("pasos de agente")
    ax.set_ylabel("puntaje medio de evaluación (greedy)")
    ax.set_title("Evaluación periódica (política greedy, episodios completos)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curvas_evaluacion.png"), dpi=150)
    plt.close(fig)

    # --- comparación: pérdida y Q
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in runs:
        tr = r["train"]
        if tr is not None and "loss" in tr and len(tr):
            axes[0].plot(tr["step"], tr["loss"], label=r["name"], lw=1)
            axes[1].plot(tr["step"], tr["q_mean"], label=r["name"], lw=1)
    axes[0].set_title("Pérdida (media por ventana de log)")
    axes[0].set_yscale("log")
    axes[1].set_title("Q medio del batch")
    for ax in axes:
        ax.set_xlabel("pasos de agente")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curvas_perdida.png"), dpi=150)
    plt.close(fig)

    # --- detalle por corrida
    for r in runs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ep, ev, tr = r["episodes"], r["eval"], r["train"]
        if ep is not None and len(ep):
            axes[0, 0].plot(ep["step"], ep["score"], ".", ms=2, alpha=0.3, label="episodio")
            axes[0, 0].plot(ep["step"], rolling(ep["score"], args.window), lw=2, label=f"media móvil {args.window}")
            axes[0, 0].legend()
        axes[0, 0].set_title("Puntaje por episodio (entrenamiento)")
        if ev is not None and len(ev):
            axes[0, 1].errorbar(ev["step"], ev["mean"], yerr=ev["std"], marker="o", ms=3, capsize=2, label="media ± std")
            axes[0, 1].plot(ev["step"], ev["max"], "--", label="máximo")
            axes[0, 1].legend()
        axes[0, 1].set_title("Evaluación greedy periódica")
        if tr is not None and len(tr) and "loss" in tr:
            axes[1, 0].plot(tr["step"], tr["loss"], lw=1)
            axes[1, 0].set_yscale("log")
            axes[1, 1].plot(tr["step"], tr["q_mean"], lw=1)
        axes[1, 0].set_title("Pérdida")
        axes[1, 1].set_title("Q medio")
        for ax in axes.ravel():
            ax.set_xlabel("pasos de agente")
            ax.grid(alpha=0.3)
        fig.suptitle(f"{r['name']}  ({describe_config(r['config'])})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{r['name']}_detalle.png"), dpi=150)
        plt.close(fig)

    # --- tabla resumen
    df = pd.DataFrame([summarize(r) for r in runs])
    df.to_csv(os.path.join(args.out, "resumen.csv"), index=False)
    with open(os.path.join(args.out, "resumen.md"), "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".1f") if hasattr(df, "to_markdown") else df.to_string(index=False))
    print(df.to_string(index=False))
    print(f"\nFiguras y tabla guardadas en {args.out}/")


if __name__ == "__main__":
    main()
