"""Utilidades: dispositivo, semillas, schedules y logging (CSV + TensorBoard opcional)."""
from __future__ import annotations

import csv
import json
import os
import random
import time

import numpy as np
import torch


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_schedule(start: float, end: float, duration: float, t: float) -> float:
    if duration <= 0:
        return end
    frac = min(max(t / duration, 0.0), 1.0)
    return start + frac * (end - start)


class CSVLogger:
    """Escribe filas (dict) a un CSV; crea el encabezado con las claves de la primera fila."""

    def __init__(self, path: str, fieldnames: list[str] | None = None, append: bool = False):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self.file = open(path, "a" if append else "w", newline="")
        self.writer = None
        self.fieldnames = fieldnames
        if append and exists:
            with open(path, newline="") as f:
                header = next(csv.reader(f), None)
            if header:
                self.fieldnames = header
                self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="ignore")

    def log(self, row: dict) -> None:
        if self.writer is None:
            self.fieldnames = self.fieldnames or list(row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="ignore")
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class TBLogger:
    """Envoltorio de TensorBoard que no falla si `tensorboard` no está instalado."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except Exception as exc:  # pragma: no cover
            print(f"[TBLogger] TensorBoard deshabilitado ({exc}). Instala `tensorboard` para habilitarlo.")

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0

    def elapsed(self) -> float:
        return time.time() - self.t0

    def lap(self) -> float:
        now = time.time()
        dt = now - self.last
        self.last = now
        return dt


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"
