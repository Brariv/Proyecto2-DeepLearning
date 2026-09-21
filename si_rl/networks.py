"""Redes neuronales para el agente (PyTorch).

Encoders:
    NatureCNN  -> Mnih et al. 2015 (conv 8x8/4 -> 4x4/2 -> 3x3/1, 3136 features)
    ImpalaCNN  -> Espeholt et al. 2018 (3 bloques conv+maxpool+2 residuales), variante
                  "large" con canales x`model_size`, spectral norm opcional y
                  AdaptiveMaxPool (6x6) como en Schmidt & Schmied 2021.
Cabezas:
    Lineal o Dueling (Wang et al. 2016): Q = V + A - mean(A)
    NoisyLinear (Fortunato et al. 2018) en las capas densas (exploración aprendida)
    C51 (Bellemare et al. 2017): `atoms` logits por acción (distribución del retorno)

La entrada es uint8 (B, 4, 84, 84); la red la convierte a float y divide por 255.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Capa lineal con ruido gaussiano factorizado: w = mu + sigma * eps."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))
        self.register_buffer("bias_eps", torch.zeros(out_features))
        self.noise_enabled = True
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 * bound)
        nn.init.constant_(self.bias_sigma, sigma0 * bound)
        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_eps.copy_(torch.outer(eps_out, eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_enabled:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


def _linear(in_f: int, out_f: int, noisy: bool, sigma0: float) -> nn.Module:
    return NoisyLinear(in_f, out_f, sigma0) if noisy else nn.Linear(in_f, out_f)


class NatureCNN(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.out_features = 64 * 7 * 7

    def forward(self, x):
        return self.net(x)


def _maybe_sn(conv: nn.Conv2d, use_sn: bool) -> nn.Module:
    if use_sn:
        return nn.utils.parametrizations.spectral_norm(conv)
    return conv


class ImpalaResidual(nn.Module):
    def __init__(self, ch: int, use_sn: bool):
        super().__init__()
        self.conv1 = _maybe_sn(nn.Conv2d(ch, ch, 3, padding=1), use_sn)
        self.conv2 = _maybe_sn(nn.Conv2d(ch, ch, 3, padding=1), use_sn)

    def forward(self, x):
        y = self.conv1(F.relu(x))
        y = self.conv2(F.relu(y))
        return x + y


class ImpalaBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ImpalaResidual(out_ch, use_sn)
        self.res2 = ImpalaResidual(out_ch, use_sn)

    def forward(self, x):
        x = self.pool(self.conv(x))
        return self.res2(self.res1(x))


class ImpalaCNN(nn.Module):
    """IMPALA CNN 'large'. model_size=1 -> canales (16, 32, 32); =2 -> (32, 64, 64)."""

    def __init__(self, in_channels: int = 4, model_size: int = 2, spectral_norm: str = "all", pool_out: int = 6):
        super().__init__()
        sn_all = spectral_norm == "all"
        sn_last = spectral_norm in ("all", "last")
        chs = (16 * model_size, 32 * model_size, 32 * model_size)
        self.blocks = nn.Sequential(
            ImpalaBlock(in_channels, chs[0], sn_all),
            ImpalaBlock(chs[0], chs[1], sn_all),
            ImpalaBlock(chs[1], chs[2], sn_last),
        )
        self.pool = nn.AdaptiveMaxPool2d((pool_out, pool_out))
        self.out_features = chs[2] * pool_out * pool_out

    def forward(self, x):
        x = self.blocks(x)
        x = F.relu(x)
        return torch.flatten(self.pool(x), 1)


class QNetwork(nn.Module):
    """Encoder + cabeza (lineal/dueling, opcionalmente noisy y/o distribucional C51)."""

    def __init__(
        self,
        num_actions: int,
        in_channels: int = 4,
        arch: str = "nature",
        model_size: int = 2,
        spectral_norm: str = "all",
        hidden: int = 512,
        dueling: bool = False,
        noisy: bool = False,
        noisy_sigma0: float = 0.5,
        atoms: int = 1,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.noisy = noisy
        self.atoms = atoms
        self.distributional = atoms > 1
        if arch == "nature":
            self.encoder = NatureCNN(in_channels)
        elif arch == "impala":
            self.encoder = ImpalaCNN(in_channels, model_size=model_size, spectral_norm=spectral_norm)
        else:
            raise ValueError(f"arquitectura desconocida: {arch}")
        feat = self.encoder.out_features

        self.adv = nn.Sequential(
            _linear(feat, hidden, noisy, noisy_sigma0), nn.ReLU(),
            _linear(hidden, num_actions * atoms, noisy, noisy_sigma0),
        )
        if dueling:
            self.val = nn.Sequential(
                _linear(feat, hidden, noisy, noisy_sigma0), nn.ReLU(),
                _linear(hidden, atoms, noisy, noisy_sigma0),
            )
        if self.distributional:
            self.register_buffer("support", torch.linspace(v_min, v_max, atoms))

    # -------------------------------------------------------------- utilidades noisy
    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def set_noise(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.noise_enabled = enabled

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve Q (B, A) o, si es distribucional, log-probabilidades (B, A, atoms)."""
        x = x.float() / 255.0
        h = self.encoder(x)
        a = self.adv(h).view(-1, self.num_actions, self.atoms)
        if self.dueling:
            v = self.val(h).view(-1, 1, self.atoms)
            q = v + a - a.mean(dim=1, keepdim=True)
        else:
            q = a
        if self.distributional:
            return F.log_softmax(q, dim=2)
        return q.squeeze(2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        if self.distributional:
            return (out.exp() * self.support).sum(dim=2)
        return out


def build_network(num_actions: int, cfg: dict) -> QNetwork:
    """Construye la red a partir del diccionario de configuración (mismo que se guarda en el checkpoint)."""
    return QNetwork(
        num_actions=num_actions,
        in_channels=cfg.get("frame_stack", 4),
        arch=cfg.get("arch", "nature"),
        model_size=cfg.get("model_size", 2),
        spectral_norm=cfg.get("spectral_norm", "all"),
        hidden=cfg.get("hidden", 512),
        dueling=cfg.get("dueling", False),
        noisy=cfg.get("noisy", False),
        noisy_sigma0=cfg.get("noisy_sigma0", 0.5),
        atoms=cfg.get("atoms", 51) if cfg.get("c51", False) else 1,
        v_min=cfg.get("v_min", -10.0),
        v_max=cfg.get("v_max", 10.0),
    )
