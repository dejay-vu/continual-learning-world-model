"""Project-wide constants and math helpers.

The contents were previously hidden inside ``clwm/utils/common`` – moving
them into a dedicated *top-level* module removes the need for the generic
``utils`` package and makes the public API clearer.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

# -------------------------------------------------------------------------
# VQ-VAE / tokeniser constants -------------------------------------------
# -------------------------------------------------------------------------


from .models.vqvae import K  # pylint: disable=cyclic-import

MAX_ACTIONS = 18                         # Max Atari action-space size
ACTION_ID_START = K                      # First action id (128)
PAD_TOKEN = K + MAX_ACTIONS              # Mask token (rarely used)
VOCAB_SIZE = PAD_TOKEN + 1               # Embedding size 147
VQVAE_CHECKPOINT = "vqvae_atari.safetensors"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# Miscellaneous helpers ----------------------------------------------------
# -------------------------------------------------------------------------


def set_global_seed(seed: int = 0) -> None:
    """Seed *torch*, *numpy* and the Python RNG."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def symlog(x: Any):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x: Any):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return np.sign(x) * (np.expm1(np.abs(x)))


# -------------------------------------------------------------------------
# Reward/value two-hot encoding helpers -----------------------------------
# -------------------------------------------------------------------------


_BIN_VALUES = torch.arange(-20, 21, dtype=torch.float32)
REWARD_BINS = _BIN_VALUES  # CPU-resident baseline copy
_BINS_CACHE: dict[str, torch.Tensor] = {}


def _get_reward_bins(device: torch.device | str) -> torch.Tensor:  # noqa: D401
    key = str(device)
    cached = _BINS_CACHE.get(key)
    if cached is None:
        cached = REWARD_BINS.to(device)
        _BINS_CACHE[key] = cached
    return cached


def unimix(logits: torch.Tensor, p: float = 0.01):
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / len(_get_reward_bins(logits.device))


def unimix_generic(logits: torch.Tensor, p: float = 0.01):
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / logits.size(-1)


def encode_two_hot(y: torch.Tensor, *, bins: torch.Tensor | None = None):
    bins = _get_reward_bins(y.device) if bins is None else bins

    y = torch.clamp(y, bins[0], bins[-1])
    k = torch.searchsorted(bins, y)
    k0 = torch.clamp(k - 1, 0, len(bins) - 2)
    k1 = k0 + 1
    w1 = (y - bins[k0]) / (bins[k1] - bins[k0])
    w0 = 1.0 - w1

    enc = torch.zeros((*y.shape, len(bins)), device=y.device)
    enc.scatter_(-1, k0.unsqueeze(-1), w0.unsqueeze(-1))
    enc.scatter_(-1, k1.unsqueeze(-1), w1.unsqueeze(-1))
    return enc


def expect_symlog(logits: torch.Tensor):
    probs = unimix(logits)
    bins = _get_reward_bins(logits.device)
    return (probs * bins).sum(-1)


def expect_raw(logits: torch.Tensor):
    return symexp(expect_symlog(logits))


# -------------------------------------------------------------------------
# Training helpers --------------------------------------------------------
# -------------------------------------------------------------------------


import torch.nn.functional as F  # placed after torch import


def split_cross_entropy(logits: torch.Tensor, tgt: torch.Tensor):
    flat_logits = logits.view(-1, VOCAB_SIZE)
    flat_tgt = tgt.reshape(-1)

    m_img = flat_tgt < ACTION_ID_START
    m_act = (flat_tgt >= ACTION_ID_START) & (flat_tgt != PAD_TOKEN)

    ce_img = (
        F.cross_entropy(flat_logits[m_img], flat_tgt[m_img], reduction="mean")
        if m_img.any()
        else torch.tensor(0.0, device=logits.device)
    )
    ce_act = (
        F.cross_entropy(flat_logits[m_act], flat_tgt[m_act], reduction="mean")
        if m_act.any()
        else torch.tensor(0.0, device=logits.device)
    )
    return ce_img, ce_act


def fisher_diagonal(
    model: torch.nn.Module, batch: torch.Tensor, chunk: int = 64
):
    """GPU-friendly approximation of the Fisher information *diagonal*."""

    device = next(model.parameters()).device

    diag_gpu = [
        torch.zeros_like(p, dtype=torch.float32, device=device)
        for p in model.parameters()
        if p.requires_grad
    ]

    for i in range(0, batch.size(0), chunk):
        sub = batch[i : i + chunk].to(device, non_blocking=True)
        loss = F.cross_entropy(
            model(sub[:, :-1]).view(-1, VOCAB_SIZE),
            sub[:, 1:].reshape(-1),
            ignore_index=PAD_TOKEN,
            reduction="sum",
        )

        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)

        for d_gpu, g in zip(diag_gpu, grads):
            if g is not None:
                d_gpu.add_(g.detach().pow(2))

    N = (batch[:, 1:] != PAD_TOKEN).sum().to(device=device, dtype=torch.float32)
    diag = [d / N for d in diag_gpu]
    return diag
