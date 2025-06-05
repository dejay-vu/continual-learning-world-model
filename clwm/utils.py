import random
import torch
import numpy as np

from .vqvae import K  # tokenizer constants

MAX_ACT = 18  # max Atari action‑space size
ACT_PAD = K  # first action id (128)
PAD = K + MAX_ACT  # mask token (rarely used)
VOCAB = PAD + 1  # embedding size 147
CHECKPOINT = "vqvae_atari.safetensors"  # path to weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def symlog(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    else:  # float or np.ndarray
        return np.sign(x) * np.log1p(np.abs(x))


def symexp(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    else:
        return np.sign(x) * (np.expm1(np.abs(x)))


_I = torch.arange(-20, 21, dtype=torch.float32)
BINS = _I


def unimix(logits, p=0.01):
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / len(BINS)


def unimix_generic(logits, p=0.01):
    """Uniform-mix that works for any vocabulary size."""
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / logits.size(-1)


def twohot(y, bins=BINS):
    bins = BINS.to(y.device)  # <── match device
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


def expected_symlog(logits):
    """
    Expectation in SYMLOG space; returns scalar symlog value.
    """
    probs = unimix(logits)
    return (probs * BINS.to(logits.device)).sum(-1)


def expected_raw(logits):
    """
    Expectation converted back to raw reward/value domain.
    """
    return symexp(expected_symlog(logits))
