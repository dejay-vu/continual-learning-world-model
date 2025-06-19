import random

import numpy as np
import torch
import torch.nn.functional as F  # placed after torch import

# -------------------------------------------------------------------------
# VQ-VAE / tokeniser constants -------------------------------------------
# -------------------------------------------------------------------------

D_LAT = 64  # latent dimension
RES = 84  # resolution of the input images
PATCH = 16  # patch size for the VQ-VAE
H16 = W16 = RES // PATCH  # 5×5 → 25 tokens
K = 128  # number of VQ codes
EMA = 0.9  # decay for the EMA in VectorQuantize


MAX_ACTIONS = 18  # Max Atari action-space size
ACTION_ID_START = K  # First action id (128)
PAD_TOKEN = K + MAX_ACTIONS  # Mask token (rarely used)
VOCAB_SIZE = PAD_TOKEN + 1  # Embedding size 147
VQVAE_CHECKPOINT = "vqvae_atari.safetensors"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# Miscellaneous helpers ----------------------------------------------------
# -------------------------------------------------------------------------


def set_global_seed(seed: int = 0) -> None:
    """Seed **all** relevant random number generators.

    The helper aims to make *every* source of randomness in the codebase
    deterministic.  It covers

    • Python's ``random`` module
    • NumPy
    • PyTorch (CPU and CUDA)

    and additionally configures PyTorch/CUDA back-ends for bit-wise
    repeatability.  Every training script **must** call this function *once*
    at start-up **before** any other library (especially PyTorch) performs
    operations that rely on an RNG.
    """

    # ------------------------------------------------------------------
    # Python, NumPy -----------------------------------------------------
    # ------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)

    # A reproducible hash seed ensures repeatable order of objects when
    # iterating over e.g. dictionaries (Python ≥3.3 randomises hash salts by
    # default).  This must be **set before** the interpreter creates new
    # hash-based objects, therefore we do it here even though setting an
    # environment variable at runtime is technically *too late* for objects
    # that have already been instantiated.  As this function is supposed to
    # be called at the very top of scripts this limitation is acceptable.
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    # ------------------------------------------------------------------
    # PyTorch -----------------------------------------------------------
    # ------------------------------------------------------------------
    torch.manual_seed(seed)

    # For completeness also seed *all* CUDA devices explicitly.  This is a
    # no-op on CPU-only machines but harmless.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Deterministic cuDNN / CUDA behaviour -------------------------
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

        # Disable TF32 which can introduce non-deterministic rounding
        torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def np_symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def np_symexp(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


# -------------------------------------------------------------------------
# Reward/value two-hot encoding helpers -----------------------------------
# -------------------------------------------------------------------------


_BIN_VALUES = torch.linspace(-20.0, 20.0, 255, dtype=torch.float32)
REWARD_BINS = _BIN_VALUES  # CPU-resident baseline copy
_BINS_CACHE: dict[str, torch.Tensor] = {}


def _get_reward_bins(device: torch.device | str) -> torch.Tensor:  # noqa: D401
    key = str(device)
    cached = _BINS_CACHE.get(key)

    if cached is None:
        cached = REWARD_BINS.to(device)
        _BINS_CACHE[key] = cached

    return cached


def unimix(logits: torch.Tensor, p: float = 0.01) -> torch.Tensor:
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / len(_get_reward_bins(logits.device))


def unimix_generic(logits: torch.Tensor, p: float = 0.01) -> torch.Tensor:
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / logits.size(-1)


def encode_two_hot(
    target_values: torch.Tensor, *, bins: torch.Tensor | None = None
) -> torch.Tensor:
    """Continuous-to-two-hot encoding.

    Each *target* scalar is projected onto the two *nearest* bin centers and
    represented as a weighted mixture ("two-hot" vector).  The weights are
    chosen such that linear expectation over the two-hot encoding recovers the
    original scalar (up to clipping at the bin edges).

    Parameters
    ----------
    target_values : torch.Tensor
        Tensor of real-valued numbers that will be encoded.
    bins : torch.Tensor, optional
        Centers of the discretisation bins.  When not provided the global
        ``REWARD_BINS`` for the *current* device are used.
    """

    bins = _get_reward_bins(target_values.device) if bins is None else bins

    # Clamp to the valid bin range to avoid index errors --------------------
    clamped_values = torch.clamp(target_values, bins[0], bins[-1])

    # Indices of the *right* bin such that bins[idx-1] ≤ value ≤ bins[idx]
    upper_indices = torch.searchsorted(bins, clamped_values)

    left_indices = torch.clamp(upper_indices - 1, 0, len(bins) - 2)
    right_indices = left_indices + 1

    # Linear interpolation weights between the two neighbouring bins --------
    right_weight = (clamped_values - bins[left_indices]) / (
        bins[right_indices] - bins[left_indices]
    )
    left_weight = 1.0 - right_weight

    # Dense (…, |bins|) tensor with zeros everywhere except the two active
    # neighbouring bins.
    two_hot = torch.zeros(
        (*clamped_values.shape, len(bins)), device=target_values.device
    )

    two_hot.scatter_(-1, left_indices.unsqueeze(-1), left_weight.unsqueeze(-1))
    two_hot.scatter_(
        -1, right_indices.unsqueeze(-1), right_weight.unsqueeze(-1)
    )

    return two_hot


def expect_symlog(logits: torch.Tensor) -> torch.Tensor:
    probs = unimix(logits)
    bins = _get_reward_bins(logits.device)

    return (probs * bins).sum(-1)


def expect_raw(logits: torch.Tensor) -> torch.Tensor:
    return symexp(expect_symlog(logits))


# -------------------------------------------------------------------------
# Training helpers --------------------------------------------------------
# -------------------------------------------------------------------------


def split_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_logits = logits.view(-1, VOCAB_SIZE)
    flat_target = target.reshape(-1)

    m_img = flat_target < ACTION_ID_START
    m_act = (flat_target >= ACTION_ID_START) & (flat_target != PAD_TOKEN)

    ce_img = (
        F.cross_entropy(
            flat_logits[m_img], flat_target[m_img], reduction="mean"
        )
        if m_img.any()
        else torch.tensor(0.0, device=logits.device)
    )
    ce_act = (
        F.cross_entropy(
            flat_logits[m_act], flat_target[m_act], reduction="mean"
        )
        if m_act.any()
        else torch.tensor(0.0, device=logits.device)
    )
    return ce_img, ce_act


def fisher_diagonal(
    model: torch.nn.Module, batch: torch.Tensor, chunk: int = 64
) -> list[torch.Tensor]:
    """GPU-friendly approximation of the Fisher information *diagonal*."""

    device = next(model.parameters()).device

    fisher_diag_buffers = [
        torch.zeros_like(p, dtype=torch.float32, device=device)
        for p in model.parameters()
        if p.requires_grad
    ]

    total_target_tokens = torch.tensor(0.0, device=device, dtype=torch.float32)

    for start_idx in range(0, batch.size(0), chunk):
        batch_chunk = batch[start_idx : start_idx + chunk].to(
            device, non_blocking=True
        )
        # ------------------------------------------------------------------
        # The stored replay sequences have shape (B, C, T) where C is the
        # context window and T the number of tokens *per* time-step (N_PATCH
        # image tokens plus the action token).  During regular training the
        # two right-most dimensions are **flattened** before being fed into
        # the Transformer (see *Trainer._train_single_task_loop*).  We mirror
        # the same pre-processing here to keep the input format consistent
        # with what the world model expects.
        # ------------------------------------------------------------------

        input_tokens = batch_chunk[:, :-1]
        target_tokens = batch_chunk[:, 1:]

        # Collapse (context, tokens_per_step) → (sequence_length)
        if input_tokens.ndim == 3:
            batch_size_current = input_tokens.size(0)
            input_tokens = input_tokens.reshape(batch_size_current, -1)
            target_tokens = target_tokens.reshape(batch_size_current, -1)

        loss = F.cross_entropy(
            model(input_tokens).view(-1, VOCAB_SIZE),
            target_tokens.reshape(-1),
            ignore_index=PAD_TOKEN,
            reduction="sum",
        )

        # Keep track of how many target tokens contributed to the summed CE so
        # that we can later *average* the Fisher estimate per token.
        total_target_tokens += (target_tokens != PAD_TOKEN).sum()

        trainable_parameters = [
            p for p in model.parameters() if p.requires_grad
        ]
        grads = torch.autograd.grad(
            loss, trainable_parameters, allow_unused=True
        )

        for fisher_buf, grad in zip(fisher_diag_buffers, grads):
            if grad is not None:
                fisher_buf.add_(grad.detach().pow(2))

    # Average over the *total* number of target tokens that contributed to the
    # Fisher estimate across **all** processed chunks.
    fisher_diagonal_estimate = [
        buf / total_target_tokens for buf in fisher_diag_buffers
    ]
    return fisher_diagonal_estimate


class RewardEMA:
    """
    Track the 5- and 95-percentile of the environment reward with
    exponential smoothing and return an (offset, scale) tuple that
    can be used to normalise λ-returns.

    Calling convention:
    >>> offset, scale = reward_ema(batch_rewards, ema_vals)
    >>> normed = (returns - offset) / scale
    """

    def __init__(self, device: torch.device, alpha: float = 1e-2) -> None:
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.alpha = alpha

    @torch.no_grad()
    def __call__(
        self, x: torch.Tensor, ema_vals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clamp(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]

        return offset.detach(), scale.detach()
