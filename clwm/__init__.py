"""Top-level package for continual-learning-world-model (clwm)."""

from .flash_attn_block import FlashAttentionBlock
from .lora import LoRA
from .utils import (
    ACT_PAD,
    BINS,
    CHECKPOINT,
    DEVICE,
    MAX_ACT,
    PAD,
    VOCAB,
    expected_raw,
    expected_symlog,
    seed_everything,
    symexp,
    symlog,
    twohot,
    unimix,
    unimix_generic,
)
from .vq_utils import get_vqvae, vqvae, frame_to_ids, frames_to_ids
from .train_utils import split_ce, fisher_diag
from .envs import make_atari, make_atari_vectorized
from .eval_utils import build_eval_seq, eval_on_sequences, eval_policy
from .vqvae import VQVAE, H16, W16, K, RES, D_LAT
from .wm import WorldModel, Actor, Critic, Buffer

__all__ = [
    "FlashAttentionBlock",
    "LoRA",
    "WorldModel",
    "Actor",
    "Critic",
    "Buffer",
    "VQVAE",
    "H16",
    "W16",
    "K",
    "RES",
    "D_LAT",
    "symlog",
    "symexp",
    "unimix",
    "unimix_generic",
    "twohot",
    "split_ce",
    "frame_to_ids",
    "frames_to_ids",
    "seed_everything",
    "build_eval_seq",
    "eval_on_sequences",
    "eval_policy",
    "expected_raw",
    "expected_symlog",
    "fisher_diag",
    "make_atari",
    "make_atari_vectorized",
    "DEVICE",
    "VOCAB",
    "ACT_PAD",
    "MAX_ACT",
    "PAD",
    "CHECKPOINT",
    "vqvae",
    "get_vqvae",
    "BINS",
]
