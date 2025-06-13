"""Convenient imports for the clwm package."""

from .models.flash_attention import FlashAttentionBlock
from .models.lora_layer import LoRA
# Public re-exports -------------------------------------------------------
from .common import (
    ACTION_ID_START,
    REWARD_BINS,
    VQVAE_CHECKPOINT,
    TORCH_DEVICE,
    MAX_ACTIONS,
    PAD_TOKEN,
    VOCAB_SIZE,
    expect_raw,
    expect_symlog,
    set_global_seed,
    symexp,
    symlog,
    encode_two_hot,
    unimix,
    unimix_generic,
)
from .models.vqvae_utils import (
    get_vqvae,
    vqvae,
    frame_to_indices,
    frames_to_indices,
)

# migrated to common
# Additional math helpers ----------------------------------------------
from .common import split_cross_entropy, fisher_diagonal
from .env.atari_envs import make_atari_env, make_atari_vectorized_envs

# (migrated to Trainer; thin wrappers added below)
from .models.vqvae import VQVAE, H16, W16, K, RES, D_LAT
from .models.world_model import (
    WorldModel,
    ActorNetwork,
    CriticNetwork,
    Replay,
)

__all__ = [
    "FlashAttentionBlock",
    "LoRA",
    "WorldModel",
    "ActorNetwork",
    "CriticNetwork",
    "Replay",
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
    "encode_two_hot",
    "split_cross_entropy",
    "frame_to_indices",
    "frames_to_indices",
    "set_global_seed",
    # evaluation helpers now live in Trainer but re-export for convenience
# evaluation helpers now live in Trainer – removed from public namespace
    "expect_raw",
    "expect_symlog",
    "fisher_diagonal",
    "make_atari_env",
    "make_atari_vectorized_envs",
    "TORCH_DEVICE",
    "VOCAB_SIZE",
    "ACTION_ID_START",
    "MAX_ACTIONS",
    "PAD_TOKEN",
    "VQVAE_CHECKPOINT",
    "vqvae",
    "get_vqvae",
    "REWARD_BINS",
    # ----------------- new abstractions ----------------------------------
    "Config",
    "Trainer",
    "AsyncExecutor",
    "StreamManager",
]

# Expose the *new* classes -------------------------------------------------

from .config import Config  # noqa: E402 (import after __all__)
from .trainer import Trainer  # noqa: E402
from .concurrency import AsyncExecutor, StreamManager  # noqa: E402
