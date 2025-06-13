import math
import random
from collections import deque
from typing import Iterable, Tuple, List

import numpy as np
import torch
from torch import nn
import torch.utils.checkpoint as ckpt

import gymnasium as gym

from ..env.atari_envs import make_atari_vectorized_envs

# Project-local helpers --------------------------------------------------
from ..concurrency import StreamManager
from ..common import (
    MAX_ACTIONS,
    VOCAB_SIZE,
    REWARD_BINS,
    unimix_generic,
    TORCH_DEVICE,
    ACTION_ID_START,
    PAD_TOKEN,
)

from .flash_attention import FlashAttentionBlock
from .vqvae_utils import frames_to_indices, vqvae


class Replay:
    """Experience replay buffer with two sampling scopes.

    The class stores *current-task* transitions in an **instance-local**
    bounded deque *and* keeps an **all-tasks** rehearsal memory that is
    shared across *all* instances.

    Sampling helpers allow callers to draw exclusively from the current task
    (``sample_current``), from the shared rehearsal memory
    (``sample_global``) or from a mixture of both pools
    (``sample_mixed``).

    Example
    -------
    >>> replay = Replay(capacity=30_000)
    >>> replay.fill("Breakout", wm, actor)   # populates both pools
    >>> batch = replay.sample_mixed(128, ratio=0.2)  # 80/20 split
    """

    # ------------------------------------------------------------------
    # Shared rehearsal memory (class attribute, persists across tasks)
    # ------------------------------------------------------------------

    _GLOBAL_CAPACITY = 100_000  # hard cap on the cross-task memory size
    _global_buffer: deque[Tuple[torch.Tensor, float]] | None = None

    def __init__(self, cap: int) -> None:
        """Create a *task-local* replay buffer with capacity *cap*."""

        # Instance-local storage for the *current* task only.
        self._buffer: deque[Tuple[torch.Tensor, float]] = deque(maxlen=cap)

        # Lazily initialise the **shared** rehearsal memory once.
        if Replay._global_buffer is None:
            Replay._global_buffer = deque(maxlen=Replay._GLOBAL_CAPACITY)

        # Separate CUDA stream for auxiliary GPU work (e.g. VQ-VAE encoding).
        # Using a dedicated stream allows replay collection – which runs in a
        # *background* thread – to overlap its GPU kernels and memory copies
        # with the main training stream.  The extra stream is only created
        # when a CUDA device is available so that CPU-only execution remains
        # unaffected.

        if torch.cuda.is_available():
            self._encode_mgr = StreamManager()
        else:
            self._encode_mgr = None

    # ------------------------------------------------------------------
    # Basic container operations
    # ------------------------------------------------------------------

    def get_buffer_size(self) -> int:
        """Return the current size of the replay buffer."""

        return len(self._buffer)

    def add(self, seq: torch.Tensor, reward: float) -> None:
        """Append a transition to the *current* buffer **and** the rehearsal memory."""

        self._buffer.append((seq, reward))

        # Push into the shared rehearsal store as well.
        if Replay._global_buffer is not None:
            Replay._global_buffer.append((seq, reward))

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def sample(self, k: int) -> list[Tuple[torch.Tensor, float]]:
        """Return *k* random samples from the *current-task* buffer."""

        return random.sample(self._buffer, min(k, len(self._buffer)))

    def sample_current(self, k: int) -> list[Tuple[torch.Tensor, float]]:
        """Alias for :py:meth:`sample` (kept for readability)."""

        return self.sample(k)

    @classmethod
    def sample_global(cls, k: int) -> list[Tuple[torch.Tensor, float]]:
        """Return *k* random samples from the shared rehearsal memory."""

        if cls._global_buffer is None:
            return []
        return random.sample(
            cls._global_buffer, min(k, len(cls._global_buffer))
        )

    def sample_mixed(
        self,
        k: int,
        *,
        ratio: float = 0.2,
        min_global: int = 13,
    ) -> list[Tuple[torch.Tensor, float]]:
        """Return a mixture of current-task and rehearsal samples.

        The method draws ``int(k * ratio)`` elements from the global buffer
        (if it contains at least *min_global* transitions) and fills the
        remainder with current-task samples.  If one of the pools holds fewer
        elements than requested the other pool automatically tops up so that
        the returned list contains exactly *k* samples (or fewer if **both**
        buffers are empty).
        """

        num_global = (
            int(k * ratio)
            if (
                Replay._global_buffer is not None
                and len(Replay._global_buffer) >= min_global
            )
            else 0
        )

        num_current = k - num_global

        samples: list[Tuple[torch.Tensor, float]] = []

        if num_current > 0 and len(self._buffer) > 0:
            samples.extend(
                random.sample(
                    self._buffer, min(num_current, len(self._buffer))
                )
            )

        if num_global > 0 and Replay._global_buffer is not None:
            samples.extend(
                random.sample(
                    Replay._global_buffer,
                    min(num_global, len(Replay._global_buffer)),
                )
            )

        # Top-up if one of the buffers had insufficient data.
        shortfall = k - len(samples)
        if shortfall > 0:
            # Prefer current buffer first, then global.
            if len(self._buffer) > len(Replay._global_buffer or []):
                source = self._buffer
            else:
                source = Replay._global_buffer or self._buffer
            if len(source) > 0:
                samples.extend(
                    random.sample(source, min(shortfall, len(source)))
                )

        return samples

    # ------------------------------------------------------------------
    # High-level data collection helper (previously *fill_replay_buffer*)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fill(
        self,
        game: str,
        wm,
        actor,
        *,
        steps: int = 512,
        context_length: int = 64,
        num_envs: int = 16,
        eps: float = 0.05,
    ) -> None:
        """Interact with *game* and push collected trajectories.

        Parameters
        ----------
        game : str
            Atari game name, e.g. "Breakout".
        wm, actor
            World-model and policy network used for action selection.
        steps : int, default 512
            Number of *vectorised* environment steps.  The actual amount of
            data pushed to the buffer equals ``steps * num_envs``.
        context_length : int, default 64
            Context length used when constructing token sequences.
        num_envs : int, default 16
            Number of parallel emulator instances (affects CPU throughput).
        eps : float, default 0.05
            Epsilon for epsilon-greedy exploration.
        """

        # Create asynchronous vector environments (CPU-side).
        envs = make_atari_vectorized_envs(
            game,
            max_episode_steps=None,
            render_mode=None,  # observations already include RGB frames
            num_envs=num_envs,
        )

        obs, _ = envs.reset()

        # ------------------------------------------------------------------
        #  The previous implementation accumulated *all* frames, actions …
        #  in Python lists and then encoded the entire 4096-frame tensor at
        #  once, creating large temporary GPU allocations.  We now encode
        #  each mini-batch of observations on-the-fly and keep all
        #  non-image data on the CPU until the final push into the replay
        #  buffer.  This reduces peak GPU memory by several gigabytes and
        #  unblocks the training thread earlier.
        # ------------------------------------------------------------------

        token_list: list[torch.Tensor] = []
        actions_all, rewards_all, dones_all = [], [], []

        for _ in range(steps):
            # 1) VQ-VAE encoding of *current* observations only (≤ batch_size)
            #    The indices remain on the GPU so we can feed them directly to
            #    the policy network without an additional host→device copy.
            # ------------------------------------------------------------------
            # Encode observations → VQ-VAE codebook indices on the **dedicated**
            # CUDA stream so that the kernels can run concurrently with the
            # default stream used by the main training loop.
            # ------------------------------------------------------------------

            if self._encode_mgr is not None:
                ids_gpu = self._encode_mgr.run(
                    frames_to_indices,
                    obs,
                    vqvae,
                    batch_size=256,
                    device=TORCH_DEVICE,
                )
            else:
                ids_gpu = frames_to_indices(
                    obs, vqvae, batch_size=256, device=TORCH_DEVICE
                )

            # 2) Action selection (uses the on-device tokens)
            actions = self._select_actions(
                actor, wm, ids_gpu, envs.single_action_space, eps=eps
            )

            # 3) Environment step (actions need to be Python ints)
            next_obs, reward, term, trunc, _ = envs.step(actions.tolist())
            done = np.logical_or(term, trunc)

            # 4) Append to host-side buffers (store **CPU** copy only once)
            token_list.append(ids_gpu.cpu())
            actions_all.append(actions.astype(np.int16))
            rewards_all.append(reward.astype(np.float32))
            dones_all.append(done.astype(np.bool_))

            obs = next_obs

            # Auto-reset for older Gymnasium versions
            if hasattr(envs, "reset_done"):
                envs.reset_done()

        # ------------------- post-processing -----------------------------
        ids_cpu = torch.cat(token_list, 0)  # (T, N_PATCH)  on CPU

        actions_np = np.concatenate(actions_all, 0)
        rewards_np = np.concatenate(rewards_all, 0)
        dones_np = np.concatenate(dones_all, 0)

        # In-place population of the replay buffer (frames remain on CPU)
        self._fill(
            ids_cpu,
            actions_np,
            rewards_np,
            dones_np,
            context_length=context_length,
        )

        envs.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _select_actions(
        actor,
        wm,
        frame_ids: torch.Tensor,
        action_space: gym.spaces.Discrete,
        *,
        eps: float = 0.05,
    ) -> np.ndarray:
        """Epsilon-greedy action selection using the current *actor* net."""

        latent_state = wm.tok(frame_ids.to(TORCH_DEVICE)).mean(1)
        action_probabilities = actor(latent_state)
        action_distribution = torch.distributions.Categorical(
            action_probabilities
        )
        greedy_actions = action_distribution.sample().cpu().numpy()

        # Epsilon-greedy mix with random actions from the env's native space
        random_actions = np.array(
            [action_space.sample() for _ in range(len(frame_ids))],
            dtype=np.int32,
        )

        mask = np.random.rand(len(frame_ids)) < eps
        return np.where(mask, random_actions, greedy_actions)

    @torch.no_grad()
    def _fill(
        self,
        frames: torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
        *,
        context_length: int = 64,
    ) -> None:
        """Vectorised construction of context_length sequences and storage.

        Parameters are expected on the *CPU*.  Only the final concatenated
        sequence is moved to ``TORCH_DEVICE`` right before being stored so
        that we do not keep unnecessary tensors resident on the GPU.
        """

        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions

        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = rewards

        if isinstance(dones, torch.Tensor):
            dones_np = dones.cpu().numpy()
        else:
            dones_np = dones

        # Pre-allocate a pad row for efficiency (CPU tensor)
        pad_row = torch.full(
            (frames.shape[1] + 1,), PAD_TOKEN, dtype=torch.long
        )

        episode_start = 0  # index of first step in current episode

        for t in range(len(frames)):
            # Determine slice for the last ≤context_length steps within current episode
            # (episode boundary handled after storing the transition)

            s = max(episode_start, t - context_length + 1)
            idx = slice(s, t + 1)

            # Build (len, N_PATCH+1) tensor : [frame_tokens ‖ action_token]
            seq_frames = frames[idx]  # (L, N_PATCH)
            seq_actions = torch.as_tensor(
                ACTION_ID_START + actions_np[idx], dtype=torch.long
            ).unsqueeze(1)
            seq = torch.cat((seq_frames, seq_actions), 1)

            # Left-pad if episode length < context_length
            if seq.size(0) < context_length:
                pad_needed = context_length - seq.size(0)
                pad = pad_row.unsqueeze(0).expand(pad_needed, -1)
                seq = torch.cat((pad, seq), 0)

            # Keep sequences on the *CPU* to avoid occupying precious GPU
            # memory.  Downstream training code will transfer only the sampled
            # mini-batches to the device.
            rew = float(rewards_np[t])

            self.add(seq, rew)

            # Mark start of new episode *after* storing transition t so that
            # the final frame of the episode is still included in context_length-windows.
            if dones_np[t]:
                episode_start = t + 1


class WorldModel(nn.Module):
    """Transformer world model predicting tokens and rewards."""

    def __init__(
        self, dim: int = 256, layers: int = 6, heads: int = 8
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, dim)
        self.blocks = nn.ModuleList(
            [FlashAttentionBlock(dim=dim, heads=heads) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_SIZE, bias=False)
        self.reward_head = nn.Linear(dim, len(REWARD_BINS), bias=False)
        nn.init.zeros_(self.reward_head.weight)

    def add_task(self) -> None:
        for b in self.blocks:
            b.add()

    def forward(
        self,
        seq: torch.Tensor,
        return_ent: bool = False,
        return_reward: bool = False,
    ):
        x = self.tok(seq)
        ent_sum = 0.0
        for i, b in enumerate(self.blocks):
            # Gradient-checkpoint every second block during training to lower
            # activation memory.  We keep evaluation untouched to avoid the
            # runtime overhead when gradients are not required.
            if self.training:
                x, g = ckpt.checkpoint(b, x, use_reentrant=False)

            if return_ent:
                kl = (g * (g + 1e-8).log()).sum(-1) + math.log(g.size(-1))
                ent_sum += kl

        h = self.ln(x)

        out_logits = self.head(h)
        out = (out_logits,)

        if return_ent:
            out += (ent_sum.mean(),)
        if return_reward:
            out += (h,)

        return out if len(out) > 1 else out[0]


class ActorNetwork(nn.Module):
    """Policy network mapping latent state to action distribution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, MAX_ACTIONS)
        )

    def forward(self, z: torch.Tensor, p_unimix: float = 0.01) -> torch.Tensor:
        """
        Args
        ----
        z : torch.Tensor
            Latent state from the world model, shape (B, dim).
        p_unimix : float
            Fraction of probability mass to spread uniformly (default 1 %).

        Returns
        -------
        probs : torch.Tensor
            Smoothed action distribution, shape (B, MAX_ACTIONS), ∑=1.
        """
        logits = self.net(z)  # (B, MAX_ACTIONS)
        probs = unimix_generic(logits, p_unimix)  # 1 % uniform mix
        return probs


class CriticNetwork(nn.Module):
    """Value network estimating future rewards."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.SiLU(),
            nn.Linear(512, len(REWARD_BINS), bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
