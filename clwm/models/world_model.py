import math
import random
from collections import deque
from typing import Iterable, Tuple, List

import torch
from torch import nn

import numpy as np
import gymnasium as gym

from .flash_attention import FlashAttentionBlock
from ..utils.common import (
    MAX_ACTIONS,
    VOCAB_SIZE,
    REWARD_BINS,
    unimix_generic,
    TORCH_DEVICE,
    ACTION_ID_START,
    PAD_TOKEN,
)

from ..env.atari_envs import make_atari_vectorized_envs
from .vqvae_utils import frames_to_indices, vqvae


class Replay:
    """Experience replay buffer with integrated collection helper.

    The class combines the original *ReplayBuffer* container with the
    *fill_replay_buffer* utility so that downstream code can simply call

        >>> replay = Replay(capacity)
        >>> replay.fill(game, wm, actor, ...)

    to gather fresh experience from the environment and push it directly into
    the buffer.
    """

    def __init__(self, cap: int) -> None:
        # Internal storage is a bounded deque of (sequence, reward) tuples
        self._buffer: deque[Tuple[torch.Tensor, float]] = deque(maxlen=cap)

    # ------------------------------------------------------------------
    # Basic container operations
    # ------------------------------------------------------------------

    def get_buffer_size(self) -> int:
        """Return the current size of the replay buffer."""

        return len(self._buffer)

    def add(self, seq: torch.Tensor, reward: float) -> None:
        """Append one transition sequence to the buffer."""

        self._buffer.append((seq, reward))

    def sample(self, k: int) -> Iterable[Tuple[torch.Tensor, float]]:
        """Return *k* random samples (without replacement)."""

        return random.sample(self._buffer, min(k, len(self._buffer)))

    # ------------------------------------------------------------------
    # High-level data collection helper (previously *fill_replay_buffer*)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fill(
        self,
        game: str,
        wm,
        actor,
        global_buffer: List | None = None,
        *,
        steps: int = 512,
        ctx: int = 64,
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
        global_buffer : list | None
            Optional list that stores a subset of transitions across tasks for
            the EWC implementation.  The method appends the same structures as
            *add* (tuples of (sequence, reward)).
        steps : int, default 512
            Number of *vectorised* environment steps.  The actual amount of
            data pushed to the buffer equals ``steps × num_envs``.
        ctx : int, default 64
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

        # Accumulators for later batch processing
        frames_all, actions_all, rewards_all, dones_all = [], [], [], []

        for _ in range(steps):
            # ---------------------------------------------------------
            # 1) Encode current observations via VQ-VAE → discrete ids
            # ---------------------------------------------------------
            ids_np = frames_to_indices(obs, vqvae, batch_size=obs.shape[0])
            ids = torch.tensor(ids_np, device=TORCH_DEVICE)

            # ---------------------------------------------------------
            # 2) Policy inference & ε-greedy exploration
            # ---------------------------------------------------------
            actions = self._select_actions(
                actor, wm, ids, envs.single_action_space, eps=eps
            )

            # ---------------------------------------------------------
            # 3) Environment step
            # ---------------------------------------------------------
            next_obs, reward, term, trunc, _ = envs.step(actions.tolist())

            done = np.logical_or(term, trunc)

            # ---------------------------------------------------------
            # 4) Store results for batched post-processing
            # ---------------------------------------------------------
            frames_all.append(obs)
            actions_all.append(actions)
            rewards_all.append(reward.astype(np.float32))
            dones_all.append(done)

            obs = next_obs

            # Auto-reset (Gymnasium handles this in NEXT_STEP mode – fallback
            # kept for older versions).
            if hasattr(envs, "reset_done"):
                envs.reset_done()

        # -------------------------------------------------------------
        # Batched post-processing (mimics the offline dataset layout)
        # -------------------------------------------------------------

        frames_np = np.concatenate(frames_all, axis=0)
        actions_np = np.concatenate(actions_all, axis=0).astype(np.int16)
        rewards_np = np.concatenate(rewards_all, axis=0).astype(np.float32)
        dones_np = np.concatenate(dones_all, axis=0).astype(np.bool_)

        # Convert frames to discrete VQ tokens on GPU
        ids = frames_to_indices(frames_np, vqvae, batch_size=4096)
        ids_t = torch.tensor(ids, device=TORCH_DEVICE)

        # Remaining arrays → GPU for further processing
        actions_t = torch.tensor(actions_np, device=TORCH_DEVICE)
        rewards_t = torch.tensor(
            rewards_np, device=TORCH_DEVICE, dtype=torch.float16
        )
        dones_t = torch.tensor(dones_np, device=TORCH_DEVICE)

        # In-place population of the replay buffer
        self._fill(
            ids_t,
            actions_t,
            rewards_t,
            dones_t,
            ctx=ctx,
            global_buffer=global_buffer,
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

        z = wm.tok(frame_ids.to(TORCH_DEVICE)).mean(1)  # (N, d_lat)
        probs = actor(z)  # (N, A)
        dist = torch.distributions.Categorical(probs)
        greedy_actions = dist.sample().cpu().numpy()

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
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        *,
        ctx: int = 64,
        global_buffer: List | None = None,
    ) -> None:
        """Convert raw arrays into token sequences and store them."""

        current_episode = []
        for frame_ids, action, reward, done in zip(
            frames, actions, rewards, dones
        ):
            current_episode.append((frame_ids, int(action)))

            sequence = torch.stack(
                [
                    torch.cat(
                        (
                            t,
                            torch.tensor(
                                [ACTION_ID_START + act_], device=TORCH_DEVICE
                            ),
                        )
                    )
                    for t, act_ in current_episode[-ctx:]
                ]
            )

            # Left-pad so that all sequences are exactly *ctx* tokens long
            if sequence.size(0) < ctx:
                pad = torch.full(
                    (ctx - sequence.size(0), sequence.size(1)),
                    PAD_TOKEN,
                    device=TORCH_DEVICE,
                )
                sequence = torch.cat((pad, sequence), 0)

            self.add(sequence, float(reward))

            if global_buffer is not None:
                global_buffer.append((sequence, float(reward)))

            if done:
                current_episode.clear()


class WorldModel(nn.Module):
    """Transformer world model predicting tokens and rewards."""

    def __init__(self, d: int = 256, layers: int = 6, heads: int = 8) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, d)
        self.blocks = nn.ModuleList(
            [
                FlashAttentionBlock(d_model=d, n_head=heads)
                for _ in range(layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        self.reward_head = nn.Linear(d, len(REWARD_BINS), bias=False)
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
        for b in self.blocks:
            x, g = b(x)  # g is [batch, n_adapter]
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

    def __init__(self, d_lat: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lat, 512), nn.ReLU(), nn.Linear(512, MAX_ACTIONS)
        )

    def forward(self, z: torch.Tensor, p_unimix: float = 0.01) -> torch.Tensor:
        """
        Args
        ----
        z : torch.Tensor
            Latent state from the world model, shape (B, d_lat).
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

    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512),
            nn.SiLU(),
            nn.Linear(512, len(REWARD_BINS), bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
