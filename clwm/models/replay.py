import random
from collections import deque
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from ..common import ACTION_ID_START, PAD_TOKEN, TORCH_DEVICE
from ..concurrency import StreamManager
from ..env.atari_envs import make_atari_vectorized_envs
from ..models.world_model import ActorNetwork, WorldModel
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

        self._encode_mgr = StreamManager()

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

    @torch.no_grad()
    def fill(
        self,
        game: str,
        wm: "WorldModel",
        actor: "ActorNetwork",
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

        # Pass a deterministic, *per-call* seed to the environment manager so
        # that the exact same trajectories are reproduced when the global
        # seed (see ``set_global_seed``) is identical.  We draw the seed from
        # Python's RNG which has been initialised by ``set_global_seed`` –
        # this guarantees repeatability across runs while still yielding a
        # unique seed for *each* call to :pyfunc:`fill`.

        obs, _ = envs.reset(seed=random.randint(0, 2**32 - 1))

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
                wm,
                actor,
                frame_ids=ids_gpu,
                action_space=envs.single_action_space,
                eps=eps,
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

        # ------------------- post-processing -----------------------------
        ids_cpu = torch.cat(token_list, 0)  # (T, N_PATCH)  on CPU

        actions_np = np.concatenate(actions_all, 0)
        rewards_np = np.concatenate(rewards_all, 0)
        dones_np = np.concatenate(dones_all, 0)

        # In-place population of the replay buffer (frames remain on CPU)
        self._fill(
            frames=ids_cpu,
            actions=actions_np,
            rewards=rewards_np,
            dones=dones_np,
            context_length=context_length,
        )

        envs.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _select_actions(
        wm: WorldModel,
        actor: ActorNetwork,
        frame_ids: torch.Tensor,
        action_space: gym.Space,
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
