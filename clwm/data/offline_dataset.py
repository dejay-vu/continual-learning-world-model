import numpy as np
import torch
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

from ..env.atari_envs import make_atari_env, make_atari_vectorized_envs
from tqdm import tqdm


def gather_offline_dataset(
    game: str,
    steps: int,
    out_dir: str,
    *,
    reso: int = 84,
    shard: int = 1000,
    policy: str | None = None,
):
    """Collect frames, actions and rewards to build an offline dataset.

    If ``policy`` is provided, the path is loaded as a PPO agent from
    ``stable-baselines3`` and used to act in the environment; otherwise actions
    are sampled uniformly. The resulting dataset is stored as compressed ``.npz``
    shards compatible with :func:`load_dataset_to_gpu`.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = make_atari_env(game, max_episode_steps=None, render_mode="rgb_array")
    obs, _ = env.reset()

    agent = PPO.load(policy, env=env) if policy is not None else None

    frames_buffer: list[np.ndarray] = []
    actions_buffer: list[int] = []
    rewards_buffer: list[float] = []
    dones_buffer: list[bool] = []
    shard_idx = 0

    for _ in range(steps):
        frame = env.render().transpose(2, 0, 1) / 255.0
        if frame.shape[1] != reso:
            frame = torch.nn.functional.interpolate(
                torch.tensor(frame)[None],
                (reso, reso),
                mode="bilinear",
                align_corners=False,
            )[0]
        frame_np = frame.numpy()
        if agent is None:
            action = env.action_space.sample()
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)

        frames_buffer.append(frame_np)
        actions_buffer.append(int(action))
        rewards_buffer.append(float(reward))
        dones_buffer.append(bool(term or trunc))

        if len(frames_buffer) == shard:
            np.savez_compressed(
                out / f"{shard_idx:04d}.npz",
                frames=np.stack(frames_buffer),
                actions=np.array(actions_buffer, dtype=np.int16),
                rewards=np.array(rewards_buffer, dtype=np.float32),
                dones=np.array(dones_buffer, dtype=np.bool_),
            )
            frames_buffer.clear()
            actions_buffer.clear()
            rewards_buffer.clear()
            dones_buffer.clear()
            shard_idx += 1

        if term or trunc:
            obs, _ = env.reset()

    if frames_buffer:
        np.savez_compressed(
            out / f"{shard_idx:04d}.npz",
            frames=np.stack(frames_buffer),
            actions=np.array(actions_buffer, dtype=np.int16),
            rewards=np.array(rewards_buffer, dtype=np.float32),
            dones=np.array(dones_buffer, dtype=np.bool_),
        )

    env.close()
    print("✓ Dataset collected")


def gather_datasets_parallel(
    games: list[str],
    steps: int,
    base_dir: str,
    *,
    reso: int = 84,
    shard: int = 1000,
):
    """Collect datasets for multiple games in parallel using vectorized envs."""

    envs = make_atari_vectorized_envs(
        games,
        max_episode_steps=None,
        render_mode="rgb_array",
    )
    obs, _ = envs.reset()

    out_dirs = [Path(base_dir) / g for g in games]
    for p in out_dirs:
        p.mkdir(parents=True, exist_ok=True)

    buffers = [
        dict(frames=[], actions=[], rewards=[], dones=[], idx=0, count=0)
        for _ in games
    ]

    pbar = tqdm(total=steps * len(games), desc="collect")
    while any(b["count"] < steps for b in buffers):
        # Gymnasium vector environments may return a tuple/list of per-env
        # frames when using the "rgb_array" render mode. Convert this into a
        # single numpy array with shape (n_env, C, H, W) expected by the
        # subsequent code path.
        frame = envs.render()
        if isinstance(frame, (list, tuple)):
            frame = np.stack(frame, axis=0)
        # Convert from (N, H, W, C) → (N, C, H, W)
        frame = frame.transpose(0, 3, 1, 2)
        if frame.shape[2] != reso:
            frame = torch.nn.functional.interpolate(
                torch.tensor(frame),
                (reso, reso),
                mode="bilinear",
                align_corners=False,
            ).numpy()

        actions = [envs.single_action_space.sample() for _ in games]
        obs, reward, term, trunc, _ = envs.step(actions)
        done = np.logical_or(term, trunc)

        for i, buf in enumerate(buffers):
            if buf["count"] >= steps:
                continue
            buf["frames"].append(frame[i])
            buf["actions"].append(int(actions[i]))
            buf["rewards"].append(float(reward[i]))
            buf["dones"].append(bool(done[i]))
            buf["count"] += 1
            pbar.update(1)
            if len(buf["frames"]) >= shard:
                np.savez_compressed(
                    out_dirs[i] / f"{buf['idx']:04d}.npz",
                    frames=np.stack(buf["frames"]),
                    actions=np.array(buf["actions"], dtype=np.int16),
                    rewards=np.array(buf["rewards"], dtype=np.float32),
                    dones=np.array(buf["dones"], dtype=np.bool_),
                )
                buf["frames"].clear()
                buf["actions"].clear()
                buf["rewards"].clear()
                buf["dones"].clear()
                buf["idx"] += 1

        # Gymnasium's vector environments used to provide a `reset_done()`
        # helper that reset only the environments that reached a terminal
        # or truncated state.  In newer versions this utility has been
        # removed because the library can automatically perform the reset
        # on the next ``step`` call when ``autoreset_mode`` is set to
        # ``NEXT_STEP`` (the current default).  To stay compatible with
        # both older and newer gymnasium releases we only call the method
        # when it is actually present.

        if hasattr(envs, "reset_done"):
            envs.reset_done()

    pbar.close()

    for buf, out_p in zip(buffers, out_dirs):
        if buf["frames"]:
            np.savez_compressed(
                out_p / f"{buf['idx']:04d}.npz",
                frames=np.stack(buf["frames"]),
                actions=np.array(buf["actions"], dtype=np.int16),
                rewards=np.array(buf["rewards"], dtype=np.float32),
                dones=np.array(buf["dones"], dtype=np.bool_),
            )

    envs.close()
    print("✓ Datasets collected")


from ..utils.common import TORCH_DEVICE, ACTION_ID_START, PAD_TOKEN
from ..models.vqvae_utils import frames_to_indices, vqvae


def read_npz_dataset(folder: str):
    """Load offline dataset saved by collect_offline_dataset.py."""
    p = Path(folder)
    frames, actions, rewards, dones = [], [], [], []
    for f in sorted(p.glob("*.npz")):
        data = np.load(f)
        f_arr = data["frames"]
        # Ensure frames are in (N, H, W, C) format. Some older dataset shards
        # were saved using channel-first layout (N, C, H, W). We detect that
        # case and transpose on-the-fly so that downstream VQ-VAE encoding
        # always receives a consistent channel-last representation.
        if f_arr.ndim == 4 and f_arr.shape[-1] == 3:
            # Already (N, H, W, 3)
            pass
        elif f_arr.ndim == 4 and f_arr.shape[1] == 3:
            # Convert from (N, 3, H, W) → (N, H, W, 3)
            f_arr = f_arr.transpose(0, 2, 3, 1)
        else:
            raise ValueError(
                f"Unexpected frame array shape {f_arr.shape} in {f}. Expected 4-D with channel dimension size 3."
            )

        frames.append(f_arr)
        actions.append(data["actions"])
        rewards.append(data["rewards"])
        dones.append(data["dones"])
    if not frames:
        raise FileNotFoundError(f"no .npz files found in {folder}")
    return (
        np.concatenate(frames, 0),
        np.concatenate(actions, 0),
        np.concatenate(rewards, 0),
        np.concatenate(dones, 0),
    )


@torch.no_grad()
def load_dataset_to_gpu(folder: str, batch_size: int = 2048):
    """Load dataset and convert all arrays to GPU tensors.

    Processes frames in batches of size ``batch_size`` to reduce peak memory
    usage when encoding to VQ-VAE indices.
    """
    frames, actions, rewards, dones = read_npz_dataset(folder)
    ids = frames_to_indices(frames, vqvae, batch_size=batch_size)
    return (
        torch.tensor(ids, device=TORCH_DEVICE),
        torch.tensor(actions, device=TORCH_DEVICE),
        torch.tensor(rewards, device=TORCH_DEVICE, dtype=torch.float16),
        torch.tensor(dones, device=TORCH_DEVICE),
    )


@torch.no_grad()
def fill_replay_buffer(
    frames,
    actions,
    rewards,
    dones,
    replay,
    *,
    ctx=64,
    global_buffer=None,
):
    """Push offline data into a replay buffer and optional global buffer."""
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
        if sequence.size(0) < ctx:
            pad = torch.full(
                (ctx - sequence.size(0), sequence.size(1)),
                PAD_TOKEN,
                device=TORCH_DEVICE,
            )
            sequence = torch.cat((pad, sequence), 0)
        replay.add(sequence, float(reward))
        if global_buffer is not None:
            global_buffer.append((sequence, float(reward)))
        if done:
            current_episode.clear()
