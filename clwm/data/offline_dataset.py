import numpy as np
import torch
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

from ..env.atari_envs import make_atari_env


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

    env = make_atari_env(game, max_episode_steps=None)
    env = gym.wrappers.RecordRGBArray(env)
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
                torch.tensor(frame)[None], (reso, reso), mode="bilinear", align_corners=False
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

from ..utils.common import TORCH_DEVICE, ACTION_ID_START, PAD_TOKEN
from ..models.vqvae_utils import frames_to_indices, vqvae


def read_npz_dataset(folder: str):
    """Load offline dataset saved by collect_offline_dataset.py."""
    p = Path(folder)
    frames, actions, rewards, dones = [], [], [], []
    for f in sorted(p.glob("*.npz")):
        data = np.load(f)
        frames.append(data["frames"])
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
def load_dataset_to_gpu(folder: str):
    """Load dataset and convert all arrays to GPU tensors."""
    frames, actions, rewards, dones = read_npz_dataset(folder)
    ids = frames_to_indices(frames, vqvae)
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
    for frame_ids, action, reward, done in zip(frames, actions, rewards, dones):
        current_episode.append((frame_ids, int(action)))
        sequence = torch.stack(
            [
                torch.cat((t, torch.tensor([ACTION_ID_START + act_], device=TORCH_DEVICE)))
                for t, act_ in current_episode[-ctx:]
            ]
        )
        if sequence.size(0) < ctx:
            pad = torch.full((ctx - sequence.size(0), sequence.size(1)), PAD_TOKEN, device=TORCH_DEVICE)
            sequence = torch.cat((pad, sequence), 0)
        replay.add(sequence, float(reward))
        if global_buffer is not None:
            global_buffer.append((sequence, float(reward)))
        if done:
            current_episode.clear()
