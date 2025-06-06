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
    num_envs: int = 1,
):
    """Collect frames, actions and rewards to build an offline dataset.

    If ``policy`` is provided, the path is loaded as a PPO agent from
    ``stable-baselines3`` and used to act in the environment; otherwise actions
    are sampled uniformly. The resulting dataset is stored as compressed ``.npz``
    shards compatible with :func:`load_dataset_to_gpu`.

    Parameters
    ----------
    num_envs:
        Number of environment instances to run in parallel when collecting the
        dataset. ``1`` replicates the previous single‑environment behaviour.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if num_envs > 1:
        env = make_atari_vectorized_envs(
            game, num_envs=num_envs, max_episode_steps=None, render_mode="rgb_array"
        )
    else:
        env = make_atari_env(game, max_episode_steps=None, render_mode="rgb_array")
    obs, _ = env.reset()

    agent = PPO.load(policy, env=env) if policy is not None else None

    frames_buffer: list[np.ndarray] = []
    actions_buffer: list[int] = []
    rewards_buffer: list[float] = []
    dones_buffer: list[bool] = []
    shard_idx = 0

    pbar = tqdm(total=steps, desc=game)
    collected = 0
    while collected < steps:
        frame = env.render()
        if num_envs == 1:
            frame = frame.transpose(2, 0, 1)[None]
        else:
            frame = frame.transpose(0, 3, 1, 2)

        if frame.shape[2] != reso:
            frame = torch.nn.functional.interpolate(
                torch.tensor(frame),
                (reso, reso),
                mode="bilinear",
                align_corners=False,
            )

        if agent is None:
            action = env.action_space.sample()
        else:
            action, _ = agent.predict(obs, deterministic=True)

        obs, reward, term, trunc, _ = env.step(action)
        if num_envs == 1:
            iter_range = [0]
        else:
            iter_range = range(num_envs)

        for i in iter_range:
            frames_buffer.append(frame[i].cpu().numpy())
            actions_buffer.append(int(action[i] if num_envs > 1 else action))
            rewards_buffer.append(float(reward[i] if num_envs > 1 else reward))
            done_i = bool(term[i] or trunc[i]) if num_envs > 1 else bool(term or trunc)
            dones_buffer.append(done_i)
            collected += 1
            pbar.update(1)
            if collected >= steps:
                break

        if len(frames_buffer) >= shard:
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

        if num_envs > 1:
            env.reset_done()
        elif term or trunc:
            obs, _ = env.reset()

    pbar.close()

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
def load_dataset_to_gpu(folder: str, *, batch_size: int = 2048):
    """Load dataset and convert all arrays to GPU tensors.

    Parameters
    ----------
    folder:
        Path to the directory containing ``.npz`` shards.
    batch_size:
        Number of frames processed at once when converting them to VQ-VAE
        indices. Reducing this value lowers peak GPU memory usage.
    """
    frames, actions, rewards, dones = read_npz_dataset(folder)

    # Datasets collected with ``gather_offline_dataset`` store frames in
    # ``(C, H, W)`` order, whereas :func:`frames_to_indices` expects the
    # channel dimension last. Convert the layout if necessary to avoid
    # shape mismatches when passing the frames through the VQ-VAE encoder.
    if frames.ndim == 4 and frames.shape[-1] not in (1, 3) and frames.shape[1] in (1, 3):
        frames = frames.transpose(0, 2, 3, 1)

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
