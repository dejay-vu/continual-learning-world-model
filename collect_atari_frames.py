#!/usr/bin/env python
"""
collect_atari_frames.py
-----------------------
Random-policy screenshot dump for every ALE/… Gymnasium ID that ends in '-v5'
(ignores RAM variants).  Frames are resized to 84*84 and saved in shards of
SHARD=1 000 for later VQ-VAE training.

Run:
    pip install gymnasium[box2d,atari] tqdm numpy torch
    python collect_atari_frames.py --frames 10000   # per game
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import ale_py

gym.register_envs(ale_py)

# ---------------- CLI -----------------
cli = argparse.ArgumentParser()
cli.add_argument("--frames", type=int, default=10_000, help="frames per game")
cli.add_argument("--out", type=str, default="atari_frames_84x84")
cli.add_argument("--reso", type=int, default=84)
cli.add_argument("--shard", type=int, default=1_000)
args = cli.parse_args()

RES = args.reso
PER_GAME = args.frames
SHARD = args.shard
OUT_DIR = Path(args.out)
OUT_DIR.mkdir(exist_ok=True)


# ------------- helper ----------------
def resize84(rgb: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 CHW in [0,1], resized"""
    t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    t = F.interpolate(
        t[None], (RES, RES), mode="bilinear", align_corners=False
    )[0]
    return t.numpy()  # 3×84×84 float32


# ----------- list ALE games ----------
all_ids = [
    gid
    for gid in gym.registry
    if gid.startswith("ALE/")
    and gid.endswith("-v5")
    and "-ram" not in gid
    and "Combat" not in gid
    and "Joust" not in gid
    and "MazeCraze" not in gid
    and "Warlords" not in gid
]
print(f"{len(all_ids)} ALE tasks found")

# ------------- main loop -------------
for game in tqdm(all_ids, desc="Games"):
    game_dir = OUT_DIR / game.replace("/", "_")
    game_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(game, render_mode="rgb_array")

    buf, count, shard_idx = [], 0, 0
    obs, _ = env.reset()

    pbar = tqdm(total=PER_GAME, desc=game, leave=False)
    while count < PER_GAME:
        buf.append(resize84(env.render()))
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()

        count += 1
        pbar.update(1)

        if len(buf) == SHARD:
            np.savez_compressed(
                game_dir / f"{shard_idx:02d}.npz", frames=np.stack(buf)
            )
            buf.clear()
            shard_idx += 1

    # save any remainder
    if buf:
        np.savez_compressed(
            game_dir / f"{shard_idx:02d}.npz", frames=np.stack(buf)
        )

    env.close()
    pbar.close()

print("✓ All games collected.")
