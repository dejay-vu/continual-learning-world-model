#!/usr/bin/env python
"""
collect_offline_dataset.py
--------------------------
Collect observations, actions and rewards from Atari environments to create
an offline dataset. Frames are resized to 84x84 and stored together with the
sampled action and resulting reward.

Example:
    python collect_offline_dataset.py --game Breakout --steps 10000 --out offline_data
"""

import argparse
from clwm.data import gather_offline_dataset


cli = argparse.ArgumentParser()
cli.add_argument("--game", type=str, required=True, help="Atari game name")
cli.add_argument("--steps", type=int, default=10000, help="number of steps")
cli.add_argument("--out", type=str, default="offline_dataset")
cli.add_argument("--reso", type=int, default=84)
cli.add_argument("--shard", type=int, default=1000)
cli.add_argument("--policy", type=str, default=None, help="optional PPO agent")
args = cli.parse_args()

gather_offline_dataset(
    args.game,
    args.steps,
    args.out,
    reso=args.reso,
    shard=args.shard,
    policy=args.policy,
)
