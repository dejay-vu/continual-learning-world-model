import gymnasium as gym
from gymnasium.wrappers import TransformReward
import ale_py
from .utils import symlog

gym.register_envs(ale_py)


def _wrap_reward(env):
    return TransformReward(env, lambda r: symlog(r))


def make_atari(
    name: str,
    *,
    frameskip: int = 4,
    sticky: bool = True,
    max_episode_steps: int | None = None,
):
    base = f"ALE/{name}-v5"
    env = gym.make(
        base,
        max_episode_steps=max_episode_steps,
        frameskip=frameskip,
        repeat_action_probability=0.25 if sticky else 0.0,
        full_action_space=True,
        render_mode=None,
    )
    env = _wrap_reward(env)
    return env


def make_atari_vectorized(
    name: str,
    *,
    frameskip: int = 4,
    sticky: bool = True,
    max_episode_steps: int | None = None,
    num_envs: int = 128,
):
    base = f"ALE/{name}-v5"
    envs = gym.make_vec(
        base,
        num_envs=num_envs,
        vectorization_mode="async",
        wrappers=[_wrap_reward],
        max_episode_steps=max_episode_steps,
        frameskip=frameskip,
        repeat_action_probability=0.25 if sticky else 0.0,
        full_action_space=True,
        render_mode=None,
    )
    return envs
