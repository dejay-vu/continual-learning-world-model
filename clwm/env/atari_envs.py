import gymnasium as gym
from gymnasium.wrappers import TransformReward
import ale_py
from ..utils.common import symlog

gym.register_envs(ale_py)


def wrap_reward_symlog(env: gym.Env) -> gym.Env:
    """Return a wrapper that applies :func:`symlog` to rewards."""

    return TransformReward(env, lambda r: symlog(r))


def make_atari_env(
    name: str,
    *,
    frameskip: int = 4,
    sticky: bool = True,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a single Atari environment with symlog reward transformation."""
    base = f"ALE/{name}-v5"
    env = gym.make(
        base,
        max_episode_steps=max_episode_steps,
        frameskip=frameskip,
        repeat_action_probability=0.25 if sticky else 0.0,
        full_action_space=True,
        render_mode=render_mode,
    )
    env = wrap_reward_symlog(env)
    return env


def make_atari_vectorized_envs(
    name: str,
    *,
    frameskip: int = 4,
    sticky: bool = True,
    max_episode_steps: int | None = None,
    num_envs: int = 128,
    render_mode: str | None = None,
) -> gym.vector.VectorEnv:
    """Create vectorized Atari environments."""
    base = f"ALE/{name}-v5"

    return gym.make_vec(
        base,
        num_envs=num_envs,
        vectorization_mode="async",
        wrappers=[wrap_reward_symlog],
        max_episode_steps=max_episode_steps,
        frameskip=frameskip,
        repeat_action_probability=0.25 if sticky else 0.0,
        full_action_space=True,
        render_mode=render_mode,
    )
