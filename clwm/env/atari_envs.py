import gymnasium as gym
from gymnasium.wrappers import TransformReward
import ale_py
from ..utils.common import symlog

gym.register_envs(ale_py)


def wrap_reward_symlog(env):
    return TransformReward(env, lambda r: symlog(r))


def make_atari_env(
    name: str,
    *,
    frameskip: int = 4,
    sticky: bool = True,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
):
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
    name: str | list[str],
    *,
    frameskip: int = 4,
    sticky: bool = True,
    max_episode_steps: int | None = None,
    num_envs: int = 128,
    render_mode: str | None = None,
):
    if isinstance(name, str):
        base = f"ALE/{name}-v5"
        envs = gym.make_vec(
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
        return envs

    env_fns = []
    for game in name:
        base = f"ALE/{game}-v5"

        def _make(base=base):
            env = gym.make(
                base,
                max_episode_steps=max_episode_steps,
                frameskip=frameskip,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=True,
                render_mode=render_mode,
            )
            return wrap_reward_symlog(env)

        env_fns.append(_make)

    return gym.vector.AsyncVectorEnv(env_fns)
