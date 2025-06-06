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
    name: str | list[str],
    *,
    frameskip: int = 4,
    sticky: bool = True,
    max_episode_steps: int | None = None,
    num_envs: int = 128,
    render_mode: str | None = None,
) -> gym.vector.VectorEnv:
    """Create vectorized Atari environments.

    When ``name`` is a list, each element becomes one environment in the
    vector. The function automatically falls back to a synchronous vector
    environment if asynchronous creation fails.
    """
    if isinstance(name, str):
        base = f"ALE/{name}-v5"
        envs = gym.make_vec(
            base,
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={"shared_memory": False},  # Avoid /dev/shm issues
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

    # Using multiprocessing based async vector envs can still cause
    # `PermissionError: [Errno 13] Permission denied` on platforms where the
    # Python multiprocessing module is not allowed to create POSIX semaphores
    # (e.g. when /dev/shm is not writable). In such cases we gracefully fall
    # back to the in-process `SyncVectorEnv` implementation which does not rely
    # on multiprocessing and therefore works in more restricted execution
    # environments. The performance impact is negligible for offline dataset
    # collection where throughput is not the primary bottleneck.

    try:
        return gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)
    except PermissionError:
        # Fall back to synchronous vector environment.
        return gym.vector.SyncVectorEnv(env_fns)
