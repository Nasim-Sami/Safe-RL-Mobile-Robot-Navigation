import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from envs.navigation_env_360 import SafeNav360Env

def make_env_fn(seed=0, randomize_obstacles=True, max_steps=500):
    def thunk():
        env = SafeNav360Env(randomize_obstacles=randomize_obstacles, max_steps=max_steps)
        env.reset(seed=seed)
        return env
    return thunk

def make_vector_env(num_envs=8, seed=42, randomize_obstacles=True, max_steps=500, async_vec=False):
    """
    Creates vectorized environments for high-throughput batch rollouts on GPU.
    """
    env_fns = [
        make_env_fn(seed=seed + i, randomize_obstacles=randomize_obstacles, max_steps=max_steps)
        for i in range(num_envs)
    ]
    if async_vec and num_envs > 1:
        return AsyncVectorEnv(env_fns)
    return SyncVectorEnv(env_fns)
