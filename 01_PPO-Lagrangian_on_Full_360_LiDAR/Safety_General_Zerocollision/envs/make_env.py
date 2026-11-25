import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from envs.navigation_env_general import SafeNavGeneralEnv

def make_env(env_id=0, randomize_obstacles=True, max_steps=500, seed=42):
    def _thunk():
        env = SafeNavGeneralEnv(randomize_obstacles=randomize_obstacles, max_steps=max_steps)
        env.reset(seed=seed + env_id)
        return env
    return _thunk

def make_vector_env(num_envs=8, randomize_obstacles=True, max_steps=500, seed=42):
    env_fns = [
        make_env(env_id=i, randomize_obstacles=randomize_obstacles, max_steps=max_steps, seed=seed)
        for i in range(num_envs)
    ]
    return SyncVectorEnv(env_fns)
