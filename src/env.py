import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformObservation 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor


def create_pruned_cartpole():
    
    base_env = gym.make("CartPole-v1" )

    low = np.array([base_env.observation_space.low[0], base_env.observation_space.low[1]])
    high = np.array([base_env.observation_space.high[0], base_env.observation_space.high[1]])
    pruned_space = Box(low=low, high=high, dtype=np.float32)

    # Create a wrapper environment that removes the pole data from the observation space
    return Monitor(TransformObservation(base_env, lambda obs: obs[:2], pruned_space))

def make_stacked_cartpole(n_envs=1, num_frames=4):
    env_fn = lambda: create_pruned_cartpole()
    vec_env = make_vec_env(env_fn, n_envs=n_envs)
    stacked_env = VecFrameStack(vec_env, n_stack=num_frames)
    return stacked_env

