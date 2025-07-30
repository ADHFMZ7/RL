import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TransformObservation 
from gymnasium.wrappers import FrameStackObservation

from gymnasium.spaces import Box

def create_pruned_cartpole():
    
    base_env = gym.make("CartPole-v1" )

    low = np.array([base_env.observation_space.low[0], base_env.observation_space.low[1]])
    high = np.array([base_env.observation_space.high[0], base_env.observation_space.high[1]])
    pruned_space = Box(low=low, high=high, dtype=np.float32)

    # Create a wrapper environment that removes the pole data from the observation space
    return TransformObservation(base_env, lambda obs: obs[:2], pruned_space)

def create_stacked_cartpole(num_frames: int = 4):
    return FrameStackObservation(create_pruned_cartpole(), num_frames)

