
import os
import torch
from stable_baselines3.common.torch_layers import FlattenExtractor
# from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

from run import Run
from env import create_pruned_cartpole, make_stacked_cartpole


def main():

    url = os.environ['MLFLOW_TRACKING_URI']

    env = make_stacked_cartpole(n_envs = 8, num_frames = 6)

    params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.06,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "policy_kwargs": None,
        "seed": None,
        "device": "auto"
    }

    # del params["num_stacked"]
    model = RecurrentPPO(
        MlpLstmPolicy, env, verbose=1, **params
    )
    # model = PPO("MlpPolicy", env, verbose=1, **params)

    new_run = Run(env, model, params, url, "test")

if __name__ == "__main__":
    main()

