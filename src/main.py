import os
import sys

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.torch_layers import FlattenExtractor
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
import gymnasium as gym
import numpy as np
import mlflow
import torch

from env import create_pruned_cartpole, create_stacked_cartpole
from log import MLflowOutputFormat

# Monkey patch gym.__version__
import gym as gym_shim

gym_shim.__version__ = gym.__version__

# Hyperparameters

# params = {
#     "learning_rate" : 3e-4 ,
#     "gamma" : 0.95,
#     "n_steps" : 2048,
#     "batch_size" : 128,
#     "ent_coef" : 0.01,
#     "n_epochs" : 40,
#     "normalize_advantage": True,
#     "policy_kwargs": {"net_arch": [128, 128]},
#     "num_stacked": 6
# }

params = {
    "learning_rate": 1e-4,
    "gamma": 0.98,
    "n_steps": 16384,
    "batch_size": 2048,
    "ent_coef": 0.05,
    "n_epochs": 30,
    "normalize_advantage": True,
    "max_grad_norm": 0.5,
    "vf_coef": 0.7,
    "gae_lambda": 0.95,
    "device": "cuda",
    "policy_kwargs": {
        "net_arch": [512, 512],
        "lstm_hidden_size": 512,
        "n_lstm_layers": 2,
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "activation_fn": torch.nn.Tanh,
        "ortho_init": True,
        "normalize_images": False,
        "features_extractor_class": FlattenExtractor,
    },
    "num_stacked": 6,
}

if params["device"] == "cuda":
    torch.set_float32_matmul_precision("high")

mlflow.set_experiment("CartPoleStackedRecurrent")
mlflow.set_tracking_uri()


loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)


# env = gym.make('CartPole-v1')
env = create_stacked_cartpole(params["num_stacked"])
del params["num_stacked"]

# Wrap env in monitor for logging
env = Monitor(env)


with mlflow.start_run():
    mlflow.log_params(params)

    model = RecurrentPPO(MlpLstmPolicy, env, verbose=1, **params)

    # model = PPO("MlpPolicy",
    #             env,
    #             verbose=1,
    #             **params
    #         )

    # Set custom logger
    model.set_logger(loggers)
    model.learn(total_timesteps=300000, log_interval=1)

    # Evaluate the model after training
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, return_episode_rewards=False
    )

    # Log evaluation metrics
    mlflow.log_metric("mean_reward", mean_reward)
    mlflow.log_metric("std_reward", std_reward)

    model.save("model.zip")
    mlflow.log_artifact("model.zip")
