# ppo_cartpole.py
import os
import sys
from typing import Dict, Union, Tuple, Any

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
import gymnasium as gym
import numpy as np
import mlflow

from env import create_pruned_cartpole, create_stacked_cartpole
from log import create_logger, plot_rewards

mlflow.set_experiment('CartPoleStacked')
mlflow.set_tracking_uri('http://localhost:8080')

class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)

# Hyperparameters

params = {
    "learning_rate" : 3e-4 ,
    "gamma" : 0.95,
    "n_steps" : 2048,
    "batch_size" : 64,
    "ent_coef" : 0.01,
    "n_epochs" : 20,
}



env = gym.make('CartPole-v1')


# Wrap env in monitor for logging
env = Monitor(env)


with mlflow.start_run():
    mlflow.log_params(params)

    # model = RecurrentPPO(MlpLstmPolicy,
    #             env, 
    #             learning_rate=params['learning_rate'],
    #             gamma=params['gamma'],
    #             n_steps=params['n_steps'],
    #             n_epochs=params['n_epochs'],
    #             batch_size=params['batch_size'],
    #             ent_coef=params['ent_coef'],
    #             verbose=1
    #         )

    model = PPO("MlpPolicy",
                env, 
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                n_steps=params['n_steps'],
                n_epochs=params['n_epochs'],
                batch_size=params['batch_size'],
                ent_coef=params['ent_coef'],
                verbose=1
            )

    # Set custom logger
    model.set_logger(loggers)
    model.learn(total_timesteps=100000, log_interval=1)

    




