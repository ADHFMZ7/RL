from typing import Optional
from mlflow.tracking import MlflowClient
from gymnasium import Env
import mlflow
import torch
import sys

from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from log import MLflowOutputFormat
from env import make_stacked_cartpole


class Run:
    def __init__(
        self,
        env_func,
        policy,
        params: dict,
        url: str,
        exp_name: str,
        timesteps: int = 30000,
    ):
        self.loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )

        self.env = env_func()

        self.params = params

        if isinstance(policy, str):
            self.model = PPO(policy, self.env, verbose=1, **params)
        else:
            self.model = RecurrentPPO(policy, self.env, verbose=1, **params)

        if self.model.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        mlflow.set_experiment(exp_name)
        mlflow.set_tracking_uri(url)

        client = MlflowClient()

        # Try to get the experiment by name
        experiment = client.get_experiment_by_name(exp_name)

        # Create if it doesn't exist
        if experiment is None:
            experiment_id = client.create_experiment(exp_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id):
            params["policy"] = policy
            mlflow.log_params(params)

            # Set custom logger
            self.model.set_logger(self.loggers)
            self.model.learn(total_timesteps=timesteps, log_interval=1)

            # Evaluate the model after training
            eval_env = env_func()
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.env,
                n_eval_episodes=10,
                return_episode_rewards=False,
                deterministic=True,
            )

            # Log evaluation metrics
            mlflow.log_metric("mean_reward", mean_reward)
            mlflow.log_metric("std_reward", std_reward)

            self.model.save("model.zip")
            mlflow.log_artifact("model.zip")
