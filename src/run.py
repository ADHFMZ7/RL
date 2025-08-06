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

from log import MLflowOutputFormat
from env import create_stacked_cartpole

# Monkey patch gym.__version__
# import gym as gym_shim
#
# gym_shim.__version__ = gym.__version__


class Run:
    def __init__(
        self, env: Env, model: BaseAlgorithm, params: dict, url: str, exp_name: str
    ):
        self.loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )

        self.env = Monitor(env)

        self.params = params
        self.model = model

        if model.device.type == "cuda":
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
            mlflow.log_params(params)

            # Set custom logger
            model.set_logger(self.loggers)
            model.learn(total_timesteps=10000, log_interval=1)

            # Evaluate the model after training
            eval_env = create_stacked_cartpole()
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=10,
                return_episode_rewards=False,
                deterministic=True,
            )

            # Log evaluation metrics
            mlflow.log_metric("mean_reward", mean_reward)
            mlflow.log_metric("std_reward", std_reward)

            # model.save("model.zip")
            mlflow.log_artifact("model.zip")
