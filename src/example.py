import os
import torch
from stable_baselines3.common.torch_layers import FlattenExtractor
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

from run import Run
from env import create_pruned_cartpole, create_stacked_cartpole


def main():
    print(os.environ)

    url = "http://mlflow.aldasouqi.com"

    env = create_stacked_cartpole()

    params = {
        "learning_rate": 1e-4,
        "gamma": 0.98,
        "n_steps": 16384,
        "batch_size": 512,
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

    # params = {
    #     "learning_rate": 1e-4,
    #     "gamma": 0.98,
    #     "n_steps": 16384,
    #     "batch_size": 2048,
    #     "ent_coef": 0.05,
    #     "n_epochs": 30,
    #     "normalize_advantage": True,
    #     "max_grad_norm": 0.5,
    #     "vf_coef": 0.7,
    #     "gae_lambda": 0.95,
    #     "device": "cuda",
    #     "policy_kwargs": {
    #         "net_arch": [512, 512],
    #         "lstm_hidden_size": 512,
    #         "n_lstm_layers": 2,
    #         "shared_lstm": False,
    #         "enable_critic_lstm": True,
    #         "activation_fn": torch.nn.Tanh,
    #         "ortho_init": True,
    #         "normalize_images": False,
    #         "features_extractor_class": FlattenExtractor,
    #     },
    #     "num_stacked": 6,
    # }

    del params["num_stacked"]
    model = RecurrentPPO(MlpLstmPolicy, env, verbose=1, **params)

    new_run = Run(env, model, params, url, "test")


if __name__ == "__main__":
    main()
