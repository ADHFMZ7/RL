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

    # params = {'learning_rate': 0.000400221439537741, 'n_steps': 512, 'gamma': 0.9757367668177018, 'gae_lambda': 0.8745117955609628, 'ent_coef': 0.0010093238389732085, 'vf_coef': 0.1845885889475887, 'max_grad_norm': 0.6961241873745438, 'clip_range': 0.127594201231583, "policy_kwargs": {"net_arch": [512, 512, 512]}}

    # params = {
    #     "learning_rate": 2.38e-5,
    #     "n_steps": 128,
    #     "gamma": 0.915,
    #     'batch_size': 2048,
    #     "gae_lambda": 0.889,
    #     "ent_coef": 0.065,
    #     "vf_coef": 0.14,
    #     "max_grad_norm": 0.72,
    #     "clip_range": 0.17,
    #     "policy_kwargs": {"net_arch": [512, 512, 512]}
    # }

    # params = {
    #     "learning_rate": 1e-4,
    #     "gamma": 0.9757367668177018,
    #     "n_steps": 2048,
    #     "ent_coef": 0.001,
    #     "n_epochs": 30,
    #     "normalize_advantage": True,
    #     "max_grad_norm": 0.6961,
    #     "vf_coef": 0.1845,
    #     "gae_lambda": 0.8745,
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
    #     'batch_size': 8 * 2048
    # }

    params = {
        'learning_rate': 0.000328323884080817,
        'n_steps': 128,
        'gamma': 0.9636242163718899,
        'gae_lambda': 0.888288034714926,
        'ent_coef': 0.020963306906827742,
        'vf_coef': 0.24336227091003873,
        'max_grad_norm': 0.5176571493287323,
        'clip_range':0.24239374366478478,
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
        'batch_size': 8 * 2048
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

    # del params["num_stacked"]
    model = RecurrentPPO(MlpLstmPolicy, env, verbose=1, **params)
    # model = PPO("MlpPolicy", env, verbose=1, **params)


    new_run = Run(env, model, params, url, "test")


if __name__ == "__main__":
    main()
