import torch
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy


from env import make_stacked_cartpole

import cartpole
import gymnasium as gym

def objective(trial: optuna.Trial) -> float:

    env = make_stacked_cartpole(n_envs=8, num_frames=6)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)

    policy_kwargs = {
        "net_arch": [512, 512],
        "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [256, 512]),
        "n_lstm_layers": trial.suggest_int("n_lstm_layers", 1, 2),
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "activation_fn": torch.nn.Tanh,
        "ortho_init": True,
        "normalize_images": False,
        "features_extractor_class": FlattenExtractor,
    }

    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        clip_range=clip_range,
        n_epochs=n_epochs,
        verbose=0,
        policy_kwargs=policy_kwargs,
        device='cuda',
        batch_size=8*n_steps
    )

    model.learn(total_timesteps=500_000)

    eval_env = make_stacked_cartpole(n_envs=1, num_frames=6)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    return mean_reward


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=1)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Reward: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
