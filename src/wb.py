import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import cartpole
from env import create_pruned_cartpole, make_stacked_cartpole

params = {
    "learning_rate": 1e-3,
    "n_steps": 4096,
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "policy_kwargs": None,
    "seed": 43,
    "device": "auto"
}

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 50_000_000,
    "env_id": "CartPole-v1",
    "num_frames": 16
}
run = wandb.init(
    project="cartpole-pruned",
    config={**config, **params},
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

# def make_env():
    # env = create_pruned_cartpole(render_mode='rgb_array')
    # env = gym.make(config["env_id"], render_mode="rgb_array")
    # env = Monitor(env)  # record stats such as returns
    # return env

# env = DummyVecEnv([make_env])
env = make_stacked_cartpole(n_envs=8, num_frames=config['num_frames'], env_id=config['env_id'], render_mode='rgb_array')
# env = VecVideoRecorder(
#     env,
#     f"videos/{run.id}",
#     record_video_trigger=lambda x: x % 8000 == 0,
#     video_length=200,
# )

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", **params)
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
