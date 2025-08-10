import cartpole
# import gymnasium as gym
#
# env = gym.make("custom_cartpole", render_mode="human")
#
# obs, info = env.reset()
#
# done = False
#
# while not done:
#     action = env.action_space.sample()
#
#     obs, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         done = True

import gymnasium as gym

from stable_baselines3 import PPO

# env = gym.make("custom_cartpole", render_mode="human")
env = gym.make("custom_cartpole")

model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.06, n_steps=2048)
model.learn(total_timesteps=300_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

model.save("custom.zip")
