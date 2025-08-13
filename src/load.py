import cartpole
import gymnasium as gym
from stable_baselines3 import PPO  # or replace with your algo (A2C, SAC, etc.)

# Load the environment
# Replace "YourCustomCartPole-v0" with your registered custom env name
env = gym.make("custom_cartpole", render_mode="human")

# Load the trained model
model = PPO.load("custom.zip", env=env)

# Number of episodes to watch
num_episodes = 2

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print(f"Episode {ep + 1} reward: {total_reward}")

env.close()
