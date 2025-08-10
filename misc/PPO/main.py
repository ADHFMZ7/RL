import gymnasium as gym
from reinforce import agent

env = gym.make('CartPole-v1', render_mode='human')

obs, info = env.reset()

done = False
total_reward = 0

while not done:

    action = None

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += float(reward)
    done = terminated or truncated

print('episode over')
env.close()
