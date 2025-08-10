import gymnasium as gym
from torch.distributions import Categorical
import torch.nn as nn
import torch


class Agent:
    '''implements simple policy gradient'''
    def __init__(self, env: gym.Env, lr=1e-4):

        self.env = env
        self.pi_net = PolicyNetwork(4, 2) # TODO: Hardcoded for now, switch this later
        self.optim = torch.optim.Adam(self.pi_net.parameters(), lr=lr, maximize=True)
        
    def step(self):

        obs, info = self.env.reset()
        episode_reward = 0

        store = []
        rewards = []

        done = False
        while not done:

            action = self.get_action(obs)
            log_prob = self.get_policy(torch.tensor(obs)).log_prob(action)
       
            store.append((action, log_prob))

            obs, rew, term, trun, info = self.env.step(action.item())

            rewards.append(rew)

            if term or trun:
                done = True

            episode_reward += float(rew)

        loss = torch.zeros(1)
        self.optim.zero_grad()

        cum_reward = 0
        for (action, log_prob), reward in zip(reversed(store), reversed(rewards)):
            cum_reward += reward
            loss += log_prob * cum_reward

        print(f'episode terminated with return: {episode_reward} and loss {loss}')
        loss.backward()
        self.optim.step()
        return episode_reward

    def get_policy(self, obs):
        logits = self.pi_net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        obs = torch.from_numpy(obs)
        return self.get_policy(obs).sample()

    def learn(self, episodes):
        for ep in range(episodes):
            episode_return = self.step()

        self.env.close()

    def view(self, episodes = 20):
        env = gym.make('CartPole-v1', render_mode='human')
        for ep in range(episodes):
            self.view_step(env)

    def view_step(self, env):
        obs, info = env.reset()
        done = False
        while not done:

            action = self.get_action(obs)
            obs, rew, term, trun, info = self.env.step(action.item())

            if term or trun:
                done = True


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == '__main__':

    env = gym.make('CartPole-v1') 

    agent = Agent(env)

    agent.learn(1000)
    print('Done training')
    agent.view()


