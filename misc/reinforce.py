import gymnasium as gym
from torch.distributions import Categorical
import torch.nn as nn
import torch

LR = 1e-4


class Agent:
    """implements simple policy gradient"""

    def __init__(self, env: gym.Env, lr=LR):
        self.env = env
        self.eval_env = gym.make("CartPole-v1", render_mode="human")
        self.pi_net = PolicyNetwork(4, 2)  # TODO: Hardcoded for now, switch this later
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
            if ep % 100 == 0:
                self.evaluate()
            episode_return = self.step()

        self.env.close()

    def evaluate(self, episodes=5, render=False):
        """Run evaluation episodes on self.eval_env without training.

        Args:
            episodes (int): Number of episodes to run.
            render (bool): Whether to render the environment.

        Returns:
            float: Average episode return.
        """
        total_return = 0.0

        for ep in range(episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_return = 0.0

            while not done:
                with torch.no_grad():
                    action = self.get_action(obs)  # same policy as training

                obs, reward, term, trunc, info = self.eval_env.step(action.item())
                episode_return += float(reward)

                if render:
                    self.eval_env.render()

                if term or trunc:
                    done = True

            total_return += episode_return
            print(f"[Eval] Episode {ep + 1}: return={episode_return:.2f}")

        avg_return = total_return / episodes
        print(f"[Eval] Average return over {episodes} episodes: {avg_return:.2f}")
        return avg_return


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    agent = Agent(env)

    agent.learn(1000)
    print("Done training")
    agent.view()
