
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import yaml
from envs.tron_three_level_env import TronBaseEnvMultiChannel

with open("../configs/field_settings.yaml") as f:
    config = yaml.safe_load(f)

env = TronBaseEnvMultiChannel(config)
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_shape[0]*in_shape[1]*in_shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class REINFORCEAgent:
    def __init__(self, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNet(obs_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = self.policy(state_v)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update_policy(self, log_probs, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_v = torch.tensor(returns, dtype=torch.float32).to(device)
        returns_v = (returns_v - returns_v.mean()) / (returns_v.std() + 1e-8)
        logp_v = torch.stack(log_probs)
        loss = -(logp_v * returns_v).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class QNet(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_shape[0]*in_shape[1]*in_shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64, target_update=100):
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.q_net = QNet(obs_shape, n_actions).to(device)
        self.target_net = QNet(obs_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.learn_step = 0
        self.target_update = target_update

    def select_action(self, state, eps):
        if random.random() < eps:
            return random.randrange(n_actions)
        s_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q = self.q_net(s_v)
        return int(q.argmax().item())

    def store(self, s,a,r,s2,done):
        self.buffer.append((s,a,r,s2,done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_s, dones = zip(*batch)
        s_v = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        a_v = torch.tensor(actions).to(device)
        r_v = torch.tensor(rewards).to(device)
        s2_v = torch.tensor(np.array(next_s), dtype=torch.float32).to(device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

        q_vals = self.q_net(s_v).gather(1, a_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_max = self.target_net(s2_v).max(1)[0]
            target = r_v + self.gamma * next_max * (~done_mask)
        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


def train(agent_type, episodes=500):
    if agent_type == 'reinforce':
        agent = REINFORCEAgent()
    else:
        agent = DQNAgent()
    eps = 1.0
    eps_decay = 0.995
    eps_min = 0.01

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        log_probs, rewards = [], []
        total_reward = 0

        while not done:
            if agent_type == 'reinforce':
                action, logp = agent.select_action(state)
            else:
                action = agent.select_action(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if agent_type == 'reinforce':
                log_probs.append(logp)
                rewards.append(reward)
            else:
                agent.store(state, action, reward, next_state, done)
                loss = agent.learn()
            state = next_state

        if agent_type == 'reinforce':
            loss = agent.update_policy(log_probs, rewards)
        if agent_type == 'dqn':
            eps = max(eps*eps_decay, eps_min)

        print(f"Episode {ep} [{agent_type}]: Reward={total_reward:.2f}, Loss={loss:.4f}, Eps={eps:.3f}")

    if agent_type == 'reinforce':
        torch.save(agent.policy.state_dict(), "reinforce_policy.pth")
    else:
        torch.save(agent.q_net.state_dict(), "dqn_qnet.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['reinforce','dqn'], default='reinforce')
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()
    train(args.agent, args.episodes)
