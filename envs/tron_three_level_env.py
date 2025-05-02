import numpy as np
from gymnasium import spaces
from envs import TronEnvWithEnemy

class TronBaseEnvMultiChannel(TronEnvWithEnemy):
    def __init__(self, config):
        super().__init__(config)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width, 3), dtype=np.float32)

    def get_observation(self):
        agent_channel = (self.field.state == 1).astype(np.float32)

        tail_channel = (self.field.state == 2).astype(np.float32)

        enemy_channel = (self.field.state == 3).astype(np.float32)
        return np.stack([agent_channel, tail_channel, enemy_channel], axis=-1)

    def reset(self, seed=None, options=None):
        state, info = super().reset(seed, options)
        return self.get_observation(), info

    def step(self, action):
        state, reward, done, truncated, info = super().step(action)
        return self.get_observation(), reward, done, truncated, info
