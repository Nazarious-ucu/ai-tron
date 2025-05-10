# tron_two_agent_env.py

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.tron_env_enemy import TronEnvWithEnemy
from envs.tron_three_level_env import TronBaseEnvMultiChannel

class TronTwoAgentEnv(gym.Env):
    """
    Two-agent Tron env:
      - obs_space: Box(0,1,(H,W,6)) stacking two 3-channel views
      - action_space: MultiDiscrete([4,4]) for (agent0, agent1)
      - step returns (obs, (r0,r1), done, info)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = TronEnvWithEnemy(config)
        self.viz = TronBaseEnvMultiChannel(config)

        H, W = config["height"], config["width"]
        self.observation_space = spaces.Box(0.0, 1.0, (H, W, 6), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([4, 4])

    def reset(self, **kwargs):
        self.base.reset(**kwargs)
        obs3 = self.viz.get_observation()

        return np.concatenate([obs3, obs3], axis=-1), {}

    def step(self, actions):
        a0, a1 = int(actions[0]), int(actions[1])

        orig_move = self.base.enemy.move
        self.base.enemy.move = lambda *args, **kw: a1

        obs0, r0, done, truncated, info = self.base.step(a0)


        self.base.enemy.move = orig_move

        obs3 = self.viz.get_observation()
        stacked = np.concatenate([obs3, obs3], axis=-1)

        r1 = -r0

        return stacked, (r0, r1), done,truncated, info

    def render(self, mode="human"):
        self.base.render()

    def close(self):
        self.base.close()
