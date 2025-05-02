import gymnasium as gym
import numpy as np
from gymnasium import spaces
from envs.field_manager import FieldManager

class BaseTronEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.width = config["width"]
        self.height = config["height"]
        self.cell_size = config["cell_size"]
        self.tail_length = config["tail_length"]
        self.reward_for_step = config["reward_for_step"]
        self.penalty_for_death = config["penalty_for_death"]
        self.enemy_count = config.get("enemy_count", 0)

        self.field = FieldManager(self.width, self.height)

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.int8)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        super().reset()
        self.field.reset_field()


    def step(self, action):
        raise NotImplementedError()