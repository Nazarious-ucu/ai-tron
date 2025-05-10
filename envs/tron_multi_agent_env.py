
import yaml
from pathlib import Path
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.two_agent_env import TronTwoAgentEnv
class SelfPlayEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, opponent):
        super().__init__()
        self._last_obs = None
        self.config = config
        self.opponent = opponent
        self.ma_env = TronTwoAgentEnv(config)

        self.observation_space = self.ma_env.observation_space
        self.action_space = spaces.Discrete(4)

    def reset(self, *args, **kwargs):
        obs, info = self.ma_env.reset(*args, **kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        a1, _ = self.opponent.predict(self._last_obs, deterministic=True)
        obs, (r0, r1), done, truncated, info = self.ma_env.step((action, int(a1)))
        self._last_obs = obs
        return obs, r0, done, truncated, info

    def step(self, action):
        a1, _ = self.opponent.predict(self._last_obs, deterministic=True)

        obs, (r0, r1), done, truncated, info = self.ma_env.step((action, int(a1)))
        self._last_obs = obs
        return obs, r0, done, truncated, info

    def render(self, *args, **kwargs):
        self.ma_env.render()

    def close(self):
        self.ma_env.close()

def train_self_play(config_path, rounds=5, timesteps=100_000, save_dir="selfplay_sb3"):
    Path(save_dir).mkdir(exist_ok=True)
    config = yaml.safe_load(open(config_path))

    rnd_env = DummyVecEnv([lambda: SelfPlayEnv(config,
        PPO("MlpPolicy", DummyVecEnv([lambda:TronTwoAgentEnv(config)]), verbose=0)
    )])
    opponent = PPO("MlpPolicy", rnd_env, verbose=0)

    for r in range(rounds):
        print(f"=== Round {r} ===")
        vec_env = DummyVecEnv([lambda: Monitor(SelfPlayEnv(config, opponent))])
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./selfplay_sb3_tb/")
        model.learn(total_timesteps=timesteps)
        path = Path(save_dir) / f"agent_round{r}.zip"
        model.save(path)
        print(f"Saved {path}")
        opponent = PPO.load(str(path))


if __name__ == "__main__":
    train_self_play("../configs/field_settings.yaml", rounds=3, timesteps=100_000)
