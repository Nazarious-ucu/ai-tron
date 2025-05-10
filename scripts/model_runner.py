from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

import time
import yaml

from envs import TronBaseEnv
from envs.tron_three_level_env import TronBaseEnvMultiChannel, TronBaseEnvSimpleMultiChannel


def run_game_with_agent(config: str | Path, model: str | Path, game_evn, num_episodes = 1, show_ui=True):
    def make_env():
        return game_evn(config)

    vec_env = DummyVecEnv([make_env])


    model = PPO.load(model, env=vec_env)

    for i in range(num_episodes):
        obs = vec_env.reset()
        dones = [False]
        total_reward = 0.0
        while not dones[0]:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            if show_ui : vec_env.envs[0].render(dones[0])

            total_reward += rewards[0]
            time.sleep(0.2)

            if dones[0]:
                # print(obs)
                print("End: ", total_reward)


if __name__ == "__main__":
    with open("../configs/field_settings.yaml") as f:
        config1 = yaml.safe_load(f)
    run_game_with_agent(config1, "tron_ppo_model10.zip", game_evn= TronBaseEnvSimpleMultiChannel, num_episodes = 100, show_ui=False)