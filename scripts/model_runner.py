from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from envs import TronBaseEnv
import time
import yaml

from envs.tron_env_enemy import TronBaseEnv
from envs.tron_three_level_env import TronBaseEnvMultiChannel

with open("../configs/field_settings.yaml") as f:
    config = yaml.safe_load(f)

def make_env():
    return TronBaseEnvMultiChannel(config)

vec_env = DummyVecEnv([make_env])


model = PPO.load("tron_ppo_model6.zip   ", env=vec_env)

obs = vec_env.reset()
# print(obs)
total_reward = 0
for i in range(2000):

    action, _states = model.predict(obs)
    obs, rewards, dones, infos = vec_env.step(action)
    vec_env.envs[0].render(dones[0])
    total_reward += rewards[0]
    # print("Total reward: ", )
    time.sleep(0.05)

    # if (rewards[0] > 0):
    #     print("Reward: ", rewards[0])

    if dones[0]:
        obs = vec_env.reset()
        print("End: ", total_reward)
        total_reward = 0
