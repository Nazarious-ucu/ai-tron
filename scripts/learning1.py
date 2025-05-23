﻿from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import yaml
import multiprocessing

from envs.tron_three_level_env import TronBaseEnvSimpleMultiChannel

with open("../configs/new.yaml") as f:
    config = yaml.safe_load(f)

def make_env(rank):
    def _init():
        return Monitor(TronBaseEnvSimpleMultiChannel(config))
    return _init

if __name__ == "__main__":
    multiprocessing.freeze_support()

    env = SubprocVecEnv([make_env(i) for i in range(4)])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.995,
        clip_range=0.15,
        ent_coef=0.05,
        verbose=1,
        tensorboard_log="./tron_tensorboard/",
        device="cpu"
    )

    model.learn(total_timesteps=200000)
    model.save("tron_ppo_model10")