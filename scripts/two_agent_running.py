import time, yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.two_agent_env import TronTwoAgentEnv
from envs.tron_multi_agent_env import SelfPlayEnv


def run_single_agent(
    cfg,
    model_path: str,
    episodes: int = 1,
    frame_time: float = 0.05
):

    model = PPO.load(model_path, verbose=0)

    stub = PPO.load(model_path, verbose=0)
    vec = DummyVecEnv([lambda: SelfPlayEnv(cfg, stub)])

    model.set_env(vec)

    for ep in range(episodes):
        reset_ret = vec.reset()
        if isinstance(reset_ret, tuple):
            obs, _ = reset_ret
        else:
            obs = reset_ret

        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, trunc, _ = vec.step(action)
            total_r += rew[0] if isinstance(rew, (list, tuple)) else rew
            vec.render()
            time.sleep(frame_time)

        print(f"[Single] Episode {ep+1}/{episodes} → reward {total_r:.2f}")

    vec.close()


def run_agent_vs_agent(
    cfg,
    model_path: str,
    episodes: int = 1,
    frame_time: float = 0.05
):
    model = PPO.load(model_path, verbose=0)

    vec = DummyVecEnv([lambda: TronTwoAgentEnv(cfg)])

    model.set_env(vec)

    for ep in range(episodes):
        reset_ret = vec.reset()
        if isinstance(reset_ret, tuple):
            obs, _ = reset_ret
        else:
            obs = reset_ret

        done = False
        tot0 = tot1 = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # e.g. [a0,a1]
            obs, (r0, r1), done, trunc, _ = vec.step(action)
            tot0 += r0
            tot1 += r1
            vec.render()
            time.sleep(frame_time)

        print(f"[Two‐Agent] Ep {ep+1}/{episodes} → A0={tot0:.1f}, A1={tot1:.1f}")

    vec.close()
