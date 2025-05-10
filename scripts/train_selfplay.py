import os
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

import yaml
import ray
from ray import tune
from envs.tron_multi_agent_env import TronMultiAgentEnv

if __name__ == "__main__":
    ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)

    config = yaml.safe_load(open("configs/field_settings.yaml"))

    tune.register_env("tron_ma", lambda ctx: TronMultiAgentEnv(config))

    policies = {
        "shared_policy": (
            None,
            TronMultiAgentEnv(config).observation_space,
            TronMultiAgentEnv(config).action_space,
            {}
        )
    }
    def policy_mapping_fn(agent_id):
        return "shared_policy"

    tune.run(
        "PPO",
        stop={"training_iteration": 50},
        config={
            "env": "tron_ma",
            "framework": "torch",
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "rollout_fragment_length": 200,
            "num_sgd_iter": 10,
            "lr": 3e-4,
        },
        local_dir="rllib_selfplay_logs",
        checkpoint_freq=5,
        verbose=1,
    )
