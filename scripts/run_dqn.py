import torch, time, yaml
from envs.tron_three_level_env import TronBaseEnvMultiChannel
from own_model import DQNAgent

with open("../configs/field_settings.yaml") as f:
    config = yaml.safe_load(f)

env = TronBaseEnvMultiChannel(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent()
agent.q_net.load_state_dict(torch.load("dqn_qnet.pth", map_location=device))
agent.q_net.eval()

for ep in range(20):
    state, _ = env.reset()
    done = False
    total = 0
    while not done:
        action = agent.select_action(state, eps=0.0)  # fully greedy
        state, reward, done, _, _ = env.step(action)
        total += reward
        print("Total reward: {}, reward of step: {}".format(total, reward))
        env.render()
        time.sleep(0.2)
    print(f"Episode {ep} return: {total:.2f}")
