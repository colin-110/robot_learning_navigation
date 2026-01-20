import yaml
import numpy as np
from env.grid_world import GridWorldEnv
from agents.q_learning import QLearningAgent
from tqdm import trange
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

env = GridWorldEnv(
    size=cfg["env"]["size"],
    n_static=cfg["env"]["static_obstacles"],
    n_dynamic=cfg["env"]["dynamic_obstacles"],
    max_steps=cfg["env"]["max_steps"]
)

agent = QLearningAgent(
    alpha=cfg["agent"]["alpha"],
    gamma=cfg["agent"]["gamma"],
    eps=cfg["agent"]["epsilon_start"],
    eps_min=cfg["agent"]["epsilon_min"],
    eps_decay=cfg["agent"]["epsilon_decay"],
    actions=env.action_space
)

rewards = []
success = []

for ep in trange(cfg["training"]["episodes"]):
    s = env.reset()
    total = 0
    done = False

    while not done:
        a = agent.select_action(s)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, s_next)
        s = s_next
        total += r

    agent.decay_eps()
    rewards.append(total)
    success.append(1 if r == 100 else 0)

plt.plot(rewards)
plt.title("Episode Reward")
plt.savefig("results/plots/reward_curve.png")

plt.plot(np.cumsum(success) / np.arange(1, len(success)+1))
plt.title("Success Rate")
plt.savefig("results/plots/success_rate.png")
