import numpy as np
from env.grid_world import GridWorldEnv
from agents.q_learning import QLearningAgent

env = GridWorldEnv()
agent = QLearningAgent(0, 0, 0, 0, 0, env.action_space)

episodes = 100
success = 0

for _ in range(episodes):
    s = env.reset()
    done = False
    while not done:
        a = np.argmax(agent.q[s])
        s, r, done = env.step(a)
    if r == 100:
        success += 1

print(f"Success Rate: {success/episodes:.2f}")
