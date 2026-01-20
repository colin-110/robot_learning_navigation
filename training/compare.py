import yaml
import numpy as np
from env.grid_world import GridWorldEnv
from agents.q_learning import QLearningAgent
from planners.astar import astar

EPISODES = 100

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

env = GridWorldEnv(
    size=cfg["env"]["size"],
    n_static=cfg["env"]["static_obstacles"],
    n_dynamic=cfg["env"]["dynamic_obstacles"],
    max_steps=cfg["env"]["max_steps"]
)

# ---------- Q-LEARNING EVALUATION ----------
agent = QLearningAgent(
    alpha=cfg["agent"]["alpha"],
    gamma=cfg["agent"]["gamma"],
    eps=0.0,  # no exploration during eval
    eps_min=0.0,
    eps_decay=1.0,
    actions=env.action_space
)

ql_success = 0

for _ in range(EPISODES):
    s = env.reset()
    done = False
    while not done:
        a = np.argmax(agent.q[s])
        s, r, done = env.step(a)
    if r == 100:
        ql_success += 1

# ---------- A* EVALUATION ----------
astar_success = 0

for _ in range(EPISODES):
    env.reset()
    path = astar(
        env.robot,
        env.goal,
        env.size,
        env.static_obs
    )

    if path is None:
        continue

    collided = False
    for pos in path[1:]:
        if pos in env.dynamic_obs:
            collided = True
            break

    if not collided:
        astar_success += 1

# ---------- RESULTS ----------
print("===== COMPARISON RESULTS =====")
print(f"Q-Learning Success Rate: {ql_success / EPISODES:.2f}")
print(f"A* Success Rate        : {astar_success / EPISODES:.2f}")
