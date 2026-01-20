import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha, gamma, eps, eps_min, eps_decay, actions):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.actions = actions
        self.q = defaultdict(lambda: np.zeros(actions))

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.actions - 1)
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s_next):
        best_next = np.max(self.q[s_next])
        self.q[s][a] += self.alpha * (r + self.gamma * best_next - self.q[s][a])

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
