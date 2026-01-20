import numpy as np
import random

class GridWorldEnv:
    def __init__(self, size=10, n_static=10, n_dynamic=3, max_steps=200):
        self.size = size
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self.max_steps = max_steps
        self.action_space = 4  # up, down, left, right
        self.reset()

    def reset(self):
        self.steps = 0

        self.robot = self._rand_pos()
        self.goal = self._rand_pos(exclude=[self.robot])

        self.static_obs = []
        for _ in range(self.n_static):
            self.static_obs.append(self._rand_pos(exclude=self.static_obs + [self.robot, self.goal]))

        self.dynamic_obs = []
        for _ in range(self.n_dynamic):
            self.dynamic_obs.append(self._rand_pos(exclude=self.static_obs + self.dynamic_obs + [self.robot, self.goal]))

        return self._state()

    def step(self, action):
        self.steps += 1
        reward = -1
        done = False

        self._move_dynamic()

        new_pos = self._move(self.robot, action)

        if new_pos in self.static_obs or new_pos in self.dynamic_obs:
            reward = -100
            done = True
        else:
            self.robot = new_pos

        if self.robot == self.goal:
            reward = 100
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._state(), reward, done

    def _state(self):
        return (self.robot[0], self.robot[1], self.goal[0], self.goal[1])

    def _move(self, pos, action):
        x, y = pos
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        return (np.clip(x, 0, self.size - 1), np.clip(y, 0, self.size - 1))

    def _move_dynamic(self):
        new_positions = []
        for obs in self.dynamic_obs:
            a = random.randint(0, 3)
            new = self._move(obs, a)
            if new not in self.static_obs and new != self.goal and new != self.robot:
                new_positions.append(new)
            else:
                new_positions.append(obs)
        self.dynamic_obs = new_positions

    def _rand_pos(self, exclude=None):
        exclude = exclude or []
        while True:
            p = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if p not in exclude:
                return p
