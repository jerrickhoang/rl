import copy
import numpy as np
from scipy.stats import uniform

from .policy import Policy


class RandomShooting(Policy):

    def __init__(self, env, popsize=1000, depth=30):
        self._popsize = popsize
        self._env = copy.deepcopy(env)
        self._depth = depth

    def is_differentible(self):
        return False

    def evaluate_action(self, env, init_ob, action_seq):
        ob = copy.deepcopy(init_ob)
        env.reset()
        env.env.state = ob
        total_reward, step, done = 0, 0, False
        while step < self._depth:
            action = action_seq[step]
            _, r, done, _ = env.step(action)
            total_reward += r
            step += 1
        del env
        return total_reward

    def plan(self, ob):
        mean = 0.
        action_dim = self._env.action_space.shape[0]
        action_lb = self._env.action_space.low
        action_ub = self._env.action_space.high
        scale = action_ub - action_lb
        samples = uniform.rvs(size=[self._popsize, self._depth, action_dim], loc=action_lb, scale=scale)

        env_copy = copy.deepcopy(self._env)
        costs = np.array([self.evaluate_action(env_copy, ob, action_seq) for action_seq in samples])
        del env_copy
        return samples[np.argmax(costs)][0]


