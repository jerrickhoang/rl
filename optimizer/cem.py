import copy
import numpy as np
import math

from .optimizer import Optimizer

import scipy.stats as stats


class CEM(Optimizer):

    def __init__(self, env, plan_hor=20, popsize=100, max_iters=5, num_elites=50, alpha=0.1, discount=0.9):
        self._plan_hor = plan_hor
        self._popsize = popsize
        self._env = env
        self._max_iters = max_iters
        self._num_elites = num_elites
        self._alpha = alpha
        self._discount = discount

    def plan(self, ob):
        action_space = self._env.action_space.shape
        action_dim = action_space[0] * self._plan_hor
        action_lb = self._env.action_space.low
        action_ub = self._env.action_space.high
        lb = np.tile(action_lb, [self._plan_hor])
        ub = np.tile(action_ub, [self._plan_hor])
        mean, var, t = 0., 1., 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self._max_iters) and np.max(var) > 1e-3:
            lb_dist, ub_dist = mean - lb, ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            samples = X.rvs(size=[self._popsize, action_dim]) * np.sqrt(constrained_var) + mean
            costs = -1 * self.evaluate_action_sequence(ob, samples)
            elites = samples[np.argsort(costs)][:self._num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self._alpha * mean + (1 - self._alpha) * new_mean
            var = self._alpha * var + (1 - self._alpha) * new_var

            t += 1
        
        return mean

    def evaluate_action_sequence(self, init_ob, actions):
        env = copy.deepcopy(self._env)
        reward_list = []
        # TODO(jhoang): figure out the canonical way to set state
        for i in range(self._popsize):
            env.reset()
            env.state = init_ob
            ob = copy.deepcopy(init_ob)
            total_reward = 0.
            for j in range(self._plan_hor):
                action = actions[i][j]
                ob, r, done, _ = env.step(action)
                total_reward += math.pow(self._discount, j) * r

            reward_list.append(total_reward)
        return np.array(reward_list)


    def improve_policy(self, policy, all_obs, all_actions, all_rewards):
        pass
