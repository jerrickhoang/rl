import numpy as np

from .estimator import DifferentibleEstimator


class LinearEstimator(DifferentibleEstimator):

    def __init__(self, env, lr=0.1, deterministic=False):
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        self._theta = np.random.normal(scale=0.1, size=(action_dim, obs_dim + 1))
        self._deterministic = deterministic
        self._lr = lr

    def predict(self, ob):
        ob_with_bias = np.concatenate([ob, np.ones_like(ob[..., :1])], axis=-1)
        mean = self._theta.dot(ob_with_bias)
        if self._deterministic:
            return mean
        return np.random.normal(loc=mean, scale=1.)

    def grad_log(self, action, ob):
        ob = np.append(np.asarray(ob), 1)
        mean = self._theta.dot(ob)
        delta = action - mean
        return np.outer(ob, delta).T

    def is_differentible(self):
        return True

    def init_grad(self):
        return np.zeros_like(self._theta)

    def improve(self, grad):
        self._theta += self._lr * grad
