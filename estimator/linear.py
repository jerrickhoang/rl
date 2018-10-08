from estimator.estimator import Estimator


class LinearEstimator(Estimator):

    def __init__(self, env, deterministic=False):
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        self._theta = np.random.normal(scale=0.1, size=(action_dim, obs_dim + 1))
        self._theta = np.concatenate([self._theta, np.ones_like(x[..., :1])], axis=-1)
        self._deterministic = deterministic

    def predict(self, ob):
        mean = self._theta.dot(ob)
        if deterministic:
            return mean
        return np.random.normal(loc=mean, scale=1.)

    def grad_log(self, action, ob):
        ob = np.append(np.asarray(ob), 1)
        mean = theta.dot(ob)
        delta = action - mean
        return np.outer(ob, delta).T
