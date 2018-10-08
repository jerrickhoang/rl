class Policy(object):

    def __init__(self, estimator):
        self._estimator = estimator

    def act(self, obs):
        return self._estimator.predict(obs)

    def grad_log(self, action, obs):
        return self._estimator.grad_log(action, obs)
