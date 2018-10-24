class DifferentiblePolicy(object):

    def __init__(self, estimator):
        self._estimator = estimator

    def plan(self, obs):
        return self._estimator.predict(obs)

    def grad_log(self, action, obs):
        return self._estimator.grad_log(action, obs)

    def is_differentible(self):
        return self._estimator.is_differentible

    def init_grad(self):
        return self._estimator.init_grad()

    def improve(self, grad):
        self._estimator.improve(grad)
