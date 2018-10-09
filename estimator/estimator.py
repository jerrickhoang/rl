
class Estimator(object):

    def predict(self, obs):
        raise NotImplementedError

    def grad_log(self, action, obs):
        raise NotImplementedError

    def is_differentible(self):
        raise NotImplementedError


class DifferentibleEstimator(object):

    def is_differentible(self):
        return True

    def init_grad(self):
        raise NotImplementedError

    def improve(self, grad):
        raise NotImplementedError
