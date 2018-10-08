
class Estimator(object):

    def predict(self, obs):
        raise NotImplementedError

    def grad_log(self, action, obs):
        raise NotImplementedError
