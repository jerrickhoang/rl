
class Policy(object):

    def is_differentible(self):
        raise NotImplementedError

    def plan(self, ob):
        raise NotImplementedError
