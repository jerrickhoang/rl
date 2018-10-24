from .optimizer import Optimizer


class NoopOptimizer(Optimizer):

    def improve_policy(self, policy, all_obs, all_actions, all_rewards):
        pass
