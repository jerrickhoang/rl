from optimizer.optimizer import Optimizer



class PolicyGradient(Optimizer):

    def __init__(self, env, gamma=0.99, use_baseline=False):
        self._gamma = gamma
        self._use_baseline = use_baseline
        self._all_returns = [[] for _ in range(env.spec.timestamp_limit)]]
        self._baselines = np.zeroes(env.spec.timestamp_limit)

    def update_baselines(self):
        baselines = []
        for t in range(len(self._all_returns)):
            b_t = np.mean(self._all_returns[t]) if len(self._all_returns[t]) != 0 else 0
            baselines.append(b_t)
        return np.array(baselines)

    def compute_grad(self, policy, next_R, s_t, a_t, r_t, b_t):
        R_t = self._gamma * next_R + r_t
        A_t = R_t - b_t
        return R_t, policy.grad_log(a_t, s_t) * A_t

    def improve_policy(self, policy, all_obs, all_actions, all_rewards):
        grad = 
        for episode_i in range(all_obs[i]):
            obs, actions, rewards = all_obs[episode_i], all_actions[episode_i], all_rewards[episode_i]

            R = 0.
            for t in reversed(range(len(obs))):
                R, grad_t = self.compute_grad(policy, R, obs[t], actions[t], rewards[t], self._baselines[t])
                self._all_returns[t].append(R)
                grad += grad_t
            if self._use_baselines:
                self.update_baselines()

