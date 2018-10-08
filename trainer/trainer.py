class Trainer(object):

    def __init__(self, env, policy, optimizer):
        self._env = env
        self._policy = policy
        self._optimizer = optimizer

    
    def rollout(self, render=False):
        obs, actions, rewards, done, n_samples = [], [], [], False, 0
        ob = env.reset()
        
        while not done:
            action = self._policy.act(ob)
            next_ob, reward, done, info = self._env.step(action)
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            ob = next_ob
            n_samples += 1

            if render:
                env.render()

        return obs, actions, rewards, n_samples

    
    def train(self):
        for itr in range(n_iters):
            all_obs, all_actions, episode_rewards, n_samples = [], [], [], 0

            while n_samples < batch_size:
                obs, action, rewards, n_samples = self.rollout()
                all_obs.append(observations)
                all_actions.append(actions)
                all_rewards.append(rewards)

            self._optimizer.improve_policy(self.policy, all_obs, all_actions, all_rewards)

            print("Iteration: {}, Average Return {}".format(itr, np.mean(all_rewards))


