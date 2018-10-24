import numpy as np


class Trainer(object):

    def __init__(self, env, policy, optimizer):
        self._env = env
        self._policy = policy
        self._optimizer = optimizer

    
    def rollout(self, render=False):
        obs, actions, rewards, done = [], [], [], False
        ob = self._env.reset()
        
        while not done:
            action = self._policy.plan(ob)
            next_ob, reward, done, info = self._env.step(action)
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            ob = next_ob

            if render:
                self._env.render()

        return obs, actions, rewards

    
    def train(self, n_iters=100, batch_size=20):
        for itr in range(n_iters):
            all_obs, all_actions, all_rewards, episode_rewards, n_samples = [], [], [], [], 0

            while n_samples < batch_size:
                obs, actions, rewards = self.rollout()
                all_obs.append(obs)
                all_actions.append(actions)
                all_rewards.append(rewards)
                episode_rewards.append(np.sum(rewards))
                n_samples += len(obs)

            self._optimizer.improve_policy(self._policy, all_obs, all_actions, all_rewards)

            print("Iteration: {}, Average Return {}".format(itr, np.mean(episode_rewards)))

