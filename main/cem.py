import gym
import click
import copy
import numpy as np

from optimizer.cem import CEM
from scipy.stats import uniform


from env import point

class RandomShooting(object):

    def __init__(self, env, popsize=1000):
        self._popsize = popsize
        self._env = env

    def evaluate_action(self, init_ob, action):
        env = copy.deepcopy(self._env)
        ob = copy.deepcopy(init_ob)
        env.reset()
        env.state = ob
        _, r, done, _ = env.step(action)
        return r

    def plan(self, ob):
        mean = 0.
        action_dim = self._env.action_space.shape[0]
        action_lb = self._env.action_space.low
        action_ub = self._env.action_space.high
        scale = action_ub - action_lb
        samples = uniform.rvs(size=[self._popsize, action_dim], loc=action_lb, scale=scale)
        costs = np.array([self.evaluate_action(ob, action) for action in samples])
        return samples[np.argmax(costs)]

def rollout(env, policy, render=False):
    env = copy.deepcopy(env)
    obs, actions, rewards, done = [], [], [], False
    ob = env.reset()
    
    while not done:
        action = policy.plan(ob)
        # TODO(jhoang): make this work with other env
        # action = action[:2]
        next_ob, reward, done, info = env.step(action)
        obs.append(ob)
        actions.append(action)
        rewards.append(reward)
        ob = next_ob

        if render:
            env.render()

    return obs, actions, rewards


def main():
    # TODO(jhoang): test other envs
    env = gym.make("Point-v0")
    # optimizer = CEM(env, plan_hor=40, popsize=50, max_iters=10, num_elites=10, alpha=0.5)
    optimizer = RandomShooting(env)
    n_iters = 100
    
    obs, actions, rewards = rollout(env, optimizer, render=True)

    # self._optimizer.improve_policy(self._policy, all_obs, all_actions, all_rewards)

    # print("Iteration: {}, Average Return {}".format(itr, np.sum(rewards)))
    print ("Averaige Return {} ".format(np.sum(rewards)))


  
if __name__ == "__main__":
    main()
