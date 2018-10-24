import gym
import click
import copy
import numpy as np
import objgraph
import gc

#from optimizer.cem import CEM
from scipy.stats import uniform


from env import point

class RandomShooting(object):

    def __init__(self, env, popsize=1000, depth=30):
        self._popsize = popsize
        self._env = copy.deepcopy(env)
        self._depth = depth

    def evaluate_action(self, env, init_ob, action_seq):
        # env = copy.deepcopy(self._env)
        ob = copy.deepcopy(init_ob)
        env.reset()
        env.env.state = ob
        total_reward = 0
        #print("current state {}, action {}".format(env.state, action))
        step = 0
        done = False
        while step < self._depth:
            action = action_seq[step]
            _, r, done, _ = env.step(action)
            total_reward += r
            step += 1
        #print("next state {}".format(next_ob))
        del env
        return total_reward

    def plan(self, ob):
        mean = 0.
        action_dim = self._env.action_space.shape[0]
        action_lb = self._env.action_space.low
        action_ub = self._env.action_space.high
        scale = action_ub - action_lb
        samples = uniform.rvs(size=[self._popsize, self._depth, action_dim], loc=action_lb, scale=scale)

        env_copy = copy.deepcopy(self._env)
        costs = np.array([self.evaluate_action(env_copy, ob, action_seq) for action_seq in samples])
        #for i in range(len(samples)):
        #    print("Action: {}, cost {}".format(samples[i], costs[i]))
        #print("best action {}".format(samples[np.argmax(costs)]))
        del env_copy
        # pick the first action
        return samples[np.argmax(costs)][0]

class CEM(object):
    def __init__(self, env, popsize=1000, n_elites=10, depth=10):
        self._popsize = popsize
        self._env = copy.deepcopy(env)
        self._n_elites = n_elites
        self._depth = depth

    def evaluate_action(self, env, init_ob, action_seq):
        # env = copy.deepcopy(self._env)
        ob = copy.deepcopy(init_ob)
        env.reset()
        env.env.state = ob
        total_reward = 0
        #print("current state {}, action {}".format(env.state, action))
        step = 0
        done = False
        while step < self._depth:
            action = action_seq[step]
            _, r, done, _ = env.step(action)
            total_reward += r
            step += 1
        #print("next state {}".format(next_ob))
        del env
        return total_reward

    def plan(self, ob):
        mean = 0.
        action_dim = self._env.action_space.shape[0]
        action_lb = self._env.action_space.low
        action_ub = self._env.action_space.high
        scale = action_ub - action_lb
        samples = uniform.rvs(size=[self._popsize, self._depth, action_dim], loc=action_lb, scale=scale)
        
        env_copy = copy.deepcopy(self._env)
        costs = np.array([self.evaluate_action(env_copy, ob, action_seq) for action_seq in samples])
        elites = samples[np.argsort(costs)][::-1][:self._n_elites]
        first_actions = [elite[0] for elite in elites]

        del env_copy
        return np.mean(first_actions)
        

def rollout(env, policy, render=False):
    # env = copy.deepcopy(env)
    obs, actions, rewards, done = [], [], [], False
    ob = env.reset()
    env.reset()
    
    #env.state = [1., 0.]
    # ob = [1., 0.]
    # print("starting state is {}".format(env.state))
    # env.render()
    step = 0
    
    while not done:
        step += 1
        if step % 100 == 0:
            print "taking a step"
        action = policy.plan(ob)
        #if step % 100 == 0:
        #    print "done picking action"
        
        # TODO(jhoang): make this work with other env
        # action = action[:2]
        # print("Outer loop state is {}".format(ob))
        next_ob, reward, done, info = env.step(action)
        #if step % 100 == 0:
        #    print "done stepping through env"
        if done:
            break
        # obs.append(ob)
        # actions.append(action)
        rewards.append(reward)
        ob = next_ob

        #if step % 100 == 0:
        #    objgraph.show_most_common_types(limit=10)

        if render:
            env.render()
        gc.collect()

    return obs, actions, rewards


def main():
    # TODO(jhoang): test other envs
    env = gym.make("Point-v0")
    # optimizer = CEM(env, plan_hor=40, popsize=50, max_iters=10, num_elites=10, alpha=0.5)
    # optimizer = RandomShooting(env, popsize=100, depth=30)
    optimizer = CEM(env)
    n_iters = 100
    
    obs, actions, rewards = rollout(env, optimizer, render=True)

    # self._optimizer.improve_policy(self._policy, all_obs, all_actions, all_rewards)

    # print("Iteration: {}, Average Return {}".format(itr, np.sum(rewards)))
    print ("Averaige Return {} ".format(np.sum(rewards)))


  
if __name__ == "__main__":
    main()
