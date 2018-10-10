import numpy as np
import scipy.stats as stats
from env import env_register


num_elites = 50
popsize = 500
max_iters = 5
alpha = 0.1
epsilon = 0.001
plan_hor = 30
ac_lb = np.array([-1.0]*6)
ac_ub = np.array([1.0]*6)

def reward_function(env, curr_ob, actions):
    action_list = np.split(actions, plan_hor, axis=1)
    reward_list = []
    observation = curr_ob.copy()
    for i in range(popsize):
        curr_reward = 0.
        curr_ob = observation.copy()
        for j in range(plan_hor):
            curr_ob = env.fdynamics(
                {'start_state': curr_ob, 'action': action_list[j][i]})
            curr_reward += env.reward(
                {'start_state': curr_ob, 'action': action_list[j][i]})
        reward_list.append(curr_reward)
    return np.array(reward_list)

def cem_plan(curr_ob, action_space):
    sol_dim = action_space * plan_hor
    lb = np.tile(ac_lb, [plan_hor])
    ub = np.tile(ac_ub, [plan_hor])
    mean, var, t = 0., 1., 0
    X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

    while (t < max_iters) and np.max(var) > epsilon:
        lb_dist, ub_dist = mean - lb, ub - mean
        constrained_var = np.minimum(
            np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

        samples = X.rvs(
            size=[popsize, sol_dim]) * np.sqrt(constrained_var) + mean
        costs = -1 * reward_function(env, curr_ob, samples)
        elites = samples[np.argsort(costs)][:num_elites]

        new_mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)

        mean = alpha * mean + (1 - alpha) * new_mean
        var = alpha * var + (1 - alpha) * new_var

        t += 1
    sol, solvar = mean, var
    return sol

def play_episode_with_env(env):

    # init the variables
    obs, rewards, actions = [], [], []

    # start the env (reset the environment)
    ob, _, _, _ = env.reset()
    obs.append(ob)
    action_space = 6
    rollout_step = 0
    reward = 0.

    while rollout_step < 1000:
        if rollout_step % 50 == 0:
            print('rollout step {}, current reward {}'.format(
                rollout_step, reward))
        # generate the policy
        action_sequence = cem_plan(ob, action_space)
        action = action_sequence[:6]
        ob = env.fdynamics(
            {'start_state': ob, 'action': action})
        one_step_reward = env.reward(
            {'start_state': ob , 'action': action})
        reward += one_step_reward

        # record the stats
        rewards.append((one_step_reward))
        obs.append(ob)
        actions.append(action)

        rollout_step += 1

    traj_episode = {
        "obs": np.array(obs, dtype=np.float64),
        "rewards": np.array(rewards, dtype=np.float64),
        "actions":  np.array(actions, dtype=np.float64),
    }
        #if done:  # terminated
        #    pdb.set_trace()
        #    # append one more for the raw_obs
        #    traj_episode = {
        #        "obs": np.array(obs, dtype=np.float64),
        #        "rewards": np.array(rewards, dtype=np.float64),
        #        "actions":  np.array(actions, dtype=np.float64),
        #    }
        #    break
    return traj_episode

env, env_info  = env_register.make_env('gym_cheetah', 666)
traj_episode = play_episode_with_env(env)
