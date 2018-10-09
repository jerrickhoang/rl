import gym

from trainer.trainer import Trainer
from trainer.policy import Policy
from estimator.linear import LinearEstimator
from optimizer.policy_gradient import PolicyGradient

from env import point


def main():
    # TODO(jhoang): make this a flag
    env = gym.make("Point-v0")
    policy = Policy(LinearEstimator(env))
    optimizer = PolicyGradient(env)
    trainer = Trainer(env, policy, optimizer)
    trainer.train()


if __name__ == "__main__":
    main()
