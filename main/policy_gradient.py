import gym
import click

from env import point
from estimator.linear import LinearEstimator
from optimizer.policy_gradient import PolicyGradient
from policy.differentible_policy import DifferentiblePolicy
from trainer.trainer import Trainer


@click.command()
@click.argument("env_id", type=str, default="Point-v0")
def main(env_id):
    env = gym.make(env_id)
    policy = DifferentiblePolicy(LinearEstimator(env))
    optimizer = PolicyGradient(env)
    trainer = Trainer(env, policy, optimizer)
    trainer.train()


if __name__ == "__main__":
    main()
