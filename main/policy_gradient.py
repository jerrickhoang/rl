import gym
import click

from trainer.trainer import Trainer
from trainer.policy import Policy
from estimator.linear import LinearEstimator
from optimizer.policy_gradient import PolicyGradient

from env import point


@click.command()
@click.argument("env_id", type=str, default="Point-v0")
def main(env_id):
    env = gym.make(env_id)
    policy = Policy(LinearEstimator(env))
    optimizer = PolicyGradient(env)
    trainer = Trainer(env, policy, optimizer)
    trainer.train()


if __name__ == "__main__":
    main()
