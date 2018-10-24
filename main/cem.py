import gym
import click

from env import point
from optimizer.noop import NoopOptimizer
from policy.cem import CEM
from trainer.trainer import Trainer


@click.command()
@click.argument("env_id", type=str, default="Point-v0")
def main(env_id):
    env = gym.make(env_id)
    policy = CEM(env)
    optimizer = NoopOptimizer()
    trainer = Trainer(env, policy, optimizer)
    trainer.train()


if __name__ == "__main__":
    main()
