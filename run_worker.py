import mapo
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy
from mapo.envs.registry import ENVS

from ray.rllib.evaluation import RolloutWorker

mapo.register_all_agents()
mapo.register_all_environments()

worker = RolloutWorker(
    env_creator=ENVS["Navigation-v0"],
    policy=MAPOTFPolicy,
    policy_config={"env": "Navigation-v0"},
)

samples = worker.sample()
# print("SAMPLES", samples)
worker.learn_on_batch(samples)
