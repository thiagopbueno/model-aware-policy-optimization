# pylint: disable=all
import mapo
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy, get_default_config
from mapo.envs.registry import ENVS

from ray.rllib.evaluation import RolloutWorker
from ray.tune.util import merge_dicts

mapo.register_all_agents()
mapo.register_all_environments()


def main():
    worker = RolloutWorker(
        env_creator=ENVS["Navigation-v0"],
        policy=MAPOTFPolicy,
        policy_config=merge_dicts(
            get_default_config(),
            {
                "env": "Navigation-v0",
                "model": {"custom_options": {"actor": {"input_layer_norm": False}}},
            },
        ),
    )

    samples = worker.sample()
    # print("SAMPLES", samples)
    worker.learn_on_batch(samples)


if __name__ == "__main__":
    main()
