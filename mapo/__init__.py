"""MAPO: Model-Aware Policy Optimization in RLlib."""

from gym.envs.registration import register

register(
    id="Navigation-v0",
    entry_point="mapo.envs:NavigationEnv",
    max_episode_steps=100,
    kwargs={"deceleration_zones": {}},
)

register(
    id="Navigation-v1",
    entry_point="mapo.envs:NavigationEnv",
    max_episode_steps=100,
    kwargs={},
)


def register_all_agents():
    """Register all trainer names in Tune."""
    from ray.tune import register_trainable
    from mapo.agents.registry import ALGORITHMS

    for name, trainer_import in ALGORITHMS.items():
        register_trainable(name, trainer_import())
