"""MAPO: Model-Aware Policy Optimization in RLlib."""


def register_all_agents():
    """Register all trainer names in Tune."""
    from ray.tune import register_trainable
    from mapo.agents.registry import ALGORITHMS

    for name, trainer_import in ALGORITHMS.items():
        register_trainable(name, trainer_import())
