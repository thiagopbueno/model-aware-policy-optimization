"""Test script to check if on-policy trajectories are collected by MAPO."""
import ray
from ray import tune
import mapo


def main():
    """Run MAPO in several environments with continuous action spaces."""
    mapo.register_all_agents()

    ray.init()
    tune.run(
        "MAPO",
        stop={"timesteps_total": 1000},
        config={
            "env": tune.grid_search(
                ["Pendulum-v0", "LunarLanderContinuous-v2", "HalfCheetah-v3"]
            ),
            "num_workers": 0,
        },
    )


if __name__ == "__main__":
    main()
