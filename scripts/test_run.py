"""Test script to check if MAPO successfully learns in environments."""

import numpy as np


def main():
    """Run MAPO."""
    import ray
    from ray import tune
    import mapo

    mapo.register_all_agents()
    mapo.register_all_environments()

    ray.init()
    tune.run(
        "MAPO",
        stop={"timesteps_total": int(2e5)},
        num_samples=20,
        config={
            # === COMMON CONFIG ===
            "env": "Navigation-v0",
            "horizon": 20,
            "num_workers": 0,
            "sample_batch_size": 1,
            "batch_mode": "complete_episodes",
            "num_envs_per_worker": 8,
            "train_batch_size": 5000,
            "optimizer": {"num_sgd_iter": 80},
            "observation_filter": "NoFilter",
            # "seed": 123,
            # === MAPO CONFIG ===
            "branching_factor": 10,
            "use_true_dynamics": True,
            "model_loss": "mle",
            "model": {
                "custom_options": {
                    "actor": {"activation": "elu", "layers": [32]},
                    "critic": {"activation": "elu", "layers": [32]},
                    "dynamics": {"activation": "elu", "layers": [32]},
                }
            },
            "dynamics_lr": tune.function(
                lambda: np.random.uniform(low=1e-3, high=1e-2)
            ),
            "critic_lr": tune.function(lambda: np.random.uniform(low=1e-3, high=1e-2)),
            "actor_lr": tune.function(lambda: np.random.uniform(low=1e-3, high=1e-2)),
            "actor_delay": 80,
            "critic_delay": 1,
        },
        checkpoint_freq=20,
        checkpoint_at_end=True,
    )


if __name__ == "__main__":
    main()
