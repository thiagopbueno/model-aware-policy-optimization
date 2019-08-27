"""Test script to check if MAPO successfully learns in environments."""


def main():
    """Run MAPO."""
    import ray
    import mapo
    from mapo.agents.mapo import MAPOTrainer

    mapo.register_all_agents()
    mapo.register_all_environments()

    ray.init()

    config = {
        # === COMMON CONFIG ===
        "debug": True,
        "log_level": "WARN",
        "env": "Navigation-v0",
        "num_workers": 0,
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "train_batch_size": 2,
        "optimizer": {"num_sgd_iter": 1},
        "observation_filter": "NoFilter",
        # "seed": 123,
        # Specify where experiences should be saved:
        #  - None: don't save any experiences
        #  - "logdir" to save to the agent log dir
        #  - a path/URI to save to a custom output directory (e.g., "s3://bucket/")
        #  - a function that returns a rllib.offline.OutputWriter
        "output": None,
        # What sample batch columns to LZ4 compress in the output data.
        # RLlib's default is ["obs", "new_obs"]
        "output_compress_columns": [],
        # === MAPO CONFIG ===
        "apply_gradients": "sgd_iter",
        "branching_factor": 10,
        "use_true_dynamics": False,
        "model_loss": "pga",
        "madpg_estimator": "sf",
        "model": {
            "custom_options": {
                "actor": {"activation": "elu", "layers": [64]},
                "critic": {"activation": "elu", "layers": [64]},
                "dynamics": {"activation": "elu", "layers": []},
            }
        },
        "dynamics_lr": 1e-4,
        "critic_lr": 1e-4,
        "actor_lr": 1e-4,
        "critic_sgd_iter": 20,
        "dynamics_sgd_iter": 10,
    }
    trainer = MAPOTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
