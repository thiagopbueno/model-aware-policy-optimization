# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.tracking_dict import UsageTrackingDict


import mapo.agents.mapo.losses as losses
from mapo.agents.mapo import MAPOModel


@pytest.fixture
def batch_tensors(spaces):
    obs_space, action_space = spaces
    obs = tf.compat.v1.placeholder(
        tf.float32, shape=[None] + list(obs_space.shape), name="observation"
    )
    actions = ModelCatalog.get_action_placeholder(action_space)
    rewards = tf.compat.v1.placeholder(tf.float32, [None], name="reward")
    dones = tf.compat.v1.placeholder(tf.bool, [None], name="done")
    next_obs = tf.compat.v1.placeholder(
        tf.float32, shape=[None] + list(obs_space.shape), name="next_observation"
    )
    returns = tf.compat.v1.placeholder(tf.float32, [None], name="return")
    return UsageTrackingDict(
        {
            SampleBatch.CUR_OBS: obs,
            SampleBatch.ACTIONS: actions,
            SampleBatch.REWARDS: rewards,
            SampleBatch.DONES: dones,
            SampleBatch.NEXT_OBS: next_obs,
            Postprocessing.ADVANTAGES: returns,
        }
    )


@pytest.fixture
def model_config():
    return {
        "custom_options": {
            "actor": {"activation": "relu", "layers": [32, 32]},
            "critic": {"activation": "relu", "layers": [32, 32]},
            "dynamics": {"activation": "relu", "layers": [32, 32]},
        }
    }


@pytest.fixture(params=[True, False])
def target_networks(request):
    return request.param


@pytest.fixture(params=[True, False])
def smooth_target_policy(request):
    return request.param


@pytest.fixture(params=[True, False])
def twin_q(request):
    return request.param


@pytest.fixture
def kernel_fn():
    return lambda x, y: tf.norm(x - y, axis=0)


# skip true dynamics test until env dynamics log_prob is done
@pytest.fixture(params=[False, pytest.param(True, marks=pytest.mark.skip)])
def use_true_dynamics(request):
    return request.param


@pytest.fixture
def config(kernel_fn, model_config, twin_q, smooth_target_policy, use_true_dynamics):
    return {
        "gamma": 0.99,
        "kernel": kernel_fn,
        "model_config": model_config,
        "twin_q": twin_q,
        "use_true_dynamics": use_true_dynamics,
        "branching_factor": 1,
        "smooth_target_policy": smooth_target_policy,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,
    }


@pytest.fixture
def model_and_config_fn():
    def make_model_and_config(spaces, target_networks, config):
        obs_space, action_space = spaces
        return (
            MAPOModel(
                obs_space,
                action_space,
                1,
                model_config=config["model_config"],
                name="mapo_model",
                target_networks=target_networks,
                twin_q=config["twin_q"],
            ),
            config,
        )

    return make_model_and_config


@pytest.fixture
def model_and_config(model_and_config_fn, spaces, target_networks, config):
    return model_and_config_fn(spaces, target_networks, config)


@pytest.fixture
def model_and_config_with_targets(model_and_config_fn, spaces, config):
    return model_and_config_fn(spaces, True, config)


@pytest.fixture
def model(model_and_config):
    return model_and_config[0]


@pytest.fixture
def env(env_creator):
    return env_creator()


def test_dynamics_mle_loss(batch_tensors, model):
    loss = losses.dynamics_mle_loss(batch_tensors, model)

    assert all(
        grad is not None for grad in tf.gradients(loss, model.dynamics_variables)
    )
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
    assert SampleBatch.ACTIONS in batch_tensors.accessed_keys
    assert SampleBatch.NEXT_OBS in batch_tensors.accessed_keys


def test_dynamics_pga_loss(batch_tensors, env, model_and_config):
    model, config = model_and_config
    actor_loss = losses.actor_model_aware_loss(batch_tensors, model, env, config)
    batch_tensors = UsageTrackingDict(batch_tensors)
    loss = losses.dynamics_pga_loss(batch_tensors, model, actor_loss, config)

    assert all(
        grad is not None for grad in tf.gradients(loss, model.dynamics_variables)
    )
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys


def test_build_critic_targets(batch_tensors, model_and_config_with_targets):
    # pylint: disable=protected-access
    model, config = model_and_config_with_targets
    targets = losses._build_critic_targets(batch_tensors, model, config)
    target_vars = [targ for main, targ in model.main_and_target_variables]

    assert all(grad is None for grad in tf.gradients(targets, model.critic_variables))
    assert all(grad is None for grad in tf.gradients(targets, model.actor_variables))
    assert all(grad is not None for grad in tf.gradients(targets, target_vars))
    assert SampleBatch.REWARDS in batch_tensors.accessed_keys
    assert SampleBatch.DONES in batch_tensors.accessed_keys
    assert SampleBatch.NEXT_OBS in batch_tensors.accessed_keys


def test_critic_1step_loss(batch_tensors, model_and_config_with_targets):
    model, config = model_and_config_with_targets
    loss, _ = losses.critic_1step_loss(batch_tensors, model, config)

    assert all(grad is not None for grad in tf.gradients(loss, model.critic_variables))
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
    assert SampleBatch.ACTIONS in batch_tensors.accessed_keys


def test_critic_return_loss(batch_tensors, model):
    loss = losses.critic_return_loss(batch_tensors, model)

    critic_variables = model.critic_variables
    if model.twin_q:
        critic_variables = critic_variables[: len(critic_variables) // 2]
    assert all(grad is not None for grad in tf.gradients(loss, critic_variables))
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
    assert SampleBatch.ACTIONS in batch_tensors.accessed_keys
    assert Postprocessing.ADVANTAGES in batch_tensors.accessed_keys


def test_actor_dpg_loss(batch_tensors, model):
    loss = losses.actor_dpg_loss(batch_tensors, model)

    assert all(grad is not None for grad in tf.gradients(loss, model.actor_variables))
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys


def test_actor_model_aware_loss(batch_tensors, env, model_and_config):
    model, config = model_and_config
    loss = losses.actor_model_aware_loss(batch_tensors, model, env, config)

    assert all(grad is not None for grad in tf.gradients(loss, model.actor_variables))
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
