# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from ray.rllib.models.model import restore_original_dimensions


import mapo.agents.mapo.losses as losses
from mapo.agents.mapo import MAPOModel


@pytest.fixture
def env(env_name, env_creator):
    return env_creator(env_name)


@pytest.fixture
def batch_tensors_fn(spaces):
    def get_batch_tensors(env):
        obs_space, action_space = spaces(env)
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
                SampleBatch.CUR_OBS: restore_original_dimensions(obs, obs_space),
                SampleBatch.ACTIONS: actions,
                SampleBatch.REWARDS: rewards,
                SampleBatch.DONES: dones,
                SampleBatch.NEXT_OBS: restore_original_dimensions(next_obs, obs_space),
                Postprocessing.ADVANTAGES: returns,
            }
        )

    return get_batch_tensors


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
def smooth_target_policy(request):
    return request.param


@pytest.fixture(params=[True, False])
def twin_q(request):
    return request.param


@pytest.fixture
def config(model_config, smooth_target_policy, twin_q):
    return {
        "gamma": 0.99,
        "kernel": "l2",
        "model_config": model_config,
        "twin_q": twin_q,
        "branching_factor": 4,
        "use_true_dynamics": False,
        "smooth_target_policy": smooth_target_policy,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,
    }


@pytest.fixture(params=[False, True])
def use_true_dynamics(request):
    return request.param


@pytest.fixture(params=[0, 4])
def branching_factor(request):
    return request.param


@pytest.fixture(params=[True, False])
def target_networks(request):
    return request.param


@pytest.fixture
def model_fn(spaces, target_networks):
    def make_model(env, config):
        obs_space, action_space = spaces(env)
        return MAPOModel(
            obs_space,
            action_space,
            1,
            model_config=config["model_config"],
            name="mapo_model",
            target_networks=target_networks,
            twin_q=config["twin_q"],
        )

    return make_model


@pytest.fixture
def model_with_targets_fn(spaces):
    def make_model_with_targets(env, config):
        obs_space, action_space = spaces(env)
        return MAPOModel(
            obs_space,
            action_space,
            1,
            model_config=config["model_config"],
            name="mapo_model",
            target_networks=True,
            twin_q=config["twin_q"],
        )

    return make_model_with_targets


def assert_consistent_shapes_and_grads(loss, variables):
    assert loss.shape == ()
    grads = tf.gradients(loss, variables)
    assert all(grad is not None for grad in grads)
    assert all(
        tuple(grad.shape) == tuple(var.shape) for grad, var in zip(grads, variables)
    )


def test_dynamics_mle_loss(env, config, model_fn, batch_tensors_fn):
    model = model_fn(env, config)
    batch_tensors = batch_tensors_fn(env)
    loss = losses.dynamics_mle_loss(batch_tensors, model)
    variables = model.dynamics_variables

    assert_consistent_shapes_and_grads(loss, variables)
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
    assert SampleBatch.ACTIONS in batch_tensors.accessed_keys
    assert SampleBatch.NEXT_OBS in batch_tensors.accessed_keys


def test_dynamics_pga_loss(env, config, model_fn, batch_tensors_fn):
    model = model_fn(env, config)
    batch_tensors = batch_tensors_fn(env)
    actor_loss = losses.actor_model_aware_loss(batch_tensors, model, env, config)
    batch_tensors = UsageTrackingDict(batch_tensors)
    loss = losses.dynamics_pga_loss(batch_tensors, model, actor_loss, config)
    variables = model.dynamics_variables

    assert_consistent_shapes_and_grads(loss, variables)
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys


def test_build_critic_targets(env, config, model_with_targets_fn, batch_tensors_fn):
    # pylint: disable=protected-access
    model = model_with_targets_fn(env, config)
    batch_tensors = batch_tensors_fn(env)
    targets = losses._build_critic_targets(batch_tensors, model, config)
    target_vars = [targ for main, targ in model.main_and_target_variables]

    assert all(grad is None for grad in tf.gradients(targets, model.critic_variables))
    assert all(grad is None for grad in tf.gradients(targets, model.actor_variables))
    assert all(grad is not None for grad in tf.gradients(targets, target_vars))
    assert SampleBatch.REWARDS in batch_tensors.accessed_keys
    assert SampleBatch.DONES in batch_tensors.accessed_keys
    assert SampleBatch.NEXT_OBS in batch_tensors.accessed_keys


def test_critic_1step_loss(env, config, model_with_targets_fn, batch_tensors_fn):
    model = model_with_targets_fn(env, config)
    batch_tensors = batch_tensors_fn(env)
    loss, _ = losses.critic_1step_loss(batch_tensors, model, config)
    variables = model.critic_variables

    assert_consistent_shapes_and_grads(loss, variables)
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
    assert SampleBatch.ACTIONS in batch_tensors.accessed_keys


def test_critic_return_loss(env, config, model_fn, batch_tensors_fn):
    model = model_fn(env, config)
    batch_tensors = batch_tensors_fn(env)
    loss, _ = losses.critic_return_loss(batch_tensors, model)

    variables = model.critic_variables
    if model.twin_q:
        variables = variables[: len(variables) // 2]

    assert_consistent_shapes_and_grads(loss, variables)
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
    assert SampleBatch.ACTIONS in batch_tensors.accessed_keys
    assert Postprocessing.ADVANTAGES in batch_tensors.accessed_keys


def test_actor_dpg_loss(env, config, model_fn, batch_tensors_fn):
    model = model_fn(env, config)
    batch_tensors = batch_tensors_fn(env)
    loss = losses.actor_dpg_loss(batch_tensors, model)
    variables = model.actor_variables

    assert_consistent_shapes_and_grads(loss, variables)
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys


def test_actor_model_aware_loss(
    use_true_dynamics, branching_factor, env, config, model_fn, batch_tensors_fn
):
    # pylint: disable=too-many-arguments
    model = model_fn(env, config)
    batch_tensors = batch_tensors_fn(env)
    config["branching_factor"] = branching_factor
    config["use_true_dynamics"] = use_true_dynamics
    loss = losses.actor_model_aware_loss(batch_tensors, model, env, config)
    variables = model.actor_variables

    assert_consistent_shapes_and_grads(loss, variables)
    assert SampleBatch.CUR_OBS in batch_tensors.accessed_keys
