# pylint: disable=missing-docstring, redefined-outer-name
import itertools

import pytest
from gym.spaces import Box
import numpy as np
import tensorflow as tf

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog

from mapo.agents.mapo.mapo_model import MAPOModel


def get_spaces():
    return (
        Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        # Action range must be limited so as to not break ActionSquashingLayer
        Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )


def get_models():
    obs_space, action_space = get_spaces()
    model_config = {
        "custom_options": {
            "actor": {"activation": "relu", "layers": [32, 32]},
            "critic": {"activation": "relu", "layers": [32, 32]},
            "dynamics": {"activation": "relu", "layers": [32, 32]},
        }
    }
    models = [
        MAPOModel(
            obs_space,
            action_space,
            1,
            model_config,
            "test_model",
            target_networks=build_targets,
            twin_q=build_twin,
        )
        for build_targets, build_twin in itertools.product([True, False], [True, False])
    ]
    return models


def get_models_with_targets():
    return [model for model in get_models() if hasattr(model, "target_models")]


def get_models_with_twin_qs():
    return [model for model in get_models() if model.twin_q]


@pytest.fixture
def input_dict():
    obs_space, action_space = get_spaces()
    obs = tf.compat.v1.placeholder(
        tf.float32, shape=[None] + list(obs_space.shape), name="observation"
    )
    prev_actions = ModelCatalog.get_action_placeholder(action_space)
    prev_rewards = tf.compat.v1.placeholder(tf.float32, [None], name="prev_reward")

    input_dict = {
        SampleBatch.CUR_OBS: obs,
        SampleBatch.PREV_ACTIONS: prev_actions,
        SampleBatch.PREV_REWARDS: prev_rewards,
        "is_training": tf.compat.v1.placeholder_with_default(False, ()),
    }
    return input_dict


@pytest.fixture
def obs_ph():
    obs_space, _ = get_spaces()
    return tf.compat.v1.placeholder(
        tf.float32, shape=[None] + list(obs_space.shape), name="observation"
    )


@pytest.fixture
def action_ph():
    _, action_space = get_spaces()
    return ModelCatalog.get_action_placeholder(action_space)


@pytest.mark.parametrize("model", get_models())
def test_variables_are_created_on_init(model):
    assert model.variables()


@pytest.mark.parametrize("model", get_models())
def test_component_variables_are_in_all_variables(model):
    all_variables = set(model.variables())
    actor_vars = set(model.actor_variables)
    critic_vars = set(model.critic_variables)
    dynamics_vars = set(model.dynamics_variables)
    assert actor_vars and actor_vars.issubset(all_variables)
    assert critic_vars and critic_vars.issubset(all_variables)
    assert dynamics_vars and dynamics_vars.issubset(all_variables)


@pytest.mark.parametrize("model", get_models())
def test_output_is_flattened_obs(model, input_dict):
    model_out, _ = model(input_dict, [], None)
    assert model_out.shape[1:] == model.obs_space.shape


@pytest.mark.parametrize("model", get_models())
def test_compute_action(model, obs_ph, action_ph):
    actions = model.compute_actions(obs_ph)
    assert actions.shape[1:] == action_ph.shape[1:]

    all_grads = tf.gradients(tf.reduce_sum(actions), model.variables())
    filtered_all_vars = [
        var for grad, var in zip(all_grads, model.variables()) if grad is not None
    ]
    actor_grads = tf.gradients(tf.reduce_sum(actions), model.actor_variables)
    filtered_actor_vars = [
        var for grad, var in zip(actor_grads, model.actor_variables) if grad is not None
    ]
    assert set(filtered_all_vars) == set(filtered_actor_vars)


@pytest.mark.parametrize("model", get_models())
def test_compute_q_values(model, obs_ph, action_ph):
    values = model.compute_q_values(obs_ph, action_ph)
    assert values.shape[1:] == (1,)

    all_grads = tf.gradients(tf.reduce_sum(values), model.variables())
    filtered_all_vars = [
        var for grad, var in zip(all_grads, model.variables()) if grad is not None
    ]
    critic_grads = tf.gradients(tf.reduce_sum(values), model.critic_variables)
    filtered_critic_vars = [
        var
        for grad, var in zip(critic_grads, model.critic_variables)
        if grad is not None
    ]
    assert set(filtered_all_vars) == set(filtered_critic_vars)


@pytest.mark.parametrize("model", get_models())
def test_sample_next_states(model, obs_ph, action_ph):
    states = model.sample_next_states(obs_ph, action_ph)
    assert states.shape[1:] == obs_ph.shape[1:]

    all_grads = tf.gradients(tf.reduce_sum(states), model.variables())
    filtered_all_vars = [
        var for grad, var in zip(all_grads, model.variables()) if grad is not None
    ]
    dynamics_grads = tf.gradients(tf.reduce_sum(states), model.dynamics_variables)
    filtered_dynamics_vars = [
        var
        for grad, var in zip(dynamics_grads, model.dynamics_variables)
        if grad is not None
    ]
    assert set(filtered_all_vars) == set(filtered_dynamics_vars)


@pytest.mark.parametrize("model", get_models_with_targets())
def test_uses_target_models(model, obs_ph, action_ph):
    actions = model.compute_actions(obs_ph, target=True)
    assert actions.shape[1:] == action_ph.shape[1:]
    assert all(
        grad is None
        for grad in tf.gradients(tf.reduce_sum(actions), model.actor_variables)
    )

    values = model.compute_q_values(obs_ph, action_ph, target=True)
    assert values.shape[1:] == (1,)
    assert all(
        grad is None
        for grad in tf.gradients(tf.reduce_sum(values), model.critic_variables)
    )


@pytest.mark.parametrize("model", get_models_with_twin_qs())
def test_creates_twin_q_vars(model):
    critic_vars = model.critic_variables
    q_vars, twin_q_vars = (
        critic_vars[: len(critic_vars)],
        critic_vars[len(critic_vars) :],
    )
    assert all(
        var.shape == twin_var.shape for var, twin_var in zip(q_vars, twin_q_vars)
    )
