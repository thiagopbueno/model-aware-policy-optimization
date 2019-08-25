"""Collection of loss functions in tensorflow."""
import tensorflow as tf
from tensorflow import keras
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.explained_variance import explained_variance

from mapo.kernels import KERNELS


def dynamics_mle_loss(batch_tensors, model):
    """Compute dynamics loss via Maximum Likelihood Estimation."""
    obs, actions, next_obs = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
        batch_tensors[SampleBatch.NEXT_OBS],
    )
    return -tf.reduce_mean(model.compute_states_log_prob(obs, actions, next_obs))


def dynamics_pga_loss(batch_tensors, model, actor_loss, config):
    """Compute Policy Gradient Aware dynamics loss"""
    gmapo = tf.gradients(actor_loss, model.actor_variables)
    dpg_loss = actor_dpg_loss(batch_tensors, model)
    dpg = tf.gradients(dpg_loss, model.actor_variables)
    kernel = KERNELS[config["kernel"]]
    return kernel(gmapo, dpg)


def _build_critic_targets(batch_tensors, model, config):
    rewards, dones, next_obs = (
        batch_tensors[SampleBatch.REWARDS],
        batch_tensors[SampleBatch.DONES],
        batch_tensors[SampleBatch.NEXT_OBS],
    )
    gamma = config["gamma"]
    next_action = model.compute_actions(next_obs, target=True)
    if config["smooth_target_policy"]:
        action_space = model.action_space
        epsilon = tf.random.normal(tf.shape(next_action), stddev=config["target_noise"])
        epsilon = tf.clip_by_value(
            epsilon, -config["target_noise_clip"], config["target_noise_clip"]
        )
        next_action = next_action + epsilon
        next_action = tf.clip_by_value(next_action, action_space.low, action_space.high)
    next_q_values = tf.squeeze(
        model.compute_q_values(next_obs, next_action, target=True)
    )
    if config["twin_q"]:
        twin_q_values = model.compute_twin_q_values(next_obs, next_action, target=True)
        next_q_values = tf.math.minimum(next_q_values, tf.squeeze(twin_q_values))
    # Do not bootstrap if the state is terminal
    bootstrapped = rewards + gamma * next_q_values
    return tf.compat.v2.where(dones, x=rewards, y=bootstrapped)


def critic_1step_loss(batch_tensors, model, config):
    """Compute mean squared error between Q network and 1 step returns."""
    obs, actions = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
    )
    target_q_values = _build_critic_targets(batch_tensors, model, config)
    q_loss_criterion = keras.losses.MeanSquaredError()
    q_pred = tf.squeeze(model.compute_q_values(obs, actions))
    fetches = {
        "q_mean": tf.reduce_mean(q_pred),
        "q_max": tf.reduce_max(q_pred),
        "q_min": tf.reduce_min(q_pred),
    }
    critic_loss = q_loss_criterion(q_pred, target_q_values)
    if config["twin_q"]:
        twin_q_pred = tf.squeeze(model.compute_twin_q_values(obs, actions))
        fetches.update(
            {
                "twin_q_mean": tf.reduce_mean(twin_q_pred),
                "twin_q_max": tf.reduce_max(twin_q_pred),
                "twin_q_min": tf.reduce_min(twin_q_pred),
            }
        )
        twin_q_loss = q_loss_criterion(twin_q_pred, target_q_values)
        critic_loss += twin_q_loss
    return critic_loss, fetches


def critic_return_loss(batch_tensors, model):
    """Compute mean squared error between Q network and trajectory returns."""
    obs, actions, returns = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
        batch_tensors[Postprocessing.ADVANTAGES],
    )
    q_loss_criterion = keras.losses.MeanSquaredError()
    q_pred = tf.squeeze(model.compute_q_values(obs, actions))
    fetches = {
        "explained_variance": explained_variance(
            batch_tensors[Postprocessing.ADVANTAGES], q_pred
        ),
        "q_mean": tf.reduce_mean(q_pred),
        "q_max": tf.reduce_max(q_pred),
        "q_min": tf.reduce_min(q_pred),
    }
    return q_loss_criterion(q_pred, returns), fetches


def actor_dpg_loss(batch_tensors, model):
    """Compute deterministic policy gradient loss."""
    obs = batch_tensors[SampleBatch.CUR_OBS]
    policy_action = model.compute_actions(obs)
    policy_action_value = model.compute_q_values(obs, policy_action)
    return -tf.reduce_mean(policy_action_value)


def pd_madpg_estimator(batch_tensors, model, env, config):
    """
    Compute model-aware dpg using the next state value pathwise derivative estimator.
    """
    # pylint: disable=protected-access
    state = batch_tensors[SampleBatch.CUR_OBS]
    action = model.compute_actions(state)
    n_samples = config["branching_factor"]
    gamma = config["gamma"]

    if n_samples == 0:
        raise ValueError(
            "Using true environment samples with the pathwise derivative"
            "estimator is currently not supported."
        )
    if config["use_true_dynamics"]:
        raise ValueError(
            "Cannot calculate pathwise derivative estimator using real dynamics."
        )
    next_state = model.rsample_next_states(state, action, n_samples)
    next_state_value = tf.squeeze(model.compute_state_values(next_state), axis=-1)
    reward = tf.reduce_mean(env._reward_fn(state, action, next_state), axis=0)
    return tf.reduce_mean(reward + gamma * tf.reduce_mean(next_state_value, axis=0))


def sf_madpg_estimator(batch_tensors, model, env, config):
    """Compute model-aware dpg using the next state value score function estimator."""
    # pylint: disable=protected-access
    state = batch_tensors[SampleBatch.CUR_OBS]
    action = model.compute_actions(state)
    n_samples = config["branching_factor"]
    gamma = config["gamma"]

    if n_samples == 0:
        next_state = batch_tensors[SampleBatch.NEXT_OBS]
        if isinstance(next_state, (tuple, list)):
            next_state = tuple(tf.expand_dims(x, 0) for x in next_state)
        else:
            next_state = tf.expand_dims(next_state, 0)
        if config["use_true_dynamics"]:
            log_prob = env._transition_log_prob_fn(state, action, next_state)
        else:
            log_prob = model.compute_states_log_prob(state, action, next_state)
    else:
        if config["use_true_dynamics"]:
            next_state, log_prob = env._transition_fn(state, action, n_samples)
        else:
            next_state, log_prob = model.compute_log_prob_sampled(
                state, action, n_samples
            )

    next_state_value = tf.squeeze(model.compute_state_values(next_state), axis=-1)
    next_state_value_sf = log_prob * tf.stop_gradient(next_state_value)
    reward = tf.reduce_mean(env._reward_fn(state, action, next_state), axis=0)
    return tf.reduce_mean(reward + gamma * tf.reduce_mean(next_state_value_sf, axis=0))


def actor_model_aware_loss(batch_tensors, model, env, config):
    """Compute model-aware deterministic policy gradient loss."""
    if config["madpg_estimator"] == "sf":
        madpg_grad_estimator = sf_madpg_estimator(batch_tensors, model, env, config)
    elif config["madpg_estimator"] == "pd":
        madpg_grad_estimator = pd_madpg_estimator(batch_tensors, model, env, config)
    else:
        raise ValueError(
            "Invalid gradient estimator option '{}'. Try one of "
            "[sf, pd]".format(config["madpg_estimator"])
        )

    return -madpg_grad_estimator
