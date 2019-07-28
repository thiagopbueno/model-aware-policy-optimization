"""Tests regarding delayed policy updates in OffMAPO."""

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from mapo.tests.mock_env import MockEnv
from mapo.agents.off_mapo.off_mapo_policy import OffMAPOTFPolicy


def get_dummy_batch(policy):
    """Get fake batch based on the policy's input."""
    # pylint: disable=protected-access
    def fake_array(tensor):
        shape = tensor.shape.as_list()
        shape[0] = 1
        return np.zeros(shape, dtype=tensor.dtype.as_numpy_dtype)

    dummy_batch = {
        SampleBatch.CUR_OBS: fake_array(policy._obs_input),
        SampleBatch.NEXT_OBS: fake_array(policy._obs_input),
        SampleBatch.DONES: np.array([False], dtype=np.bool),
        SampleBatch.ACTIONS: fake_array(
            ModelCatalog.get_action_placeholder(policy.action_space)
        ),
        SampleBatch.REWARDS: np.array([0], dtype=np.float32),
    }
    if policy._obs_include_prev_action_reward:
        dummy_batch.update(
            {
                SampleBatch.PREV_ACTIONS: fake_array(policy._prev_action_input),
                SampleBatch.PREV_REWARDS: fake_array(policy._prev_reward_input),
            }
        )
    state_init = policy.get_initial_state()
    for idx, hid in enumerate(state_init):
        dummy_batch["state_in_{}".format(idx)] = np.expand_dims(hid, 0)
        dummy_batch["state_out_{}".format(idx)] = np.expand_dims(hid, 0)
    if state_init:
        dummy_batch["seq_lens"] = np.array([1], dtype=np.int32)
    for key, val in policy.extra_compute_action_fetches().items():
        dummy_batch[key] = fake_array(val)
    return dummy_batch


def test_update_optimizer_global_step():
    """Check that apply operations are run the correct number of times."""
    # pylint: disable=protected-access
    env = MockEnv()
    policy = OffMAPOTFPolicy(
        env.observation_space, env.action_space, {"policy_delay": 2}
    )

    dummy_batch = get_dummy_batch(policy)
    for _ in range(6):
        policy.learn_on_batch(dummy_batch)

    sess = policy.get_session()
    assert sess.run(policy.global_step) == 6
    assert sess.run(policy._actor_optimizer.iterations) == 3
    assert sess.run(policy._critic_optimizer.iterations) == 6
