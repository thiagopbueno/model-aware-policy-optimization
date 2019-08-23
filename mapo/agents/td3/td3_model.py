"""ModelV2 for TD3."""
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override

from mapo.models.policy import build_deterministic_policy
from mapo.models.q_function import build_continuous_q_function


class TD3Model(TFModelV2):  # pylint: disable=abstract-method
    """Extension of standard TFModel for TD3."""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, twin_q=False
    ):
        # pylint: disable=too-many-arguments,arguments-differ
        # Ignore num_outputs as we don't use a shared state preprocessor
        output_dim = sum(obs_space.shape)
        super().__init__(obs_space, action_space, output_dim, model_config, name)

        self.options = model_config["custom_options"]
        self.policy = build_deterministic_policy(
            obs_space, action_space, **self.options["actor"]
        )
        self.register_variables(self.policy.variables)
        self.q_net = build_continuous_q_function(
            obs_space, action_space, **self.options["critic"]
        )
        self.register_variables(self.q_net.variables)

        self.twin_q = twin_q
        if twin_q:
            self.twin_q_net = build_continuous_q_function(
                obs_space, action_space, **self.options["critic"]
            )
            self.register_variables(self.twin_q_net.variables)

    @override(TFModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Return the flattened observation.

        This is a hack so that `__call__` does not complain when building model
        output in `DynamicTFPolicy.__init__`.
        """
        # pylint: disable=unused-argument,no-self-use
        return input_dict["obs_flat"], state

    @property
    def actor_variables(self):
        """List of actor variables."""
        return self.policy.variables

    @property
    def critic_variables(self):
        """List of critic variables."""
        return self.q_net.variables + (self.twin_q_net.variables if self.twin_q else [])

    def get_actions(self, obs_tensor):
        """Compute actions using policy network."""
        return self.policy(obs_tensor)

    def get_q_values(self, obs_tensor, action_tensor):
        """Compute action values using main Q network."""
        return self.q_net([obs_tensor, action_tensor])

    def get_twin_q_values(self, obs_tensor, action_tensor):
        """Compute action values using twin Q network."""
        return self.twin_q_net([obs_tensor, action_tensor])
