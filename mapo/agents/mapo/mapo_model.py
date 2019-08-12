"""ModelV2 for MAPO."""

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override

from mapo.models.policy import build_deterministic_policy
from mapo.models.q_function import build_continuous_q_function
from mapo.models.dynamics import GaussianDynamicsModel


class MAPOModel(TFModelV2):  # pylint: disable=abstract-method
    """Extension of standard TFModel for MAPO.

    Encapsulates all the networks necessary for Model-Aware Policy Optimization.
    Exposes the necessary API for loss calculation and variable tracking.

    Keyword arguments:
        target_networks(bool): Whether or not to create target actor and critic.
        twin_q=(bool): Whether or not to create twin Q networks.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        target_networks=False,
        twin_q=False,
    ):
        # pylint: disable=too-many-arguments,arguments-differ
        prep = get_preprocessor(obs_space)(obs_space)
        # Ignore num_outputs as we don't use a shared state preprocessor
        super().__init__(obs_space, action_space, prep.size, model_config, name)

        models = {}
        models["policy"] = build_deterministic_policy(
            obs_space, action_space, model_config["custom_options"]["actor"]
        )
        models["q_net"] = build_continuous_q_function(
            obs_space, action_space, model_config["custom_options"]["critic"]
        )
        self.twin_q = twin_q
        if twin_q:
            models["twin_q_net"] = build_continuous_q_function(
                obs_space, action_space, model_config["custom_options"]["critic"]
            )

        models["dynamics"] = GaussianDynamicsModel(
            obs_space, action_space, **model_config["custom_options"]["dynamics"]
        )
        self.models = models
        self.register_variables(
            [variable for model in models.values() for variable in model.variables]
        )

        if target_networks:
            target_models = {}
            target_models["policy"] = build_deterministic_policy(
                obs_space, action_space, model_config["custom_options"]["actor"]
            )
            target_models["q_net"] = build_continuous_q_function(
                obs_space, action_space, model_config["custom_options"]["critic"]
            )
            if twin_q:
                target_models["twin_q_net"] = build_continuous_q_function(
                    obs_space, action_space, model_config["custom_options"]["critic"]
                )
            self.target_models = target_models
            self.register_variables(
                [var for model in target_models.values() for var in model.variables]
            )

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
        return self.models["policy"].variables

    @property
    def critic_variables(self):
        """List of critic variables."""
        return self.models["q_net"].variables + (
            self.models["twin_q_net"].variables if self.twin_q else []
        )

    @property
    def dynamics_variables(self):
        """List of dynamics model variables."""
        return self.models["dynamics"].variables

    @property
    def main_and_target_variables(self):
        """List of (main, target) variable pairs for applicable models."""
        return [
            (main, target)
            for key in self.target_models
            for main, target in zip(
                self.models[key].variables, self.target_models[key].variables
            )
        ]

    def get_actions(self, obs, target=False):
        """Compute actions using policy network.

        Keyword arguments:
            target(bool): Whether or not to use the target model.
        """
        return self._get_model("policy", target)(obs)

    def get_q_values(self, obs, action, target=False):
        """Compute action values using main Q network.

        Keyword arguments:
            target(bool): Whether or not to use the target model.
        """
        return self._get_model("q_net", target)([obs, action])

    def get_twin_q_values(self, obs, action, target=False):
        """Compute action values using twin Q network.

        Keyword arguments:
            target(bool): Whether or not to use the target model.
        """
        return self._get_model("twin_q_net", target)([obs, action])

    def get_state_values(self, obs):
        """Compute state values by composing policy and Q networks."""
        return self.get_q_values(obs, self.get_actions(obs))

    def next_state_dist(self, obs, action):
        """Compute the the dynamics model's conditional distribution of
        the next state."""
        return self.models["dynamics"].dist(obs, action)

    def sample_next_state(self, obs, action, shape=()):
        """Sample the next state from the dynamics model."""
        return self.models["dynamics"].sample(obs, action, shape=shape)

    def next_state_log_prob(self, obs, action, next_obs):
        """Compute the log-likelihood of a transition using the dynamics model."""
        return self.models["dynamics"].log_prob(obs, action, next_obs)

    def _get_model(self, name, is_target):
        return self.target_models[name] if is_target else self.models[name]
