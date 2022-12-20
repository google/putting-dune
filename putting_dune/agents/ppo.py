# Copyright 2022 The Putting Dune Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyformat: mode=pyink
"""PPO utilities."""


import functools
from typing import NamedTuple, Optional, Sequence

from acme import specs
from acme.agents.jax import ppo
from acme.jax import experiments as acme_experiments
from acme.jax import networks as networks_lib
from acme.jax import utils as acme_utils
import frozendict
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
from putting_dune import run_helpers
from putting_dune.experiments import experiments


class MVNDiagParams(NamedTuple):
  """Parameters for a diagonal multi-variate normal distribution."""

  loc: jnp.ndarray
  scale_diag: jnp.ndarray


class TanhNormalParams(NamedTuple):
  """Parameters for a tanh squashed diagonal MVN distribution."""

  loc: jnp.ndarray
  scale: jnp.ndarray


def make_conv_networks(
    environment_spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (64, 64),
    value_layer_sizes: Sequence[int] = (64, 64),
    use_tanh_gaussian_policy: bool = True,
) -> ppo.PPONetworks:
  """Creates PPONetworks to be used for continuous action environments."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

  def forward_fn(inputs: networks_lib.Observation):
    nature_dqn = hk.Sequential([
        # 128x128 -> 31x31
        hk.Conv2D(output_channels=32, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        # 31x31 -> 14x14
        hk.Conv2D(output_channels=64, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        # 14x14 -> 12x12
        hk.Conv2D(output_channels=64, kernel_shape=(3, 3), stride=(1, 1)),
        jax.nn.relu,
        hk.Flatten(),
        hk.Linear(output_size=512),
        jax.nn.relu,  # TODO(joshgreaves): Keep this relu?
    ])

    def _policy_and_value_networks(obs: networks_lib.Observation):
      conv_output = nature_dqn(obs['image'])

      # Concatenate the goal info.
      conv_output = jnp.concatenate(
          (conv_output, obs['goal_delta_angstroms']), axis=-1
      )

      # First, get value outputs (because it's short).
      value_prediction = hk.Sequential([
          acme_utils.batch_concat,
          hk.nets.MLP(value_layer_sizes, activate_final=True),
          hk.Linear(1),
          lambda x: jnp.squeeze(x, axis=-1),
      ])(conv_output)

      # From here on out, it's all policy network.
      h = hk.nets.MLP(policy_layer_sizes, activate_final=True)(conv_output)

      # tfd distributions have a weird bug in jax when vmapping is used, so the
      # safer implementation in general is for the policy network to output the
      # distribution parameters, and for the distribution to be constructed
      # in a method such as make_ppo_networks above
      if not use_tanh_gaussian_policy:
        # Following networks_lib.MultivariateNormalDiagHead
        init_scale = 0.3
        min_scale = 1e-6
        w_init = hk.initializers.VarianceScaling(1e-4)
        b_init = hk.initializers.Constant(0.0)
        loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

        loc = loc_layer(h)
        scale = jax.nn.softplus(scale_layer(h))
        scale *= init_scale / jax.nn.softplus(0.0)
        scale += min_scale

        return MVNDiagParams(loc=loc, scale_diag=scale), value_prediction

      # Following networks_lib.NormalTanhDistribution
      min_scale = 1e-3
      w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform')
      b_init = hk.initializers.Constant(0.0)
      loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
      scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

      loc = loc_layer(h)
      scale = scale_layer(h)
      scale = jax.nn.softplus(scale) + min_scale

      return TanhNormalParams(loc=loc, scale=scale), value_prediction

    policy_output, value = _policy_and_value_networks(inputs)
    return (policy_output, value)

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = acme_utils.zeros_like(environment_spec.observations)
  dummy_obs = acme_utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply
  )

  # Create PPONetworks to add functionality required by the agent.

  if not use_tanh_gaussian_policy:
    return ppo.make_mvn_diag_ppo_networks(network)

  return ppo.make_tanh_normal_ppo_networks(network)


NETWORKS = frozendict.frozendict({
    'conv': make_conv_networks,
    'dense': lambda spec: ppo.make_networks(spec, (256, 256, 256)),
})


def build_experiment_config(
    seed: int,
    experiment: experiments.TrainExperiment,
    num_steps: int,
    network: str = 'dense',
    checkpoint_dir: Optional[str] = None,
) -> acme_experiments.ExperimentConfig:
  """Builds PPO experiment config which can be executed in different ways.

  Args:
    seed: The seed for the agent.
    experiment: The experiment to run.
    num_steps: The maximum number of steps to run the agent for.
    network: A string for the type of network to use. The options are the kets
      to the NETWORKS dictionary.
    checkpoint_dir: The checkpoint directory to save checkpoints to.

  Returns:
    An experiment configuration for a PPO experiment.
  """
  # Create an environment, grab the spec, and use it to create networks.

  config = ppo.PPOConfig(entropy_cost=0, learning_rate=1e-4)
  ppo_builder = ppo.PPOBuilder(config)

  if checkpoint_dir is not None:
    checkpointing_config = acme_experiments.CheckpointingConfig(
        directory=checkpoint_dir
    )
  else:
    checkpointing_config = None

  return acme_experiments.ExperimentConfig(
      builder=ppo_builder,
      environment_factory=functools.partial(
          run_helpers.create_putting_dune_env,
          get_adapters_and_goal=experiment.get_adapters_and_goal,
          get_simulator_config=experiment.get_simulator_config,
      ),
      network_factory=NETWORKS[network],
      seed=seed,
      checkpointing=checkpointing_config,
      logger_factory=run_helpers.make_logger,
      max_num_actor_steps=num_steps,
  )
