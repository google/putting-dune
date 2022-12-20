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
"""Helpers for loading Acme checkpoints."""

import os.path as osp
from typing import Optional

from absl import logging
from acme import core
from acme import specs
from acme.jax import experiments
from acme.tf import savers
import dm_env
import jax
import numpy as np
from putting_dune.agents import agent_lib


class AcmeCheckpointer(savers.Checkpointer):
  """Helper class for loading checkpoint of a trained Acme agent."""

  def restore(self):
    """Overrides this to avoid loading the latest checkpoint by default."""
    # `savers.Checkpointer` always restores the latest agent, so we `pass` it.
    pass

  def restore_checkpoint(self, checkpoint_number: Optional[int] = None):
    if checkpoint_number is None:
      checkpoint_to_reload = self._checkpoint_manager.latest_checkpoint
      if checkpoint_to_reload is None:
        raise FileNotFoundError(
            f"Couldn't find any checkpoints at {self.checkpoint_dir}"
        )
    else:
      checkpoint_to_reload = osp.join(
          self.checkpoint_dir, f'ckpt-{checkpoint_number}'
      )
    logging.info('Attempting to restore checkpoint: %s', checkpoint_to_reload)
    self._checkpoint.restore(checkpoint_to_reload).expect_partial()

  @property
  def checkpoint_dir(self):
    return self._checkpoint_manager.directory


class _AcmeEvalAgentBuilder(object):
  """Helper class for loading checkpoint of a trained Acme agent.

  This class can be used as follows:
    builder = _AcmeEvalAgentBuilder(...)  # Pass the appropriate arguments.
    actor = builder.load_checkpoint(checkpoint_dir)
    actor.select_action(time_step.observation)
  """

  def __init__(self, experiment_config: experiments.ExperimentConfig):
    """Initializes the evaluation agent.

    Args:
      experiment_config: Experiment config for the agent to load.
    """
    self.learner: core.Learner
    self._env_spec = specs.make_environment_spec(
        experiment_config.environment_factory(0)
    )
    self._network_fn = experiment_config.network_factory
    self._agent_builder = experiment_config.builder
    self._logger_factory = experiment_config.logger_factory
    self._setup_learner()

  def _setup_learner(self) -> None:
    """Creates actor and learner."""
    # Create learner using the agent_builder and network_fn.
    self._network = self._network_fn(self._env_spec)
    self.learner = self._agent_builder.make_learner(
        jax.random.PRNGKey(0),
        self._network,
        iter([]),
        self._logger_factory,
        self._env_spec,
    )
    # Create evaluation policy.
    self._policy = self._agent_builder.make_policy(
        self._network, self._env_spec, evaluation=True
    )

  def make_actor(self) -> core.Actor:
    return self._agent_builder.make_actor(
        jax.random.PRNGKey(0),
        self._policy,
        self._env_spec,
        variable_source=self.learner,
    )

  def load_checkpoint(
      self, checkpoint_dir: str, checkpoint_number: Optional[int] = None
  ) -> core.Actor:
    """Loads checkpoint from the specified directory and returns an Actor.

    Args:
      checkpoint_dir: Directory from which to load the agent checkpoint.
      checkpoint_number: Checkpoint to load. If None, defaults to loading the
        latest checkpoint.

    Returns:
      Acme actor that can be used in an interactive environment loop.
    """
    try:
      checkpointer = AcmeCheckpointer(
          {'learner': self.learner},
          subdirectory='learner',
          directory=checkpoint_dir,
          add_uid=False,
      )
      checkpointer.restore_checkpoint(checkpoint_number)
      return self.make_actor()
    except OSError as e:
      raise FileNotFoundError(
          f'Unable to load checkpoint at {checkpoint_dir}'
      ) from e


class AcmeEvalAgent(agent_lib.Agent):
  """An Acme eval agent."""

  def __init__(
      self,
      experiment_config: experiments.ExperimentConfig,
      checkpoint_dir: str,
      checkpoint_number: Optional[int] = None,
  ):
    builder = _AcmeEvalAgentBuilder(experiment_config)
    self._actor = builder.load_checkpoint(checkpoint_dir, checkpoint_number)
    self._last_action = None

  def step(self, time_step: dm_env.TimeStep) -> np.ndarray:
    if time_step.step_type == dm_env.StepType.FIRST:
      self._actor.observe_first(time_step)
    else:
      self._actor.observe(self._last_action, time_step)

    self._last_action = self._actor.select_action(time_step.observation)
    return self._last_action

  def set_mode(self, mode: agent_lib.AgentMode) -> None:
    pass  # No action required.
