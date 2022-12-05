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
"""An agent for interfacing directly with a microscope."""

from typing import List

import dm_env
import numpy as np
from putting_dune import experiment_registry
from putting_dune import microscope_utils


class MicroscopeAgent:
  """An agent for interfacing directly with a microscope."""

  def __init__(
      self,
      experiment=experiment_registry.Experiment,
  ):
    self.agent = experiment.agent
    self.env_config = experiment.env_config
    self._is_first_step = True

  def reset(
      self,
      rng: np.random.Generator,
      observation: microscope_utils.MicroscopeObservation,
  ) -> None:
    """Resets the agent."""
    self.env_config.feature_constructor.reset()
    self.env_config.goal.reset(rng, observation)
    self.env_config.action_adapter.reset()

    self._is_first_step = True

  def step(
      self,
      observation: microscope_utils.MicroscopeObservation,
  ) -> List[microscope_utils.BeamControl]:
    """Steps the agent."""
    features = self.env_config.feature_constructor.get_features(
        observation, self.env_config.goal
    )
    goal_return = self.env_config.goal.calculate_reward_and_terminal(
        observation
    )

    # TODO(joshgreaves): What discount to use?
    discount = 0.99

    if self._is_first_step:
      # We have restart override terminal or truncation.
      # There is a small chance that we could start in a terminal state,
      # but it is more important that the agent knows that we have started a
      # new episode than catching this edge case.
      time_step = dm_env.restart(features)
    if goal_return.is_terminal:
      time_step = dm_env.termination(goal_return.reward, features)
    elif goal_return.is_truncated:
      time_step = dm_env.truncation(goal_return.reward, features, discount)
    elif self._is_first_step:
      time_step = dm_env.restart(features)
    else:
      time_step = dm_env.transition(goal_return.reward, features, discount)

    action = self.agent.step(time_step)

    beam_control = self.env_config.action_adapter.get_action(
        observation.grid, action
    )

    # First step is only after immediately calling reset.
    self._is_first_step = False

    return beam_control
