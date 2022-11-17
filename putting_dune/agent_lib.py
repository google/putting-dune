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
"""Agents for Putting Dune."""

import abc
import enum
import functools
from typing import Sequence, Union

import dm_env
import numpy as np


@enum.unique
class AgentMode(enum.Enum):
  TRAIN = 'train'
  EVAL = 'eval'


class Agent(abc.ABC):
  """Abstract base class for agents."""

  @abc.abstractmethod
  def step(self, time_step: dm_env.TimeStep) -> np.ndarray:
    """Steps the agent.

    Arguments:
      time_step: The previous TimeStep object returned by the environment. From
        this object, the agent can infer whether this is the start or end of an
        episode.

    Returns:
      An action to apply. If step was called for the last time step of
        the environment, then the action won't be used.
    """

  @abc.abstractmethod
  def set_mode(self, mode: AgentMode) -> None:
    """Sets the mode of the agent."""


class UniformRandomAgent(Agent):
  """An agent that takes uniform random actions."""

  def __init__(
      self,
      rng: np.random.Generator,
      low: Union[float, np.ndarray],
      high: Union[float, np.ndarray],
      size: Sequence[int],
  ):
    """UniformRandomAgent constructor.

    Args:
      rng: The rng Generator to use for generating random actions.
      low: The lowest value to sample.
      high: The highest value to sample.
      size: The shape of the action to sample.
    """
    self._sample_action = functools.partial(rng.uniform, low, high, size)

  def step(self, time_step: dm_env.TimeStep) -> np.ndarray:
    return self._sample_action()

  def set_mode(self, mode: AgentMode) -> None:
    pass  # No action required.
