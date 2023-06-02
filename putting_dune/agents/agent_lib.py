# Copyright 2023 The Putting Dune Authors.
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

"""Agents for Putting Dune."""

import abc
import enum
import functools
from typing import Callable, Optional, Sequence, Union
import dm_env
import numpy as np
from putting_dune import geometry


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


class GreedyAgent(Agent):
  """An agent that acts greedily according to a transition function.

  Optionally, some randomization can be added, for the sake of data collection,
  and a fixed offset to the argmax may be specified.

  The argmax is assumed to be calculated for an Si with a neighbor positioned
  directly in the positive X-direction, and is the beam position most likely to
  cause a transition to that neighbor.

  If no transition function is specified, a standard offset of 1.42 A towards
  the neighbor will be used. A manual argmax may also be specified in place of
  this.

  Currently must be used with the SingleSiliconMaterialFrameFeatureConstructor,
  and the RelativeToSiliconMaterialFrameActionAdapter.
  """

  def __init__(
      self,
      rng: np.random.Generator = None,
      transition_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
      argmax: Optional[np.ndarray] = np.asarray([1.42, 0.0]),
      argmax_resolution: float = 0.05,
      position_noise_sigma: float = 0.0,
      fixed_offset: np.ndarray = np.zeros(2, dtype=np.float32),
      low: Union[float, np.ndarray] = -5,
      high: Union[float, np.ndarray] = 5,
  ):
    """GreedyAgent constructor.

    Args:
      rng: The rng Generator to use any randomness.
      transition_function: function that takes a beam position and predicts
        transition probabilities.
      argmax: manual specification of greedy position (For a silicon with a
        neighbor at (1.42, 0)).
      argmax_resolution: resolution in Angstroms for argmax-finding.
      position_noise_sigma: standard deviation for extra beam position noise.
      fixed_offset: Fixed vector to add to beam position (on top of argmax).
      low: The lowest value to sample.
      high: The highest value to sample.
    """
    self._position_noise_sigma = position_noise_sigma
    self._fixed_offset = fixed_offset
    self._rng = rng if rng is not None else np.random.default_rng()
    self._low = low
    self._high = high
    if transition_function is not None:
      self._argmax = self.find_argmax(transition_function, argmax_resolution)
    elif argmax is not None:
      self._argmax = argmax
    else:
      raise ValueError('One of transition_function or argmax must be set.')

  def find_argmax(
      self,
      transition_function: Callable[[np.ndarray], np.ndarray],
      resolution: float = 0.05,
  ) -> np.ndarray:
    """Finds the argmax of a transition function by grid search.

    Args:
      transition_function: A function taking a numpy array of shape (2,) and
        returning the probability (or rate) of transitioning to three neighbors.
      resolution: Grid resolution (in Angstroms) to use for the search.

    Returns:
      The approximate argmax of the transition function with respect to
      transitioning to neighbor 0.
    """
    num_points = int((self._high - self._low) // resolution)
    points_1d = np.linspace(self._low, self._high, num_points, dtype=np.float32)
    points_x = np.tile(points_1d[None], (num_points, 1))
    points_y = np.tile(points_1d[:, None], (1, num_points))
    points = np.stack([points_x, points_y], axis=-1)
    points = np.reshape(points, (-1, points.shape[-1]))
    transition_probabilities = np.stack(
        [transition_function(x) for x in points], 0
    )
    return points[np.argmax(transition_probabilities[..., 0], axis=-1)]

  def step(self, time_step: dm_env.TimeStep) -> np.ndarray:
    assert time_step.observation.shape == (10,)
    neighbor_deltas = time_step.observation[2:-2].reshape(3, 2)
    goal_delta = time_step.observation[-2:]

    neighbor_scores = np.linalg.norm(
        neighbor_deltas - goal_delta[None], axis=-1
    )
    best_neighbor = np.argmin(neighbor_scores, axis=-1)
    angles = geometry.get_angles(neighbor_deltas)
    angle = angles[best_neighbor]

    beam_position = self._argmax + self._fixed_offset
    beam_position_noise = self._rng.normal(
        0, self._position_noise_sigma, size=2
    )
    beam_position = beam_position + beam_position_noise

    rotated_beam_position = geometry.rotate_coordinates(beam_position, angle)

    return rotated_beam_position

  def set_mode(self, mode: AgentMode) -> None:
    pass  # No action required.
