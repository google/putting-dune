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
"""A collection of action adapters for PuttingDuneEnvironment."""

import abc
import datetime as dt
from typing import List

from absl import logging
from dm_env import specs
import numpy as np
from putting_dune import graphene
from putting_dune import microscope_utils
from shapely import geometry
from sklearn import neighbors


class ActionAdapter(abc.ABC):
  """Abstract base class for action adapters."""

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the action adapter.

    Called at the beginning of a new episode.
    """

  @abc.abstractmethod
  def get_action(
      self,
      grid: microscope_utils.AtomicGrid,
      action: np.ndarray,
  ) -> List[microscope_utils.BeamControl]:
    """Gets simulator controls from the agent action."""

  @property
  @abc.abstractmethod
  def action_spec(self) -> specs.BoundedArray:
    """Gets the action spec for the action adapter."""


class DeltaPositionActionAdapter(ActionAdapter):
  """An action adapter that moves the beam by a supplied delta.

  It uses a fixed dwell time for each action.
  """

  def __init__(self, rng: np.random.Generator):
    self.rng = rng
    self.reset()

  def reset(self):
    # Randomly place beam anywhere in [0, 1] frame.
    self.beam_pos = self.rng.uniform(0, 1, size=2)

  def get_action(
      self,
      grid: microscope_utils.AtomicGrid,
      action: np.ndarray,
  ) -> List[microscope_utils.BeamControl]:
    del grid  # Unused.

    self.beam_pos += action
    # For now, we clip the beam to ensure it doesn't leave the field
    # of view. In the future, we may want to change this.
    self.beam_pos = np.clip(self.beam_pos, 0.0, 1.0)

    return [
        microscope_utils.BeamControl(
            geometry.Point(self.beam_pos[0], self.beam_pos[1]),
            # TODO(joshgreaves): Choose/parameterize dwell time.
            dt.timedelta(seconds=1.5),
        )
    ]

  @property
  def action_spec(self) -> specs.BoundedArray:
    # x, y position of the STEM probe.
    return specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-0.1, maximum=0.1
    )


class RelativeToSiliconActionAdapter(ActionAdapter):
  """An action adapter that accepts a relative position to a silicon atom."""

  def reset(self):
    pass

  def get_action(
      self,
      grid: microscope_utils.AtomicGrid,
      action: np.ndarray,
  ) -> List[microscope_utils.BeamControl]:
    """Gets simulator controls from the agent action."""
    silicon_position = graphene.get_silicon_positions(grid)

    if silicon_position.shape != (1, 2):
      raise RuntimeError(
          'Expected to find one silicon with x, y coordinates. Instead, '
          f'got {silicon_position.shape[0]} silicon atoms with '
          f'{silicon_position.shape[1]} dimensions.'
      )

    nearest_neighbors = neighbors.NearestNeighbors(
        n_neighbors=4,  # Self and 3 nearest.
        metric='l2',
        algorithm='brute',
    ).fit(grid.atom_positions)
    neighbor_distances, _ = nearest_neighbors.kneighbors(silicon_position)
    neighbor_distances = neighbor_distances.reshape(-1)

    if abs(neighbor_distances[1] - neighbor_distances[3]) > 1e-3:
      bond_distance_difference = abs(
          neighbor_distances[1] - neighbor_distances[3]
      )
      logging.warning(
          (
              '%s is intended for pristine graphene. It expects the 1st and '
              '3rd nearest neighbors to be roughly equidistant. '
              'Difference was %.3f'
          ),
          self.__class__.__name__,
          bond_distance_difference,
      )

    # Action is [dx, dy] in unit cell terms.
    cell_radius = np.mean(neighbor_distances[1:])
    control_position = silicon_position + (action * cell_radius)

    return [
        microscope_utils.BeamControl(
            geometry.Point(*control_position), dt.timedelta(seconds=1.5)
        )
    ]

  @property
  def action_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0
    )
