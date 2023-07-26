# Copyright 2024 The Putting Dune Authors.
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

"""Classes for constructing features from simulator state."""

import abc
import typing
from typing import Dict, Union

from cvx2 import latest as cv2
from dm_env import specs
import numpy as np
from putting_dune import goals
from putting_dune import graphene
from putting_dune import microscope_utils
from sklearn import neighbors

# This supports current usage. Expand as necessary.
NestedObservation = Union[np.ndarray, Dict[str, 'NestedObservation']]
NestedObservationSpec = Union[specs.Array, Dict[str, 'NestedObservationSpec']]


class FeatureConstructor(abc.ABC):
  """Abstract base class for feature constructors.

  A feature constructor is component that translates from a simulator
  observation to a numpy array to be used by a learning agent.
  """

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the feature constructor."""

  @abc.abstractmethod
  def get_features(
      self,
      observation: microscope_utils.MicroscopeObservation,
      goal: goals.Goal,
  ) -> NestedObservation:
    """Gets features for an agent based on the osbervation and goal."""

  @abc.abstractmethod
  def observation_spec(self) -> NestedObservationSpec:
    """Gets the osbervation spec for the constructed features."""

  @property
  @abc.abstractmethod
  def requires_image(self) -> bool:
    """Returns True if the feature constructor requires an image."""


def _get_silicon_goal_delta(
    grid: microscope_utils.AtomicGridMicroscopeFrame,
    fov: microscope_utils.MicroscopeFieldOfView,
    goal: goals.SingleSiliconGoalReaching,
) -> np.ndarray:
  """Gets the delta from the current silicon position to goal in angstroms."""
  silicon_position = graphene.get_silicon_positions(grid).reshape(2)
  silicon_position_material_frame = fov.microscope_frame_to_material_frame(
      silicon_position
  )
  goal_delta_angstroms = (
      goal.goal_position_material_frame - silicon_position_material_frame
  )
  return goal_delta_angstroms


class SingleSiliconPristineGrapheneFeatureConstuctor(FeatureConstructor):
  """A feature constructor assuming pristine graphene with single dopant.

  The goal used with this class must be SingleSiliconGoalReaching.
  """

  def reset(self) -> None:
    pass

  def get_features(
      self,
      observation: microscope_utils.MicroscopeObservation,
      goal: goals.Goal,
  ) -> np.ndarray:
    """Gets features for an agent based on the osbervation and goal.

    Args:
      observation: The observation from the microscope.
      goal: The current goal we are executing.

    Returns:
      A numpy feature vector.

    Raises:
      SiliconNotFoundError: if no silicon atom was found.
    """
    if not isinstance(goal, goals.SingleSiliconGoalReaching):
      raise ValueError(
          f'{self.__class__} only usable with goals.SingleSiliconGoalReaching.'
          f' Got {goal.__class__}'
      )
    goal = typing.cast(goals.SingleSiliconGoalReaching, goal)

    silicon_position = graphene.get_single_silicon_position(observation.grid)

    # Ensure the silicon position shape is correct.
    silicon_position = silicon_position.reshape(2)

    # Get the vectors to the nearest neighbors.
    nearest_neighbors = neighbors.NearestNeighbors(
        n_neighbors=1 + 3,
        metric='l2',
        algorithm='brute',
    ).fit(observation.grid.atom_positions)
    neighbor_distances, neighbor_indices = nearest_neighbors.kneighbors(
        silicon_position.reshape(1, 2)
    )
    neighbor_positions = observation.grid.atom_positions[
        neighbor_indices[0, 1:]
    ]
    neighbor_deltas = neighbor_positions - silicon_position.reshape(1, 2)
    neighbor_distances = neighbor_distances[0, 1:].reshape(-1, 1)
    normalized_deltas = neighbor_deltas / neighbor_distances

    goal_delta_angstroms = _get_silicon_goal_delta(
        observation.grid, observation.fov, goal
    )

    obs = np.concatenate([
        silicon_position,
        normalized_deltas.reshape(-1),
        goal_delta_angstroms,
    ])

    return obs.astype(np.float32)

  def observation_spec(self) -> specs.Array:
    # 2 for silicon position.
    # 6 for 3 nearest neighbor delta vectors.
    # 2 for goal delta.
    return specs.Array((2 + 6 + 2,), np.float32)

  @property
  def requires_image(self) -> bool:
    """Returns True if the feature constructor requires an image."""
    return False


class SingleSiliconMaterialFrameFeatureConstructor(FeatureConstructor):
  """A feature constructor designed for agents that need material-frame inputs.

  The goal used with this class must be SingleSiliconGoalReaching.
  """

  def reset(self) -> None:
    pass

  def get_features(
      self,
      observation: microscope_utils.MicroscopeObservation,
      goal: goals.Goal,
  ) -> np.ndarray:
    """Gets features for an agent based on the osbervation and goal.

    Args:
      observation: The observation from the microscope.
      goal: The current goal we are executing.

    Returns:
      A numpy feature vector.

    Raises:
      SiliconNotFoundError: if no silicon atom was found.
    """
    if not isinstance(goal, goals.SingleSiliconGoalReaching):
      raise ValueError(
          f'{self.__class__} only usable with goals.SingleSiliconGoalReaching.'
          f' Got {goal.__class__}'
      )
    goal = typing.cast(goals.SingleSiliconGoalReaching, goal)
    grid = observation.fov.microscope_frame_to_material_frame(observation.grid)

    silicon_position = graphene.get_single_silicon_position(grid)

    # Ensure the silicon position shape is correct.
    silicon_position = silicon_position.reshape(2)

    # Get the vectors to the nearest neighbors.
    nearest_neighbors = neighbors.NearestNeighbors(
        n_neighbors=1 + 3,
        metric='l2',
        algorithm='brute',
    ).fit(grid.atom_positions)
    _, neighbor_indices = nearest_neighbors.kneighbors(
        silicon_position.reshape(1, 2)
    )
    neighbor_positions = grid.atom_positions[neighbor_indices[0, 1:]]
    neighbor_deltas = neighbor_positions - silicon_position.reshape(1, 2)
    goal_delta_angstroms = _get_silicon_goal_delta(
        observation.grid, observation.fov, goal
    )

    obs = np.concatenate([
        silicon_position,
        neighbor_deltas.reshape(-1),
        goal_delta_angstroms,
    ])

    return obs.astype(np.float32)

  def observation_spec(self) -> specs.Array:
    # 2 for silicon position.
    # 6 for 3 nearest neighbor delta vectors.
    # 2 for goal delta.
    return specs.Array((2 + 6 + 2,), np.float32)

  @property
  def requires_image(self) -> bool:
    """Returns True if the feature constructor requires an image."""
    return False


class ImageFeatureConstructor(FeatureConstructor):
  """An image feature constructor for single silicon goal reaching."""

  def reset(self) -> None:
    pass

  def get_features(
      self,
      observation: microscope_utils.MicroscopeObservation,
      goal: goals.Goal,
  ) -> Dict[str, np.ndarray]:
    if not isinstance(goal, goals.SingleSiliconGoalReaching):
      raise ValueError(
          f'{self.__class__} only usable with goals.SingleSiliconGoalReaching.'
          f' Got {goal.__class__}'
      )
    goal = typing.cast(goals.SingleSiliconGoalReaching, goal)

    if observation.image is None:
      raise RuntimeError(
          f'No image found in obsevation for {self.__class__}.get_features.'
      )

    resized_image = (
        cv2.resize(observation.image, (128, 128))
        .reshape(128, 128, 1)
        .astype(np.float32)
    )

    goal_delta_angstroms = _get_silicon_goal_delta(
        observation.grid, observation.fov, goal
    )

    return {
        'image': resized_image.astype(np.float32),
        'goal_delta_angstroms': goal_delta_angstroms.astype(np.float32),
    }

  def observation_spec(self) -> Dict[str, specs.Array]:
    return {
        'image': specs.Array(
            (128, 128, 1),
            np.float32,
        ),
        'goal_delta_angstroms': specs.Array((2,), np.float32),
    }

  @property
  def requires_image(self) -> bool:
    return True
