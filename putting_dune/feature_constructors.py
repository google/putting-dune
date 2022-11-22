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
"""Classes for constructing features from simulator state."""

import abc
import typing

from dm_env import specs
import numpy as np
from putting_dune import goals
from putting_dune import graphene
from putting_dune import simulator_utils
from sklearn import neighbors


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
      observation: simulator_utils.SimulatorObservation,
      goal: goals.Goal,
  ) -> np.ndarray:
    """Gets features for an agent based on the osbervation and goal."""

  @abc.abstractmethod
  def observation_spec(self) -> specs.Array:
    """Gets the osbervation spec for the constructed features."""


class SingleSiliconPristineGraphineFeatureConstuctor(FeatureConstructor):
  """A feature constructor assuming pristine graphene with single dopant.

  This goal used with this class must be SingleSiliconGoalReaching.
  """

  def reset(self) -> None:
    pass

  def get_features(
      self,
      observation: simulator_utils.SimulatorObservation,
      goal: goals.Goal,
  ) -> np.ndarray:
    """Gets features for an agent based on the osbervation and goal."""
    if not isinstance(goal, goals.SingleSiliconGoalReaching):
      raise ValueError(
          f'{self.__class__} only usable with goals.SingleSiliconGoalReaching.'
          f' Got {goal.__class__}'
      )
    goal = typing.cast(goals.SingleSiliconGoalReaching, goal)

    silicon_position = graphene.get_silicon_positions(observation.grid).reshape(
        2
    )

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

    material_frame_grid = observation.fov.microscope_grid_to_material_grid(
        observation.grid
    )
    silicon_position_material_frame = graphene.get_silicon_positions(
        material_frame_grid
    ).reshape(2)
    goal_delta_material_frame = (
        goal.goal_position_material_frame - silicon_position_material_frame
    )

    obs = np.concatenate([
        silicon_position,
        normalized_deltas.reshape(-1),
        goal_delta_material_frame,
    ])

    return obs.astype(np.float32)

  def observation_spec(self) -> specs.Array:
    # 2 for silicon position.
    # 6 for 3 nearest neighbor delta vectors.
    # 2 for goal delta.
    return specs.Array((2 + 6 + 2,), np.float32)
