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
"""Atomic manipulation goals."""

import abc
import dataclasses

import numpy as np
from putting_dune import constants
from putting_dune import graphene
from putting_dune import simulator_utils
from sklearn import neighbors


@dataclasses.dataclass(frozen=True)
class GoalReturn:
  reward: float
  is_terminal: bool
  is_truncated: bool


class Goal(abc.ABC):
  """Interface for goals."""

  @abc.abstractmethod
  def reset(
      self,
      rng: np.random.Generator,
      initial_observation: simulator_utils.SimulatorObservation,
  ):
    """Resets and picks a new goal."""

  @abc.abstractmethod
  def calculate_reward_and_terminal(
      self,
      observation: simulator_utils.SimulatorObservation,
  ) -> GoalReturn:
    """Calculates the reward and terminal signals for the goal."""


class SingleSiliconGoalReaching(Goal):
  """A single-silicon goal-reaching goal."""

  def __init__(self):
    # For now, require only that we reach the goal. This makes
    # the problem much less sparse, especially under the
    # relative-to-silicon action adapter.
    self._required_consecutive_goal_steps_for_termination = 1

    # Will be set on reset.
    self.goal_position_material_frame = np.zeros((2,), dtype=np.float32)
    self._consecutive_goal_steps = 0

  def reset(
      self,
      rng: np.random.Generator,
      initial_observation: simulator_utils.SimulatorObservation,
  ):
    """Resets the goal, picking a new position.

    Args:
      rng: The RNG to use for sampling a new goal.
      initial_observation: The initial simulator observation.
    """
    # TODO(joshgreaves): Enable ability to sample goals outside FOV.
    num_atoms, _ = initial_observation.grid.atom_positions.shape
    atom_idx = rng.choice(num_atoms)
    self.goal_position_material_frame = (
        initial_observation.fov.microscope_grid_to_material_grid(
            initial_observation.grid
        ).atom_positions[atom_idx]
    )
    self._consecutive_goal_steps = 0

  def calculate_reward_and_terminal(
      self,
      observation: simulator_utils.SimulatorObservation,
  ) -> GoalReturn:
    """Calculates the reward and terminal signals for the goal.

    Note: we assume that this is called once per simulator/agent step.

    Args:
      observation: The last observation from the simulator.

    Raises:
      RuntimeError: If the number of observed silicons is not 1.

    Returns:
      The reward, and whether the episode is terminal or should be
        truncated. Truncation happens when atoms get to the edge of
        the material and we can no longer simulate.
    """
    # Calculate the reward.
    material_frame_grid = observation.fov.microscope_grid_to_material_grid(
        observation.grid
    )
    silicon_positions = graphene.get_silicon_positions(material_frame_grid)

    num_silicon, _ = silicon_positions.shape
    if num_silicon != 1:
      raise RuntimeError(f'Found {num_silicon} silicon when 1 was expected.')
    silicon_position_material_frame = silicon_positions.reshape(2)

    cost = np.linalg.norm(
        silicon_position_material_frame - self.goal_position_material_frame
    )

    # Update whether the silicon is near the goal.
    goal_radius = constants.CARBON_BOND_DISTANCE_ANGSTROMS * 0.5
    goal_distance = np.linalg.norm(
        silicon_position_material_frame - self.goal_position_material_frame
    )
    if goal_distance < goal_radius:
      self._consecutive_goal_steps += 1
    else:
      self._consecutive_goal_steps = 0

    # Calculate whether it is a terminal state.
    is_terminal = (
        self._consecutive_goal_steps
        >= self._required_consecutive_goal_steps_for_termination
    )

    # Truncate if near the graphene edge.
    nearest_neighbors = neighbors.NearestNeighbors(
        n_neighbors=1 + 3,
        metric='l2',
        algorithm='brute',
    ).fit(material_frame_grid.atom_positions)
    si_neighbor_distances, _ = nearest_neighbors.kneighbors(
        silicon_position_material_frame.reshape(1, 2)
    )

    # If any of the neighbors are much greater than the expected bond distance,
    # then there aren't three neighbors and we're at the edge.
    # Since the neighbors are sorted, just look at the furthest neighbor.
    is_truncation = (
        si_neighbor_distances[0, -1]
        > constants.CARBON_BOND_DISTANCE_ANGSTROMS * 1.1
    )

    return GoalReturn(-cost, is_terminal, is_truncation)
