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
from putting_dune import microscope_utils


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
      initial_observation: microscope_utils.MicroscopeObservation,
  ):
    """Resets and picks a new goal."""

  @abc.abstractmethod
  def calculate_reward_and_terminal(
      self,
      observation: microscope_utils.MicroscopeObservation,
  ) -> GoalReturn:
    """Calculates the reward and terminal signals for the goal."""


class SingleSiliconGoalReaching(Goal):
  """A single-silicon goal-reaching goal."""

  def __init__(self):
    # For now, require only that we reach the goal. This makes
    # the problem much less sparse, especially under the
    # relative-to-silicon action adapter.
    self._required_consecutive_goal_steps_for_termination = 1

    # Will sample a goal within this distance range.
    self.goal_range_angstroms = (0.1, 50.0)

    # Will be set on reset.
    self.goal_position_material_frame = np.zeros((2,), dtype=np.float32)
    self._consecutive_goal_steps = 0

  def reset(
      self,
      rng: np.random.Generator,
      initial_observation: microscope_utils.MicroscopeObservation,
  ) -> None:
    """Resets the goal, picking a new position.

    Args:
      rng: The RNG to use for sampling a new goal.
      initial_observation: The initial simulator observation.

    Raises:
      RuntimeError: If the number of observed silicons is not 1.
    """
    # TODO(joshgreaves): Enable ability to sample goals outside FOV.
    silicon_position = graphene.get_silicon_positions(initial_observation.grid)
    num_silicon_atoms, _ = silicon_position.shape
    if num_silicon_atoms != 1:
      raise RuntimeError(
          f'{self.__class__} expected to find a single silicon atom. Instead,'
          f' {num_silicon_atoms} were found.'
      )

    # Get the distance of every atom from the silicon atom.
    # Center atoms on silicon atom, and then account for the FOV
    # before working out the distances in angstroms.
    shifted_atom_positions = (
        initial_observation.grid.atom_positions - silicon_position
    )

    scale = np.asarray(
        [initial_observation.fov.width, initial_observation.fov.height]
    )
    scaled_shifted_atom_positions = scale * shifted_atom_positions

    distances = np.linalg.norm(scaled_shifted_atom_positions, axis=1)

    # Select the atoms that are in the desired distance range.
    min_distance, max_distance = self.goal_range_angstroms
    valid_distances = (distances < max_distance) & (distances > min_distance)
    valid_goals = initial_observation.grid.atom_positions[valid_distances]

    num_goals, _ = valid_goals.shape
    if num_goals == 0:
      raise RuntimeError("Couldn't find any valid goals.")

    goal_idx = rng.choice(num_goals)
    goal_position = valid_goals[goal_idx]

    goal_grid = microscope_utils.AtomicGrid(
        goal_position, np.asarray([constants.CARBON])
    )
    self.goal_position_material_frame = (
        initial_observation.fov.microscope_grid_to_material_grid(
            goal_grid
        ).atom_positions[0, :]
    )
    self._consecutive_goal_steps = 0

  def calculate_reward_and_terminal(
      self,
      observation: microscope_utils.MicroscopeObservation,
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

    return GoalReturn(reward=-cost, is_terminal=is_terminal, is_truncated=False)
