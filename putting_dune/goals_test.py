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

"""Tests for goals."""

import unittest

from absl.testing import absltest
import numpy as np
from putting_dune import constants
from putting_dune import geometry
from putting_dune import goals
from putting_dune import graphene
from putting_dune import microscope_utils
from putting_dune import simulator


class SingleSiliconGoalReachingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)
    self.goal = goals.SingleSiliconGoalReaching()
    # Make a small graphene sheet to more thoroughly test what happens
    # at the edge of a graphene sheet.
    self.material = graphene.PristineSingleDopedGraphene(grid_columns=50)
    self.sim = simulator.PuttingDuneSimulator(self.material)

  def test_goal_position_is_set_to_a_lattice_position(self):
    # Reset several times to check it's always a lattice position.
    for _ in range(10):
      obs = self.sim.reset(self.rng)
      self.goal.reset(self.rng, obs)

      neighbor_distances, _ = self.material.nearest_neighbors.kneighbors(
          self.goal.goal_position_material_frame.reshape(1, -1)
      )
      self.assertLess(neighbor_distances[0, 0], 1e-3)

  def test_goal_position_is_not_set_near_an_edge(self):
    # This is enforced implicitly - the goal is an atom position within
    # the field of view, and the simulator initiates the silicon close to
    # the center of the simulated material.

    # Reset several times to check it's not near an edge.
    for _ in range(100):
      obs = self.sim.reset(self.rng)
      self.goal.reset(self.rng, obs)

      # We look at the neighbor distances in the material frame.
      neighbor_distances, _ = self.material.nearest_neighbors.kneighbors(
          self.goal.goal_position_material_frame.reshape(1, -1)
      )
      self.assertLessEqual(
          neighbor_distances[0, -1],
          constants.CARBON_BOND_DISTANCE_ANGSTROMS + 1e-3,
      )

  @unittest.skip('The reward is now sparse. If we switch back, un-skip this.')
  def test_reward_increases_when_silicon_is_nearer_goal(self):
    obs = self.sim.reset(self.rng)
    self.goal.reset(self.rng, obs)

    # Normally goals should be on the grid, but we can fake it for this test.
    silicon_position = self.material.get_silicon_position()
    closer_goal = silicon_position + np.asarray([5.0, 5.0], dtype=np.float32)
    further_goal = silicon_position + np.asarray([-8.0, 5.0], dtype=np.float32)

    self.goal.goal_position_material_frame = closer_goal
    closer_result = self.goal.calculate_reward_and_terminal(obs)
    self.goal.goal_position_material_frame = further_goal
    further_result = self.goal.calculate_reward_and_terminal(obs)

    self.assertGreater(closer_result.reward, further_result.reward)

  def test_returns_terminal_when_silicon_is_at_goal(self):
    obs = self.sim.reset(self.rng)
    self.goal.reset(self.rng, obs)

    # Make an observation with the silicon at the goal position.
    silicon_position = graphene.get_silicon_positions(obs.grid)
    self.assertEqual(silicon_position.shape, (1, 2))
    obs = microscope_utils.MicroscopeObservation(
        # Put the silicon right in the center of the fov for convenience.
        grid=microscope_utils.AtomicGridMicroscopeFrame(
            microscope_utils.AtomicGrid(
                atom_positions=obs.grid.atom_positions - silicon_position + 0.5,
                atomic_numbers=obs.grid.atomic_numbers,
            )
        ),
        fov=microscope_utils.MicroscopeFieldOfView(
            lower_left=geometry.Point(
                self.goal.goal_position_material_frame - 10.0
            ),
            upper_right=geometry.Point(
                self.goal.goal_position_material_frame + 10.0
            ),
        ),
        controls=obs.controls,
        elapsed_time=obs.elapsed_time,
    )

    result = self.goal.calculate_reward_and_terminal(obs)

    self.assertTrue(result.is_terminal)

  def test_no_goals_within_1_angstrom(self):
    obs = self.sim.reset(self.rng)
    self.goal.goal_range_angstroms = (0.1, 1.0)

    with self.assertRaises(RuntimeError):
      self.goal.reset(self.rng, obs)

  def test_three_possible_goals_one_step_away(self):
    obs = self.sim.reset(self.rng)
    # 1.42 = 1 bond distance.
    self.goal.goal_range_angstroms = (1.40, 1.44)

    # Select a goal many times. We should see each goal at least once.
    observed_goal_positions = set()
    for _ in range(30):
      self.goal.reset(self.rng, obs)
      goal_position = self.goal.goal_position_material_frame.copy()
      observed_goal_positions.add(
          (round(goal_position[0], 5), round(goal_position[1], 5))
      )

    self.assertLen(observed_goal_positions, 3)

  def test_goal_reset_raises_error_if_no_silicon_is_found(self):
    obs = self.sim.reset(self.rng)
    obs.grid.atomic_numbers[:] = constants.CARBON

    with self.assertRaises(graphene.SiliconNotFoundError):
      self.goal.reset(self.rng, obs)

  def test_goal_calculate_raises_error_if_no_silicon_is_found(self):
    obs = self.sim.reset(self.rng)
    self.goal.reset(self.rng, obs)

    obs.grid.atomic_numbers[:] = constants.CARBON

    with self.assertRaises(graphene.SiliconNotFoundError):
      self.goal.calculate_reward_and_terminal(obs)


if __name__ == '__main__':
  absltest.main()
