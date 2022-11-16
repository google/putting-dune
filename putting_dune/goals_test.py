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
"""Tests for goals."""

from unittest import mock

from absl.testing import absltest
import numpy as np
from putting_dune import constants
from putting_dune import goals
from putting_dune import graphene
from putting_dune import simulator


class SingleSiliconGoalReachingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)
    self.goal = goals.SingleSiliconGoalReaching()
    # Make a small graphene sheet to more thoroughly test what happens
    # at the edge of a graphene sheet.
    self.material = graphene.PristineSingleDopedGraphene(
        self.rng, grid_columns=10
    )
    self.sim = simulator.PuttingDuneSimulator(self.material)

  def test_goal_position_is_set_to_a_lattice_position(self):
    # Reset several times to check it's always a lattice position.
    for _ in range(10):
      obs = self.sim.reset()
      self.goal.reset(self.rng, obs)

      neighbor_distances, _ = self.material.nearest_neighbors.kneighbors(
          self.goal.goal_position_material_frame.reshape(1, -1)
      )
      self.assertLess(neighbor_distances[0, 0], 1e-3)

  def test_goal_position_is_not_set_near_an_edge(self):
    # Reset several times to check it's not near an edge.
    for _ in range(100):
      obs = self.sim.reset()
      self.goal.reset(self.rng, obs)

      neighbor_distances, _ = self.material.nearest_neighbors.kneighbors(
          self.goal.goal_position_material_frame.reshape(1, -1)
      )
      self.assertLessEqual(
          neighbor_distances[0, -1],
          constants.CARBON_BOND_DISTANCE_ANGSTROMS + 1e-3,
      )

  def test_reward_increases_when_silicon_is_nearer_goal(self):
    obs = self.sim.reset()
    self.goal.reset(self.rng, obs)

    # Normally goals should be on the grid, but we can fake it for this test.
    silicon_position = self.material.get_silicon_position()
    closer_goal = silicon_position + np.asarray([5.0, 5.0], dtype=np.float32)
    further_goal = silicon_position + np.asarray([-8.0, 5.0], dtype=np.float32)

    self.goal.goal_position_material_frame = closer_goal
    closer_result = self.goal.caluclate_reward_and_terminal(obs)
    self.goal.goal_position_material_frame = further_goal
    further_result = self.goal.caluclate_reward_and_terminal(obs)

    self.assertGreater(closer_result.reward, further_result.reward)

  def test_returns_terminal_when_silicon_is_at_goal(self):
    obs = self.sim.reset()
    self.goal.reset(self.rng, obs)
    self.material.get_silicon_position = mock.MagicMock(
        return_value=self.goal.goal_position_material_frame
    )

    result = self.goal.caluclate_reward_and_terminal(obs)

    self.assertTrue(result.is_terminal)

  def test_truncation_is_applied_at_graphene_edge(self):
    obs = self.sim.reset()
    self.goal.reset(self.rng, obs)

    # Manually position the silicon at the further position from the
    # center, which is guaranteed to be a edge (a corner, specifically).
    atom_distances = np.linalg.norm(self.material.atom_positions, axis=1)
    furthest_idx = np.argmax(atom_distances)
    self.material.atomic_numbers[:] = constants.CARBON
    self.material.atomic_numbers[furthest_idx] = constants.SILICON

    result = self.goal.caluclate_reward_and_terminal(obs)

    self.assertTrue(result.is_truncated)
