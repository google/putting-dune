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
"""Tests for putting_dune_environment."""

from unittest import mock

from absl.testing import absltest
import dm_env
import numpy as np
from putting_dune import graphene
from putting_dune import putting_dune_environment
from putting_dune import simulator


# These actions are for the DeltaPositionActionAdapter.
_ARBITRARY_ACTIONS = [
    np.array([0.04, -0.06], dtype=np.float32),
    np.array([0.08, 0.01], dtype=np.float32),
    np.array([-0.07, 0.06], dtype=np.float32),
    np.array([-0.10, 0.07], dtype=np.float32),
    np.array([0.03, -0.08], dtype=np.float32),
    np.array([0.00, 0.07], dtype=np.float32),
    np.array([-0.03, 0.10], dtype=np.float32),
    np.array([0.07, -0.01], dtype=np.float32),
    np.array([0.08, 0.04], dtype=np.float32),
    np.array([0.09, 0.06], dtype=np.float32),
]


class PuttingDuneEnvironmentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = putting_dune_environment.PuttingDuneEnvironment()
    self.env.seed(0)

  def test_environment_is_initialized_correctly(self):
    self.assertIsInstance(
        self.env, putting_dune_environment.PuttingDuneEnvironment
    )
    self.assertIsInstance(self.env, dm_env.Environment)

  def test_environment_reset(self):
    step = self.env.reset()

    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertSequenceEqual(
        step.observation.shape, self.env.observation_spec().shape
    )

  def test_environment_step(self):
    env = putting_dune_environment.PuttingDuneEnvironment()
    env.reset()
    action = np.zeros((2,), dtype=np.float32)
    step = env.step(action)

    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertSequenceEqual(
        step.observation.shape, env.observation_spec().shape
    )

  def test_environment_render(self):
    self.env.reset()
    image = self.env.render()
    # image shape should be [width, height, 3].
    self.assertLen(image.shape, 3)
    self.assertEqual(image.shape[2], 3)

  def test_environment_behaves_deterministically_with_same_seed(self):
    self.env.seed(123)
    self.env.reset()

    trajectory1 = []
    for action in _ARBITRARY_ACTIONS:
      trajectory1.append(self.env.step(action))

    self.env.seed(123)
    self.env.reset()

    trajectory2 = []
    for action in _ARBITRARY_ACTIONS:
      trajectory2.append(self.env.step(action))

    self.assertEqual(len(trajectory1), len(trajectory2))
    for step1, step2 in zip(trajectory1, trajectory2):
      self.assertTrue((step1.observation == step2.observation).all())

  def test_environment_obeys_reset_semantics_on_creation(self):
    env = putting_dune_environment.PuttingDuneEnvironment()

    step = env.step(_ARBITRARY_ACTIONS[0])

    self.assertEqual(step.step_type, dm_env.StepType.FIRST)

  def test_environment_obeys_reset_semantics_after_last_step(self):
    env = putting_dune_environment.PuttingDuneEnvironment()
    env.goal.caluclate_reward_and_terminal = mock.MagicMock(
        return_value=putting_dune_environment.GoalReturn(
            reward=0.0, is_terminal=True, is_truncated=False
        )
    )
    env.reset()

    # Mocked to be a terminal step.
    step1 = env.step(_ARBITRARY_ACTIONS[0])

    # Should trigger a reset.
    step2 = env.step(_ARBITRARY_ACTIONS[1])

    self.assertEqual(step1.step_type, dm_env.StepType.LAST)
    self.assertEqual(step2.step_type, dm_env.StepType.FIRST)

  def test_environment_truncates_correctly(self):
    env = putting_dune_environment.PuttingDuneEnvironment()
    env.goal.caluclate_reward_and_terminal = mock.MagicMock(
        return_value=putting_dune_environment.GoalReturn(
            reward=0.0, is_terminal=False, is_truncated=True
        )
    )
    env.reset()

    step = env.step(_ARBITRARY_ACTIONS[0])

    self.assertEqual(step.step_type, dm_env.StepType.LAST)
    self.assertGreater(step.discount, 0.0)


class SingleSiliconGoalReachingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)
    self.goal = putting_dune_environment.SingleSiliconGoalReaching()
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
      self.goal.reset(self.rng, obs, self.sim)

      neighbor_distances, _ = self.material.nearest_neighbors.kneighbors(
          self.goal.goal_position_material_frame.reshape(1, -1)
      )
      self.assertLess(neighbor_distances[0, 0], 1e-3)

  def test_goal_position_is_not_set_near_an_edge(self):
    # Reset several times to check it's not near an edge.
    for _ in range(100):
      obs = self.sim.reset()
      self.goal.reset(self.rng, obs, self.sim)

      neighbor_distances, _ = self.material.nearest_neighbors.kneighbors(
          self.goal.goal_position_material_frame.reshape(1, -1)
      )
      self.assertLessEqual(
          neighbor_distances[0, -1],
          graphene.CARBON_BOND_DISTANCE_ANGSTROMS + 1e-3,
      )

  def test_reward_increases_when_silicon_is_nearer_goal(self):
    obs = self.sim.reset()
    self.goal.reset(self.rng, obs, self.sim)

    # Normally goals should be on the grid, but we can fake it for this test.
    silicon_position = self.material.get_silicon_position()
    closer_goal = silicon_position + np.asarray([5.0, 5.0], dtype=np.float32)
    further_goal = silicon_position + np.asarray([-8.0, 5.0], dtype=np.float32)

    self.goal.goal_position_material_frame = closer_goal
    closer_result = self.goal.caluclate_reward_and_terminal(obs, self.sim)
    self.goal.goal_position_material_frame = further_goal
    further_result = self.goal.caluclate_reward_and_terminal(obs, self.sim)

    self.assertGreater(closer_result.reward, further_result.reward)

  def test_returns_terminal_when_silicon_is_at_goal(self):
    obs = self.sim.reset()
    self.goal.reset(self.rng, obs, self.sim)
    self.material.get_silicon_position = mock.MagicMock(
        return_value=self.goal.goal_position_material_frame
    )

    result = self.goal.caluclate_reward_and_terminal(obs, self.sim)

    self.assertTrue(result.is_terminal)

  def test_truncation_is_applied_at_graphene_edge(self):
    obs = self.sim.reset()
    self.goal.reset(self.rng, obs, self.sim)

    # Manually position the silicon at the further position from the
    # center, which is guaranteed to be a edge (a corner, specifically).
    atom_distances = np.linalg.norm(self.material.atom_positions, axis=1)
    furthest_idx = np.argmax(atom_distances)
    self.material.atomic_numbers[:] = graphene.CARBON
    self.material.atomic_numbers[furthest_idx] = graphene.SILICON

    result = self.goal.caluclate_reward_and_terminal(obs, self.sim)

    self.assertTrue(result.is_truncated)


if __name__ == '__main__':
  absltest.main()
