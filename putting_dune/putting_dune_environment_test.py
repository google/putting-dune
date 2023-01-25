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

"""Tests for putting_dune_environment."""

import typing
from unittest import mock

from absl.testing import absltest
import dm_env
from dm_env import specs
import numpy as np
from putting_dune import goals
from putting_dune import test_utils


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
    self.env = test_utils.create_simple_environment()
    self.env.seed(0)

  def test_environment_is_initialized_correctly(self):
    self.assertIsInstance(self.env, dm_env.Environment)

  def test_environment_reset(self):
    step = self.env.reset()

    observation = typing.cast(np.ndarray, step.observation)
    observation_spec = typing.cast(specs.Array, self.env.observation_spec())

    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertSequenceEqual(observation.shape, observation_spec.shape)

  def test_environment_step(self):
    self.env.reset()
    action = np.zeros((2,), dtype=np.float32)
    step = self.env.step(action)

    observation = typing.cast(np.ndarray, step.observation)
    observation_spec = typing.cast(specs.Array, self.env.observation_spec())

    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertSequenceEqual(observation.shape, observation_spec.shape)

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
    step = self.env.step(_ARBITRARY_ACTIONS[0])

    self.assertEqual(step.step_type, dm_env.StepType.FIRST)

  def test_environment_obeys_reset_semantics_after_last_step(self):
    self.env.goal.calculate_reward_and_terminal = mock.MagicMock(
        return_value=goals.GoalReturn(
            reward=0.0, is_terminal=True, is_truncated=False
        )
    )
    self.env.reset()

    # Mocked to be a terminal step.
    step1 = self.env.step(_ARBITRARY_ACTIONS[0])

    # Should trigger a reset.
    step2 = self.env.step(_ARBITRARY_ACTIONS[1])

    self.assertEqual(step1.step_type, dm_env.StepType.LAST)
    self.assertEqual(step2.step_type, dm_env.StepType.FIRST)

  def test_environment_truncates_correctly(self):
    self.env.goal.calculate_reward_and_terminal = mock.MagicMock(
        return_value=goals.GoalReturn(
            reward=0.0, is_terminal=False, is_truncated=True
        )
    )
    self.env.reset()

    step = self.env.step(_ARBITRARY_ACTIONS[0])

    self.assertEqual(step.step_type, dm_env.StepType.LAST)
    self.assertGreater(step.discount, 0.0)


if __name__ == '__main__':
  absltest.main()
