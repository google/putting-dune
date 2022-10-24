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

from unittest import mock

from absl.testing import absltest
import dm_env
import numpy as np
from putting_dune import putting_dune_environment


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
        self.env, putting_dune_environment.PuttingDuneEnvironment)
    self.assertIsInstance(self.env, dm_env.Environment)

  def test_environment_reset(self):
    step = self.env.reset()

    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertSequenceEqual(step.observation.shape,
                             self.env.observation_spec().shape)

  def test_environment_step(self):
    env = putting_dune_environment.PuttingDuneEnvironment()
    env.reset()
    action = np.zeros((2,), dtype=np.float32)
    step = env.step(action)

    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertSequenceEqual(step.observation.shape,
                             env.observation_spec().shape)

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
    env._is_terminal = mock.MagicMock(return_value=True)
    env.reset()

    # Mocked to be a terminal step.
    step1 = env.step(_ARBITRARY_ACTIONS[0])

    # Should trigger a reset.
    step2 = env.step(_ARBITRARY_ACTIONS[1])

    self.assertEqual(step1.step_type, dm_env.StepType.LAST)
    self.assertEqual(step2.step_type, dm_env.StepType.FIRST)


if __name__ == '__main__':
  absltest.main()
