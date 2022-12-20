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
"""Tests for agent."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import numpy as np
from putting_dune.agents import agent_lib

_ARBITRARY_STEP = dm_env.TimeStep(
    step_type=dm_env.StepType.MID,
    reward=1.0,
    discount=0.99,
    observation=np.zeros((4,), dtype=np.float32),
)


class AgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)

  @parameterized.named_parameters(
      dict(testcase_name='tight_bounds', low=-0.1, high=0.1),
      dict(testcase_name='loose_bounds', low=-1.0, high=1.0),
  )
  def test_uniform_agent_selects_actions_within_bounds(
      self, low: float, high: float
  ):
    agent = agent_lib.UniformRandomAgent(self._rng, low, high, (10,))

    action = agent.step(_ARBITRARY_STEP)

    self.assertTrue(((action >= low) & (action <= high)).all())

  @parameterized.named_parameters(
      dict(testcase_name='single_dimension', shape=(8,)),
      dict(testcase_name='multidimensional', shape=(4, 3, 2)),
  )
  def test_uniform_agent_selects_actions_with_correct_shape(
      self, shape: Tuple[int, ...]
  ):
    agent = agent_lib.UniformRandomAgent(self._rng, 0.0, 1.0, shape)

    action = agent.step(_ARBITRARY_STEP)

    self.assertEqual(action.shape, shape)


if __name__ == '__main__':
  absltest.main()
