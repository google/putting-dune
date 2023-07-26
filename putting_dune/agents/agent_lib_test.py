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

_CANONICAL_GREEDY_STEP = dm_env.TimeStep(
    step_type=dm_env.StepType.MID,
    reward=0.0,
    discount=0.99,
    observation=np.array(
        [0, 0, 1.42, 0, -0.71, 1.23, -0.71, -1.23, 1.42, 0], dtype=np.float32
    ),
)


class UniformRandomAgentTest(parameterized.TestCase):

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


class GreedyAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)

  @parameterized.named_parameters(
      dict(
          testcase_name='classic_argmax',
          argmax=np.array([1.42, 0]),
      ),
      dict(
          testcase_name='weird_argmax',
          argmax=np.random.randn(2),
      ),
  )
  def test_greedy_agent_selects_argmax(self, argmax: np.ndarray):
    agent = agent_lib.GreedyAgent(argmax=argmax)

    action = agent.step(_CANONICAL_GREEDY_STEP)
    np.testing.assert_allclose(action, argmax)

  @parameterized.named_parameters(
      dict(
          testcase_name='neighbor_0',
          argmax=np.array([1.42, 0]),
          neighbors=np.array([1.42, 0, -0.71, 1.23, -0.71, -1.23]),
          goal=np.array([1.42, 0]),
      ),
      dict(
          testcase_name='neighbor_1',
          argmax=np.array([1.42, 0]),
          neighbors=np.array([1.42, 0, -0.71, 1.23, -0.71, -1.23]),
          goal=np.array([-0.71, 1.23]),
      ),
      dict(
          testcase_name='neighbor_2',
          argmax=np.array([1.42, 0]),
          neighbors=np.array([1.42, 0, -0.71, 1.23, -0.71, -1.23]),
          goal=np.array([-0.71, -1.23]),
      ),
  )
  def test_greedy_agent_selects_rotated_argmax(
      self, argmax: np.ndarray, neighbors: np.ndarray, goal: np.ndarray
  ):
    agent = agent_lib.GreedyAgent(argmax=argmax)
    observation = np.concatenate([
        np.zeros(2),
        neighbors,
        goal,
    ])

    step = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=0.0,
        discount=0.99,
        observation=observation,
    )

    action = agent.step(step)
    np.testing.assert_allclose(action, goal, atol=0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name='l2_rate',
          neighbors=np.array([1.42, 0, -0.71, 1.23, -0.71, -1.23]),
          goal=np.array([1.42, 0]),
          low=-1.5,
          high=1.5,
          resolution=0.05,
      ),
  )
  def test_greedy_agent_finds_argmax(
      self,
      neighbors: np.ndarray,
      goal: np.ndarray,
      low: float,
      high: float,
      resolution: float,
  ):
    neighbor = np.reshape(neighbors, (3, 2))

    def transition_function(beam_position):
      return -np.linalg.norm(neighbor - beam_position[..., None, :], axis=-1)

    agent = agent_lib.GreedyAgent(
        argmax=None,
        transition_function=transition_function,
        argmax_resolution=resolution,
        low=low,
        high=high,
    )
    observation = np.concatenate([
        np.zeros(2),
        neighbors,
        goal,
    ])

    step = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=0.0,
        discount=0.99,
        observation=observation,
    )

    action = agent.step(step)
    np.testing.assert_allclose(action, neighbor[0], atol=resolution)


if __name__ == '__main__':
  absltest.main()
