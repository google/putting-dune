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
"""Tests for microscope_agent."""

from unittest import mock

from absl.testing import absltest
import numpy as np
from putting_dune import experiment_registry
from putting_dune import microscope_agent
from putting_dune import simulator_utils
from putting_dune import test_utils


class MicroscopeAgentTest(absltest.TestCase):

  def test_microscope_agent_resets_correctly(self):
    rng = np.random.default_rng(0)

    # Create a mock experiment.
    # This is more verbose than I would like, but it helps with the
    # type checker noticing we have mocks.
    experiment = experiment_registry.create_experiment('relative_random', rng)
    mock_agent = mock.create_autospec(experiment.agent, spec_set=True)
    mock_action_adapter = mock.create_autospec(
        experiment.action_adapter, spec_set=True
    )
    mock_feature_constructor = mock.create_autospec(
        experiment.feature_constructor, spec_set=True
    )
    mock_goal = mock.create_autospec(experiment.goal, spec_set=True)
    mock_experiment = experiment_registry.Experiment(
        agent=mock_agent,
        action_adapter=mock_action_adapter,
        feature_constructor=mock_feature_constructor,
        goal=mock_goal,
    )
    agent = microscope_agent.MicroscopeAgent(mock_experiment)

    agent.reset(
        rng,
        test_utils.create_graphene_observation_with_single_silicon_in_fov(rng),
    )

    self.assertTrue(agent._is_first_step)
    mock_feature_constructor.reset.assert_called_once()
    mock_action_adapter.reset.assert_called_once()
    mock_goal.reset.assert_called_once()

  def test_microscope_agent_steps_correctly(self):
    rng = np.random.default_rng(0)
    experiment = experiment_registry.create_experiment('relative_random', rng)
    observation = (
        test_utils.create_graphene_observation_with_single_silicon_in_fov(rng)
    )

    agent = microscope_agent.MicroscopeAgent(experiment)
    agent.reset(rng, observation)

    controls = agent.step(observation)

    self.assertIsInstance(controls, list)
    self.assertLen(controls, 1)
    self.assertIsInstance(controls[0], simulator_utils.BeamControl)


if __name__ == '__main__':
  absltest.main()
