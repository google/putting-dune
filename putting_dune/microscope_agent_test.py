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
from putting_dune import microscope_agent
from putting_dune import microscope_utils
from putting_dune import test_utils
from putting_dune.experiments import experiments
from putting_dune.experiments import registry


class MicroscopeAgentTest(absltest.TestCase):

  def test_microscope_agent_resets_correctly(self):
    rng = np.random.default_rng(0)

    # Create a mock experiment.
    # This is more verbose than I would like, but it helps with the
    # type checker noticing we have mocks.
    experiment = registry.create_microscope_experiment('relative_random')
    adapters_and_goal = experiment.get_adapters_and_goal()
    agent = experiment.get_agent(rng, adapters_and_goal)

    mock_agent = mock.create_autospec(agent, spec_set=True)
    mock_action_adapter = mock.create_autospec(
        adapters_and_goal.action_adapter, spec_set=True
    )
    mock_feature_constructor = mock.create_autospec(
        adapters_and_goal.feature_constructor, spec_set=True
    )
    mock_goal = mock.create_autospec(adapters_and_goal.goal, spec_set=True)

    def get_mock_adapters_and_goal() -> experiments.AdaptersAndGoal:
      return experiments.AdaptersAndGoal(
          action_adapter=mock_action_adapter,
          feature_constructor=mock_feature_constructor,
          goal=mock_goal,
      )

    mock_experiment = experiments.MicroscopeExperiment(
        get_agent=lambda _rng, _adapters: mock_agent,
        get_adapters_and_goal=get_mock_adapters_and_goal,
    )
    agent = microscope_agent.MicroscopeAgent(rng, mock_experiment)

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
    experiment = registry.create_microscope_experiment('relative_random')
    observation = (
        test_utils.create_graphene_observation_with_single_silicon_in_fov(rng)
    )

    agent = microscope_agent.MicroscopeAgent(rng, experiment)
    agent.reset(rng, observation)

    controls = agent.step(observation)

    self.assertIsInstance(controls, list)
    self.assertLen(controls, 1)
    self.assertIsInstance(controls[0], microscope_utils.BeamControl)


if __name__ == '__main__':
  absltest.main()
