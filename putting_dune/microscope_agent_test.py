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

"""Tests for microscope_agent."""

import datetime as dt
import typing
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
import pandas as pd
from putting_dune import constants
from putting_dune import io as pdio
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
        test_utils.create_single_silicon_observation(rng),
    )

    self.assertTrue(agent._is_first_step)
    mock_feature_constructor.reset.assert_called_once()
    mock_action_adapter.reset.assert_called_once()
    mock_goal.reset.assert_called_once()

  def test_microscope_agent_steps_correctly(self):
    rng = np.random.default_rng(0)
    experiment = registry.create_microscope_experiment('relative_random')
    observation = test_utils.create_single_silicon_observation(rng)

    agent = microscope_agent.MicroscopeAgent(rng, experiment)
    agent.reset(rng, observation)

    controls = agent.step(observation)

    self.assertIsInstance(controls, list)
    self.assertLen(controls, 1)
    self.assertIsInstance(controls[0], microscope_utils.BeamControl)

  def test_microscope_agent_rescans_if_no_silicon_found(self):
    rng = np.random.default_rng(0)
    experiment = registry.create_microscope_experiment('relative_random')
    observation = test_utils.create_single_silicon_observation(rng)

    agent = microscope_agent.MicroscopeAgent(rng, experiment)
    agent.reset(rng, observation)

    observation.grid.atomic_numbers[:] = constants.CARBON
    controls = agent.step(observation)

    self.assertIsInstance(controls, list)
    self.assertLen(controls, 1)
    self.assertEqual(controls[0].dwell_time, dt.timedelta(seconds=0))

  def test_microscope_agent_logs_data(self):
    rng = np.random.default_rng(0)
    experiment = registry.create_microscope_experiment('relative_random')

    path = epath.Path(self.create_tempdir().full_path)
    agent = microscope_agent.MicroscopeAgent(rng, experiment)
    with microscope_agent.MicroscopeAgentLogger(agent, logdir=path) as agent:
      agent.reset(rng, test_utils.create_single_silicon_observation(rng))
      for _ in range(2):
        agent.step(test_utils.create_single_silicon_observation(rng))
      agent.reset(rng, test_utils.create_single_silicon_observation(rng))
      for _ in range(3):
        agent.step(test_utils.create_single_silicon_observation(rng))

    self.assertTrue((path / 'steps.csv').exists())
    steps_df = pd.read_csv((path / 'steps.csv').as_posix())
    self.assertEqual(
        steps_df.shape,
        (5, len(typing.get_type_hints(microscope_agent.StepRecord).keys())),
        f'{steps_df!r}',
    )

    self.assertTrue((path / 'episodes.csv').exists())
    episodes_df = pd.read_csv((path / 'episodes.csv').as_posix())
    self.assertEqual(
        episodes_df.shape,
        (2, len(typing.get_type_hints(microscope_agent.EpisodeRecord).keys())),
        f'{episodes_df!r}',
    )

    trajectory_lengths = [2, 3]
    self.assertTrue((path / 'trajectories.tfrecords').exists())
    for trajectory, trajectory_length in zip(
        pdio.read_records(
            path / 'trajectories.tfrecords',
            microscope_utils.Trajectory,
        ),
        trajectory_lengths,
    ):
      self.assertLen(trajectory.observations, trajectory_length)


if __name__ == '__main__':
  absltest.main()
