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

"""Tests for eval_lib."""

import datetime as dt
from unittest import mock

from absl.testing import absltest
import dm_env
from putting_dune import agent_lib
from putting_dune import eval_lib
from putting_dune import putting_dune_environment


_TERMINAL_STEP = dm_env.termination(0.0, None)


def _add_mock_simulator_to_env(env: mock.MagicMock) -> None:
  env.sim = mock.MagicMock()
  env.sim.elapsed_time = dt.timedelta(seconds=10)


class EvalLibTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._mock_agent = mock.create_autospec(
        agent_lib.Agent, spec_set=True)
    # spec_set = False since we need to add the sim attribute after
    # creation.
    self._mock_environment = mock.create_autospec(
        putting_dune_environment.PuttingDuneEnvironment, spec_set=False)
    _add_mock_simulator_to_env(self._mock_environment)

  def test_eval_lib_runs_for_correct_number_of_seeds(self):
    self._mock_environment.step.return_value = _TERMINAL_STEP

    num_seeds = 10
    eval_suite = eval_lib.EvalSuite(tuple(range(num_seeds)))

    eval_lib.evaluate(self._mock_agent, self._mock_environment, eval_suite)

    # Since step always returns terminal, it should be called the same
    # number of times as number of seeds we are evaluating on.
    self.assertEqual(self._mock_environment.step.call_count, num_seeds)


if __name__ == '__main__':
  absltest.main()
