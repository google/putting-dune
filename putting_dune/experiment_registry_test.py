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
"""Tests for experiment_registry.py."""

from unittest import mock

from absl.testing import absltest
import numpy as np
from putting_dune import experiment_registry



class ExperimentRegistryTest(absltest.TestCase):

  def test_registry_retrieves_random_test(self):
    experiment = experiment_registry.create_experiment(
        'relative_random', np.random.default_rng(0)
    )
    self.assertIsInstance(experiment, experiment_registry.Experiment)



if __name__ == '__main__':
  absltest.main()
