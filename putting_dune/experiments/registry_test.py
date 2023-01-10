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
"""Tests for experiments/registry.py."""

from unittest import mock

from absl.testing import absltest
import numpy as np
from putting_dune.agents import acme_eval_agent
from putting_dune.agents import tf_eval_agent
from putting_dune.experiments import experiments
from putting_dune.experiments import registry


# TODO(joshgreaves): Write tests for the individual creation functions.


class ExperimentRegistryTest(absltest.TestCase):

  def test_registry_retrieves_microscope_experiment(self):
    experiment = registry.create_microscope_experiment('relative_random')
    self.assertIsInstance(experiment, experiments.MicroscopeExperiment)

  def test_invalid_microscope_experiment_raises_error(self):
    with self.assertRaises(ValueError):
      registry.create_microscope_experiment('invalid_experiment')

  def test_registry_retrieves_train_experiment(self):
    experiment = registry.create_train_experiment('relative_simple_rates')
    self.assertIsInstance(experiment, experiments.TrainExperiment)

  def test_invalid_train_experiment_raises_error(self):
    with self.assertRaises(ValueError):
      registry.create_train_experiment('invalid_experiment')

  def test_registry_retrieves_eval_experiment(self):
    experiment = registry.create_eval_experiment('relative_random_simple')
    self.assertIsInstance(experiment, experiments.EvalExperiment)

  def test_invalid_eval_experiment_raises_error(self):
    with self.assertRaises(ValueError):
      registry.create_eval_experiment('invalid_experiment')



if __name__ == '__main__':
  absltest.main()
