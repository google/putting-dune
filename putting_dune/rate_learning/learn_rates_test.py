# Copyright 2023 The Putting Dune Authors.
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

"""Tests for rate learning."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from ml_collections import config_dict
import numpy as np
import optax
from putting_dune.rate_learning import data_utils
from putting_dune.rate_learning import learn_rates


debug_defaults = config_dict.FrozenConfigDict({
    'batch_size': 32,
    'epochs': 1,
    'num_models': 2,
    'bootstrap': True,
    'hidden_dimensions': (32, 32),
    'weight_decay': 1e-1,
    'learning_rate': 1e-3,
    'val_frac': 0.0,
    'use_voltage': True,
    'use_current': True,
    'dwell_time_in_context': False,
    'class_loss_weight': 1.0,
    'rate_loss_weight': 1.0,
    'augment_data': True,
    'batchnorm': True,
    'dropout_rate': 0.0,
})


class RateLearningTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)

  @parameterized.named_parameters(
      dict(
          testcase_name='small_scale_from_prior',
          num_samples=100,
          num_states=3,
          context_dim=2,
          actual_time_range=(0.0, 5.0),
          mode='prior',
          mlp_dims=(
              16,
              16,
          ),
      ),
      dict(
          testcase_name='small_scale_from_network',
          num_samples=100,
          num_states=3,
          context_dim=2,
          actual_time_range=(0.0, 5.0),
          mode='network',
          mlp_dims=(
              16,
              16,
          ),
      ),
  )
  def test_rate_learner(
      self,
      num_samples: int,
      num_states: int,
      context_dim: int,
      actual_time_range: Tuple[float, float],
      mlp_dims: Tuple[int, ...],
      mode: str,
  ):
    train_data, test_data = data_utils.generate_synthetic_data(
        num_data=num_samples,
        data_seed=None,
        num_states=num_states,
        context_dim=context_dim,
        actual_time_range=actual_time_range,
        mode=mode,
    )

    init_fn, apply_fn = learn_rates.get_mlp_fn(mlp_dims, num_states)
    optim = optax.adamw(1e-3, weight_decay=0.1)
    key = jax.random.PRNGKey(42)
    init_params, init_state = init_fn(
        key,
        train_data['context'][0:1],
    )

    opt_state = optim.init(init_params)
    params, state, opt_state, metrics = learn_rates.train_model(
        train_data,
        test_data,
        key,
        init_params,
        init_state,
        opt_state,
        apply_fn,
        optim,
        train_args=debug_defaults,
    )

    self.assertIsNotNone(params)
    self.assertIsNotNone(state)
    self.assertIsNotNone(metrics)


class RatePredictorTest(parameterized.TestCase):

  # TODO(maxschwarzer): Fix flaky test!
  @absltest.skip('Flaky test, needs to be fixed.')
  @parameterized.named_parameters(
      dict(
          testcase_name='small_scale',
          num_samples=100,
          num_states=3,
          context_dim=2,
          actual_time_range=(0.0, 5.0),
          test_inputs=[
              np.array([1, 0]),
              np.array([-0.5, 0.866]),
              np.array([-0.5, -0.866]),
          ],
          test_argmaxes=[0, 1, 2],
          mode='prior',
      ),
      dict(
          testcase_name='large_scale',
          num_samples=1000,
          num_states=3,
          context_dim=2,
          actual_time_range=(0.0, 5.0),
          test_inputs=[
              np.array([1, 0]),
              np.array([-0.5, 0.866]),
              np.array([-0.5, -0.866]),
          ],
          test_argmaxes=[0, 1, 2],
          mode='prior',
      ),
  )
  def test_rate_learner(
      self,
      num_samples: int,
      num_states: int,
      context_dim: int,
      actual_time_range: Tuple[float, float],
      test_inputs: Tuple[np.ndarray, ...],
      test_argmaxes: Tuple[int, ...],
      mode: str,
  ):
    train_data, _ = data_utils.generate_synthetic_data(
        num_data=num_samples,
        data_seed=None,
        num_states=num_states,
        context_dim=context_dim,
        actual_time_range=actual_time_range,
        mode=mode,
    )

    key = jax.random.PRNGKey(0)
    learner = learn_rates.LearnedTransitionRatePredictor(
        init_key=key,
    )

    learner.train(train_data, key)
    self.assertIsNotNone(learner.params)

    for x, y in zip(test_inputs, test_argmaxes):
      rates = learner.apply_model(x, key)
      self.assertEqual(np.argmax(rates), y)


if __name__ == '__main__':
  absltest.main()
