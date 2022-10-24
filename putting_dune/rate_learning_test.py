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

"""Tests for rate learning."""

import time
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import optax
from putting_dune import data_utils
from putting_dune import rate_learning


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
          actual_time_range=(0., 5.0),
          mode='prior',
          mlp_dims=(
              16,
              16,
          )),
      dict(
          testcase_name='small_scale_from_network',
          num_samples=100,
          num_states=3,
          context_dim=2,
          actual_time_range=(0., 5.0),
          mode='network',
          mlp_dims=(
              16,
              16,
          )),
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

    init_fn, apply_fn = rate_learning.get_mlp_fn(mlp_dims, num_states)
    optim = optax.adamw(1e-3, weight_decay=0.1)

    params, _ = rate_learning.train_model(
        train_data,
        test_data,
        apply_fn,
        init_fn,
        jax.random.PRNGKey(int(time.time())),
        optim,
        batch_size=32,
        epochs=1,
    )

    self.assertIsNotNone(params)


if __name__ == '__main__':
  absltest.main()
