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
"""Tests for data utils."""

import collections
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from putting_dune import data_utils


class DataUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)

  @parameterized.named_parameters(
      dict(
          testcase_name='small_scale_from_prior',
          num_samples=100,
          num_states=3,
          context_dim=2,
          actual_time_range=(0.0, 1.0),
          mode='prior',
      ),
      dict(
          testcase_name='small_scale_from_network',
          num_samples=100,
          num_states=3,
          context_dim=2,
          actual_time_range=(0.0, 1.0),
          mode='network',
      ),
      dict(
          testcase_name='large_scale_from_network',
          num_samples=1000,
          num_states=20,
          context_dim=64,
          actual_time_range=(0.0, 1.0),
          mode='network',
      ),
  )
  def test_random_data_generation(
      self,
      num_samples: int,
      num_states: int,
      context_dim: int,
      actual_time_range: Tuple[float, float],
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

    self.assertEqual(train_data['context'].shape, (num_samples, context_dim))
    self.assertEqual(train_data['next_state'].shape, (num_samples, 1))
    self.assertEqual(train_data['dt'].shape, (num_samples, 1))
    self.assertEqual(test_data['context'].shape, (num_samples, context_dim))
    self.assertEqual(test_data['next_state'].shape, (num_samples, 1))
    self.assertEqual(test_data['dt'].shape, (num_samples, 1))

    self.assertTrue(
        (train_data['dt'] < actual_time_range[1]).all()
        and (train_data['dt'] > actual_time_range[0]).all()
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='small_scale_with_reflection_from_prior',
          num_samples=100,
          num_states=3,
          context_dim=2,
          actual_time_range=(0.0, 1.0),
          mode='prior',
      ),
      dict(
          testcase_name='small_scale_without_reflection_from_prior',
          num_samples=100,
          num_states=6,
          context_dim=2,
          actual_time_range=(0.0, 1.0),
          mode='prior',
      ),
  )
  def test_data_augmentation(
      self,
      num_samples: int,
      num_states: int,
      context_dim: int,
      actual_time_range: Tuple[float, float],
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

    train_data = data_utils.augment_data(
        **train_data, reflect=num_states == 3, num_states=num_states
    )

    new_len = num_states * num_samples
    if num_states == 3:
      new_len = new_len * 2
    self.assertEqual(train_data['context'].shape, (new_len, context_dim))
    self.assertEqual(train_data['next_state'].shape, (new_len, 1))
    self.assertEqual(train_data['dt'].shape, (new_len, 1))

    counts = collections.Counter(np.array(train_data['next_state'].reshape(-1)))
    for i in range(1, num_states + 1):
      self.assertEqual(counts[i], counts[1])

  @parameterized.named_parameters(
      dict(
          testcase_name='aligned',
          neighbor_positions=np.array(
              [[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]]
          ),
          target_order=np.array([1, 0, 2], dtype=np.int32),
          beam_pos=np.array([1, 0]),
          target_beam_pos=np.array([-0.5, -np.sqrt(3) / 2]),
          target_angles=np.array([
              0,
              -2 * np.pi / 3,
              2 * np.pi / 3,
          ]),
      ),
      dict(
          testcase_name='singleton',
          neighbor_positions=np.array(
              [
                  [1, 0],
              ]
          ),
          target_order=np.array([0], dtype=np.int32),
          beam_pos=np.array([1, 0]),
          target_beam_pos=np.array([1, 0]),
          target_angles=np.array([0]),
      ),
  )
  def test_neighborhood_standardization(
      self,
      neighbor_positions: np.ndarray,
      target_order: np.ndarray,
      beam_pos: np.ndarray,
      target_beam_pos: np.ndarray,
      target_angles: np.ndarray,
  ):
    (
        rotated_beam_pos,
        state_order,
        angles,
    ) = data_utils.standardize_beam_and_neighbors(beam_pos, neighbor_positions)
    np.testing.assert_allclose(rotated_beam_pos, target_beam_pos, rtol=1e-6)
    np.testing.assert_allclose(state_order, target_order, rtol=1e-6)
    np.testing.assert_allclose(angles, target_angles, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
