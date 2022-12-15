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
"""Tests for feature_constructors."""

from absl.testing import absltest
import numpy as np
from putting_dune import feature_constructors
from putting_dune import goals
from putting_dune import test_utils


class FeatureConstructorsTest(absltest.TestCase):

  def test_pristine_graphene_constructor_returns_valid_features(self):
    rng = np.random.default_rng(0)
    observation = (
        test_utils.create_graphene_observation_with_single_silicon_in_fov(rng)
    )

    goal = goals.SingleSiliconGoalReaching()
    goal.reset(rng, observation)
    fc = feature_constructors.SingleSiliconPristineGraphineFeatureConstuctor()

    features = fc.get_features(observation, goal)

    self.assertIsInstance(features, np.ndarray)
    self.assertEqual(features.shape, fc.observation_spec().shape)

  def test_image_feature_constructor_returns_valid_features(self):
    rng = np.random.default_rng(0)
    observation = (
        test_utils.create_graphene_observation_with_single_silicon_in_fov(
            rng, return_image=True
        )
    )

    goal = goals.SingleSiliconGoalReaching()
    goal.reset(rng, observation)
    fc = feature_constructors.ImageFeatureConstructor()

    features = fc.get_features(observation, goal)

    self.assertIsInstance(features, dict)
    self.assertIsInstance(features['image'], np.ndarray)
    self.assertIsInstance(features['goal_delta_angstroms'], np.ndarray)
    self.assertEqual(
        features['image'].shape, fc.observation_spec()['image'].shape
    )
    self.assertEqual(
        features['goal_delta_angstroms'].shape,
        fc.observation_spec()['goal_delta_angstroms'].shape,
    )


if __name__ == '__main__':
  absltest.main()
