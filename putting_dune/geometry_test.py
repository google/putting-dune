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

"""Unit tests for geoemtry.py."""

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from putting_dune import geometry


class GeometryTest(absltest.TestCase):

  def test_get_angles_returns_correct_values(self):
    neighbors = np.asarray([[1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]])
    angles = geometry.get_angles(neighbors)

    np.testing.assert_allclose(
        angles, np.asarray([0.0, -np.pi / 2, -3 * np.pi / 4])
    )

  def test_rotate_coordinates_returns_correct_values(self):
    # (1, 0) -> (0, 1)
    # (0, -1) -> (1, 0)
    # (-1, -1) -> (1, -1)
    coordinates = np.asarray([[1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]])
    expected_coordinates = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, -1.0]])

    rotated_coordinates = geometry.rotate_coordinates(coordinates, np.pi / 2)

    np.testing.assert_allclose(
        rotated_coordinates, expected_coordinates, atol=1e-9
    )

  def test_jnp_rotate_coordinates_returns_correct_values(self):
    # (1, 0) -> (0, 1)
    # (0, -1) -> (1, 0)
    # (-1, -1) -> (1, -1)
    coordinates = jnp.asarray([[1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]])
    expected_coordinates = jnp.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, -1.0]])

    rotated_coordinates = geometry.jnp_rotate_coordinates(
        coordinates, jnp.pi / 2
    )

    np.testing.assert_allclose(
        rotated_coordinates, expected_coordinates, atol=1e-7
    )


if __name__ == '__main__':
  absltest.main()
