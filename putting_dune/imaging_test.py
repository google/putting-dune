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

"""Tests for imaging.py."""

from absl.testing import absltest
import numpy as np
from putting_dune import geometry
from putting_dune import graphene
from putting_dune import imaging
from putting_dune import microscope_utils


class ImagingTest(absltest.TestCase):

  def test_sample_image_parameters_returns_parameters(self):
    rng = np.random.default_rng(0)

    image_params = imaging.sample_image_parameters(rng)

    self.assertIsInstance(image_params, imaging.ImageGenerationParameters)

  def test_sample_image_parameters_behaves_deterministically_with_seed(self):
    rng = np.random.default_rng(0)
    image_params1 = imaging.sample_image_parameters(rng)

    rng = np.random.default_rng(0)
    image_params2 = imaging.sample_image_parameters(rng)

    self.assertEqual(image_params1, image_params2)

  def test_sample_image_parameters_returns_different_values(self):
    rng = np.random.default_rng(0)

    image_params1 = imaging.sample_image_parameters(rng)
    image_params2 = imaging.sample_image_parameters(rng)

    self.assertNotEqual(image_params1, image_params2)

  def test_generate_stem_image_returns_an_image(self):
    # This has mostly been tested in a colab notebook, since we can
    # visually inspect the images there.
    rng = np.random.default_rng(0)
    material = graphene.PristineSingleDopedGraphene()
    material.reset(rng)
    silicon_position = material.get_silicon_position()

    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point(silicon_position - 10.0),
        upper_right=geometry.Point(silicon_position + 10.0),
    )
    grid = material.get_atoms_in_bounds(
        fov.lower_left,
        fov.upper_right,
    )

    image_params = imaging.sample_image_parameters(rng)
    stem_image = imaging.generate_stem_image(grid, fov, image_params, rng)

    self.assertIsInstance(stem_image, np.ndarray)
    self.assertEqual(stem_image.shape, (512, 512))
    self.assertLessEqual(stem_image.max(), 1.0)
    self.assertGreaterEqual(stem_image.min(), 0.0)

  def test_generate_grid_mask_generates_an_array(self):
    # This has mostly been tested in a colab notebook, since we can
    # visually inspect the images there.
    rng = np.random.default_rng(0)
    material = graphene.PristineSingleDopedGraphene()
    material.reset(rng)
    silicon_position = material.get_silicon_position()

    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point(silicon_position - 10.0),
        upper_right=geometry.Point(silicon_position + 10.0),
    )
    grid = material.get_atoms_in_bounds(
        fov.lower_left,
        fov.upper_right,
    )

    mask = imaging.generate_grid_mask(grid, fov)
    atomic_numbers = set(mask.reshape(-1))

    self.assertIsInstance(mask, np.ndarray)
    self.assertEqual(mask.shape, (512, 512))
    self.assertLen(atomic_numbers, 3)  # C, Si, and None.


if __name__ == '__main__':
  absltest.main()
