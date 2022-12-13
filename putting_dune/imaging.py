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
"""Image generators for STEM."""

import dataclasses

import numpy as np
from putting_dune import microscope_utils
from scipy import ndimage
from skimage import exposure
import skimage.util


@dataclasses.dataclass(frozen=True)
class ImageGenerationParameters:
  gaussian_variance: float
  jitter_rate: float
  poisson_rate_multiplier: float
  salt_and_pepper_amount: float
  blur_amount: float
  contrast_gamma: float


def sample_image_parameters(rng: np.random.Generator):
  return ImageGenerationParameters(
      gaussian_variance=rng.uniform(0.0, 5e-3),
      jitter_rate=rng.uniform(0.0, 5.0),
      poisson_rate_multiplier=rng.exponential(2) + 0.5,
      salt_and_pepper_amount=rng.uniform(0.0, 1e-3),
      blur_amount=rng.uniform(0.0, 1.0),
      contrast_gamma=rng.uniform(0.6, 1.0),
  )


def generate_clean_image(
    grid: microscope_utils.AtomicGrid,
    fov: microscope_utils.MicroscopeFieldOfView,
) -> np.ndarray:
  """Generates a clean image for an atomic grid."""
  # TODO(joshgreaves): Explore adding randomness to this function.
  atomic_numbers = set(grid.atomic_numbers)

  image = np.zeros((512, 512), dtype=np.float64)
  for atomic_number in atomic_numbers:
    positions = grid.atom_positions[grid.atomic_numbers == atomic_number]
    intensities, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=512,
        range=((0, 1), (0, 1)),
        density=False,
    )

    image = image + intensities * atomic_number**2

  fov_width = fov.upper_right.x - fov.lower_left.x
  fov_height = fov.upper_right.y - fov.lower_left.y

  # We assume we will only ever generate 512x512 pixel images.
  # 512 pixels with 20 angstrom FOV => sigma = 10
  # => sigma = 200 / FOV
  # Empirically, this looks about right.
  sigma = (225.0 / fov_width, 225.0 / fov_height)

  image = ndimage.gaussian_filter(image, sigma, mode='constant')

  # Normalize
  image = image / np.max(image)

  return image


def apply_gaussian_noise(
    image: np.ndarray, variance: float, rng: np.random.Generator
) -> np.ndarray:
  noisy_image = skimage.util.random_noise(
      image, mode='gaussian', var=variance, seed=rng
  )
  return noisy_image


def apply_jitter(
    image: np.ndarray, jitter_rate: float, rng: np.random.Generator
) -> np.ndarray:
  num_rows, _ = image.shape
  roll_per_row = rng.poisson(jitter_rate, size=num_rows)
  image = np.stack(
      [np.roll(image[i, :], roll_per_row[i]) for i in range(num_rows)]
  )
  return image


def apply_poisson_noise(
    image: np.ndarray, poisson_rate_multiplier: float, rng: np.random.Generator
) -> np.ndarray:
  image = rng.poisson(image * poisson_rate_multiplier)
  return image / np.max(image)


def apply_salt_and_pepper_noise(
    image: np.ndarray, amount: float, rng: np.random.Generator
) -> np.ndarray:
  return skimage.util.random_noise(image, mode='s&p', amount=amount, seed=rng)


def apply_blur(image: np.ndarray, amount: float) -> np.ndarray:
  image = ndimage.gaussian_filter(image, amount)
  return image / np.max(image)


def apply_contrast(image: np.ndarray, gamma: float) -> np.ndarray:
  return exposure.adjust_gamma(image, gamma)


def generate_stem_image(
    grid: microscope_utils.AtomicGrid,
    fov: microscope_utils.MicroscopeFieldOfView,
    image_params: ImageGenerationParameters,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generates a noisy STEM image."""
  image = generate_clean_image(grid, fov)
  image = apply_gaussian_noise(image, image_params.gaussian_variance, rng)
  image = apply_jitter(image, image_params.jitter_rate, rng)
  image = apply_poisson_noise(image, image_params.poisson_rate_multiplier, rng)
  image = apply_salt_and_pepper_noise(
      image, image_params.salt_and_pepper_amount, rng
  )
  image = apply_blur(image, image_params.blur_amount)
  image = apply_contrast(image, image_params.contrast_gamma)
  return image
