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

"""Image generators for STEM."""

import dataclasses
from typing import Tuple

import numpy as np
from putting_dune import constants
from putting_dune import microscope_utils
from scipy import ndimage
from skimage import exposure
import skimage.util


@dataclasses.dataclass(frozen=True)
class ImageGenerationParameters:
  intensity_exponent: float
  gaussian_variance: float
  jitter_rate: float
  poisson_rate_multiplier: float
  salt_and_pepper_amount: float
  blur_amount: float
  contrast_gamma: float
  exponential_lambda: float
  uniform_noise_scale: float
  image_size: int = 512


def sample_image_parameters(rng: np.random.Generator, image_size: int = 512):
  return ImageGenerationParameters(
      intensity_exponent=rng.uniform(1.4, 2.0),
      gaussian_variance=rng.uniform(0.0, 5e-3),
      jitter_rate=rng.uniform(0.0, 5.0),
      poisson_rate_multiplier=rng.exponential(15) + 1.0,
      salt_and_pepper_amount=rng.uniform(0.0, 1e-3),
      blur_amount=rng.uniform(0.0, 1.0),
      contrast_gamma=rng.uniform(0.6, 1.0),
      exponential_lambda=rng.uniform(0.0, 0.1),
      uniform_noise_scale=rng.uniform(0.0, 0.1),
      image_size=image_size,
  )


def sample_noisy_image_parameters(
    rng: np.random.Generator, image_size: int = 512
):
  """Very noisy images hand-tuned by joshgreaves@ & jfarebro@."""
  return ImageGenerationParameters(
      intensity_exponent=rng.uniform(1.4, 2.0),
      gaussian_variance=rng.uniform(0.0, 0.3),
      jitter_rate=rng.uniform(0.0, 5.0),
      poisson_rate_multiplier=rng.exponential(15) + 1.0,
      salt_and_pepper_amount=rng.uniform(0.0, 1e-2),
      blur_amount=rng.uniform(0.0, 0.25),
      contrast_gamma=rng.uniform(0.1, 1.0),
      exponential_lambda=rng.uniform(0.0, 0.15),
      uniform_noise_scale=rng.uniform(0.0, 0.15),
      image_size=image_size,
  )


def generate_grid_mask(
    grid: microscope_utils.AtomicGrid,
    fov: microscope_utils.MicroscopeFieldOfView,
    *,
    intensity_exponent: float = 1.7,
    image_dimensions: Tuple[int, int] = (512, 512),
) -> np.ndarray:
  """Generates a semantic mask for an atomic grid."""
  width, height = image_dimensions

  # Get the x and y location of the center of each pixel.
  xs = np.linspace(
      fov.lower_left.x, fov.upper_right.x, width + 1, endpoint=True
  )
  xs = (xs[:-1] + xs[1:]) / 2

  ys = np.linspace(
      fov.lower_left.y, fov.upper_right.y, height + 1, endpoint=True
  )
  ys = (ys[:-1] + ys[1:]) / 2

  xx, yy = np.meshgrid(xs, ys)

  # We need to work with a grid in the material frame to work out distances
  # in angstroms.
  material_grid = fov.microscope_frame_to_material_frame(grid)
  mask = np.zeros(image_dimensions, dtype=np.uint8)

  for pos, atomic_number in zip(
      material_grid.atom_positions, material_grid.atomic_numbers
  ):
    # The intensity exponent effects how large the mask for the atom is.
    # Empirically, multiplying by 0.1 gives a reasonable sized mask for
    # each atom.
    radius = (atomic_number / constants.CARBON) ** intensity_exponent * 0.1
    distance = (xx - pos[0]) ** 2.0 + (yy - pos[1]) ** 2.0
    mask[distance < radius] = atomic_number

  mask = np.flipud(mask)
  return mask


def generate_clean_image(
    grid: microscope_utils.AtomicGrid,
    fov: microscope_utils.MicroscopeFieldOfView,
    *,
    intensity_exponent: float = 1.7,
    image_size: int = 512,
    buffer_size: float = 0.0,
) -> np.ndarray:
  """Generates a clean image for an atomic grid."""
  # TODO(joshgreaves): Explore adding randomness to this function.
  atomic_numbers = set(grid.atomic_numbers)

  buffer_width = int(buffer_size * image_size)
  buffered_image_size = image_size + 2 * buffer_width
  image = np.zeros((buffered_image_size, buffered_image_size), dtype=np.float64)
  for atomic_number in atomic_numbers:
    positions = grid.atom_positions[grid.atomic_numbers == atomic_number]
    intensities, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=buffered_image_size,
        range=(
            (-buffer_size, 1 + buffer_size),
            (-buffer_size, 1 + buffer_size),
        ),
        density=False,
    )

    image = image + intensities * atomic_number**intensity_exponent

  # np.histogram2d returns results transposed when compared to what we expect.
  # The documentation states:
  #   "Values in x are histogrammed along the first dimension and values in
  #    y are histogrammed along the second dimension."
  # After the transpose, the image is flipped upside-down, too.
  image = np.flipud(np.transpose(image))

  fov_width = fov.upper_right.x - fov.lower_left.x
  fov_height = fov.upper_right.y - fov.lower_left.y

  # We allow images of any size to be generated, and vary sigma accordingly.
  # For example, at 512 pixels with 20 angstrom FOV => sigma = 12
  # => sigma = 200 / FOV ~= image_size / (2 * FOV)
  # Empirically, this looks about right.
  sigma = (image_size / (2 * fov_width), image_size / (2 * fov_height))

  image = ndimage.gaussian_filter(image, sigma, mode='constant')

  image = image[
      buffer_width : buffer_width + image_size,
      buffer_width : buffer_width + image_size,
  ]

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


def apply_exponential_noise(
    image: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
  noise = rng.exponential(noise_scale, size=image.shape)
  image = image + noise
  return image / np.max(image)


def apply_uniform_noise(
    image, noise_scale: float, rng: np.random.Generator
) -> np.ndarray:
  noise = rng.uniform(0.0, noise_scale, size=image.shape)
  image = image + noise
  return image / np.max(image)


def generate_stem_image(
    grid: microscope_utils.AtomicGrid,
    fov: microscope_utils.MicroscopeFieldOfView,
    image_params: ImageGenerationParameters,
    rng: np.random.Generator,
    buffer_size: float = 0.0,
) -> np.ndarray:
  """Generates a noisy STEM image."""
  image = generate_clean_image(
      grid,
      fov,
      image_size=image_params.image_size,
      intensity_exponent=image_params.intensity_exponent,
      buffer_size=buffer_size,
  )
  image = apply_poisson_noise(image, image_params.poisson_rate_multiplier, rng)
  image = apply_jitter(image, image_params.jitter_rate, rng)
  image = apply_salt_and_pepper_noise(
      image, image_params.salt_and_pepper_amount, rng
  )
  image = apply_uniform_noise(image, image_params.uniform_noise_scale, rng)
  image = apply_exponential_noise(image, image_params.exponential_lambda, rng)
  image = apply_contrast(image, image_params.contrast_gamma)
  image = apply_gaussian_noise(image, image_params.gaussian_variance, rng)
  return image
