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

"""Atom detection dataset."""
import dataclasses
from typing import Tuple

from clu import preprocess_spec
import grain.tensorflow as grain
import jax
import numpy as np
from putting_dune import constants
import tensorflow as tf


FlatFeatures = preprocess_spec.FlatFeatures


@dataclasses.dataclass(frozen=True)
class ResizeImage(grain.MapTransform):
  """Decodes the images and resizes."""

  image_size: int

  def map(self, features: FlatFeatures) -> FlatFeatures:
    images = features['images']
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    images = tf.transpose(images, [1, 2, 0, 3])
    images = tf.reshape(images, (*images.shape[:2], -1))
    if images.shape[0] != self.image_size or images.shape[1] != self.image_size:
      features['images'] = tf.image.resize(
          images,
          (self.image_size, self.image_size),
          'nearest',
      )
    else:
      features['images'] = images

    mask = features['mask']
    if mask.shape[1] != self.image_size or mask.shape[2] != self.image_size:
      mask = tf.image.resize(
          mask,
          (self.image_size, self.image_size),
          'nearest',
      )
    mask = tf.transpose(mask, [1, 2, 0, 3])
    features['mask'] = mask

    return features


@dataclasses.dataclass(frozen=True)
class RemapAndOneHotMask(grain.MapTransform):
  """One hot the mask, gather is super slow on TPUs?"""

  num_classes: int

  def map(self, features: FlatFeatures) -> FlatFeatures:
    mask = features['mask']

    # Remap class labels for Carbon and Silicon from atomic number.
    mask = tf.where(tf.equal(mask, constants.CARBON), np.uint8(1), mask)
    mask = tf.where(tf.equal(mask, constants.SILICON), np.uint8(2), mask)
    mask = tf.one_hot(mask, self.num_classes, axis=-1)
    # one-hot always adds a new dimension, squeeze out the previous output dim
    mask = tf.squeeze(mask, axis=-2)

    features['mask'] = mask
    return features


@dataclasses.dataclass(frozen=True)
class SelectFinalTimestep(grain.MapTransform):
  """If training only on the final timestep, select it now to save VRAM."""

  def map(self, features: FlatFeatures) -> FlatFeatures:
    features['mask'] = features['mask'][-1:]
    features['drift'] = features['drift'][-1:]

    return features


@dataclasses.dataclass
class ImageAlignmentDatasetConfig:
  data_dir: str
  batch_size: int = 128
  image_size: int = 512
  start_index: int = 1
  final_only: bool = True


def make_dataset(
    config: ImageAlignmentDatasetConfig, *, seed: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Make atom detection dataset."""
  options = tf.data.Options()
  options.experimental_optimization.map_parallelization = True
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  if config.final_only:
    transformations = [SelectFinalTimestep()]
  else:
    transformations = []

  transformations += [
      ResizeImage(image_size=config.image_size),
      RemapAndOneHotMask(num_classes=3),
  ]
  train_loader = grain.load_from_tfds(
      name='image_alignment',
      split='train',
      data_dir=config.data_dir,
      shuffle=True,
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=config.batch_size,
      tf_data_options=options,
  )
  test_loader = grain.load_from_tfds(
      name='image_alignment',
      split='test',
      data_dir=config.data_dir,
      shuffle=True,
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=config.batch_size,
      tf_data_options=options,
  )

  start_index = (
      config.start_index // jax.process_count()
  ) * jax.process_count()
  start_index = start_index + jax.process_index()

  train_ds = train_loader.as_dataset(start_index=start_index)
  test_ds = test_loader.as_dataset(start_index=start_index)

  return train_ds, test_ds
