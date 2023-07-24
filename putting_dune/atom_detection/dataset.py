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
import numpy as np
from putting_dune import constants
import tensorflow as tf


FlatFeatures = preprocess_spec.FlatFeatures


@dataclasses.dataclass(frozen=True)
class ResizeImage(grain.MapTransform):
  """Decodes the images and extracts a random crop."""

  image_size: int

  def map(self, features: FlatFeatures) -> FlatFeatures:
    image = features['image']
    features['image'] = tf.image.resize(
        image,
        (self.image_size, self.image_size),
        'nearest',
    )

    mask = features['mask']
    features['mask'] = tf.image.resize(
        mask,
        (self.image_size, self.image_size),
        'nearest',
    )

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
class AtomDetectionDatasetConfig:
  data_dir: str
  batch_size: int = 128
  image_size: int = 256
  start_index: int = 1


def make_dataset(
    config: AtomDetectionDatasetConfig, *, seed: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Make atom detection dataset."""
  options = tf.data.Options()
  options.experimental_optimization.map_parallelization = True
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  transformations = [
      ResizeImage(image_size=config.image_size),
      RemapAndOneHotMask(num_classes=3),
  ]
  train_loader = grain.load_from_tfds(
      name='atom_detection',
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
      name='atom_detection',
      split='test',
      data_dir=config.data_dir,
      shuffle=True,
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=config.batch_size,
      tf_data_options=options,
  )

  train_ds = train_loader.as_dataset(start_index=config.start_index)
  test_ds = test_loader.as_dataset(start_index=config.start_index)

  return train_ds, test_ds
