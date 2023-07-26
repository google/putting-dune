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

"""Atom detection dataset."""
import dataclasses
from typing import Tuple

import grain.tensorflow as grain
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class GraphAlignmentDatasetConfig:
  data_dir: str = dataclasses.field()
  batch_size: int = 16
  start_index: int = 1


def make_dataset(
    config: GraphAlignmentDatasetConfig, *, seed: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Make atom detection dataset."""
  options = tf.data.Options()
  options.experimental_optimization.map_parallelization = True
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  train_loader = grain.load_from_tfds(
      name='graph_alignment',
      split='train',
      data_dir=config.data_dir,
      shuffle=True,
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=(),
      batch_size=config.batch_size,
      tf_data_options=options,
  )
  test_loader = grain.load_from_tfds(
      name='graph_alignment',
      split='test',
      data_dir=config.data_dir,
      shuffle=True,
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=(),
      batch_size=config.batch_size,
      tf_data_options=options,
  )

  train_ds = train_loader.as_dataset(start_index=config.start_index)
  test_ds = test_loader.as_dataset(start_index=config.start_index)

  return train_ds, test_ds
