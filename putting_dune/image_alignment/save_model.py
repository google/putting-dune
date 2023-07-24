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

"""Save an atom detection model to GCS."""

import dataclasses
import datetime
import os
import tempfile
from typing import Optional, Tuple
import zipfile

from absl import app
from absl import logging
from clu import metrics as clu_metrics
from etils import eapp
from etils import epath
from flax import struct
from flax.training import train_state
import jax
from jax.experimental import jax2tf
import numpy as np
from orbax import checkpoint
from putting_dune.image_alignment import model as unet
import simple_parsing as sp
import tensorflow as tf


Array = jax.Array


@dataclasses.dataclass(frozen=True)
class Config(sp.helpers.FrozenSerializable):
  """Train config."""

  workdir: epath.Path
  model_name: Optional[str] = None
  save_path: epath.Path = epath.Path('gs://spr_data_bucket_public/alignment/')
  image_size: Tuple[int, ...] = (512, 512)
  overwrite: bool = True
  final_step_only: bool = True
  sequence_length: int = 5


@struct.dataclass
class TrainMetrics(clu_metrics.Collection):
  accuracy: clu_metrics.Average.from_output('accuracy')
  loss: clu_metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: TrainMetrics


def save(config: Config):
  """Saves an atom detection model as a tf saved model.

  Args:
    config: Config specifying save_path, checkpoint load path, and input shape.
  """

  chkpt_manager = checkpoint.CheckpointManager(
      config.workdir,
      checkpointers=checkpoint.PyTreeCheckpointer(),
      options=checkpoint.CheckpointManagerOptions(
          best_fn=lambda x: -x['eval/loss'] - x['eval/drift_error'],
          create=True,
      ),
      metadata=config.to_dict(),
  )

  if config.final_step_only:
    local_output_size = 3
    global_output_size = 2
  else:
    local_output_size = 3 * config.sequence_length
    global_output_size = 2 * (config.sequence_length - 1)

  model = unet.GlobalLocalUNet(
      local_output_size=local_output_size,
      global_output_size=global_output_size,
  )

  best_step = chkpt_manager.best_step()
  restored = chkpt_manager.restore(best_step)
  state = TrainState(**restored, apply_fn=model.apply, tx=None)

  def remove_value_layer(param_dict):
    if isinstance(param_dict, dict):
      if 'value' in param_dict:
        return param_dict['value']
      else:
        return {k: remove_value_layer(v) for k, v in param_dict.items()}
    else:
      return param_dict

  params = remove_value_layer(state.params)
  state = state.replace(params=params)

  @jax.jit
  def apply_model(image):
    return state.apply_fn(state.params, image)

  input_shape = (*config.image_size, config.sequence_length)
  predictor_tf = jax2tf.convert(apply_model, with_gradient=False)
  tf_module = tf.Module()
  tf_module.__call__ = tf.function(
      predictor_tf,
      autograph=False,
      input_signature=[
          tf.TensorSpec(shape=input_shape, dtype=np.float32, name='image')
      ],
  )

  if config.model_name is None:
    model_name = (
        datetime.datetime.now().strftime('%Y%m%d') + '-image-aligner.zip'
    )
    logging.warning('Using default model name: %s', model_name)
  else:
    model_name = config.model_name
    logging.info('Using model name: %s', model_name)

  copy_target = config.save_path.joinpath(model_name)
  logging.info('Will save model at: %s', copy_target)

  with tempfile.TemporaryDirectory() as tmpdir:
    os.chdir(tmpdir)
    save_path = epath.Path(tmpdir) / 'image-alignment-model'
    zip_path = epath.Path('image-alignment-model.zip')
    tf.saved_model.save(tf_module, save_path)

    zipped = zipfile.ZipFile(zip_path, 'w')
    for dirname, _, files in os.walk(save_path):
      dirname = dirname.replace(tmpdir + '/', '')
      zipped.write(dirname)
      for filename in files:
        zipped.write(os.path.join(dirname, filename))
    zipped.close()
    zip_path.copy(dst=copy_target, overwrite=config.overwrite)


if __name__ == '__main__':
  # Disable rank promotion
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  jax.config.parse_flags_with_absl()

  # Better logging
  eapp.better_logging()

  # Parse using simple parsing
  app.run(save, flags_parser=eapp.make_flags_parser(Config))
