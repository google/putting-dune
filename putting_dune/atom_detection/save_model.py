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
from putting_dune.atom_detection import model as unet
import simple_parsing as sp
import tensorflow as tf


Array = jax.Array


@dataclasses.dataclass(frozen=True)
class Config(sp.helpers.FrozenSerializable):
  """Train config."""

  workdir: epath.Path
  save_path: epath.Path
  model_name: Optional[str] = None
  input_shape: Tuple[int, ...] = (256, 256, 1)
  overwrite: bool = False


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
          best_fn=lambda metrics: metrics['eval/accuracy'],
          create=True,
      ),
      metadata=config.to_dict(),
  )

  model = unet.UNet()
  best_step = chkpt_manager.best_step()
  restored = chkpt_manager.restore(best_step)
  state = TrainState(**restored, apply_fn=model.apply, tx=None)  # pytype: disable=wrong-arg-types  # dataclass_transform

  @jax.jit
  def apply_model(image):
    predictions = state.apply_fn(state.params, image)
    return predictions

  predictor_tf = jax2tf.convert(apply_model, with_gradient=False)
  tf_module = tf.Module()
  tf_module.__call__ = tf.function(
      predictor_tf,
      autograph=False,
      input_signature=[
          tf.TensorSpec(
              shape=config.input_shape, dtype=np.float32, name='image'
          )
      ],
  )

  if config.model_name is None:
    model_name = (
        datetime.datetime.now().strftime('%Y%m%d') + '-atom-detector.zip'
    )
    logging.warning('Using default model name: %s', model_name)
  else:
    model_name = config.model_name
    logging.info('Using model name: %s', model_name)

  copy_target = config.save_path.joinpath(model_name)
  logging.info('Will save model at: %s', copy_target)

  with tempfile.TemporaryDirectory() as tmpdir:
    save_path = epath.Path(tmpdir) / 'packaged'
    zip_path = epath.Path(tmpdir) / 'packaged.zip'
    tf.saved_model.save(tf_module, save_path)

    zipped = zipfile.ZipFile(zip_path, 'w')
    for dirname, _, files in os.walk(save_path):
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
