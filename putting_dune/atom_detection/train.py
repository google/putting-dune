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

"""Train atom detection model."""

import dataclasses
import functools
import itertools
import logging
from typing import Any, Dict, Optional, Tuple, TypedDict

from absl import app
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
from etils import eapp
from etils import epath
from etils import etqdm
from flax import struct
from flax.training import train_state
import grain.tensorflow as grain
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint
from putting_dune.atom_detection import dataset as atom_detection_dataset
from putting_dune.atom_detection import model as unet
import simple_parsing as sp
import tensorflow as tf


Array = jax.Array


class Example(TypedDict):
  image: Array
  mask: Array


@dataclasses.dataclass(frozen=True)
class Config(sp.helpers.FrozenSerializable):
  """Train config."""

  workdir: epath.Path
  data_dir: epath.Path

  seed: Optional[int] = None

  train_steps_per_epoch: int = 500_000
  eval_steps_per_epoch: int = 50_000

  learning_rate: float = 1e-3
  batch_size: int = 128
  epochs: int = 100


@struct.dataclass
class TrainMetrics(clu_metrics.Collection):
  accuracy: clu_metrics.Average.from_output('accuracy')
  loss: clu_metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: TrainMetrics


@struct.dataclass
class EvalMetrics(clu_metrics.Collection):
  accuracy: clu_metrics.Average.from_output('accuracy')


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: TrainState,
    batch: Example,
) -> TrainState:
  """Train function."""

  @functools.partial(jax.grad, has_aux=True)
  def grad_fn(params) -> Tuple[float, Dict[str, Array]]:
    logits = jax.vmap(state.apply_fn, in_axes=(None, 0))(params, batch['image'])
    loss = optax.softmax_cross_entropy(logits, batch['mask'])
    loss = jnp.mean(loss)
    accuracy = jnp.mean(
        jnp.argmax(logits, axis=-1) == jnp.argmax(batch['mask'], axis=-1)
    )

    return loss, {'accuracy': accuracy, 'loss': loss}

  grads, infos = grad_fn(state.params)
  metrics = state.metrics.merge(state.metrics.single_from_model_output(**infos))
  state = state.apply_gradients(grads=grads, metrics=metrics)

  return state


@functools.partial(jax.jit, donate_argnums=(0,))
def eval_step(
    metrics: clu_metrics.Collection,
    batch: Example,
    *,
    state: TrainState,
) -> clu_metrics.Collection:
  """Evaluation step."""
  logits = jax.vmap(state.apply_fn, in_axes=(None, 0))(
      state.params, batch['image']
  )
  accuracy = jnp.mean(
      jnp.argmax(logits, axis=-1) == jnp.argmax(batch['mask'], axis=-1)
  )

  return metrics.merge(metrics.single_from_model_output(accuracy=accuracy))


def make_batch_sharding(
    element_spec: Dict[str, tf.TensorShape]
) -> Dict[str, jax.sharding.NamedSharding]:
  """Match batch sharding spec."""

  def _pspec_with_data_leading_axis(
      spec: tf.TensorShape,
  ) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(
        'data', *(None for _ in range(len(spec.shape) - 1))
    )

  pspec = jax.tree_util.tree_map(
      _pspec_with_data_leading_axis,
      element_spec,
  )
  devices = mesh_utils.create_device_mesh((len(jax.devices()),))
  mesh = jax.sharding.Mesh(devices, axis_names='data')
  return jax.tree_util.tree_map(
      functools.partial(jax.sharding.NamedSharding, mesh), pspec
  )


def prefix_dict_keys(mapping: Dict[str, Any], prefix: str) -> Dict[str, Any]:
  return {f'{prefix}{key}': value for key, value in mapping.items()}


def train(config: Config) -> TrainState:
  """Train function."""
  writer = metric_writers.create_default_writer(
      write_to_xm_measurements=False,
      write_to_datatable=True,
  )

  chkpt_manager = checkpoint.CheckpointManager(
      config.workdir,
      checkpointers=checkpoint.PyTreeCheckpointer(),
      options=checkpoint.CheckpointManagerOptions(
          best_fn=lambda metrics: metrics['eval/accuracy'],
          create=True,
      ),
      metadata=config.to_dict(),
  )

  seed = np.random.SeedSequence(config.seed)
  seed, dataset_seed = tuple(seed.generate_state(n_words=2))
  key = jax.random.PRNGKey(seed)

  model = unet.UNet()
  learning_rate = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.learning_rate,
      warmup_steps=(config.train_steps_per_epoch // config.batch_size) // 5,
      decay_steps=config.epochs
      * (config.train_steps_per_epoch // config.batch_size)
      - ((config.train_steps_per_epoch // config.batch_size) // 5),
  )
  optim = optax.adamw(learning_rate=learning_rate)
  params = model.init(key, jnp.zeros((256, 256, 1)))
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=optim,
      metrics=TrainMetrics.empty(),
  )

  dataset_config = atom_detection_dataset.AtomDetectionDatasetConfig(
      data_dir=config.data_dir.as_posix(),
  )

  train_ds, test_ds = atom_detection_dataset.make_dataset(
      config=dataset_config, seed=dataset_seed
  )
  shard_batch = functools.partial(
      jax.device_put, device=make_batch_sharding(train_ds.element_spec)
  )
  train_iter = map(shard_batch, train_ds.as_numpy_iterator())
  test_iter = map(shard_batch, test_ds.as_numpy_iterator())

  progress = periodic_actions.ReportProgress(
      num_train_steps=config.epochs, writer=writer
  )
  hooks = [progress]

  if latest_step := chkpt_manager.latest_step():
    with progress.timed('restore-checkpoint'):
      restored = chkpt_manager.restore(latest_step, items=state)
      state = jax.tree_util.tree_map(
          lambda o, r: o if r is None else r,
          state,
          restored,
      )

  with metric_writers.ensure_flushes(writer):
    for epoch in etqdm.tqdm(
        range(chkpt_manager.latest_step() or 0, config.epochs),
        desc='Epoch',
    ):
      state = functools.reduce(
          train_step,
          etqdm.tqdm(
              itertools.islice(
                  train_iter, config.train_steps_per_epoch // config.batch_size
              ),
              desc='Train',
              leave=False,
              unit_scale=config.batch_size,
              unit='example',
              total=config.train_steps_per_epoch // config.batch_size,
          ),
          state.replace(metrics=TrainMetrics.empty()),
      )

      eval_metrics = functools.reduce(
          functools.partial(eval_step, state=state),
          etqdm.tqdm(
              itertools.islice(
                  test_iter, config.eval_steps_per_epoch // config.batch_size
              ),
              desc='Test',
              leave=False,
              unit_scale=config.batch_size,
              unit='example',
              total=config.eval_steps_per_epoch // config.batch_size,
          ),
          EvalMetrics.empty(),
      )

      metrics = prefix_dict_keys(state.metrics.compute(), 'train/')
      metrics |= prefix_dict_keys(eval_metrics.compute(), 'eval/')
      metrics = jax.tree_util.tree_map(lambda x: x.item(), metrics)
      logging.info(metrics)

      with progress.timed('save-checkpoint'):
        chkpt_manager.save(
            step=epoch,
            items=state,
            metrics=metrics,
        )
      metric_writers.write_values(
          writer,
          step=epoch,
          metrics=metrics,
      )

      for hook in hooks:
        hook(epoch)

  chkpt_manager.save(step=config.epochs, items=state)
  return state


if __name__ == '__main__':
  # Turn on interleaved shuffle.
  grain.config.update('tf_interleaved_shuffle', True)
  # Increase the block size to 20 records for even faster reads.
  grain.config.update('tf_interleaved_shuffle_block_size', 20)

  # Disable rank promotion
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  jax.config.parse_flags_with_absl()

  # Better logging
  eapp.better_logging()

  # Parse using simple parsing
  app.run(train, flags_parser=eapp.make_flags_parser(Config))
