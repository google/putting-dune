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
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint
from putting_dune.image_alignment import dataset as image_alignment_dataset
from putting_dune.image_alignment import model as unet
import simple_parsing as sp
import tensorflow as tf


Array = jax.Array


class Example(TypedDict):
  images: Array
  mask: Array
  drift: Array


@dataclasses.dataclass(frozen=True)
class Config(sp.helpers.FrozenSerializable):
  """Train config."""

  workdir: epath.Path
  data_dir: epath.Path

  seed: Optional[int] = None

  train_steps_per_epoch: int = 50_000
  eval_steps_per_epoch: int = 50_000

  drift_loss_weight: float = 1.0
  learning_rate: float = 1e-3
  batch_size: int = 32
  epochs: int = 1000
  final_step_only: bool = True
  image_size: int = 512


@struct.dataclass
class Metrics(clu_metrics.Collection):
  accuracy: clu_metrics.Average.from_output('accuracy')
  loss: clu_metrics.Average.from_output('loss')
  drift_loss: clu_metrics.Average.from_output('drift_loss')
  drift_error: clu_metrics.Average.from_output('drift_error')
  final_step_accuracy: clu_metrics.Average.from_output('final_step_accuracy')
  final_step_loss: clu_metrics.Average.from_output('final_step_loss')
  final_step_drift_loss: clu_metrics.Average.from_output(
      'final_step_drift_loss'
  )
  final_step_drift_error: clu_metrics.Average.from_output(
      'final_step_drift_error'
  )


class TrainState(train_state.TrainState):
  metrics: Metrics


def maybe_unsqeeze(tensor):
  if not isinstance(tensor, jnp.ndarray):
    tensor = jnp.asarray(tensor)
  if not tensor.shape:
    tensor = jnp.expand_dims(tensor, axis=0)
  return tensor


tree_unsqueeze = functools.partial(jax.tree_util.tree_map, maybe_unsqeeze)


def train_step(
    state: TrainState,
    batch: Example,
    drift_loss_weight: float,
    final_only: bool,
) -> TrainState:
  """Train function."""

  @functools.partial(jax.grad, has_aux=True)
  def grad_fn(params) -> Tuple[float, Dict[str, Array]]:
    batch_apply_fn = jax.vmap(state.apply_fn, in_axes=(None, 0))
    logits, pred_drift = batch_apply_fn(params, batch['images'])
    logits = logits.reshape(*batch['mask'].shape)
    losses = optax.softmax_cross_entropy(logits, batch['mask'])
    final_loss = jnp.mean(losses[..., -1])
    loss = jnp.mean(losses)

    pred_drift = pred_drift.reshape(*batch['drift'].shape)
    drift_losses = jnp.square(batch['drift'] - pred_drift).sum(-1)
    drift_loss = jnp.mean(drift_losses)
    final_drift_loss = jnp.mean(drift_losses[..., -1])
    accuracies = jnp.argmax(logits, axis=-1) == jnp.argmax(batch['mask'])
    accuracy = jnp.mean(accuracies)
    final_accuracy = jnp.mean(accuracies[..., -1])
    drift_errors = jnp.linalg.norm(batch['drift'] - pred_drift, axis=-1)
    drift_error = jnp.mean(drift_errors)
    final_drift_error = jnp.mean(drift_errors[..., -1])

    if final_only:
      loss = final_loss
      drift_loss = final_drift_loss
      accuracy = final_accuracy
      drift_error = final_drift_error

    total_loss = loss + drift_loss_weight * drift_loss

    metric_dict = {
        'accuracy': accuracy,
        'loss': loss,
        'drift_loss': drift_loss,
        'drift_error': drift_error,
        'final_step_accuracy': final_accuracy,
        'final_step_loss': final_loss,
        'final_step_drift_loss': final_drift_loss,
        'final_step_drift_error': final_drift_error,
    }
    return total_loss, metric_dict

  grads, infos = grad_fn(state.params)
  infos = state.metrics.single_from_model_output(**infos)
  metrics = state.metrics.merge(infos)
  state = state.apply_gradients(grads=grads, metrics=metrics)

  return state


def eval_step(
    state: TrainState,
    batch: Example,
    final_only: bool,
) -> TrainState:
  """Evaluation step."""
  batch_apply_fn = jax.vmap(state.apply_fn, in_axes=(None, 0))
  logits, pred_drift = batch_apply_fn(state.params, batch['images'])
  logits = logits.reshape(*batch['mask'].shape)
  losses = optax.softmax_cross_entropy(logits, batch['mask'])
  final_loss = jnp.mean(losses[..., -1])
  loss = jnp.mean(losses)

  pred_drift = pred_drift.reshape(*batch['drift'].shape)
  drift_losses = jnp.square(batch['drift'] - pred_drift).sum(-1)
  drift_loss = jnp.mean(drift_losses)
  final_drift_loss = jnp.mean(drift_losses[..., -1])

  accuracies = jnp.argmax(logits, axis=-1) == jnp.argmax(batch['mask'])
  accuracy = jnp.mean(accuracies)
  final_accuracy = jnp.mean(accuracies[..., -1])
  drift_errors = jnp.linalg.norm(batch['drift'] - pred_drift, axis=-1)
  drift_error = jnp.mean(drift_errors)
  final_drift_error = jnp.mean(drift_errors[..., -1])

  if final_only:
    loss = final_loss
    drift_loss = final_drift_loss
    accuracy = final_accuracy
    drift_error = final_drift_error

  metric_dict = {
      'accuracy': accuracy,
      'loss': loss,
      'drift_loss': drift_loss,
      'drift_error': drift_error,
      'final_step_accuracy': final_accuracy,
      'final_step_loss': final_loss,
      'final_step_drift_loss': final_drift_loss,
      'final_step_drift_error': final_drift_error,
  }

  infos = state.metrics.single_from_model_output(**metric_dict)
  metrics = state.metrics.merge(infos)
  state = state.replace(metrics=metrics)

  return state


def make_batch_sharding(
    element_spec: Dict[str, tf.TensorShape],
    state: TrainState,
) -> Tuple[
    jax.sharding.Mesh,
    Dict[str, jax.sharding.NamedSharding],
    Dict[str, jax.sharding.PartitionSpec],
    Dict[str, jax.sharding.NamedSharding],
]:
  """Match batch sharding spec."""

  def _pspec_with_data_leading_axis(
      spec: tf.TensorShape,
  ) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(
        'data', *(None for _ in range(len(spec.shape) - 1))
    )

  def _pspec_with_none_leading_axis(
      leaf: Any,
  ) -> jax.sharding.PartitionSpec:
    if not isinstance(leaf, jax.Array):
      leaf = jnp.asarray(leaf)
    if not leaf.shape:
      return jax.sharding.PartitionSpec()
    else:
      return jax.sharding.PartitionSpec(*(None for _ in range(len(leaf.shape))))

  data_pspec = jax.tree_util.tree_map(
      _pspec_with_data_leading_axis,
      element_spec,
  )
  state_pspec = jax.tree_util.tree_map(
      _pspec_with_none_leading_axis,
      state,
  )

  devices = mesh_utils.create_device_mesh(
      (jax.device_count(),), contiguous_submeshes=True
  )
  mesh = jax.sharding.Mesh(devices, axis_names='data')
  return (
      mesh,
      jax.tree_util.tree_map(
          functools.partial(jax.sharding.NamedSharding, mesh),
          data_pspec,
      ),
      data_pspec,
      jax.tree_util.tree_map(
          functools.partial(jax.sharding.NamedSharding, mesh),
          state_pspec,
      ),
  )


def prefix_dict_keys(mapping: Dict[str, Any], prefix: str) -> Dict[str, Any]:
  return {f'{prefix}{key}': value for key, value in mapping.items()}


def train(config: Config) -> TrainState:
  """Train function."""
  writer = metric_writers.create_default_writer(
      write_to_xm_measurements=False,
      write_to_datatable=True,
      just_logging=jax.process_index() > 0,
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

  dataset_config = image_alignment_dataset.ImageAlignmentDatasetConfig(
      data_dir=config.data_dir.as_posix(),
      batch_size=config.batch_size,
      image_size=config.image_size,
      final_only=config.final_step_only,
  )

  seed = np.random.SeedSequence(config.seed)
  seed, dataset_seed = tuple(seed.generate_state(n_words=2))
  key = jax.random.PRNGKey(seed)
  train_ds, test_ds = image_alignment_dataset.make_dataset(
      config=dataset_config, seed=dataset_seed
  )

  dummy_input = next(train_ds.as_numpy_iterator())

  sequence_length = dummy_input['images'][0].shape[-1]

  if config.final_step_only:
    local_output_size = 3
    global_output_size = 2
  else:
    local_output_size = 3 * sequence_length
    global_output_size = 2 * (sequence_length - 1)

  model = unet.GlobalLocalUNet(
      local_output_size=local_output_size,
      global_output_size=global_output_size,
  )
  learning_rate = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.learning_rate,
      warmup_steps=(config.train_steps_per_epoch // config.batch_size) // 5,
      decay_steps=config.epochs
      * (config.train_steps_per_epoch // config.batch_size)
      - ((config.train_steps_per_epoch // config.batch_size) // 5),
  )
  optim = optax.adamw(learning_rate=learning_rate)
  params = model.init(key, dummy_input['images'][0])
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=optim,
      metrics=Metrics.empty(),
  )

  progress = periodic_actions.ReportProgress(
      num_train_steps=config.epochs, writer=writer
  )
  hooks = [progress]

  (
      mesh,
      batch_sharding,
      batch_pspec,
      state_sharding,
  ) = make_batch_sharding(train_ds.element_spec, state)

  if latest_step := chkpt_manager.latest_step():
    with progress.timed('restore-checkpoint'):
      restore_args = jax.tree_util.tree_map(
          lambda sharding: checkpoint.ArrayRestoreArgs(  # pylint: disable=g-long-lambda
              sharding=sharding,
          ),
          state_sharding,
      )
      state = chkpt_manager.restore(
          latest_step,
          items=state,
          restore_kwargs={'restore_args': restore_args},
      )

  concretized_train_step = functools.partial(
      train_step,
      drift_loss_weight=config.drift_loss_weight,
      final_only=config.final_step_only,
  )
  concretized_eval_step = functools.partial(
      eval_step, final_only=config.final_step_only
  )

  jitted_train_step = jax.jit(
      concretized_train_step,
      donate_argnums=(0,),
      static_argnames='drift_loss_weight,final_only',
      in_shardings=(state_sharding, batch_sharding),
      out_shardings=state_sharding,
  )

  jitted_eval_step = jax.jit(
      concretized_eval_step,
      static_argnames='final_only',
      in_shardings=(state_sharding, batch_sharding),
      out_shardings=state_sharding,
  )

  def globalize_tensor(array, spec):
    return multihost_utils.host_local_array_to_global_array(array, mesh, spec)

  def globalize_batch(batch):
    return jax.tree_util.tree_map(globalize_tensor, batch, batch_pspec)

  train_iter = map(globalize_batch, train_ds.as_numpy_iterator())
  test_iter = map(globalize_batch, test_ds.as_numpy_iterator())
  with jax.spmd_mode('allow_all'):
    with metric_writers.ensure_flushes(writer):
      for epoch in etqdm.tqdm(
          range(chkpt_manager.latest_step() or 0, config.epochs),
          desc='Epoch',
      ):
        state = functools.reduce(
            jitted_train_step,
            etqdm.tqdm(
                itertools.islice(
                    train_iter,
                    config.train_steps_per_epoch // config.batch_size,
                ),
                desc='Train',
                leave=False,
                unit_scale=config.batch_size,
                unit='example',
                total=config.train_steps_per_epoch // config.batch_size,
            ),
            state.replace(metrics=Metrics.empty()),
        )

        eval_metrics = functools.reduce(
            jitted_eval_step,
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
            state.replace(metrics=Metrics.empty()),
        ).metrics

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
  jax.config.update('jax_numpy_rank_promotion', 'allow')
  jax.config.update('jax_spmd_mode', 'allow_all')
  jax.config.parse_flags_with_absl()

  # Better logging
  eapp.better_logging()

  # Parse using simple parsing
  app.run(train, flags_parser=eapp.make_flags_parser(Config))
