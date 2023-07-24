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
from typing import Any, Callable, Dict, Optional, Tuple, TypedDict

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
from jax_md import partition
from jax_md import space
import numpy as np
from numpy import typing as npt
import optax
from orbax import checkpoint
from putting_dune.graph_alignment import dataset as graph_alignment_dataset
from putting_dune.graph_alignment import model as graph_model
import simple_parsing as sp
import tensorflow as tf

NDArray = npt.NDArray
Array = jax.Array


class GraphExample(TypedDict):
  positions: NDArray[Any]
  features: NDArray[Any]
  global_labels: NDArray[Any]
  local_labels: NDArray[Any]


def sanitize_state_keys(
    state: train_state.TrainState,
    replacements: Tuple[Tuple[str, str], ...] = (
        ('/', 'dummy_slash'),
        ('~', 'dummy_tilde'),
    ),
) -> train_state.TrainState:
  """Replaces values in all of the key names in a Haiku-style training state.

  Args:
    state: Training state to sanitize (or unsanitize).
    replacements: Tuples of (to_replace, replacement) strings.

  Returns:
    A copy of `state` with strings replaced.
  """
  for left, right in replacements:
    clean_params = {k.replace(left, right): v for k, v in state.params.items()}
    state = state.replace(params=clean_params)
    clean_opt_state_tuple = []
    for i in range(len(state.opt_state)):
      if not hasattr(state.opt_state[i], '_fields'):
        clean_opt_state_tuple.append(state.opt_state[i])
        continue
      clean_opt_fields = {}
      for field_name in state.opt_state[i]._fields:  # pytype: disable=attribute-error
        field = getattr(state.opt_state[i], field_name)
        if hasattr(field, 'items'):
          field = {k.replace(left, right): v for k, v in field.items()}
        clean_opt_fields[field_name] = field
      clean_opt_state = state.opt_state[i]._replace(**clean_opt_fields)  # pytype: disable=attribute-error
      clean_opt_state_tuple.append(clean_opt_state)
    state = state.replace(opt_state=tuple(clean_opt_state_tuple))
  return state


@dataclasses.dataclass(frozen=True)
class Config(sp.helpers.FrozenSerializable):
  """Train config."""

  workdir: epath.Path
  data_dir: epath.Path

  seed: Optional[int] = None

  train_steps_per_epoch: int = 1_000_000
  eval_steps_per_epoch: int = 100_000

  learning_rate: float = 1e-4
  batch_size: int = 128
  epochs: int = 100
  neighbor_capacity: Optional[int] = None
  box_size: float = 300.0
  cell_list_capacity: Optional[int] = None
  r_cutoff: float = 1.42 * 3


@struct.dataclass
class Metrics(clu_metrics.Collection):
  total_loss: clu_metrics.Average.from_output('total_loss')
  global_loss: clu_metrics.Average.from_output('global_loss')
  local_loss: clu_metrics.Average.from_output('local_loss')
  local_error: clu_metrics.Average.from_output('local_error')
  global_error: clu_metrics.Average.from_output('global_error')


class TrainState(train_state.TrainState):
  metrics: Metrics


@functools.partial(
    jax.jit,
    static_argnames=(
        'neighbor_list_capacity',
        'loss_start_timestep',
        'neighbor_update_fn',
        'cell_list_capacity',
        'apply_fn',
        'do_grad_step',
    ),
)
def train_step(
    state: TrainState,
    batch: GraphExample,
    apply_fn,
    neighbor_update_fn: Callable[
        [Array, partition.NeighborList], partition.NeighborList
    ],
    local_loss_weight: float = 1.0,
    loss_start_timestep: int = 1,
    neighbor_list_capacity: int = 4096,
    cell_list_capacity: Optional[int] = None,
    do_grad_step: bool = True,
) -> TrainState:
  """Performs a single training step on `batch`.

  Args:
    state: The current TrainState.
    batch: Batch of data to train.
    apply_fn: Haiku or Flax network application function
    neighbor_update_fn: Function to update the neighbor list.
    local_loss_weight: Weight for local loss (as opposed to global).
    loss_start_timestep: Timestep in series to start loss estimation at.
    neighbor_list_capacity: Maximum number of edges to allow for.
    cell_list_capacity: Capacity of cell lists used by neighbor construction.
    do_grad_step: Whether or not to take a training step. Set to `False` to
      evaluate the model.

  Returns:
    Updated TrainState after gradient step.
  """
  batch = {k: v for k, v in batch.items() if not k.startswith('_')}

  def alignment_grad_fn(
      params,
      positions,
      features,
      local_labels,
      global_labels,
  ):
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def train_alignment_fn(params, positions, features):
      neighbor = graph_model.create_sparse_neighbor_list(
          neighbor_list_capacity,
          positions,
          neighbor_update_fn,
          cell_list_capacity,
      )
      neighbor = neighbor.update(positions)
      return apply_fn(params, positions, neighbor, nodes=features)

    total_num_points = positions.shape[-2] * positions.shape[-3]
    positions = positions.reshape(
        *positions.shape[:-3], total_num_points, positions.shape[-1]
    )
    features = features.reshape(
        *features.shape[:-3], total_num_points, features.shape[-1]
    )
    global_preds, local_preds = train_alignment_fn(params, positions, features)
    global_loss = jnp.square(global_labels - global_preds).sum(-1)
    local_loss = jnp.square(local_labels - local_preds).sum(-1)

    # Calculate the local loss only for non-padding points.
    padding_mask = features[..., 0].reshape(local_loss.shape)

    # don't put anything like sum(padding_mask) in the denominator, sometimes
    # graphs are all padding!
    local_loss = (local_loss * padding_mask).mean(-1)

    # Average over time steps
    local_loss = local_loss.mean(0)[loss_start_timestep:].mean()
    global_loss = global_loss.mean(0)[loss_start_timestep:].mean()
    loss = global_loss + local_loss_weight * local_loss
    local_error = jnp.linalg.norm(local_labels - local_preds, axis=-1).mean()
    global_error = jnp.linalg.norm(global_labels - global_preds, axis=-1).mean()

    return loss, {
        'global_loss': global_loss,
        'local_loss': local_loss,
        'total_loss': loss,
        'local_error': local_error,
        'global_error': global_error,
    }

  if do_grad_step:
    alignment_grad_fn = jax.grad(alignment_grad_fn, has_aux=True)
    grads, infos = alignment_grad_fn(state.params, **batch)
    metrics = state.metrics.merge(
        state.metrics.single_from_model_output(**infos)
    )
    state = state.apply_gradients(grads=grads, metrics=metrics)
  else:
    _, infos = alignment_grad_fn(state.params, **batch)
    metrics = state.metrics.merge(
        state.metrics.single_from_model_output(**infos)
    )
    state = state.replace(metrics=metrics)
  return state


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
  seed = np.random.SeedSequence(config.seed)
  seed, dataset_seed = tuple(seed.generate_state(n_words=2))
  key = jax.random.PRNGKey(seed)

  chkpt_manager = checkpoint.CheckpointManager(
      config.workdir,
      checkpointers=checkpoint.PyTreeCheckpointer(),
      options=checkpoint.CheckpointManagerOptions(
          best_fn=lambda metrics: metrics['eval/total_loss'],
          create=True,
      ),
      metadata=config.to_dict(),
  )

  dataset_config = graph_alignment_dataset.GraphAlignmentDatasetConfig(
      data_dir=config.data_dir.as_posix(),
  )

  train_ds, test_ds = graph_alignment_dataset.make_dataset(
      config=dataset_config, seed=dataset_seed
  )
  shard_batch = functools.partial(
      jax.device_put, device=make_batch_sharding(train_ds.element_spec)
  )
  train_iter = map(shard_batch, train_ds.as_numpy_iterator())
  test_iter = map(shard_batch, test_ds.as_numpy_iterator())

  displacement, _ = space.periodic(
      jnp.ones(
          3,
      )
      * config.box_size
  )
  dummy_batch = next(train_ds.as_numpy_iterator())
  dummy_input = {k: v[0] for k, v in dummy_batch.items()}

  neighbor_fn, init_fn, apply_fn = graph_model.graph_network_neighbor_list(
      displacement,
      jnp.array(config.box_size),
      r_cutoff=config.r_cutoff,
      dr_threshold=0.0,
      disable_cell_list=not bool(config.cell_list_capacity),
      partition_format=partition.Sparse,
      n_recurrences=3,
      capacity_multiplier=1.25,
      sequence_length=dummy_input['positions'].shape[0],
  )
  # It appears that the type annotations in Jax MD are incorrect, so using
  # a disable here is necessary.
  neighbor_update_fn = neighbor_fn.update  #  pytype: disable=attribute-error
  neighbor_allocate_fn = (
      neighbor_fn.allocate  #  pytype: disable=attribute-error
  )
  if config.neighbor_capacity is None:
    neighbor_capacity = neighbor_allocate_fn(
        dummy_input['positions'].reshape((-1, 3))
    ).max_occupancy
  else:
    neighbor_capacity = config.neighbor_capacity

  learning_rate = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.learning_rate,
      warmup_steps=(config.train_steps_per_epoch // config.batch_size) // 5,
      decay_steps=config.epochs
      * (config.train_steps_per_epoch // config.batch_size)
      - ((config.train_steps_per_epoch // config.batch_size) // 5),
  )
  optim = optax.adamw(learning_rate=learning_rate)

  static_neighbor_list = graph_model.create_sparse_neighbor_list(
      neighbor_capacity,
      dummy_input['positions'],
      neighbor_update_fn,
      config.cell_list_capacity,
  )
  params = init_fn(
      key,
      dummy_input['positions'],
      static_neighbor_list,
      dummy_input['features'],
  )
  state = TrainState.create(
      apply_fn=None,
      params=params,
      tx=optim,
      metrics=Metrics.empty(),
  )

  progress = periodic_actions.ReportProgress(
      num_train_steps=config.epochs, writer=writer
  )
  hooks = [progress]

  if latest_step := chkpt_manager.latest_step():
    with progress.timed('restore-checkpoint'):
      restored = chkpt_manager.restore(latest_step, items=state)
      restored = sanitize_state_keys(
          restored, (('dummy_slash', '/'), ('dummy_tilde', '~'))
      )
      state = jax.tree_util.tree_map(
          lambda o, r: o if r is None else r,
          state,
          restored,
      )

  global train_step
  train_step = functools.partial(
      train_step,
      neighbor_update_fn=neighbor_update_fn,
      apply_fn=apply_fn,
      neighbor_list_capacity=neighbor_capacity,
      cell_list_capacity=config.cell_list_capacity,
  )
  eval_step = functools.partial(train_step, do_grad_step=False)

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
          state.replace(metrics=Metrics.empty()),
      )
      train_metrics = state.metrics

      eval_metrics = functools.reduce(
          eval_step,
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

      metrics = prefix_dict_keys(train_metrics.compute(), 'train/')
      metrics |= prefix_dict_keys(eval_metrics.compute(), 'eval/')
      metrics = jax.tree_util.tree_map(lambda x: x.item(), metrics)

      clean_state = sanitize_state_keys(
          state, (('/', 'dummy_slash'), ('~', 'dummy_tilde'))
      )

      logging.info(metrics)
      with progress.timed('save-checkpoint'):
        chkpt_manager.save(
            step=epoch,
            items=clean_state,
            metrics=metrics,
        )
      metric_writers.write_values(
          writer,
          step=epoch,
          metrics=metrics,
      )

      for hook in hooks:
        hook(epoch)

  clean_state = sanitize_state_keys(
      state, (('/', 'dummy_slash'), ('~', 'dummy_tilde'))
  )
  chkpt_manager.save(step=config.epochs, items=clean_state)
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
