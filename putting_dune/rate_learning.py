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
"""Code implementing basic KMC contextual rate learning in Haiku."""

from collections.abc import Callable, Mapping, Sequence
import functools
import os
from typing import Any, Optional

from absl import app
from etils import epath
import flax
import haiku as hk
import jax
from jax import numpy as jnp
from ml_collections import config_dict
import numpy as np
import optax
from putting_dune import constants
from putting_dune import data_utils
from putting_dune import simulator_utils
from shapely import geometry


rate_learning_defaults = config_dict.FrozenConfigDict({
    'batch_size': 64,
    'epochs': 1000,
    'synthetic_samples': 100,
    'num_models': 1,
    'bootstrap': False,
    'hidden_dimensions': (64, 64),
    'weight_decay': 1e-1,
    'learning_rate': 1e-3,
    'val_frac': 0.1,
})

Params = Mapping[str, Any]


def tree_stack(list_of_trees: Sequence[Params]) -> Params:
  return jax.tree_util.tree_map(lambda *x: jnp.stack(x, 0), *list_of_trees)


class MLP(hk.Module):
  """A simple convenience class representing a multilayer perceptron."""

  def __init__(self, hidden_dimensions: Sequence[int], nonlinearity=jax.nn.elu):
    super(MLP, self).__init__()
    self.hidden_dimensions = hidden_dimensions
    self.nonlinearity = nonlinearity

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    for dim in self.hidden_dimensions[:-1]:
      layer = hk.Linear(dim)
      x = self.nonlinearity(layer(x))

    return hk.Linear(self.hidden_dimensions[-1])(x)


def get_mlp_fn(
    hidden_dimensions: Sequence[int] = (64, 64), num_states: int = 3
):
  """Creates an MLP usable for rate learning."""

  def call_mlp(x):
    mlp = MLP([*hidden_dimensions, num_states + 1])
    # We use a softplus to force the output to lie in (0, inf).
    return jax.nn.softplus(mlp(x))

  return hk.transform(call_mlp)


def batched_loss_fn(
    params: Params,
    apply_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    next_state: jnp.ndarray,
    elapsed_time: jnp.ndarray,
    did_transition: jnp.ndarray,
    context: jnp.ndarray,
    key: jnp.ndarray,
):
  """Calculates the loss on a minibatch of transitions and return metrics.

  Args:
    params: Network parameters.
    apply_fn: Network application function.
    next_state: State the system transitioned to (index).
    elapsed_time: Elapsed time during the transition.
    did_transition: Whether or not a transition occurred.
    context: Context vector for the transition.
    key: JAX prng key.

  Returns:
    Mean loss, tuple of (predicted rates, rate loss, classification loss).
  """

  def loss_fn(params, next_state, elapsed_time, did_transition, context, key):
    predicted_rates = apply_fn(params, key, context)
    total_rate = predicted_rates[-1]
    predicted_log_rates = predicted_rates[:-1]
    predicted_probabilities = jax.nn.softmax(predicted_log_rates)

    total_rate_loss = -jax.lax.cond(
        did_transition.sum(),
        lambda: (jnp.log(1.0 - jnp.exp(-total_rate * elapsed_time))),
        lambda: (-total_rate * elapsed_time),
    )
    next_state_loss = -jnp.log(
        predicted_probabilities[next_state - 1]
    ) * did_transition.sum(())

    return (
        jnp.sum(next_state_loss) + total_rate_loss,
        predicted_probabilities * total_rate,
        total_rate_loss,
        next_state_loss,
    )

  batch_loss = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, None))
  (loss, predicted_rates, total_rate_loss, next_state_loss) = batch_loss(
      params, next_state, elapsed_time, did_transition, context, key
  )
  return jnp.mean(loss), (
      predicted_rates,
      jnp.mean(total_rate_loss),
      jnp.mean(next_state_loss),
  )


def train_epoch(
    params: Params,
    opt_state: Mapping[str, jnp.ndarray],
    optim: optax.GradientTransformation,
    apply_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    batch_size: int,
    key: jnp.ndarray,
    train_data: Mapping[str, jnp.ndarray],
):
  """Does one epoch of training, shuffling the dataset to create batches.

  Args:
    params: Rate MLP parameters.
    opt_state: Optimizer state.
    optim: Optax optimizer (static).
    apply_fn: Model apply function (static).
    batch_size: Batch size (static).
    key: JAX prng key.
    train_data: Training data (list of arrays).

  Returns:
    Updated parameters, optimizer state, prng key.
  """
  key, data_key = jax.random.split(key)
  data_size = list(train_data.values())[0].shape[0]
  indices = jax.random.shuffle(data_key, jnp.arange(data_size))
  batch_inds = [
      jax.lax.dynamic_slice_in_dim(indices, index * batch_size, batch_size)
      for index in jnp.arange(0, len(indices) // batch_size)
  ]
  batch_inds = jnp.stack(batch_inds)
  batches = {k: array[batch_inds] for k, array in train_data.items()}

  def train_step(state, batch):
    params = state[0]
    opt_state = state[1]

    grad_fn = jax.value_and_grad(batched_loss_fn, has_aux=True)
    (_, _), grad = grad_fn(
        params,
        apply_fn,
        batch['next_state'],
        batch['dt'],
        (batch['next_state'] != 0),
        batch['context'],
        key,
    )
    updates, opt_state = optim.update(grad, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return (params, opt_state), None

  (params, opt_state), _ = jax.lax.scan(
      train_step, (params, opt_state), batches
  )

  return params, opt_state, key


@functools.partial(
    jax.jit, static_argnames=('optim', 'train_args', 'apply_fn', 'init_fn')
)
def train_model(
    train_data: Mapping[str, jnp.ndarray],
    test_data: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    params: Params,
    apply_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    optim: optax.GradientTransformation,
    train_args: config_dict.FrozenConfigDict = rate_learning_defaults,
):
  """Trains a rate prediction model from scratch.

  Args:
    train_data: Training data. Dictionary of arrays.
    test_data: Testing data. Dictionary of arrays.
    key: JAX prng key.
    params: Starting model parameters from haiku's init_fn.
    apply_fn: Haiku model application function.
    optim: Optax optimizer.
    train_args: Config dictionary listing batch size and number of epochs.

  Returns:
    model parameters.
    tuple of training metrics.
  """
  opt_state = optim.init((params))

  def do_epoch(state, key):
    params, opt_state, train_data, test_data = state
    params, opt_state, key = train_epoch(
        params,
        opt_state,
        optim,
        apply_fn,
        train_args['batch_size'],
        key,
        train_data,
    )

    test_loss, (_, test_rate_loss, test_class_loss) = batched_loss_fn(
        params,
        apply_fn,
        test_data['next_state'],
        test_data['dt'],
        (test_data['next_state'] != 0),
        test_data['context'],
        key,
    )
    train_loss, (_, train_rate_loss, train_class_loss) = batched_loss_fn(
        params,
        apply_fn,
        train_data['next_state'],
        train_data['dt'],
        (train_data['next_state'] != 0),
        train_data['context'],
        key,
    )

    return (
        (params, opt_state, train_data, test_data),
        (
            train_loss,
            test_loss,
            train_rate_loss,
            test_rate_loss,
            train_class_loss,
            test_class_loss,
        ),
    )

  (
      (params, opt_state, _, _),
      (
          train_loss,
          test_loss,
          train_rate_loss,
          test_rate_loss,
          train_class_loss,
          test_class_loss,
      ),
  ) = jax.lax.scan(
      do_epoch,
      (params, opt_state, train_data, test_data),
      jax.random.split(key, num=train_args['epochs']),
  )

  (
      (params, opt_state, _, _),
      (
          train_loss,
          test_loss,
          train_rate_loss,
          test_rate_loss,
          train_class_loss,
          test_class_loss,
      ),
  ) = jax.lax.scan(
      do_epoch,
      (params, opt_state, train_data, test_data),
      jax.random.split(key, num=train_args['epochs']),
  )

  return (
      params,
      (
          train_loss,
          test_loss,
          train_rate_loss,
          test_rate_loss,
          train_class_loss,
          test_class_loss,
      ),
  )


def train_multiple_models(
    train_data: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    num_models: int,
    optim: optax.GradientTransformation,
    apply_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_fn: Callable[[jnp.ndarray, jnp.ndarray], Params],
    bootstrap: bool = True,
    test_fraction: float = 0.1,
    train_config: config_dict.FrozenConfigDict = rate_learning_defaults,
):
  """Trains a set of models on a single dataset by bootstrapping.

  Args:
    train_data: Dictionary of training arrays.
    key: PRNG key.
    num_models: How many models to train.
    optim: Optax optimizer.
    apply_fn: Haiku model application function.
    init_fn: Haiku model initialization function.
    bootstrap: whether to bootstrap or just split datasets.
    test_fraction: what fraction of data to use for eval if not bootstrapping.
    train_config: Additional training arguments (config dict).

  Returns:
    List of models and training metrics, stacked along axis 0.
  """
  data_key, train_key, init_key = jax.random.split(key, 3)
  data_keys = jax.random.split(data_key, num_models)
  train_keys = jax.random.split(train_key, num_models)

  if bootstrap:
    datasets = [
        data_utils.bootstrap_dataset(train_data, key) for key in data_keys
    ]
    train_datasets = [d[0] for d in datasets]
    test_datasets = [d[1] for d in datasets]
    test_set_len = min([a['context'].shape[0] for a in test_datasets])
    test_datasets = [
        {k: a[:test_set_len] for k, a in d.items()} for d in test_datasets
    ]
  elif 1.0 > test_fraction > 0.0:
    datasets = [
        data_utils.split_dataset(train_data, key, test_fraction)
        for key in data_keys
    ]
    train_datasets = [d[0] for d in datasets]
    test_datasets = [d[1] for d in datasets]
  else:
    assert test_fraction == 0
    train_datasets = [train_data] * num_models
    test_datasets = [train_data] * num_models

  train_datasets = {
      k: jnp.stack([d[k] for d in train_datasets]) for k in train_data.keys()
  }
  test_datasets = {
      k: jnp.stack([d[k] for d in test_datasets]) for k in train_data.keys()
  }
  init_context = train_data['context'][0]
  init_keys = jax.random.split(init_key, num_models)
  params = [init_fn(key, init_context) for key in init_keys]
  init_params = tree_stack(params)
  batch_train = jax.vmap(train_model, in_axes=(0, 0, 0, 0, None, None, None))
  return batch_train(
      train_datasets,
      test_datasets,
      train_keys,
      init_params,
      apply_fn,
      optim,
      train_config,
  )


class LearnedTransitionRatePredictor:
  """Class wrapping rate learning and prediction."""

  def __init__(
      self,
      init_key: jnp.ndarray,
      num_states: int = 3,
      context_dim: int = 2,
      config: config_dict.FrozenConfigDict = rate_learning_defaults,
  ):
    self.init_fn, self.apply_fn = get_mlp_fn(
        config.hidden_dimensions, num_states
    )
    self.rng, *keys = jax.random.split(init_key, config.num_models + 1)
    params = [self.init_fn(x=jnp.zeros(context_dim), rng=key) for key in keys]
    self.params = tree_stack(params)
    self.params = self.init_fn(x=jnp.zeros(context_dim), rng=init_key)
    self.context_dim = context_dim
    self.num_states = num_states

    self.config = config

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def batch_apply(params, x, rng):
      return self.apply_fn(params, rng, x)

    self.batch_apply = batch_apply

    @jax.jit
    def apply_single_model(model_index, params, x, rng):
      params = jax.tree_util.tree_map(lambda x: x[model_index], params)
      return self.apply_fn(params, rng, x)

    self.apply_single_model = jax.jit(apply_single_model)

  def apply_model(
      self,
      x: np.ndarray,
      key: jnp.ndarray,
      model_index: Optional[int] = None,
  ) -> np.ndarray:
    if model_index is None:
      return self.batch_apply(self.params, x, key).mean(0)
    else:
      return self.apply_single_model(model_index, self.params, x, key)

  def train(
      self,
      train_data: Mapping[str, jnp.ndarray],
      key: jnp.ndarray,
      bootstrap=True,
  ):
    """Trains one or more rate prediction models from a dataset.

    Args:
      train_data: Dictionary of training arrays.
      key: PRNG key.
      bootstrap: Whether to bootstrap training data for each model.

    Returns:
      Training metrics.
    """
    self.rng, train_key = jax.random.split(key, 2)
    optim = optax.adamw(
        self.config.learning_rate, weight_decay=self.config.weight_decay
    )
    self.params, train_metrics = train_multiple_models(
        train_data,
        train_key,
        num_models=self.config.num_models,
        optim=optim,
        init_fn=self.init_fn,
        apply_fn=self.apply_fn,
        bootstrap=bootstrap,
        test_fraction=self.config.val_frac,
        train_config=self.config,
    )

    return train_metrics

  def save(self, save_path: str, step: int) -> None:
    fn = os.path.join(save_path, str(step) + '.ckpt')
    path = epath.Path(fn)
    path.mkdir(parents=True, exist_ok=True)
    with path.open(fn, 'wb') as f:
      f.write(flax.serialization.to_bytes(self.params))

  def load(self, load_path: str, step: int = 0) -> None:
    path = epath.Path(load_path)
    with path.open(os.path.join(load_path, str(step) + '.ckpt'), 'rb') as f:
      self.params = flax.serialization.from_bytes(self.params, f.read())

  def predict(
      self,
      grid: simulator_utils.AtomicGrid,
      beam_pos: geometry.Point,
      current_position: np.ndarray,
      neighbor_indices: np.ndarray,
      model_index: Optional[int] = None,
  ) -> np.ndarray:
    """Computes rate constants for transitioning a Si atom.

    Args:
      grid: Atomic grid state.
      beam_pos: 2-dimensional beam position in [0, 1] coordinate frame.
      current_position: 2-dimensional position of current silicon atom.
      neighbor_indices: Indices of the atoms on the grid to calculate rates for.
      model_index: which model to apply, if the predictor has an ensemble.
        Defaults to averaging all of them.

    Returns:
      a 3-dimensional array of rate constants for transitioning to the 3
        nearest neighbors.
    """
    # Convert the beam_pos into a numpy array for convenience
    beam_pos = np.asarray([[beam_pos.x, beam_pos.y]])  # Shape = [1, -1]
    neighbor_positions = grid.atom_positions[neighbor_indices, :]
    neighbor_positions = neighbor_positions - current_position
    beam_pos = beam_pos - current_position
    beam_pos = beam_pos / constants.CARBON_BOND_DISTANCE_ANGSTROMS
    new_beam_pos, neighbor_order, _ = data_utils.standardize_beam_and_neighbors(
        beam_pos,
        neighbor_positions,
    )
    self.rng, key = jax.random.split(self.rng)

    rates = self.apply_model(key, new_beam_pos, model_index=model_index)

    # neighbor_order is an array mapping the canonical order of neighbors
    # (increasing angle from (1, 0)) to the order they were given in.
    # We can invert it with argsort to return the rates to the correct order.
    rates = rates[np.argsort(neighbor_order)]
    return rates


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
