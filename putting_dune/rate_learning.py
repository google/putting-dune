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
from typing import Any

from absl import app
import haiku as hk
import jax
from jax import numpy as jnp
import optax
from putting_dune import data_utils


Params = Mapping[str, Any]


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
    jax.jit,
    static_argnames=('batch_size', 'optim', 'epochs', 'apply_fn', 'init_fn'),
)
def train_model(
    train_data: Mapping[str, jnp.ndarray],
    test_data: Mapping[str, jnp.ndarray],
    apply_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    init_fn,
    key: jnp.ndarray,
    optim: optax.GradientTransformation,
    batch_size: int = 32,
    epochs: int = 10,
):
  """Trains a rate prediction model from scratch.

  Args:
    train_data: Training data. Dictionary of arrays.
    test_data: Testing data. Dictionary of arrays.
    apply_fn: Haiku model application function.
    init_fn: Haiku model initialization function.
    key: JAX prng key.
    optim: Optax optimizer.
    batch_size: Batch size to use in training.
    epochs: Number of epochs to train for.

  Returns:
    model parameters.
    tuple of training metrics.
  """
  context_dim = train_data['context'].shape[1]
  init_key, key = jax.random.split(key)
  params = init_fn(x=jnp.zeros(context_dim), rng=init_key)
  opt_state = optim.init((params))

  def do_epoch(state, key):
    params, opt_state, train_data, test_data = state
    params, opt_state, key = train_epoch(
        params, opt_state, optim, apply_fn, batch_size, key, train_data
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
      jax.random.split(key, num=epochs),
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


def bootstrap_train_models(
    train_data: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    num_models: int,
    hidden_dimensions: Sequence[int],
    num_states: int,
    *args,
):
  """Trains a set of models on a single dataset by bootstrapping.

  Args:
    train_data: Dictionary of training arrays.
    key: PRNG key.
    num_models: How many models to train.
    hidden_dimensions: tuple, arguments to pass for model size.
    num_states: int, how many outputs the model should predict.
    *args: Additional arguments for the training function (batch size, etc).

  Returns:
    List of models and training metrics, stacked along axis 0.
  """
  data_key, train_key = jax.random.split(key, 2)
  data_keys = jax.random.split(data_key, num_models)
  train_keys = jax.random.split(train_key, num_models)

  datasets = [
      data_utils.bootstrap_dataset(train_data, key) for key in data_keys
  ]
  train_datasets = [d[0] for d in datasets]
  test_datasets = [d[1] for d in datasets]
  test_set_len = min([a[0].shape[0] for a in test_datasets])
  test_datasets = [[a[:test_set_len] for a in d] for d in test_datasets]

  train_datasets = {
      k: jnp.stack([d[k] for d in train_datasets]) for k in train_data.keys()
  }
  train_datasets = {
      k: jnp.stack([d[k] for d in test_datasets]) for k in train_data.keys()
  }
  init_fn, apply_fn = get_mlp_fn(hidden_dimensions, num_states)
  batch_train = jax.vmap(train_model, in_axes=(0, 0, 0, None, None, None))
  return batch_train(
      train_datasets, test_datasets, apply_fn, init_fn, train_keys, *args
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
