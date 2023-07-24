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

"""Code implementing basic KMC contextual rate learning in Haiku."""

from collections.abc import Callable, Mapping, Sequence
import functools
import json
import os
from typing import Any, Optional, Tuple

from absl import logging
from etils import epath
import flax
import haiku as hk
import jax
from jax import numpy as jnp
from jax.experimental import jax2tf
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
import optax
from putting_dune import constants
from putting_dune import geometry
from putting_dune import microscope_utils
from putting_dune.rate_learning import data_utils
import tensorflow as tf

rate_learning_defaults = config_dict.FrozenConfigDict({
    'batch_size': 256,
    'epochs': 500,
    'num_models': 50,
    'bootstrap': True,
    'hidden_dimensions': (256, 256),
    'weight_decay': 1e-3,
    'learning_rate': 1e-3,
    'val_frac': 0.0,
    'use_voltage': True,
    'use_current': True,
    'dwell_time_in_context': False,
    'class_loss_weight': 1.0,
    'rate_loss_weight': 1.0,
    'augment_data': True,
    'batchnorm': True,
    'dropout_rate': 0.0,
})


distillation_defaults = config_dict.FrozenConfigDict({
    'batch_size': 4096,
    'epochs': 10000,
    'batches_per_epoch': 10,
})


State = Mapping[str, Any]
Params = Mapping[str, Any]
OptaxState = Mapping[str, jnp.ndarray]
ApplyFn = Callable[
    [Params, State, jnp.ndarray, jnp.ndarray, Optional[bool]],
    Tuple[jnp.ndarray, State],
]


def tree_stack(list_of_trees: Sequence[Params]) -> Params:
  return jax.tree_util.tree_map(lambda *x: jnp.stack(x, 0), *list_of_trees)


def get_mlp_fn(
    hidden_dimensions: Sequence[int] = (64, 64),
    num_states: int = 3,
    batchnorm: bool = True,
    dropout_rate: float = 0.0,
):
  """Creates an MLP usable for rate learning."""

  def call_mlp(x, is_training=True):
    if batchnorm:
      normalization = hk.BatchNorm(True, True, 0.9)
      x = normalization(x, is_training=is_training)
    mlp = hk.nets.MLP(
        [*hidden_dimensions, num_states + 1], activation=jax.nn.swish
    )
    dropout = 0 if not is_training else dropout_rate
    # We use a softplus to force the output to lie in (0, inf).
    return jax.nn.softplus(mlp(x, dropout_rate=dropout, rng=hk.next_rng_key()))

  return hk.transform_with_state(call_mlp)


def batched_loss_fn(
    params: Params,
    network_state: State,
    apply_fn: ApplyFn,
    next_state: jnp.ndarray,
    elapsed_time: jnp.ndarray,
    did_transition: jnp.ndarray,
    context: jnp.ndarray,
    key: jnp.ndarray,
    is_training: bool = True,
    class_loss_weight: float = 1.0,
    rate_loss_weight: float = 1.0,
):
  """Calculates the loss on a minibatch of transitions and return metrics.

  Args:
    params: Network parameters.
    network_state: Network state (e.g., EMA).
    apply_fn: Network application function.
    next_state: State the system transitioned to (index).
    elapsed_time: Elapsed time during the transition.
    did_transition: Whether or not a transition occurred.
    context: Context vector for the transition.
    key: JAX prng key.
    is_training: bool, whether network should be in train mode.
    class_loss_weight: Weight for classification loss.
    rate_loss_weight: Weight for total rate loss.

  Returns:
    Mean loss, tuple of (predicted rates, rate loss, classification loss).
  """
  predicted_rates, network_state = apply_fn(
      params, network_state, key, context, is_training
  )
  predicted_total_rate = predicted_rates[:, -1]
  no_transition_prob = jnp.exp(-predicted_total_rate * elapsed_time)
  no_transition_prob = jnp.clip(no_transition_prob, a_max=1 - 1e-6)
  did_transition_logprob = jnp.log(1 - no_transition_prob)
  no_transition_logprob = -predicted_total_rate * elapsed_time
  total_rate_loss = -(
      did_transition * did_transition_logprob
      + (1 - did_transition) * no_transition_logprob
  )

  next_state_logprobs = jax.nn.log_softmax(predicted_rates[:, :-1], axis=-1)
  next_state_loss = -(
      next_state_logprobs[jnp.arange(next_state.shape[0]), next_state - 1]
      * did_transition
  )
  next_state_probs = jax.nn.softmax(predicted_rates[:, -1:], axis=-1)

  losses = (
      next_state_loss * class_loss_weight + total_rate_loss * rate_loss_weight
  )
  return (
      jnp.mean(losses),
      (
          network_state,
          next_state_probs * predicted_rates[:, -1:],
          total_rate_loss,
          next_state_loss,
      ),
  )


def train_epoch(
    params: Params,
    network_state: State,
    opt_state: OptaxState,
    optim: optax.GradientTransformation,
    apply_fn: ApplyFn,
    batch_size: int,
    key: jnp.ndarray,
    train_data: Mapping[str, jnp.ndarray],
    train_args: config_dict.FrozenConfigDict,
):
  """Does one epoch of training, shuffling the dataset to create batches.

  Args:
    params: Rate MLP parameters.
    network_state: Rate network state (e.g., BatchNorm).
    opt_state: Optimizer state.
    optim: Optax optimizer (static).
    apply_fn: Model apply function (static).
    batch_size: Batch size (static).
    key: JAX prng key.
    train_data: Training data (list of arrays).
    train_args: Config dictionary listing extra training parameters.

  Returns:
    Updated parameters, optimizer state, prng key.
  """
  key, data_key = jax.random.split(key)
  data_size = list(train_data.values())[0].shape[0]
  indices = jax.random.permutation(
      data_key, jnp.arange(data_size), independent=True
  )
  batch_inds = [
      jax.lax.dynamic_slice_in_dim(indices, index * batch_size, batch_size)
      for index in jnp.arange(0, len(indices) // batch_size)
  ]
  batch_inds = jnp.stack(batch_inds)
  batches = {k: array[batch_inds] for k, array in train_data.items()}

  def train_step(state, batch):
    params = state[0]
    network_state = state[1]
    opt_state = state[2]

    grad_fn = jax.value_and_grad(batched_loss_fn, has_aux=True)
    (_, (network_state, _, _, _)), grad = grad_fn(
        params,
        network_state,
        apply_fn,
        batch['next_state'],
        batch['dt'],
        (batch['next_state'] != 0),
        batch['context'],
        key,
        True,
        train_args.class_loss_weight,
        train_args.rate_loss_weight,
    )
    updates, opt_state = optim.update(grad, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return (params, network_state, opt_state), None

  (params, network_state, opt_state), _ = jax.lax.scan(
      train_step, (params, network_state, opt_state), batches
  )

  return params, network_state, opt_state, key


@functools.partial(
    jax.jit,
    static_argnames=(
        'optim',
        'train_args',
        'apply_fn',
    ),
)
def train_model(
    train_data: Mapping[str, jnp.ndarray],
    test_data: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    params: Params,
    network_state: State,
    opt_state: OptaxState,
    apply_fn: ApplyFn,
    optim: optax.GradientTransformation,
    train_args: config_dict.FrozenConfigDict = rate_learning_defaults,
) -> Tuple[Params, State, OptaxState, Mapping[str, jnp.ndarray]]:
  """Trains a rate prediction model from scratch.

  Args:
    train_data: Training data. Dictionary of arrays.
    test_data: Testing data. Dictionary of arrays.
    key: JAX prng key.
    params: Starting model parameters from Haiku's init_fn.
    network_state: Starting model state from Haiku's init_fn.
    opt_state: Starting optimizer state, from optim.init.
    apply_fn: Haiku model application function.
    optim: Optax optimizer.
    train_args: Config dictionary listing batch size and number of epochs.

  Returns:
    Model parameters, network state, optimizer state,
      dictionary of training metrics.
  """

  def do_epoch(state, key):
    params, network_state, opt_state, train_data, test_data = state
    params, network_state, opt_state, key = train_epoch(
        params,
        network_state,
        opt_state,
        optim,
        apply_fn,
        train_args['batch_size'],
        key,
        train_data,
        train_args,
    )

    test_loss, (_, _, test_rate_loss, test_class_loss) = batched_loss_fn(
        params,
        network_state,
        apply_fn,
        test_data['next_state'],
        test_data['dt'],
        (test_data['next_state'] != 0),
        test_data['context'],
        key,
        is_training=False,
    )
    train_loss, (_, _, train_rate_loss, train_class_loss) = batched_loss_fn(
        params,
        network_state,
        apply_fn,
        train_data['next_state'],
        train_data['dt'],
        (train_data['next_state'] != 0),
        train_data['context'],
        key,
        is_training=False,
    )
    metrics = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_rate_loss': jnp.mean(train_rate_loss),
        'train_class_loss': jnp.mean(train_class_loss),
        'test_rate_loss': jnp.mean(test_rate_loss),
        'test_class_loss': jnp.mean(test_class_loss),
    }

    return (
        (params, network_state, opt_state, train_data, test_data),
        metrics,
    )

  ((params, network_state, opt_state, _, _), metrics) = jax.lax.scan(
      do_epoch,
      (params, network_state, opt_state, train_data, test_data),
      jax.random.split(key, num=train_args['epochs']),
  )

  return params, network_state, opt_state, metrics


@functools.partial(jax.jit, static_argnames=('batch_size', 'apply_fn'))
def distill_loss(
    params: Params,
    network_state: State,
    ensemble_params: Params,
    ensemble_state: State,
    key: jnp.ndarray,
    batch_size: int,
    apply_fn: ApplyFn,
    data_mean: jnp.ndarray,
    data_scale: jnp.ndarray,
) -> Tuple[jnp.ndarray, State]:
  """A distillation loss that uses an L2 loss on random data.

  Args:
    params: Parameters to train.
    network_state: State of the network to distill.
    ensemble_params: Parameters of the ensemble of models to distill. Should be
      stacked along the first dimension.
    ensemble_state: State of the ensemble of models to distill. Should be
      stacked along the first dimension.
    key: Jax RNG key.
    batch_size: Batch size for distillation.
    apply_fn: Flax/Haiku apply function (for both student and teachers).
    data_mean: Array of same shape as data specifying per-dimension means.
    data_scale: Array of same shape as data specifying per-dimension stds.

  Returns:
    Mean squared error between the student and teachers' predictions on
    a random batch of data.
  """
  rng, data_key, eval_key = jax.random.split(key, 3)
  datapoints = (
      jax.random.normal(
          data_key, shape=(batch_size, *data_mean.shape), dtype=jnp.float32
      )
      * data_scale
      + data_mean
  )

  @functools.partial(jax.vmap, in_axes=(0, 0, None, None))
  def batch_apply(params, state, datapoints, key):
    rates, state = apply_fn(params, state, key, datapoints, False)
    # Convert rates from directional probabilities and a total rate to be
    # per-neighbor.
    rates = jax.nn.softmax(rates[..., :-1], axis=-1) * rates[..., -1:]
    return rates

  targets = batch_apply(
      ensemble_params, ensemble_state, datapoints, eval_key
  ).mean(0)

  pred_rates, network_state = apply_fn(
      params, network_state, rng, datapoints, True
  )
  pred_rates = (
      jax.nn.softmax(pred_rates[..., :-1], axis=-1) * pred_rates[..., -1:]
  )

  loss = ((pred_rates - targets) ** 2).sum(-1).mean(0)
  return loss, network_state


@functools.partial(
    jax.jit,
    static_argnames=(
        'optim',
        'batch_size',
        'apply_fn',
        'batches',
    ),
)
def distill_train_epoch(
    params: Params,
    network_state: State,
    ensemble_params: Params,
    ensemble_state: State,
    opt_state: OptaxState,
    key: jnp.ndarray,
    batches: int,
    optim: optax.GradientTransformation,
    batch_size: int,
    apply_fn: ApplyFn,
    data_mean: jnp.ndarray,
    data_scale: jnp.ndarray,
) -> Tuple[Params, State, OptaxState, jnp.ndarray, float]:
  """Does one 'epoch' of distillation training using a jax scan.

  Args:
    params: Parameters to train.
    network_state: State of the network to distill.
    ensemble_params: Parameters of the ensemble of models to distill. Should be
      stacked along the first dimension.
    ensemble_state: State of the ensemble of models to distill. Should be
      stacked along the first dimension.
    opt_state: Optimizer state.
    key: JAX PRNG Key.
    batches: How many batches to run.
    optim: Optax optimizer.
    batch_size: Size of each batch.
    apply_fn: Haiku network application function.
    data_mean: Array of same shape as data specifying per-dimension means.
    data_scale: Array of same shape as data specifying per-dimension stds.

  Returns:
    Updated tuple of parameters, optimization state, RNG key and training loss.
  """

  @functools.partial(
      jax.jit,
      donate_argnums=(0,),
  )
  def distill_train_step(
      state,
      key,
  ):
    params, network_state, opt_state = state
    grad_fn = jax.value_and_grad(distill_loss, has_aux=True)
    (loss, network_state), grad = grad_fn(
        params,
        network_state,
        ensemble_params,
        ensemble_state,
        key,
        batch_size,
        apply_fn,
        data_mean,
        data_scale,
    )
    updates, opt_state = optim.update(grad, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return (params, network_state, opt_state), loss

  keys = jax.random.split(key, batches)
  (params, network_state, opt_state), losses = jax.lax.scan(
      distill_train_step, (params, network_state, opt_state), keys
  )

  return params, network_state, opt_state, keys[-1], jnp.mean(losses)


def distill_multiple_models_to_single(
    key: jnp.ndarray,
    optim: optax.GradientTransformation,
    apply_fn: ApplyFn,
    init_fn: Callable[[jnp.ndarray, jnp.ndarray], Tuple[Params, State]],
    ensemble_params: Params,
    ensemble_state: State,
    batch_size: int,
    epochs: int,
    batches_per_epoch: int,
    data_mean: jnp.ndarray,
    data_scale: jnp.ndarray,
) -> Tuple[Params, State, jnp.ndarray, Mapping[str, jnp.ndarray]]:
  """Distills an ensemble of teachers to a single student on synthetic data.

  Args:
    key: Jax PRNG Key.
    optim: Optax optimizer.
    apply_fn: Flax apply function (for teachers and student).
    init_fn: Flax initialization function for student.
    ensemble_params: Teacher parameters, stacked along first dimension.
    ensemble_state: Teacher state, stacked along first dimension.
    batch_size: Batch size to use per gradient step.
    epochs: How many 'epochs' to train for.
    batches_per_epoch: How many batches to use per 'epoch'.
    data_mean: Array of same shape as data specifying per-dimension means.
    data_scale: Array of same shape as data specifying per-dimension stds.

  Returns:
    Student parameters, PRNG Key, Dictionary of distillation metrics.
  """
  train_key, init_key = jax.random.split(key, 2)

  (params, network_state) = init_fn(init_key, data_mean[None])
  opt_state = optim.init(params)

  losses = np.zeros(epochs)
  for i in range(epochs):
    params, network_state, opt_state, train_key, loss = distill_train_epoch(
        params,
        network_state,
        ensemble_params,
        ensemble_state,
        opt_state,
        train_key,
        batches_per_epoch,
        optim,
        batch_size,
        apply_fn,
        data_mean,
        data_scale,
    )
    losses[i] = loss

  return params, network_state, train_key, {'distill_loss': losses}


def create_dataset_splits(
    train_data: Mapping[str, jnp.ndarray],
    num_splits: int,
    key: jnp.ndarray,
    bootstrap: bool = True,
    augment_data: bool = True,
    test_fraction: float = 0.1,
) -> Tuple[Mapping[str, jnp.ndarray], Mapping[str, jnp.ndarray]]:
  """Create multiple splits from a dataset, possibly bootstrapped and augmented.

  Args:
    train_data: Dictionary of training arrays.
    num_splits: How many splits to create.
    key: Jax PRNG key.
    bootstrap: whether to bootstrap or just split datasets.
    augment_data: whether to augment data after splitting, to ensure that
      symmetry holds in each resulting dataset.
    test_fraction: what fraction of data to use for eval if not bootstrapping.

  Returns:
    Training datasets and testing datasets, stacked along axis 0.
  """

  data_keys = jax.random.split(key, num_splits)
  if bootstrap:
    datasets = [
        data_utils.bootstrap_dataset(train_data, key) for key in data_keys  # pytype: disable=wrong-arg-types  # jax-ndarray
    ]
    train_datasets = [d[0] for d in datasets]
    test_datasets = [d[1] for d in datasets]
  elif 1.0 > test_fraction > 0.0:
    datasets = [
        data_utils.split_dataset(train_data, key, test_fraction)  # pytype: disable=wrong-arg-types  # jax-ndarray
        for key in data_keys
    ]
    train_datasets = [d[0] for d in datasets]
    test_datasets = [d[1] for d in datasets]
  else:
    assert test_fraction == 0
    train_datasets = [train_data] * num_splits
    test_datasets = [train_data] * num_splits

  if augment_data:
    train_datasets = [
        data_utils.augment_data(**data) for data in train_datasets
    ]
    test_datasets = [data_utils.augment_data(**data) for data in test_datasets]

  test_set_len = min([a['context'].shape[0] for a in test_datasets])
  test_datasets = [
      {k: a[:test_set_len] for k, a in d.items()} for d in test_datasets
  ]

  train_datasets = {
      k: jnp.stack([d[k] for d in train_datasets]) for k in train_data.keys()
  }
  test_datasets = {
      k: jnp.stack([d[k] for d in test_datasets]) for k in train_data.keys()
  }

  if 'position' in train_datasets and 'context' in train_datasets:
    train_datasets['context'] = jnp.concatenate(
        [train_datasets['context'], train_datasets['position']], -1
    )
    test_datasets['context'] = jnp.concatenate(
        [test_datasets['context'], test_datasets['position']], -1
    )
    del train_datasets['position']
    del test_datasets['position']

  return train_datasets, test_datasets


def train_multiple_models(
    train_datasets: Mapping[str, jnp.ndarray],
    test_datasets: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    num_models: int,
    optim: optax.GradientTransformation,
    apply_fn: ApplyFn,
    init_fn: Callable[[jnp.ndarray, jnp.ndarray], Tuple[Params, State]],
    train_config: config_dict.FrozenConfigDict = rate_learning_defaults,
) -> Tuple[Params, State, OptaxState, Mapping[str, jnp.ndarray]]:
  """Trains a set of models on a single dataset by bootstrapping.

  Args:
    train_datasets: Dictionary of training arrays, stacked along axis 0.
    test_datasets: Dictionary of testing arrays, stacked along axis 0.
    key: PRNG key.
    num_models: How many models to train.
    optim: Optax optimizer.
    apply_fn: Haiku model application function.
    init_fn: Haiku model initialization function.
    train_config: Additional training arguments (config dict).

  Returns:
    List of models and training metrics, stacked along axis 0.
  """
  assert train_datasets['context'].shape[0] == num_models
  assert test_datasets['context'].shape[0] == num_models

  train_key, init_key = jax.random.split(key, 2)
  train_keys = jax.random.split(train_key, num_models)
  init_context = train_datasets['context'][0, 0:1]
  init_keys = jax.random.split(init_key, num_models)
  params = [init_fn(key, init_context)[0] for key in init_keys]
  states = [init_fn(key, init_context)[1] for key in init_keys]
  init_params = tree_stack(params)
  init_states = tree_stack(states)
  init_opt_states = tree_stack([optim.init(x) for x in params])
  batch_train = jax.vmap(
      train_model, in_axes=(0, 0, 0, 0, 0, 0, None, None, None)
  )
  return batch_train(
      train_datasets,
      test_datasets,
      train_keys,
      init_params,
      init_states,
      init_opt_states,
      apply_fn,
      optim,
      train_config,
  )


class LearnedTransitionRatePredictor:
  """Class wrapping rate learning and prediction."""

  def __init__(
      self,
      init_key: Optional[jnp.ndarray] = None,
      num_states: int = 3,
      position_dim: int = 2,
      config: config_dict.FrozenConfigDict = rate_learning_defaults,
  ):
    self.num_models = config.num_models
    if init_key is None:
      init_key = jax.random.PRNGKey(0)
    self.init_fn, self.apply_fn = get_mlp_fn(
        config.hidden_dimensions,
        num_states,
        batchnorm=config.batchnorm,
        dropout_rate=config.dropout_rate,
    )
    self.context_dim = position_dim + config.use_current + config.use_voltage
    self.rng, *keys = jax.random.split(init_key, self.num_models + 1)
    all_params = []
    all_states = []
    for key in keys:
      params, state = self.init_fn(x=jnp.zeros(self.context_dim), rng=key)
      all_params.append(params)
      all_states.append(state)
    self.params = tree_stack(all_params)
    self.state = tree_stack(all_states)
    self.num_states = num_states

    self.config = config

    @functools.partial(jax.jit, static_argnames='is_training')
    @functools.partial(jax.vmap, in_axes=(0, 0, None, None, None))
    def batch_call(params, state, x, rng, is_training):
      return self.apply_fn(params, state, rng, x, is_training=is_training)

    self.batch_apply = batch_call

    @functools.partial(jax.jit, static_argnames='is_training')
    def call_single_model(model_index, params, state, x, rng, is_training):
      params = jax.tree_util.tree_map(lambda x: x[model_index], params)
      state = jax.tree_util.tree_map(lambda x: x[model_index], state)
      return self.apply_fn(params, state, rng, x, is_training)

    self.apply_single_model = call_single_model

  def apply_model(
      self,
      x: np.ndarray,
      key: Optional[jnp.ndarray] = None,
      model_index: Optional[int] = None,
  ) -> np.ndarray:
    """Apply the learned networks, or optionally only a single network.

    Args:
      x: The beam position(s) to be predicted for, plus any other context.
      key: A Jax PRNG key.
      model_index: Which model to use. Defaults to None, all models.

    Returns:
      The predicted rates as a jax numpy array.
    """
    if key is None:
      key, self.rng = jax.random.split(self.rng)
    if model_index is None:
      rates, _ = self.batch_apply(self.params, self.state, x, key, False)
    else:
      rates, _ = self.apply_single_model(
          model_index, self.params, self.state, x, key, False
      )
      rates = rates[None]

    total_rate = rates[..., -1:]
    weights = jax.nn.softmax(rates[..., :-1], axis=-1)
    return (total_rate * weights).mean(0)

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
    self.rng, data_key, train_key = jax.random.split(key, 3)
    optim = optax.adamw(
        self.config.learning_rate, weight_decay=self.config.weight_decay
    )
    train_datasets, test_datasets = create_dataset_splits(
        train_data,
        self.num_models,
        data_key,
        bootstrap=bootstrap,
        augment_data=self.config.augment_data,
        test_fraction=self.config.val_frac,
    )

    (self.params, self.state, self.opt_state, train_metrics) = (
        train_multiple_models(
            train_datasets,
            test_datasets,
            train_key,
            num_models=self.num_models,
            optim=optim,
            init_fn=self.init_fn,
            apply_fn=self.apply_fn,
            train_config=self.config,
        )
    )

    return train_metrics

  def distill(
      self,
      train_data: Mapping[str, jnp.ndarray],
      config: config_dict.FrozenConfigDict = distillation_defaults,
  ) -> Mapping[str, jnp.ndarray]:
    """Distills this model's ensemble to a single neural network on fake data.

    Parameters are updated in-place rather than returned.

    Args:
      train_data: Data to model synthetic data on.
      config: Parameters to use for distillation, specifying batch size, epochs,
        and batches_per_epoch.

    Returns:
      Metrics from distillation training.
    """
    optim = optax.adamw(
        self.config.learning_rate, weight_decay=self.config.weight_decay
    )

    data_mean = np.concatenate(
        [train_data['context'].mean(0), train_data['position'].mean(0)], 0
    )

    data_scale = np.concatenate(
        [train_data['context'].std(0), train_data['position'].std(0)], 0
    )

    distillation_params, distillation_state, self.rng, distill_metrics = (
        distill_multiple_models_to_single(
            self.rng,
            optim,
            self.apply_fn,
            self.init_fn,
            ensemble_params=self.params,
            ensemble_state=self.state,
            batches_per_epoch=config.batches_per_epoch,
            epochs=config.epochs,
            batch_size=config.batch_size,
            data_mean=data_mean,
            data_scale=data_scale,
        )
    )
    self.params = tree_stack([distillation_params])
    self.state = tree_stack([distillation_state])

    return distill_metrics

  def save(
      self,
      save_dir: str,
      step: int = 0,
      fixed_context: Optional[np.ndarray] = None,
  ) -> None:
    """Saves the model as a TF saved model and its config dictionary as json.

    Args:
      save_dir: Directory to save model in.
      step: Step to save model with.
      fixed_context: Optional fixed input to prepend to all inputs to the saved
        model, used (for example) to fix a given current or voltage.
    """
    checkpoint_path = os.path.join(save_dir, str(step) + '.ckpt')
    logging.info('Saving checkpoint to %s', checkpoint_path)
    epath.Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = epath.Path(checkpoint_path)
    with path.open('wb') as f:
      f.write(flax.serialization.to_bytes(self.params))

    self.package_model(fixed_context=fixed_context)
    logging.info('Saving tf saved model to %s', save_dir)
    tf.saved_model.save(
        self.packaged_model,
        save_dir,
    )

    config_path = epath.Path(os.path.join(save_dir, 'config.json'))
    json_str = self.config.to_json()
    config_path.write_text(json_str)

  def load(
      self, load_dir: str, step: int = 0, load_params=True, load_config=True
  ) -> None:
    """Load in model, optionally including parameter dictionary and config.

    Args:
      load_dir: Directory to load model from.
      step: Checkpoint ID to load.
      load_params: Bool, whether to load parameters.
      load_config: Bool, whether to load config.
    """
    if load_params:
      checkpoint_path = os.path.join(load_dir, str(step) + '.ckpt')
      path = epath.Path(checkpoint_path)
      with path.open('rb') as f:
        self.params = flax.serialization.from_bytes(self.params, f.read())
    self.packaged_model = tf.saved_model.load(load_dir)

    if load_config:
      config_path = epath.Path(os.path.join(load_dir, 'config.json'))
      config_str = config_path.read_text()
      config_json = json.loads(config_str)
      config = config_dict.ConfigDict(config_json)
      self.config = config_dict.FrozenConfigDict(config)

  def package_model(self, fixed_context: Optional[np.ndarray] = None) -> None:
    """Packages the rate predictor network into a Tensorflow saved model."""

    logging.info('Converting rate predictor from jax to tf.')

    if fixed_context is None:
      input_shape = (1, self.context_dim)
    else:
      input_shape = (1, self.context_dim - fixed_context.shape[0])
      fixed_context = jnp.asarray(fixed_context)

    def apply_model(context, fixed_context):
      if fixed_context is not None:
        fixed_context = jnp.repeat(fixed_context[None], context.shape[0], 0)
        context = jnp.concatenate([fixed_context, context], axis=-1)

      rates, _ = self.batch_apply(
          self.params, self.state, context, self.rng, False
      )
      total_rate = rates[..., -1:]
      weights = jax.nn.softmax(rates[..., :-1], axis=-1)
      return (total_rate * weights).mean(0)

    apply_model = functools.partial(apply_model, fixed_context=fixed_context)
    apply_model = jax.jit(apply_model)

    predictor_tf = jax2tf.convert(apply_model, with_gradient=False)
    tf_module = tf.Module()

    tf_module.__call__ = tf.function(
        predictor_tf,
        autograph=False,
        input_signature=[
            tf.TensorSpec(
                shape=input_shape,
                dtype=np.float32,
                name='beam_position',
            )
        ],
    )

    self.packaged_model = tf_module

  def predict(
      self,
      grid: microscope_utils.AtomicGridMaterialFrame,
      beam_pos: geometry.Point,
      current_position: np.ndarray,
      neighbor_indices: np.ndarray,
      voltage_kv: float = 60,
      current_na: float = 0.1,
  ) -> np.ndarray:
    """Computes rate constants for transitioning a Si atom.

    Args:
      grid: Atomic grid state.
      beam_pos: 2-dimensional beam position in [0, 1] coordinate frame.
      current_position: 2-dimensional position of current silicon atom.
      neighbor_indices: Indices of the atoms on the grid to calculate rates for.
      voltage_kv: beam voltage in kV.
      current_na: beam current in nA.

    Returns:
      a 3-dimensional array of rate constants for transitioning to the 3
        nearest neighbors.
    """
    if not hasattr(self, 'packaged_model'):
      self.package_model()
    # Convert the beam_pos into a numpy array for convenience
    beam_pos = np.asarray([[beam_pos.x, beam_pos.y]])  # Shape = [1, -1]
    neighbor_positions = grid.atom_positions[neighbor_indices, :]
    neighbor_positions = neighbor_positions - current_position
    beam_pos = beam_pos - current_position
    beam_pos = beam_pos / constants.CARBON_BOND_DISTANCE_ANGSTROMS
    new_beam_pos, _, neighbor_order = data_utils.standardize_beam_and_neighbors(
        beam_pos,
        neighbor_positions,
    )
    context = new_beam_pos
    if self.config.use_voltage:
      context = np.concatenate([voltage_kv, context], axis=-1)
    if self.config.use_current:
      context = np.concatenate([current_na, context], axis=-1)

    rates = np.asarray(self.packaged_model(context))[0]

    # neighbor_order is an array mapping the canonical order of neighbors
    # (increasing angle from (1, 0)) to the order they were given in.
    # We can invert it with argsort to return the rates to the correct order.
    rates = rates[np.argsort(neighbor_order)]
    return rates


def visualize_rates(
    save_path: Optional[str],
    predict_rates: Callable[[np.ndarray], np.ndarray],
    grid_range: float = 1.5,
    num_points: int = 40_000,
    fixed_context: Optional[np.ndarray] = None,
):
  """Produce a contour plot visualizing a rate predictor.

  Args:
    save_path: A file to save the plot to.
    predict_rates: A function that maps a beam position in R^2 to predicted
      rates for each neighbor, as an array of rates for the neighbors in
      counterclockwise order. This may need to be a custom function made via
      functools.partial, for certain rate predictors.
    grid_range: The expanse of the grid to generate points over. The grid will
      span from (-range, range) on each axis.
    num_points: How many total points to use in the grid.
    fixed_context: Additional context inputs to provide at all locations.

  Returns:
    Matplotlib figure, the plot.
  """
  per_side_resolution = int(np.sqrt(num_points))
  xs, ys = np.meshgrid(
      np.linspace(-grid_range, grid_range, per_side_resolution),
      np.linspace(-grid_range, grid_range, per_side_resolution),
  )
  coords = np.stack([xs, ys], -1)
  context = coords.reshape(-1, *coords.shape[2:])
  if fixed_context is not None:
    fixed_context = np.repeat(fixed_context[None], num_points, 0)
    context = np.concatenate([fixed_context, context], axis=-1)

  pred_rates = predict_rates(context)

  pred_rates = pred_rates.reshape(*xs.shape, 3)

  fig = plt.figure(figsize=(5, 5))

  neighbor_positions = np.array([
      [1.42, 0],
      [-0.71, 1.23],
      [-0.71, -1.23],
  ]) * (1.6 / 1.42)
  cmaps = [
      plt.get_cmap('Blues'),
      plt.get_cmap('Reds'),
      plt.get_cmap('Greens'),
  ]
  colors = ['blue', 'red', 'green']

  max_rates = np.zeros((3,))
  for i in range(0, 3):
    f = pred_rates[..., i]
    plt.contourf(xs, ys, f, levels=10, cmap=cmaps[i], alpha=0.2)
    argmax = f.argmax()
    plt.scatter(
        xs.reshape(-1)[argmax], ys.reshape(-1)[argmax], c=colors[i], alpha=0.2
    )
    print(
        f'Maximum for {i} is {f.max()} at ({xs.reshape(-1)[argmax]},'
        f' {ys.reshape(-1)[argmax]}))'
    )
    plt.scatter(neighbor_positions[i, 0], neighbor_positions[i, 1], c=colors[i])
    max_rates[i] = f.max()
  plt.scatter([0], [0], c='black')

  if save_path is not None:
    with epath.Path(save_path).open('wb') as f:
      plt.savefig(f, bbox_inches='tight')

  return fig, max_rates
