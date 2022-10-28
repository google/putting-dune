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

import enum
import functools
import time
from typing import Mapping, Optional, Tuple

import jax
from jax import numpy as jnp
import numpy as np
from putting_dune import rate_learning


# TODO(joshgreaves): Avoid duplicating these values.
PRIOR_RATE_MEAN = np.array((0.85, 0))
PRIOR_RATE_COV = np.array(((0.1, 0), (0, 0.1)))
PRIOR_MAX_RATE = np.log(2) / 3


class SyntheticDataType(str, enum.Enum):
  NETWORK = 'network'
  PRIOR = 'prior'


@jax.jit
def sample_multivariate_context(
    key: jnp.ndarray,
    mean: jnp.ndarray,
    cov: jnp.ndarray,
):
  return jax.random.multivariate_normal(
      key,
      mean=mean,
      cov=cov,
  )


def get_all_context_rotations(context: jnp.ndarray, num_states: int = 3):
  """Gets all possible context rotations."""
  rot_contexts = [
      rotate_coordinates(context, 2 * n * jnp.pi / num_states)
      for n in range(num_states)
  ]
  return jnp.stack(rot_contexts, 0)


def rotate_coordinates(coord: jnp.ndarray, theta: float):
  """Rotates a set of coordinates by theta radians."""
  rotation_matrix = jnp.array(
      [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
  )
  return rotation_matrix @ coord


def rotate_attributes(x: jnp.ndarray, n: int):
  """Rotates a matrix n steps along its first dimension."""
  return jnp.roll(x, n, 0)


def get_all_rate_rotations(rates: jnp.ndarray, num_states: int = 3):
  """Gets all possible rate rotations."""
  rot_rates = [rotate_attributes(rates, n) for n in range(num_states)]
  return jnp.stack(rot_rates, 0)


def rotate_index(ind: jnp.ndarray, n: int, num_states: int = 3):
  """Rotates a state index by n."""
  return (ind + n) % num_states


def get_all_state_rotations(states: jnp.ndarray, num_states: int = 3):
  """Get all possible state rotations."""
  return jnp.stack(
      [
          rotate_index(states, n, num_states=num_states)
          for n in jnp.arange(num_states)
      ],
      0,
  )


def reflect_transition(
    states: jnp.ndarray,
    times: jnp.ndarray,
    rates: jnp.ndarray,
    context: jnp.ndarray,
    num_states: int = 3,
):
  """Reflects a transition along the y=0 axis."""
  if num_states != 3:
    raise NotImplementedError('Reflection currently only supported for n=3.')
  attr_reflection_matrix = jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
  ref_rates = attr_reflection_matrix @ rates

  coord_reflection_matrix = jnp.array(((1, 0), (0, -1)))
  context = coord_reflection_matrix @ context

  ref_states = jnp.array([0, 2, 1])[states - 1]
  ref_states = (ref_states + 1) * jnp.sign(states)

  return ref_states, times, ref_rates, context


reflect_dataset = jax.vmap(reflect_transition)


def get_transition_rotations(
    states: jnp.ndarray,
    times: jnp.ndarray,
    rates: jnp.ndarray,
    contexts: jnp.ndarray,
    num_states: int = 3,
):
  """Gets all valid rotations of an entire transition."""
  rot_states = get_all_state_rotations(states - 1, num_states=num_states)
  rot_states = (rot_states + 1) * jnp.sign(states[None])
  rot_rates = get_all_rate_rotations(rates, num_states=num_states)
  times = jnp.stack([times] * num_states)
  rot_context = get_all_context_rotations(contexts, num_states=num_states)

  return rot_states, times, rot_rates, rot_context


def rotate_dataset(
    states: jnp.ndarray,
    times: jnp.ndarray,
    rates: jnp.ndarray,
    contexts: jnp.ndarray,
    num_states: int = 3,
):
  """Get all valid rotations of an entire dataset."""
  rotate = functools.partial(get_transition_rotations, num_states=num_states)
  map_rotate = jax.vmap(rotate, in_axes=(0, 0, 0, 0))

  rot_states, rot_times, rot_rates, rot_contexts = map_rotate(
      states, times, rates, contexts
  )

  return (
      rot_states.reshape(-1, *states.shape[1:]),
      rot_times.reshape(-1, *times.shape[1:]),
      rot_rates.reshape(-1, *rates.shape[1:]),
      rot_contexts.reshape(-1, *contexts.shape[1:]),
  )


def prior_rates(
    context: jnp.ndarray,
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    max_rate: float,
):
  """Gets transition rates as following a Gaussian curve with given maximum."""
  norm = max_rate / jax.scipy.stats.multivariate_normal.pdf(mean, mean, cov)
  rate = jax.scipy.stats.multivariate_normal.pdf(context, mean, cov)
  return rate * norm


def generate_synthetic_data(
    num_data: int = 100,
    data_seed: Optional[int] = None,
    num_states: int = 3,
    context_dim: int = 2,
    actual_time_range: Tuple[float, float] = (0, 5),
    mode=SyntheticDataType.PRIOR,
) -> Tuple[Mapping[str, jnp.ndarray], Mapping[str, jnp.ndarray]]:
  """Generates a synthetic dataset to use when testing rate learning.

  Arguments:
    num_data: Number of transitions to generate.
    data_seed: Optional random seeding for data.
    num_states: How many states to transition between.
    context_dim: Dimensionality of the random context vectors.
    actual_time_range: How long of windows to sample. Controls frequency of
      empty transitions.
    mode: Whether to use a random network ("network") or an informed prior
      ("prior") to generate data.

  Returns:
    train_data: Training dataset
    test_data: Testing dataset
  """
  if data_seed is None:
    data_seed = int(time.time())
  key = jax.random.PRNGKey(data_seed)
  key, init_key = jax.random.split(key)

  if mode == SyntheticDataType.NETWORK:
    (init_mlp, apply_mlp) = rate_learning.get_mlp_fn((64,), num_states)
    init_params = init_mlp(x=jnp.zeros(context_dim), rng=init_key)

  @functools.partial(jax.jit, static_argnames='shape')
  def sample_exp(sample_key, k, shape):
    return (
        -jnp.log(jax.random.uniform(sample_key, shape, dtype=jnp.float32)) / k
    )

  @jax.jit
  def sample_network_rates(element_key):
    state_key, time_key, actual_time_key, context_key = jax.random.split(
        element_key, 4
    )
    context = jax.random.normal(context_key, shape=(1, context_dim))
    rates = apply_mlp(init_params, context_key, context)[0, :-1]

    total_rate = jnp.sum(rates)
    p = rates / total_rate
    next_state = jax.random.choice(state_key, len(rates), (1,), p=p)
    next_time = sample_exp(time_key, total_rate, (1,))
    actual_time = jax.random.uniform(
        actual_time_key,
        (1,),
        minval=actual_time_range[0],
        maxval=actual_time_range[1],
    )

    transitioned = next_time < actual_time
    next_state = transitioned * (next_state + 1)

    return {
        'next_state': next_state,
        'dt': actual_time,
        'rates': rates,
        'context': context[0],
    }

  @jax.jit
  def sample_from_prior(key):
    (
        state_key,
        rot_key,
        time_key,
        actual_time_key,
        context_key,
    ) = jax.random.split(key, 5)
    context = sample_multivariate_context(
        context_key, PRIOR_RATE_MEAN, PRIOR_RATE_COV * 1.5
    )
    rates = prior_rates(
        get_all_context_rotations(context, num_states=num_states),
        mean=PRIOR_RATE_MEAN,
        cov=PRIOR_RATE_COV,
        max_rate=PRIOR_MAX_RATE,
    )
    total_rate = jnp.sum(rates, -1)
    p = rates / total_rate
    next_state = jax.random.choice(state_key, len(rates), (), p=p)

    rotation_factor = jax.random.randint(rot_key, (), 0, num_states)
    context = rotate_coordinates(
        context, 2 * rotation_factor * jnp.pi / num_states
    )
    next_state = rotate_index(
        next_state, rotation_factor, num_states=num_states
    )
    rates = rotate_attributes(rates, rotation_factor)
    next_time = sample_exp(time_key, total_rate, (1,))
    actual_time = jax.random.uniform(
        actual_time_key,
        (1,),
        minval=actual_time_range[0],
        maxval=actual_time_range[1],
    )
    transitioned = next_time < actual_time
    next_state = transitioned * (next_state + 1)
    return {
        'next_state': next_state,
        'dt': actual_time,
        'rates': rates,
        'context': context,
    }

  vmap_sample_from_prior = jax.vmap(sample_from_prior, axis_name='batch')
  vmap_sample_network = jax.vmap(sample_network_rates, axis_name='batch')

  def sample_dataset(key, num_data, mode):
    keys = jax.random.split(key, num_data)
    if mode == SyntheticDataType.PRIOR:
      data = vmap_sample_from_prior(keys)
    elif mode == SyntheticDataType.NETWORK:
      data = vmap_sample_network(keys)
    return data

  train_key, test_key = jax.random.split(key)

  train_data = sample_dataset(train_key, num_data, mode=mode)
  test_data = sample_dataset(test_key, num_data, mode=mode)

  return train_data, test_data


def bootstrap_dataset(data: Mapping[str, np.ndarray], rng: jnp.ndarray):
  """Bootstraps a dataset to generate a training and testing dataset.

  Args:
    data: training dataset. Dictionary of arrays with same first dim.
    rng: JAX prngkey.

  Returns:
    train_data: boostrapped training data.
    test_data: Testing data (samples excluded from train_data).
  """
  original_length = list(data.values())[0].shape[0]
  indices = jax.random.choice(
      rng, a=original_length, shape=[original_length], replace=True
  )
  train_data = {k: a[indices] for k, a in data.items()}
  test_indices = set(range(original_length)) - set(np.array(indices))
  test_indices = np.array(list(test_indices))
  test_data = {k: a[test_indices] for k, a in test_indices}
  return train_data, test_data


def augment_data(
    next_state: jnp.ndarray,
    dt: jnp.ndarray,
    rates: jnp.ndarray,
    context: jnp.ndarray,
    reflect: bool = True,
    num_states: int = 3,
):
  """Augments a dataset by adding all valid reflections and rotations."""
  if reflect:
    ref_next_state, ref_dt, ref_rates, ref_context = reflect_dataset(
        next_state, dt, rates, context
    )
    next_state = jnp.concatenate([next_state, ref_next_state])
    dt = jnp.concatenate([dt, ref_dt])
    rates = jnp.concatenate([rates, ref_rates])
    context = jnp.concatenate([context, ref_context])

  next_state, dt, rates, context = rotate_dataset(
      next_state, dt, rates, context, num_states=num_states
  )

  return {
      'next_state': next_state,
      'dt': dt,
      'rates': rates,
      'context': context,
  }


def get_neighbor_angles(neighbor_positions: np.ndarray) -> np.ndarray:
  """Gets the angles from a current atom to neighboring atoms."""
  angles = np.arctan2(-neighbor_positions[:, 1], neighbor_positions[:, 0])
  return angles


def standardize_beam_and_neighbors(
    beam_pos: np.ndarray, neighbor_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Standardizes local graphene rotation, adjusts beam and sorts neighbors."""
  angles = get_neighbor_angles(neighbor_positions)
  angle = np.min(angles)
  state_order = np.argsort(angles)
  beam_pos = rotate_coordinates(beam_pos, angle)
  return beam_pos, state_order, angles
