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

"""Code for managing transition data for rate learning."""

import enum
import functools
import time
from typing import Mapping, Optional, Tuple

import jax
from jax import numpy as jnp
import numpy as np
from putting_dune import constants
from putting_dune import geometry
from putting_dune import graphene
from putting_dune.rate_learning import learn_rates


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


def get_all_position_rotations(context: jnp.ndarray, num_states: int = 3):
  """Gets all possible context rotations."""
  rot_context = [
      geometry.jnp_rotate_coordinates(context, 2 * n * jnp.pi / num_states)  # pytype: disable=wrong-arg-types  # jax-ndarray
      for n in range(num_states)
  ]
  return jnp.stack(rot_context, 0)


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
    position: jnp.ndarray,
    context: Optional[jnp.ndarray] = None,
    num_states: int = 3,
):
  """Reflects a transition along the y=0 axis."""
  if num_states != 3:
    raise NotImplementedError('Reflection currently only supported for n=3.')
  attr_reflection_matrix = jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
  ref_rates = attr_reflection_matrix @ rates

  coord_reflection_matrix = jnp.array(((1, 0), (0, -1)))
  position = coord_reflection_matrix @ position

  ref_states = jnp.array([0, 2, 1])[states - 1]
  ref_states = (ref_states + 1) * jnp.sign(states)

  return ref_states, times, ref_rates, position, context


reflect_dataset = jax.vmap(reflect_transition)


def get_transition_rotations(
    states: jnp.ndarray,
    times: jnp.ndarray,
    rates: jnp.ndarray,
    position: jnp.ndarray,
    context: Optional[jnp.ndarray] = None,
    num_states: int = 3,
):
  """Gets all valid rotations of an entire transition."""
  rot_states = get_all_state_rotations(states - 1, num_states=num_states)
  rot_states = (rot_states + 1) * jnp.sign(states[None])
  rot_rates = get_all_rate_rotations(rates, num_states=num_states)
  times = jnp.stack([times] * num_states)
  if context is not None:
    context = jnp.stack([context] * num_states)
  rot_position = get_all_position_rotations(position, num_states=num_states)

  return rot_states, times, rot_rates, rot_position, context


def rotate_dataset(
    states: jnp.ndarray,
    times: jnp.ndarray,
    rates: jnp.ndarray,
    position: jnp.ndarray,
    context: Optional[jnp.ndarray] = None,
    num_states: int = 3,
):
  """Get all valid rotations of an entire dataset."""
  rotate = functools.partial(get_transition_rotations, num_states=num_states)
  map_rotate = jax.vmap(rotate, in_axes=(0, 0, 0, 0, 0))

  rot_states, rot_times, rot_rates, rot_position, rot_context = map_rotate(
      states, times, rates, position, context
  )
  if context is not None:
    rot_context = rot_context.reshape(-1, *context.shape[1:])

  return (
      rot_states.reshape(-1, *states.shape[1:]),
      rot_times.reshape(-1, *times.shape[1:]),
      rot_rates.reshape(-1, *rates.shape[1:]),
      rot_position.reshape(-1, *position.shape[1:]),
      rot_context,
  )


def generate_synthetic_data(
    num_data: int = 100,
    data_seed: Optional[int] = None,
    num_states: int = 3,
    position_dim: int = 2,
    context_dim: int = 2,
    actual_time_range: Tuple[float, float] = (0, 5),
    mode=SyntheticDataType.PRIOR,
) -> Tuple[Mapping[str, jnp.ndarray], Mapping[str, jnp.ndarray]]:
  """Generates a synthetic dataset to use when testing rate learning.

  Arguments:
    num_data: Number of transitions to generate.
    data_seed: Optional random seeding for data.
    num_states: How many states to transition between.
    position_dim: Dimensionality of the random position vectors.
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
    (init_mlp, apply_mlp) = learn_rates.get_mlp_fn(
        (1, 64), num_states, batchnorm=False
    )
    init_params, init_state = init_mlp(
        x=jnp.zeros(context_dim + position_dim), rng=init_key
    )

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
    context = jax.random.normal(
        context_key, shape=(1, context_dim + position_dim)
    )
    rates, _ = apply_mlp(init_params, init_state, context_key, context)
    rates = rates[0, :-1]

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
        'context': context[0, :context_dim],
        'position': context[0, context_dim:],
    }

  @jax.jit
  def sample_from_prior(key):
    (
        state_key,
        rot_key,
        time_key,
        actual_time_key,
        position_key,
        context_key,
    ) = jax.random.split(key, 6)
    position = sample_multivariate_context(
        position_key,
        constants.SIGR_PRIOR_RATE_MEAN,
        constants.SIGR_PRIOR_RATE_COV * 1.5,
    )
    context = jax.random.normal(context_key, shape=(context_dim,))
    rates = graphene.single_silicon_prior_rates(  # pytype: disable=wrong-arg-types  # jnp-type
        get_all_position_rotations(position, num_states=num_states),
        mean=constants.SIGR_PRIOR_RATE_MEAN,
        cov=constants.SIGR_PRIOR_RATE_COV,
        max_rate=constants.SIGR_PRIOR_MAX_RATE,
    )
    total_rate = jnp.sum(rates, -1)
    p = rates / total_rate
    next_state = jax.random.choice(state_key, len(rates), (), p=p)

    rotation_factor = jax.random.randint(rot_key, (), 0, num_states)
    position = geometry.jnp_rotate_coordinates(  # pytype: disable=wrong-arg-types  # jax-types
        position, 2 * rotation_factor * jnp.pi / num_states
    )
    next_state = rotate_index(  # pytype: disable=wrong-arg-types  # jax-types
        next_state, rotation_factor, num_states=num_states
    )
    rates = rotate_attributes(rates, rotation_factor)  # pytype: disable=wrong-arg-types  # jax-types
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
        'position': position,
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
  test_data = {k: a[test_indices] for k, a in data.items()}
  return train_data, test_data


def split_dataset(
    data: Mapping[str, np.ndarray],
    rng: jnp.ndarray,
    test_fraction: float = 0.1,
) -> Tuple[Mapping[str, np.ndarray], ...]:
  """Splits a dataset to generate a training and testing dataset.

  Args:
    data: training dataset. Dictionary of arrays with same first dim.
    rng: JAX prngkey.
    test_fraction: float in [0, 1]. Fraction of data to put in val set.

  Returns:
    train_data: training data.
    test_data: Testing data (samples excluded from train_data).
  """
  original_length = list(data.values())[0].shape[0]
  indices = jax.random.choice(
      rng, a=original_length, shape=[original_length], replace=False
  )
  train_indices = indices[int(original_length * test_fraction) :]
  test_indices = indices[: int(original_length * test_fraction)]
  train_data = {k: a[train_indices] for k, a in data.items()}
  test_data = {k: a[test_indices] for k, a in data.items()}
  return train_data, test_data


def augment_data(
    next_state: jnp.ndarray,
    dt: jnp.ndarray,
    rates: jnp.ndarray,
    position: jnp.ndarray,
    context: Optional[jnp.ndarray] = None,
    reflect: bool = True,
    num_states: int = 3,
):
  """Augments a dataset by adding all valid reflections and rotations."""
  if reflect:
    ref_next_state, ref_dt, ref_rates, ref_position, ref_context = (
        reflect_dataset(next_state, dt, rates, position, context)
    )
    next_state = jnp.concatenate([next_state, ref_next_state])
    dt = jnp.concatenate([dt, ref_dt])
    rates = jnp.concatenate([rates, ref_rates])
    position = jnp.concatenate([position, ref_position])
    if context is not None:
      context = jnp.concatenate([context, ref_context])

  next_state, dt, rates, position, context = rotate_dataset(
      next_state, dt, rates, position, context, num_states=num_states
  )

  return {
      'next_state': next_state,
      'dt': dt,
      'rates': rates,
      'context': context,
      'position': position,
  }


def standardize_beam_and_neighbors(
    beam_position: np.ndarray, neighbor_position: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Standardizes local graphene rotation, adjusts beam and sorts neighbors.

  To standardize the representation for a single silicon atom, the
  STEM probe, and the neighboring atoms, we first find which neighbor the
  is closest to the probe. We then orient the observation so that atom is
  directly to the right (x, 0).

  Args:
    beam_position: The position of the STEM probe relative to the silicon atom
      in angstroms.
    neighbor_position: The position of the neighboring carbon atoms, relative to
      the silicon atoms in angstroms.

  Returns:
    A tuple of (updated beam position, updated neighbor position,
      and neighbor_order). The updated position are the position after
      rotating to the canonical rotation. The order of the neighbors in the
      canonical representation is the index of the atoms starting from
      the rightmost atom, then sweeping through the circle counter-clockwise.
  """
  neighbor_distances_from_beam = np.linalg.norm(
      neighbor_position.reshape(-1, 2) - beam_position.reshape(1, 2), axis=1
  )
  min_distance_from_beam_idx = np.argmin(neighbor_distances_from_beam)

  # Get the (negative) angle that the beam is nearest as the rotation angle.
  neighbor_angles = geometry.get_angles(neighbor_position)
  rotation_angle = -neighbor_angles[min_distance_from_beam_idx]

  # Perform the rotation.
  new_neighbor_position = geometry.rotate_coordinates(
      neighbor_position, rotation_angle
  )
  new_beam_position = geometry.rotate_coordinates(beam_position, rotation_angle)

  # To decide the state order, get the new angles in (0, 2pi),
  # and sort ascending.
  positive_angles = (neighbor_angles + rotation_angle) % (2 * np.pi)
  state_order = np.argsort(positive_angles)

  return new_beam_position, new_neighbor_position, state_order
