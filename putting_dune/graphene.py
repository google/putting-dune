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

"""Doped graphene materials."""

import abc
import dataclasses
import datetime as dt
import functools
import os
import pathlib
from typing import Iterable, Protocol, Sequence

from absl import logging
from etils import epath
from jax.scipy import stats
import msgpack_numpy as msgpack
import numpy as np
import numpy.typing as npt
from putting_dune import constants
from putting_dune import geometry
from putting_dune import microscope_utils
import scipy.stats


@dataclasses.dataclass(frozen=True)
class SuccessorState:
  grid: microscope_utils.AtomicGridMaterialFrame
  rate: float


@dataclasses.dataclass(frozen=True)
class Rates:
  successor_states: Sequence[SuccessorState]

  @property
  def total_rate(self) -> float:
    return sum([x.rate for x in self.successor_states])


class RateFunction(Protocol):

  def __call__(
      self,
      grid: microscope_utils.AtomicGridMaterialFrame,
      beam_position: geometry.PointMaterialFrame,
  ) -> Rates:
    ...


class CanonicalRatePredictionFn(Protocol):
  """A rate predictor for use with PristineSingleSiGrRatePredictor.

  Takes an atomic grid representing the current material
  state, a probe position, a silicon atom position, and positions of the
  3 nearest neighbors, and returns the rate at which the silicon atom
  swaps places with its nearest neighbors.
  """

  def __call__(
      self,
      grid: microscope_utils.AtomicGridMaterialFrame,
      beam_position: geometry.PointMaterialFrame,
      silicon_position: np.ndarray,
      neighbor_indices: np.ndarray,
  ) -> np.ndarray:
    ...


class SiliconNotFoundError(RuntimeError):
  ...


# TODO(joshgreaves): Move interface if we support more than graphene.
class Material(abc.ABC):
  """Abstract base class for materials."""

  @abc.abstractmethod
  def get_atoms_in_bounds(
      self,
      lower_left: geometry.PointMaterialFrame,
      upper_right: geometry.PointMaterialFrame,
  ) -> microscope_utils.AtomicGridMicroscopeFrame:
    """Gets the atomic grid for a particular field of view.

    Args:
      lower_left: The lower left position of the field of view.
      upper_right: The upper right position of the field of view.

    Returns:
      The observed atomic grid within the supplied bounds. Atom positions
        are normalized in [0, 1], where (0, 0) corresponds to lower_left and
        (1, 1) corresponds to upper_right.
    """

  @abc.abstractmethod
  def reset(self, rng: np.random.Generator) -> None:
    """Resets the material."""

  @abc.abstractmethod
  def apply_control(
      self,
      rng: np.random.Generator,
      control: microscope_utils.BeamControlMaterialFrame,
      observers: Iterable[microscope_utils.SimulatorObserver] = (),
  ) -> None:
    """Simulates controls applied to the material."""


def single_silicon_prior_rates(
    context: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    max_rate: float,
):
  """Gets transition rates as following a Gaussian curve with given maximum."""
  norm = max_rate / stats.multivariate_normal.pdf(mean, mean, cov)
  rate = stats.multivariate_normal.pdf(context, mean, cov)
  return rate * norm


def simple_canonical_rate_function(
    grid: microscope_utils.AtomicGridMaterialFrame,
    beam_position: geometry.PointMaterialFrame,
    silicon_position: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
  """Computes rate constants for transitioning a Si atom.

  Args:
    grid: Atomic grid state.
    beam_position: 2-dimensional beam position in [0, 1] coordinate frame.
    silicon_position: 2-dimensional position of the current silicon atom.
    neighbor_indices: Indices of the atoms on the grid to calculate rates for.

  Returns:
    a 3-dimensional array of rate constants for transitioning to the 3
      nearest neighbors.
  """
  # Convert the beam_position into a numpy array for convenience
  beam_position = np.asarray([[beam_position.x, beam_position.y]])

  neighbor_positions = grid.atom_positions[neighbor_indices, :]
  neighbor_positions = neighbor_positions - silicon_position
  beam_position = beam_position - silicon_position

  # Distance between neighbors and beam position.
  dist = np.linalg.norm(beam_position - neighbor_positions, axis=-1)
  # Normalize by carbon bond distance.
  dist = dist / constants.CARBON_BOND_DISTANCE_ANGSTROMS
  # Inverse square falloff for beam displacement.
  rates = 1.0 / (np.square(dist * 4) + 1.0)  # Maximum rate = 1.

  # Rates for moving to the nearest neighbor positions.
  return rates


class HumanPriorRatePredictor:
  """Implements rate prediction according to a human-designed prior.

  Attributes:
    mean: Point for which transition is most likely, relative to a neighbor
      located at (1, 0).
    cov: Assuming Gaussian falloff of transition rates, the covariance matrix
      describing the shape of the distribution.
    max_rate: The maximum rate (as lambda of an exponential distribution) at the
      center of the peak.
  """

  def __init__(
      self,
      mean: np.ndarray = constants.SIGR_PRIOR_RATE_MEAN,
      cov: np.ndarray = constants.SIGR_PRIOR_RATE_COV,
      max_rate: float = constants.SIGR_PRIOR_MAX_RATE,
  ):
    self.mean = mean
    self.cov = cov
    self.max_rate = max_rate

  def predict(
      self,
      grid: microscope_utils.AtomicGridMaterialFrame,
      beam_position: geometry.PointMaterialFrame,
      silicon_position: np.ndarray,
      neighbor_indices: np.ndarray,
  ) -> np.ndarray:
    """Computes rate constants for transitioning a Si atom.

    Args:
      grid: Atomic grid state.
      beam_position: 2-dimensional beam position in material coordinate frame.
      silicon_position: 2-dimensional position of the current silicon atom.
      neighbor_indices: Indices of the atoms on the grid to calculate rates for.

    Returns:
      a 3-dimensional array of rate constants for transitioning to the 3
        nearest neighbors.
    """
    # Convert the beam_pos into a numpy array for convenience
    beam_position = np.asarray(
        [[beam_position.x, beam_position.y]]
    )  # Shape = [1, -1]

    neighbor_positions = grid.atom_positions[neighbor_indices, :]
    relative_neighbor_positions = neighbor_positions - silicon_position
    angles = geometry.get_angles(relative_neighbor_positions)

    relative_beam_position = beam_position - silicon_position
    relative_beam_position /= constants.CARBON_BOND_DISTANCE_ANGSTROMS
    rates = np.zeros((neighbor_indices.shape), dtype=float)
    for i, angle in enumerate(angles):
      rotated_mean = geometry.rotate_coordinates(self.mean, -angle)
      rate = single_silicon_prior_rates(
          relative_beam_position, rotated_mean, self.cov, self.max_rate
      )
      rates[i] = rate

    return rates


@dataclasses.dataclass(frozen=True)
class PristineSingleSiGrRatePredictor(RateFunction):
  """A single silicon, pristine graphene rate predictor."""

  canonical_rate_prediction_fn: CanonicalRatePredictionFn

  def __call__(
      self,
      grid: microscope_utils.AtomicGridMaterialFrame,
      beam_position: geometry.PointMaterialFrame,
  ) -> Rates:
    silicon_position = get_single_silicon_position(grid)

    si_neighbor_indices = geometry.nearest_neighbors3(
        grid.atom_positions, silicon_position
    ).neighbor_indices
    si_neighbor_indices = si_neighbor_indices.reshape(-1)

    transition_rates = self.canonical_rate_prediction_fn(
        grid,
        beam_position,
        silicon_position,
        si_neighbor_indices,
    )
    transition_rates = np.asarray(transition_rates).astype(np.float32)

    assert (transition_rates >= 0).all(), 'transition_rates were not positive.'
    assert transition_rates.size == si_neighbor_indices.size

    # Create successor grids associated with the rates.
    atom_positions = grid.atom_positions  # Atom positions remain fixed.
    successor_states = []
    for next_si_idx, rate in zip(si_neighbor_indices, transition_rates):
      atomic_numbers = np.full_like(grid.atomic_numbers, constants.CARBON)
      atomic_numbers[next_si_idx] = constants.SILICON
      successor_states.append(
          SuccessorState(
              microscope_utils.AtomicGridMaterialFrame(
                  microscope_utils.AtomicGrid(atom_positions, atomic_numbers)
              ),
              rate,
          )
      )

    return Rates(successor_states)


@dataclasses.dataclass(frozen=True)
class GaussianMixtureRateFunction(RateFunction):
  """A rate function that uses a mixture of Gaussians."""

  max_rate: float
  mixture_weights: npt.NDArray[np.float32]  # Shape (n_mixtures,)
  loc_distances: npt.NDArray[np.float32]  # Shape (n_mixtures,)
  variances: npt.NDArray[np.float32]  # Shape (n_mixtures, 2)

  @functools.cached_property
  def _normalizing_factor(self) -> float:
    """Computes the normalizing factor the mixture."""
    num_mixtures = len(self.mixture_weights)

    # Get the peak values of the components to work out a normalizing factor.
    max_mode_prob = 0.0
    for i in range(num_mixtures):
      covariance_matrix = np.diag(self.variances[i])
      mode_prob = scipy.stats.multivariate_normal.pdf(
          np.zeros(2), np.zeros(2), covariance_matrix
      )
      mode_prob = mode_prob * self.mixture_weights[i]
      max_mode_prob = max(max_mode_prob, mode_prob)
    return self.max_rate / max_mode_prob

  def __call__(
      self,
      grid: microscope_utils.AtomicGridMaterialFrame,
      beam_position: geometry.PointMaterialFrame,
  ) -> Rates:
    # TODO(joshgreaves): Break this function apart.
    # I will imminently be refactoring the materials part of this project,
    # and once that is done I can break this up nicely.
    num_mixtures = len(self.mixture_weights)

    # Get the silicon position and its 3 nearest neighbors.
    si_pos = get_single_silicon_position(grid)
    neighbor_indices = geometry.nearest_neighbors3(
        grid.atom_positions, si_pos
    ).neighbor_indices
    neighbor_indices = neighbor_indices.reshape(-1)
    neighbor_positions = grid.atom_positions[neighbor_indices]

    # Compute the vectors from silicon to each of the three nearest neighbors.
    # Then, construct vectors orthogonal to these. This will give us
    # the principal components of the covariance matrix (which we will
    # construct later).
    neighbor_delta_vectors = neighbor_positions - si_pos.reshape(1, 2)
    orthog_vectors = np.empty_like(neighbor_delta_vectors)
    orthog_vectors[:, 0] = -neighbor_delta_vectors[:, 1]
    orthog_vectors[:, 1] = neighbor_delta_vectors[:, 0]

    # Normalize the basis vectors.
    basis_vectors1 = neighbor_delta_vectors / np.linalg.norm(
        neighbor_delta_vectors, axis=-1, keepdims=True
    )
    basis_vectors2 = orthog_vectors / np.linalg.norm(
        orthog_vectors, axis=-1, keepdims=True
    )

    successor_states = []

    # Calculate the rate for each neighbor.
    for i, neighbor_idx in enumerate(neighbor_indices):
      # To compute the covariance matrix, we will need the matrix of
      # eigenvectors, along with the inverse of that matrix.
      covariance_eigenvectors = np.transpose(
          np.vstack((basis_vectors1[i], basis_vectors2[i]))
      )
      covariance_eigenvectors_inv = np.linalg.pinv(covariance_eigenvectors)

      # Accumulate the rate across all mixtures.
      rate = 0.0
      for mixture_idx in range(num_mixtures):
        # The gaussian mean is found a fixed distance along the vector
        # to the neighbor from the silicon atom.
        loc = (
            si_pos + neighbor_delta_vectors[i] * self.loc_distances[mixture_idx]
        )

        # The covariance is calculated using the eigenvector matrix
        # and a diagonal matrix indicating the variance in the direction
        # of each eigenvector.
        diagonal_variance = np.diag(self.variances[mixture_idx])
        covariance_matrix = (
            covariance_eigenvectors
            @ diagonal_variance
            @ covariance_eigenvectors_inv
        )

        # Calculate the multivariate gaussian probability density.
        probability_density = scipy.stats.multivariate_normal.pdf(
            np.asarray(beam_position.coords), loc, covariance_matrix
        )

        # Weight the probability by the normalizing factor, along with
        # the mixture weight to get its contribution to the total weight.
        mixture_rate = probability_density * self._normalizing_factor
        rate += mixture_rate * self.mixture_weights[mixture_idx]

      # Construct the new atomic grid for this neighbor's transition.
      new_atomic_numbers = np.full_like(grid.atomic_numbers, constants.CARBON)
      new_atomic_numbers[neighbor_idx] = constants.SILICON
      new_grid = microscope_utils.AtomicGridMaterialFrame(
          microscope_utils.AtomicGrid(
              grid.atom_positions,
              new_atomic_numbers,
          )
      )
      successor_states.append(SuccessorState(new_grid, rate))

    return Rates(successor_states)

  def serialize_to_directory(self, save_dir: pathlib.Path | str, /) -> None:
    """Serializes this rate function to the specified directory."""
    logging.info(
        'Serializing %s to %s.', self.__class__.__name__, str(save_dir)
    )
    path = epath.Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    bundle = {
        'sem_ver': '1.0.0',
        'max_rate': self.max_rate,
        'mixture_weights': self.mixture_weights,
        'loc_distances': self.loc_distances,
        'variances': self.variances,
    }

    output_file = path / 'gmm_parameters.mpk'
    output_file.write_bytes(msgpack.packb(bundle))

  @classmethod
  def deserialize_from_directory(
      cls, load_dir: os.PathLike[str] | str, /
  ) -> 'GaussianMixtureRateFunction':
    """Deserializes a rate function from a specified directory."""
    logging.info('Deserializing %s from %s', cls.__name__, str(load_dir))
    path = epath.Path(load_dir)

    input_file = path / 'gmm_parameters.mpk'
    bundle = msgpack.unpackb(input_file.read_bytes())

    return cls(
        max_rate=bundle['max_rate'],
        mixture_weights=bundle['mixture_weights'],
        loc_distances=bundle['loc_distances'],
        variances=bundle['variances'],
    )

  @classmethod
  def sample_new(
      cls, rng: np.random.Generator, /
  ) -> 'GaussianMixtureRateFunction':
    num_mixtures = rng.poisson(2.0) + 1
    max_rate = rng.uniform(0.01, 1.0)
    mixture_weights = rng.uniform(0.0, 10.0, size=(num_mixtures,))
    mixture_weights = mixture_weights / np.sum(mixture_weights)
    loc_distances = rng.uniform(-2.0, 3.0, size=(num_mixtures,))
    variances = rng.uniform(0.1, 5.0, size=(num_mixtures, 2))

    return GaussianMixtureRateFunction(
        max_rate=max_rate,
        mixture_weights=mixture_weights,
        loc_distances=loc_distances,
        variances=variances,
    )

  def __eq__(self, other: 'GaussianMixtureRateFunction') -> bool:
    # We consider very similar models to be the same, to account
    # for any floating point precision errors.
    if (
        self.mixture_weights.shape != other.mixture_weights.shape
        or self.loc_distances.shape != other.loc_distances.shape
        or self.variances.shape != other.variances.shape
        or abs(self.max_rate - other.max_rate) > 1e-3
        or (np.abs(self.mixture_weights - other.mixture_weights) > 1e-3).any()
        or (np.abs(self.loc_distances - other.loc_distances) > 1e-3).any()
        or (np.abs(self.variances - other.variances) > 1e-3).any()
    ):
      return False

    return True


def _generate_hexagonal_grid(num_cols: int = 50) -> np.ndarray:
  """Generates a hexagonal grid.

  This will generate a square grid based on the specified number of columns.
  The number of rows will be int(num_cols * 2 / sqrt(3)).
  See www.redblobgames.com/grids/hexagons/ for reference.

  Args:
    num_cols: The number of columns to generate.

  Returns:
    A 2d hexagonal grid of x, y coordinates. This will have shape
      (num_atoms, 2).
  """
  # Since we will scale the y coords by ratio, we may need to generate
  # more rows than columns to fill the unit frame.
  ratio = np.sqrt(3) / 2
  num_rows = int(num_cols / ratio)

  coord_x, coord_y = np.meshgrid(
      np.arange(num_cols), np.arange(num_rows), indexing='xy'
  )

  # Generate the grid.
  coord_y = coord_y * ratio
  coord_x = coord_x.astype(np.float32)
  coord_x[1::2, :] += 0.5
  # Remove extra points
  coord_x[0::2, 0::3] = np.inf
  coord_x[1::2, 1::3] = np.inf
  coord_y[0::2, 0::3] = np.inf
  coord_y[1::2, 1::3] = np.inf
  coord_x = coord_x.flatten()
  coord_y = coord_y.flatten()
  coord_x = coord_x[np.where(coord_x != np.inf)]
  coord_y = coord_y[np.where(coord_y != np.inf)]

  return np.stack((coord_x, coord_y), axis=1)


def canonical_pristine_graphene_with_centered_silicon(
    num_columns: int = 10,
) -> microscope_utils.AtomicGridMaterialFrame:
  """Generate a canonical pristine graphene sheet."""
  atom_positions = _generate_hexagonal_grid(num_columns)
  atom_positions *= constants.CARBON_BOND_DISTANCE_ANGSTROMS
  atom_positions -= np.mean(atom_positions, axis=0, keepdims=True)

  rotation_angle = 0
  rotation_angle_rad = np.deg2rad(rotation_angle)
  rotation_matrix = np.asarray([
      [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
      [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)],
  ])
  atom_positions @= rotation_matrix

  atomic_numbers = np.full((atom_positions.shape[0],), constants.CARBON)

  si_idx = np.argmin(np.sum(atom_positions**2, axis=1))  # Nearest (0, 0)
  atomic_numbers[si_idx] = constants.SILICON

  # Center silicon on (0, 0)
  atom_positions -= atom_positions[si_idx].reshape(1, -1)

  return microscope_utils.AtomicGridMaterialFrame(  # pylint: disable=no-value-for-parameter
      microscope_utils.AtomicGrid(atom_positions, atomic_numbers)
  )


def generate_pristine_graphene(
    rng: np.random.Generator, num_columns: int = 50
) -> np.ndarray:
  """Generates the positions of carbon atoms in a pristine graphene sheet."""
  positions = _generate_hexagonal_grid(num_columns)

  # Scale the positions to have the correct distance.
  positions = positions * constants.CARBON_BOND_DISTANCE_ANGSTROMS

  # Center the grid, but add a small random offset.
  positions = positions - np.mean(positions, axis=0, keepdims=True)
  positions += rng.uniform(
      -constants.CARBON_BOND_DISTANCE_ANGSTROMS / 2,
      constants.CARBON_BOND_DISTANCE_ANGSTROMS / 2,
      size=(1, 2),
  )

  # Apply a random rotation to the grid.
  rotation_angle = rng.uniform(0.0, 2 * np.pi)
  rotation_matrix = np.asarray([
      [np.cos(rotation_angle), -np.sin(rotation_angle)],
      [np.sin(rotation_angle), np.cos(rotation_angle)],
  ])
  # Apply the rotation matrix on the rhs, since grid is shape (num_atoms, 2).
  positions = positions @ rotation_matrix

  return positions


class PristineSingleDopedGraphene(Material):
  """A pristine graphene sheet with a single silicon dopant.

  All distances in this class are measured in angstroms, and data is
  stored in numpy arrays for efficiency.
  """

  def __init__(
      self,
      *,
      rate_function: RateFunction = PristineSingleSiGrRatePredictor(
          canonical_rate_prediction_fn=simple_canonical_rate_function,
      ),
      grid_columns: int = 50,
  ):
    self._grid_columns = grid_columns

    # Set in reset, declared here to help type-checkers.
    self._has_been_reset = False
    self.grid: microscope_utils.AtomicGridMaterialFrame
    self._rate_function = rate_function

  def reset(self, rng: np.random.Generator) -> None:
    self._has_been_reset = True

    atom_positions = generate_pristine_graphene(rng, self._grid_columns)
    num_atoms = atom_positions.shape[0]
    atomic_numbers = np.full(num_atoms, constants.CARBON)

    # Choose the point closest to the center for the silicon starting point.
    distances = np.linalg.norm(atom_positions, axis=1)
    si_index = np.argmin(distances)
    atomic_numbers[si_index] = constants.SILICON

    self.grid = microscope_utils.AtomicGridMaterialFrame(
        microscope_utils.AtomicGrid(atom_positions, atomic_numbers)
    )

  def get_atoms_in_bounds(
      self,
      lower_left: geometry.PointMaterialFrame,
      upper_right: geometry.PointMaterialFrame,
  ) -> microscope_utils.AtomicGridMicroscopeFrame:
    """Gets the atomic grid for a particular field of view.

    Args:
      lower_left: The lower left position of the field of view.
      upper_right: The upper right position of the field of view.

    Returns:
      The observed atomic grid within the supplied bounds. Atom positions
        are normalized in [0, 1], where (0, 0) corresponds to lower_left and
        (1, 1) corresponds to upper_right.

    Raises:
      RuntimeError: If called before reset.
    """
    self._assert_has_been_reset('get_atoms_in_bounds')
    lower_left = np.asarray(lower_left.coords)
    upper_right = np.asarray(upper_right.coords)

    indices_in_bounds = np.all(
        (
            (lower_left <= self.grid.atom_positions)
            & (self.grid.atom_positions <= upper_right)
        ),
        axis=1,
    )

    selected_atom_positions = self.grid.atom_positions[indices_in_bounds]
    selected_atomic_numbers = self.grid.atomic_numbers[indices_in_bounds]

    # Normalize atom positions in [0, 1]
    delta = (upper_right - lower_left).reshape(1, -1)
    selected_atom_positions = (
        selected_atom_positions - lower_left.reshape(1, -1)
    ) / delta

    return microscope_utils.AtomicGridMicroscopeFrame(
        microscope_utils.AtomicGrid(
            selected_atom_positions, selected_atomic_numbers
        )
    )

  def apply_control(
      self,
      rng: np.random.Generator,
      control: microscope_utils.BeamControlMaterialFrame,
      observers: Iterable[microscope_utils.SimulatorObserver] = (),
  ) -> None:
    """Simulates applying a beam exposure to the material."""
    self._assert_has_been_reset('apply_control')
    # There is a chance that, if the dwell time is long enough, there
    # will be multiple state transitions. We simulate these transitions
    # until the dwell time is over.
    elapsed_time = dt.timedelta(seconds=0)
    while elapsed_time < control.dwell_time:
      rates = self._rate_function(
          self.grid, geometry.PointMaterialFrame(control.position)
      )

      # The time at which the next transition takes place is modeled
      # by an exponential distribution, with the total rate as the
      # parameter. Scale is the inverse rate.
      transition_seconds = rng.exponential(scale=1.0 / rates.total_rate)
      # Avoid np.inf when using very small rates. Clip arbitrarily to 1 hour.
      transition_seconds = min(transition_seconds, 3600)
      transition_time = dt.timedelta(seconds=transition_seconds)
      elapsed_time += transition_time

      # If elapsed_time is less than the dwell time, a transition has
      # ocurred while we are still dwelling.
      # If elapsed_time is greater than the dwell time, a transition would
      # have ocurred if we had a longer exposure, but we didn't wait long
      # enough. Therefore, don't transition and break out of the loop.
      if elapsed_time <= control.dwell_time:
        # Update the silicon position
        successor_states_rates = np.asarray(
            [ss.rate for ss in rates.successor_states], dtype=np.float32
        )
        successor_states_prob = successor_states_rates / rates.total_rate

        # Pick index to transition to.
        successor_state_idx = rng.choice(
            successor_states_prob.size, p=successor_states_prob
        )
        self.grid = rates.successor_states[successor_state_idx].grid

        for observer in observers:
          observer.observe_transition(
              time_since_control_was_applied=elapsed_time,
              grid=self.grid,
          )

  def get_silicon_position(self) -> np.ndarray:
    self._assert_has_been_reset('get_silicon_position')
    return self.grid.atom_positions[
        self.grid.atomic_numbers == constants.SILICON
    ].reshape(-1)

  def _assert_has_been_reset(self, fn_name: str) -> None:
    if not self._has_been_reset:
      raise RuntimeError(
          f'Must call reset on {self.__class__} before {fn_name}.'
      )


def get_silicon_positions(grid: microscope_utils.AtomicGrid) -> np.ndarray:
  return grid.atom_positions[grid.atomic_numbers == constants.SILICON]


def get_single_silicon_position(
    grid: microscope_utils.AtomicGrid,
) -> np.ndarray:
  """Gets the silicon position, assuming there is only one.

  If there are no silicon atoms, it raises a SiliconNotFoundError.
  If there is 1 silicon atom, it returns its position.
  If there are more than 1 silicon atoms, it returns the position of the
  silicon atom nearest the center of the FOV.

  Args:
    grid: The grid to find the silicon atom on.

  Returns:
    A numpy array of the silicon atoms position.

  Raises:
    SiliconNotFoundError: If there are no silicon atoms found.
  """
  silicon_position = get_silicon_positions(grid)

  num_silicon_atoms = silicon_position.size // 2
  if num_silicon_atoms == 0:
    raise SiliconNotFoundError()
  elif num_silicon_atoms > 1:
    logging.warning('Expected 1 silicon atom. Found %d', num_silicon_atoms)

    # Select the silicon nearest the middle of the FOV.
    distance_from_center_fov = np.linalg.norm(
        np.asarray([[0.5, 0.5]]) - silicon_position, axis=1
    )
    silicon_position = silicon_position[np.argmin(distance_from_center_fov)]

  return silicon_position.reshape(-1)
