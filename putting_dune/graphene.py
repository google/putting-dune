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
"""Doped graphene materials."""

import abc
import datetime as dt
from typing import Callable, Iterable, Tuple

import numpy as np
from putting_dune import constants
from putting_dune import data_utils
from putting_dune import simulator_utils
from shapely import geometry
from sklearn import neighbors


def simple_transition_rates(
    grid: simulator_utils.AtomicGrid,
    beam_pos: geometry.Point,
    current_position: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
  """Computes rate constants for transitioning a Si atom.

  Args:
    grid: Atomic grid state.
    beam_pos: 2-dimensional beam position in [0, 1] coordinate frame.
    current_position: 2-dimensional position of the current silicon atom.
    neighbor_indices: Indices of the atoms on the grid to calculate rates for.

  Returns:
    a 3-dimensional array of rate constants for transitioning to the 3
      nearest neighbors.
  """
  # Convert the beam_pos into a numpy array for convenience
  beam_pos = np.asarray([[beam_pos.x, beam_pos.y]])  # Shape = [1, -1]

  neighbor_positions = grid.atom_positions[neighbor_indices, :]
  neighbor_positions = neighbor_positions - current_position
  beam_pos = beam_pos - current_position
  # Distance between neighbors and beam position.
  dist = np.linalg.norm(beam_pos - neighbor_positions, axis=-1)
  # Normalize by carbon bond distance.
  dist = dist / constants.CARBON_BOND_DISTANCE_ANGSTROMS
  # Inverse square falloff for beam displacement.
  rates = 1.0 / (np.square(dist) + 0.5)  # Maximum rate = 2.

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
      grid: simulator_utils.AtomicGrid,
      beam_pos: geometry.Point,
      current_position: np.ndarray,
      neighbor_indices: np.ndarray,
  ) -> np.ndarray:
    """Computes rate constants for transitioning a Si atom.

    Args:
      grid: Atomic grid state.
      beam_pos: 2-dimensional beam position in material coordinate frame.
      current_position: 2-dimensional position of the current silicon atom.
      neighbor_indices: Indices of the atoms on the grid to calculate rates for.

    Returns:
      a 3-dimensional array of rate constants for transitioning to the 3
        nearest neighbors.
    """
    # Convert the beam_pos into a numpy array for convenience
    beam_pos = np.asarray([[beam_pos.x, beam_pos.y]])  # Shape = [1, -1]

    neighbor_positions = grid.atom_positions[neighbor_indices, :]
    relative_neighbor_positions = neighbor_positions - current_position
    angles = data_utils.get_neighbor_angles(relative_neighbor_positions)

    relative_beam_position = beam_pos - current_position
    relative_beam_position /= constants.CARBON_BOND_DISTANCE_ANGSTROMS
    rates = np.zeros((neighbor_indices.shape), dtype=float)
    for i, angle in enumerate(angles):
      rotated_mean = data_utils.rotate_coordinates(self.mean, -angle)
      rate = data_utils.prior_rates(
          relative_beam_position, rotated_mean, self.cov, self.max_rate
      )
      rates[i] = rate

    return rates


# TODO(joshgreaves): Move interface if we support more than graphene.
class Material(abc.ABC):
  """Abstract base class for materials."""

  @abc.abstractmethod
  def get_atoms_in_bounds(
      self, lower_left: geometry.Point, upper_right: geometry.Point
  ) -> simulator_utils.AtomicGrid:
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
  def reset(self) -> None:
    """Resets the material."""

  @abc.abstractmethod
  def apply_control(
      self,
      control: simulator_utils.BeamControl,
      start_time: dt.timedelta,
      observers: Iterable[simulator_utils.SimulatorObserver] = (),
  ) -> None:
    """Simulates controls applied to the material."""

  @property
  @abc.abstractmethod
  def atomic_grid(self) -> simulator_utils.AtomicGrid:
    """Gets the atomic grid representing the current material state."""


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


def sample_point_away_from_edge(
    rng: np.random.Generator,
    atom_positions: np.ndarray,
    nearest_neighbors: neighbors.NearestNeighbors,
    *,
    border_atom_positions: int = 8,
) -> Tuple[np.ndarray, int]:
  """Samples a point away from the edge of an atomic grid.

  Args:
    rng: The generator to use for sampling.
    atom_positions: The positions of the atoms in the atomic grid. should have
      shape (num_atoms, 2).
    nearest_neighbors: A nearest neighbors object for the supplied atom
      positions.
    border_atom_positions: The number of atomic positions to use as a border.
      i.e. the minimum number of atomic positions from the edge we should
      sample.

  Returns:
    A tuple containig the sampled coordinate from the grid with shape (2,),
      and the index it corresponds to from atom_positions.
  """
  # An easy way to pick a goal that is not near an adge is to contract
  # all the points, and then pick from the contracted points, and finally
  # pick the closest original point to the selected contracted point.
  max_atom_l2_distance = np.max(np.linalg.norm(atom_positions, axis=1, ord=2))
  border_length = (
      border_atom_positions * constants.CARBON_BOND_DISTANCE_ANGSTROMS
  )
  contraction = 1 - border_length / max_atom_l2_distance
  # Some very small grids might have a negative contraction, so clip it.
  contraction = max(contraction, 0.1)
  assert contraction < 1.0
  contracted_points = atom_positions * contraction

  # Randomly select from the contracted points.
  num_atoms, _ = contracted_points.shape
  goal_position = contracted_points[rng.choice(num_atoms, 1)]

  # Pick the point on the lattice closest to the contracted goal position.
  _, neighbor_indices = nearest_neighbors.kneighbors(
      goal_position.reshape(1, 2)
  )
  return atom_positions[neighbor_indices[0, 0]], neighbor_indices[0, 0]


# A function that takes an atomic grid representing the current material
# state, a probe position, a silicon atom position, and positions of the
# 3 nearest neighbors, and returns the rate at which the silicon atom
# swaps places with its nearest neighbors.
RatePredictionFn = Callable[
    [simulator_utils.AtomicGrid, geometry.Point, np.ndarray, np.ndarray],
    np.ndarray,
]


class PristineSingleDopedGraphene(Material):
  """A pristine graphene sheet with a single silicon dopant.

  All distances in this class are measured in angstroms, and data is
  stored in numpy arrays for efficiency.
  """

  def __init__(
      self,
      rng: np.random.Generator,
      *,
      predict_rates: RatePredictionFn = simple_transition_rates,
      grid_columns: int = 50,
  ):
    self.rng = rng
    self._grid_columns = grid_columns

    # Set in reset, declared here to help type-checkers.
    self.atom_positions: np.ndarray
    self.atomic_numbers: np.ndarray
    self.nearest_neighbors: neighbors.NearestNeighbors
    self._predict_rates = predict_rates

    self.reset()

  def reset(self) -> None:
    grid = _generate_hexagonal_grid(self._grid_columns)

    # Scale the grid to have the correct cell distance.
    grid = grid * constants.CARBON_BOND_DISTANCE_ANGSTROMS

    # Center the grid.
    # TODO(joshgreaves): Maybe add a small randomly generated offset.
    grid = grid - np.mean(grid, axis=0, keepdims=True)

    # Apply a random rotation to the grid.
    rotation_angle = self.rng.uniform(0.0, 2 * np.pi)
    rotation_matrix = np.asarray([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)],
    ])
    # Apply the rotation matrix on the rhs, since grid is shape (num_atoms, 2).
    self.atom_positions = grid @ rotation_matrix

    self.nearest_neighbors = neighbors.NearestNeighbors(
        n_neighbors=1 + 3,
        metric='l2',
        algorithm='brute',
    ).fit(self.atom_positions)

    num_atoms = self.atom_positions.shape[0]
    self.atomic_numbers = np.full(num_atoms, constants.CARBON)

    _, si_index = sample_point_away_from_edge(
        self.rng, self.atom_positions, self.nearest_neighbors
    )
    self.atomic_numbers[si_index] = constants.SILICON

  def get_atoms_in_bounds(
      self, lower_left: geometry.Point, upper_right: geometry.Point
  ) -> simulator_utils.AtomicGrid:
    """Gets the atomic grid for a particular field of view.

    Args:
      lower_left: The lower left position of the field of view.
      upper_right: The upper right position of the field of view.

    Returns:
      The observed atomic grid within the supplied bounds. Atom positions
        are normalized in [0, 1], where (0, 0) corresponds to lower_left and
        (1, 1) corresponds to upper_right.
    """
    lower_left = np.asarray(lower_left)
    upper_right = np.asarray(upper_right)

    indices_in_bounds = np.all(
        (
            (lower_left <= self.atom_positions)
            & (self.atom_positions <= upper_right)
        ),
        axis=1,
    )

    selected_atom_positions = self.atom_positions[indices_in_bounds]
    selected_atomic_numbers = self.atomic_numbers[indices_in_bounds]

    # Normalize atom positions in [0, 1]
    delta = (upper_right - lower_left).reshape(1, -1)
    selected_atom_positions = (
        selected_atom_positions - lower_left.reshape(1, -1)
    ) / delta

    return simulator_utils.AtomicGrid(
        selected_atom_positions, selected_atomic_numbers
    )

  def apply_control(
      self,
      control: simulator_utils.BeamControl,
      start_time: dt.timedelta,
      observers: Iterable[simulator_utils.SimulatorObserver] = (),
  ) -> None:
    """Simulates applying a beam exposure to the material."""
    # There is a chance that, if the dwell time is long enough, there
    # will be multiple state transitions. We simulate these transitions
    # until the dwell time is over.
    elapsed_time = dt.timedelta(seconds=0)
    while elapsed_time < control.dwell_time:
      silicon_position = self.get_silicon_position()

      # Get silicon transition probabilities.
      _, si_neighbors_index = self.nearest_neighbors.kneighbors(
          silicon_position.reshape(1, 2)
      )
      # Get the nearest neighbors, ignore the atom itself.
      si_neighbors_index = si_neighbors_index[0, 1:]

      transition_rates = self._predict_rates(
          simulator_utils.AtomicGrid(self.atom_positions, self.atomic_numbers),
          control.position,
          silicon_position,
          si_neighbors_index,
      )
      total_rate = np.sum(transition_rates)

      # The time at which the next transition takes place is modeled
      # by an exponential distribution, with the total rate as the
      # parameter. Scale is the inverse rate.
      transition_seconds = self.rng.exponential(scale=1.0 / total_rate)
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
        neighbors_prob = transition_rates / total_rate

        # Pick index to transition to.
        si_atom_index = self.rng.choice(si_neighbors_index, p=neighbors_prob)
        self.atomic_numbers[
            self.atomic_numbers == constants.SILICON
        ] = constants.CARBON
        self.atomic_numbers[si_atom_index] = constants.SILICON

        for observer in observers:
          observer.observe_transition(
              time=start_time + elapsed_time, grid=self.atomic_grid
          )

  @property
  def atomic_grid(self) -> simulator_utils.AtomicGrid:
    """Gets the atomic grid representing the current material state."""
    return simulator_utils.AtomicGrid(
        self.atom_positions, np.copy(self.atomic_numbers)
    )

  def get_silicon_position(self) -> np.ndarray:
    return self.atom_positions[
        self.atomic_numbers == constants.SILICON
    ].reshape(-1)


def get_silicon_positions(grid: simulator_utils.AtomicGrid) -> np.ndarray:
  return grid.atom_positions[grid.atomic_numbers == constants.SILICON]
