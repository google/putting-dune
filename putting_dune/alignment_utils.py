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
"""Tools for aligning and postprocessing microscope observations."""

import copy
from typing import Optional

import networkx as nx
import numpy as np
from putting_dune import constants
from putting_dune import data_utils
import scipy.spatial
import scipy.stats
from sklearn import cluster


def get_graphene_scale_factor(coordinates: np.ndarray) -> float:
  """Estimates the scale of a graphene lattice relative to standard.

  Args:
    coordinates: (num_atoms, 2) coordinate array.

  Returns:
    scale relative to standard 1.42 angstrom graphene.
  """
  distances = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
  distances = np.sort(distances, axis=-1)
  neighbor_distances = distances[:, 1:4].reshape(-1)
  estimated_scale = scipy.stats.trim_mean(
      neighbor_distances,
      0.25,
  )

  return estimated_scale / constants.CARBON_BOND_DISTANCE_ANGSTROMS


def get_offsets(
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    mask_above: float = np.inf,
) -> np.ndarray:
  """Estimates the closest-point offset between two sets of coordinates.

  Optionally, masks distances above a certain threshold to avoid outliers.

  Args:
    left_coords: (left_num_atoms, 2)
    right_coords: (right_num_atoms, 2)
    mask_above: Float, distance above which to mask out offsets.

  Returns:
    Offsets: array of length at most left_num_atoms
      (exactly equal without masking)
  """
  distances = np.linalg.norm(left_coords[:, None] - right_coords[None], axis=-1)
  closest_pairs = np.argmin(distances, -1)
  closest_distances = distances[np.arange(len(closest_pairs)), closest_pairs]
  mask = closest_distances < mask_above
  offsets = right_coords[closest_pairs] - left_coords
  offsets = offsets[mask]
  return offsets


def align_latest(
    new_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    new_classes: np.ndarray,
    reference_classes: np.ndarray,
    iterations: int = 20,
    noise_scale: float = 0.0,
    max_shift: float = 2.0,
    mask_above: float = np.inf,
    trim: float = 0.0,
    init_shift: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Calculates a vector that will align coordinates with a reference.

  Uses the iterative closest points (ICP) method with an optional simulated
  annealing to help escape local minima.

  Args:
    new_coordinates: Coordinates to align.
    reference_coordinates: Fixed coordinates to align to.
    new_classes: Classes for the input coordinates.
    reference_classes: Classes for the reference coordinates.
    iterations: How many alignment iterations to perform.
    noise_scale: What scale of noise to use if doing annealing (0 to disable).
    max_shift: The maximum shift to permit. If set too low, alignment may fail.
      If set too high, the periodic nature of graphene could permit spuriously
      large shifts.
    mask_above: Distance above which to mask out comparisons. Can prevent ICP
      from forcing odd alignments when point clouds do not fully overlap.
    trim: Optional, fraction of outliers to trim when calculating an offset.
    init_shift: Optional, an initial guess at a shift value.

  Returns:
    A shift that will align new_coordinates and reference_coordinates when
    added to new_coordinates or subtracted from reference_coordinates.
  """
  if init_shift is None:
    cum_shift = np.zeros(new_coordinates.shape[-1])
  else:
    cum_shift = init_shift
  noise_scales = np.linspace(noise_scale, 0, num=iterations)
  class_masks = [new_classes == i for i in set(new_classes)]
  reference_class_masks = [reference_classes == i for i in set(new_classes)]

  for i in range(iterations):
    noise_scale = noise_scales[i]
    noise_shift = 0 if noise_scale == 0 else np.random.normal(size=(2,))
    noise_shift = noise_scale * noise_shift
    current_coords = new_coordinates + cum_shift + noise_shift

    offsets = [
        get_offsets(
            current_coords[mask], reference_coordinates[ref_mask], mask_above
        )
        for mask, ref_mask in zip(class_masks, reference_class_masks)
    ]
    offsets = np.concatenate(offsets)
    shift = scipy.stats.trim_mean(offsets, trim, axis=0)
    cum_shift += noise_shift + shift
    norm_shift = np.linalg.norm(cum_shift)
    if norm_shift > max_shift:
      cum_shift = max_shift * cum_shift / norm_shift
    current_coords = new_coordinates + cum_shift
  return cum_shift


def clique_merge(
    coordinates: np.ndarray,
    min_distance: float = 1.0,
    max_iterations: int = 100,
    counts: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Iteratively merges groups of nearby points.

  Groups points into cliques based on min_distance, and then sets each
  clique to be a new point positioned at the average location of its members.

  Args:
    coordinates: (num_atoms, 2) ndarray of coordinates
    min_distance: Distance below which to consider two points to have an edge.
    max_iterations: maximum number of iterations to run before forcing
      termination.
    counts: Optional (num_atoms,) array of

  Returns:
    Array of merged coordinates and an array listing how many points were
    joined into each.
  """
  if counts is None:
    counts = np.ones(coordinates.shape[0])
  for _ in range(max_iterations):
    tree = scipy.spatial.cKDTree(coordinates)
    close = tree.query_pairs(r=min_distance, output_type='ndarray')

    # if no collisions just return immediately
    if not close.shape[0]:
      return coordinates, counts

    g = nx.Graph()
    g.add_nodes_from(range(len(coordinates)))
    g.add_edges_from(close)
    cliques = list(nx.find_cliques(g))

    new_coordinates = [
        np.sum(coordinates[c] * counts[c, None] / np.sum(counts[c]), axis=0)
        for c in cliques
    ]
    coordinates = np.stack(new_coordinates, 0)
    counts = np.stack([np.sum(counts[c]) for c in cliques])

  return coordinates, counts


class IterativeAlignmentFiltering:
  """Class that keeps a list of recent states and aligns new states to them.

  Attributes:
    history_length: How long a history to keep (in steps).
    alignment_iterations: How many iterations to use in the closest points
      alignment method.
    noise_scale: What noise magnitude to use in the alignment method (0 to
      disable).
    max_shift: The maximum alignment shift allowed at each iterations.
    momentum_tau: Momentum to use when initializing alignment shifts.
    merge_cutoff: Distance below which points should be merged after alignment.
    accumulate_merged: Whether to accumulate the post-merging states or the raw
      shifted inputs in the history.
    clique_merging: Whether to use a clique-based merging method instead of the
      default naive threshold-based method.
    recent_observations: List of recent observations. The core "state" of the
      object.
    recent_classes: List of atom classes accompanying the recent observations.
    classifier: SKLearn classifier to use when determining carbon classes.
    shift_momentum: The current moving average shift.
    step: How many steps have been carried out in the current trajectory.
  """

  def __init__(
      self,
      history_length: int = 10,
      alignment_iterations: int = 20,
      noise_scale: float = 0.0,
      max_shift: float = 2.0,
      momentum_tau: float = 0.9,
      merge_cutoff: float = 1.1,
      accumulate_merged: bool = False,
      clique_merging: bool = False,
  ):
    self.history_length = history_length
    self.alignment_iterations = alignment_iterations
    self.noise_scale = noise_scale
    self.max_shift = max_shift
    self.momentum_tau = momentum_tau
    self.merge_cutoff = merge_cutoff
    self.accumulate_merged = accumulate_merged
    self.clique_merging = clique_merging

    self.reset()

  def reset(self):
    self.recent_observations = list()
    self.recent_classes = list()
    self.classifier = None
    self.shift_momentum = np.zeros(2)
    self.step = 0

  def apply_shift(self, shift: np.ndarray) -> None:
    """Applies a vector shift to the aligner's state.

    The shift is assumed to be of the form (new + shift ~= old), so we convert
    this to (new ~= old - shift) to allow past shifts to be accumulated.

    Args:
      shift: (2,) ndarray containing the shift in each direction.

    Returns:
      None.
    """
    self.recent_observations = [obs - shift for obs in self.recent_observations]

  def align(self, new_observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Takes a new observation and reconciles it with the aligner's state.

    If no state exists, initializes it to the new observation and returns.
    If there is an alignment history, estimates an offset with iterative closest
    points and shifts the history by it to align it with the current
    observation, then creates a single merged grid by joining nearby points.

    Args:
      new_observation: (num_atoms, 2) ndarray of positions.

    Returns:
      Postprocessed and aligned coordinates.
      Offset by which the observation was shifted.
    """
    self.step = 1
    if not self.recent_observations:
      self.recent_observations.append(new_observation)
      self.shift_momentum = np.zeros(2)
      self.classifier = get_carbon_clusterer(new_observation)
      self.recent_classes.append(
          classify_carbon_types(new_observation, self.classifier)
      )

      joined_coords = new_observation
      offset = np.zeros((2,))

    else:
      classes = classify_carbon_types(new_observation, self.classifier)

      offset = align_latest(
          new_observation,
          np.concatenate(self.recent_observations),
          classes,
          np.concatenate(self.recent_classes),
          iterations=self.alignment_iterations,
          noise_scale=self.noise_scale,
          max_shift=self.max_shift,
          mask_above=20.0,
          init_shift=copy.deepcopy(self.shift_momentum),
      )
      self.apply_shift(offset)

      self.shift_momentum *= self.momentum_tau
      self.shift_momentum += offset * (1 - self.momentum_tau)

      to_merge = list(self.recent_observations) + [new_observation]
      if self.clique_merging:
        to_merge = np.concatenate(to_merge, 0)
        joined_coords, _ = clique_merge(to_merge, self.merge_cutoff)
      else:
        joined_coords, _ = naive_merge(to_merge, self.merge_cutoff)
      if self.accumulate_merged:
        self.recent_observations.append(joined_coords)
        self.recent_classes.append(
            classify_carbon_types(joined_coords, self.classifier)
        )
      else:
        self.recent_observations.append(new_observation)
        self.recent_classes.append(classes)

      if len(self.recent_observations) > self.history_length:
        to_cut = len(self.recent_observations) - self.history_length
        self.recent_observations = self.recent_observations[to_cut:]
        self.recent_classes = self.recent_classes[to_cut:]

    return joined_coords, offset


def naive_merge(
    coordinates: np.ndarray, cutoff: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
  """Merges lists of coordinates based on simple proximity.

  Args:
    coordinates: List of arrays of coordinates. Arrays are all of shape
      (num_atoms_i, n_dim), and may have different lengths.
    cutoff: Float, distance in L2 Norm within which to join atoms.

  Returns:
    positions: array of shape (num_atoms, n_dim) containing the output of the
      merging process.
    counts: array of shape (num_atoms,), indicating how many raw points were
      mapped to each merged point.
  """
  coordinates = [c for c in coordinates if c.shape[0]]
  positions = coordinates[0]
  counts = np.ones(coordinates[0].shape[:1])

  for m in coordinates[1:]:
    new_positions = []
    distances = ((m[None] - positions[:, None]) ** 2).sum(-1) ** 0.5
    closest = distances.argmin(0)
    for current, target in enumerate(closest):
      if distances[target, current] < cutoff:
        positions[target] = (
            positions[target] * counts[target] + m[current]
        ) / (counts[target] + 1)
        counts[target] += 1
      else:
        new_positions.append(m[current])

    if new_positions:
      new_positions = np.stack(new_positions, 0)
      new_counts = np.array([1] * len(new_positions))
      positions = np.concatenate([positions, new_positions], 0)
      counts = np.concatenate([counts, new_counts], 0)

  return positions, counts


def get_carbon_clusterer(grid: np.ndarray) -> cluster.KMeans:
  """Trains a K-Means model to classify carbon atoms based on bond angles.

  Args:
    grid: (num_atoms, 2) ndarray of atom positions.

  Returns:
    K-Means-based atom classifier.
  """
  grid = grid[:, :2] - grid[:, :2].mean(0, keepdims=True)

  distances = np.linalg.norm(
      grid[None] - grid[:, None], axis=-1, keepdims=False
  )

  # Closest neighbor is the atom itself, so get neighbors 2-4.
  neighbors = np.argsort(distances, axis=-1)[:, 1:4]

  neighbor_positions = grid[neighbors]
  relative_neighbor_positions = neighbor_positions - grid[:, None]

  angles = np.stack(
      [data_utils.get_neighbor_angles(x) for x in relative_neighbor_positions]
  )
  angles = np.sort(angles, axis=-1)

  clusters = cluster.KMeans(2)
  clusters.fit(angles)
  return clusters


def classify_carbon_types(
    grid: np.ndarray, clusters: cluster.KMeans
) -> np.ndarray:
  """Classifies graphene carbon atoms based on their angles to their neighbors.

  Args:
    grid: (num_atoms, 2) ndarray of atom positions.
    clusters: sklearn or other classifier implementing a predict() method.

  Returns:
    (num_atoms,) array of atom class labels.
  """
  grid = grid[:, :2] - grid[:, :2].mean(0, keepdims=True)

  distances = np.linalg.norm(
      grid[None] - grid[:, None], axis=-1, keepdims=False
  )
  distances = distances + np.eye(distances.shape[0]) * 1000
  neighbors = np.argsort(distances, axis=-1)[:, :3]
  neighbor_dists = np.sort(distances, axis=-1)

  assert neighbor_dists.shape[1] >= 3
  neighbor_positions = grid[neighbors]
  relative_neighbor_positions = neighbor_positions - grid[:, None]

  angles = np.stack(
      [data_utils.get_neighbor_angles(x) for x in relative_neighbor_positions]
  )
  classes = clusters.predict(angles)
  classes = propagate_graphene_classes(classes, grid)

  return classes


def propagate_graphene_classes(
    classes: np.ndarray, grid: np.ndarray
) -> np.ndarray:
  """Propagates graphene carbon atom classes to the edges of a sheet.

  Takes advantage of the fact that bond orientation is a two-coloring of the
  graphene sheet to re-assign labels to nodes with insufficiently many
  neighbors (e.g., those on the edge of a scan).

  Args:
    classes: (num_atoms,) ndarray of atom labels.
    grid: (num_atoms, 2) ndarray of atom positions.

  Returns:
    array of updated classes.
    array indicating which iteration atoms were updated on.
  """
  grid = grid[:, :2] - grid[:, :2].mean(0, keepdims=True)

  distances = np.linalg.norm(
      grid[None] - grid[:, None], axis=-1, keepdims=False
  )
  distances = distances + np.eye(distances.shape[0]) * 1000
  neighbor_dists = np.sort(distances, axis=-1)

  neighbor_mask = distances < neighbor_dists[:, :3].mean() * 1.1
  degrees = neighbor_mask.sum(-1)

  classified = degrees >= 3

  frontiers = np.zeros(classified.shape)
  i = 0
  while True:
    i += 1
    filtered_neighbor_mask = copy.deepcopy(neighbor_mask)
    filtered_neighbor_mask[:, ~classified] = False
    frontier = ~classified & (filtered_neighbor_mask.sum(-1) >= 1)
    frontiers[frontier] = i
    if frontier.sum() == 0:
      frontiers[~classified] = i
      return classes
    neighbor_classes = filtered_neighbor_mask[frontier] * classes[None]
    num_neighbors = filtered_neighbor_mask[frontier].sum(-1)
    new_classes = 1 - neighbor_classes.sum(-1) / num_neighbors
    classes[frontier] = np.nan_to_num(new_classes, True, 0, 0, 0)
    classified[frontier] = True
