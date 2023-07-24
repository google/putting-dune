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

"""Tools for aligning and postprocessing microscope observations."""

import collections
import copy
import functools
import io
import typing
from typing import Any, Deque, Optional, Sequence, Tuple
import urllib
import zipfile

import cv2
from etils import epath
import networkx as nx
import numpy as np
from putting_dune import constants
from putting_dune import geometry
from putting_dune import microscope_utils
import scipy.spatial
import scipy.stats
from skimage import exposure
from sklearn import cluster
import tensorflow as tf


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
    init_shift: Optional[np.ndarray] = np.zeros((2,)),
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
    cumulative_drift = np.zeros(new_coordinates.shape[-1])
  else:
    cumulative_drift = init_shift
  noise_scales = np.linspace(noise_scale, 0, num=iterations)
  class_masks = [new_classes == i for i in set(new_classes)]
  reference_class_masks = [reference_classes == i for i in set(new_classes)]

  for i in range(iterations):
    noise_scale = noise_scales[i]
    noise = 0 if noise_scale == 0 else np.random.normal(size=(2,)) * noise_scale
    current_coords = new_coordinates + cumulative_drift + noise

    offsets = [
        get_offsets(
            current_coords[mask], reference_coordinates[ref_mask], mask_above
        )
        for mask, ref_mask in zip(class_masks, reference_class_masks)
    ]
    offsets = np.concatenate(offsets)

    if trim > 0:
      distances = np.linalg.norm(offsets, axis=-1)
      sorted_distances = np.argsort(distances)
      offsets = offsets[sorted_distances[: int((1 - trim) * len(offsets))]]

    offset = offsets.mean(axis=0)
    cumulative_drift += noise + offset
    drift_norm = np.linalg.norm(cumulative_drift)
    if drift_norm > max_shift:
      cumulative_drift = max_shift * cumulative_drift / drift_norm
    current_coords = new_coordinates + cumulative_drift
  return cumulative_drift


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


def pad_and_crop_images_by_fov(
    image: np.ndarray,
    original_fov: microscope_utils.MicroscopeFieldOfView,
    new_fov: microscope_utils.MicroscopeFieldOfView,
) -> np.ndarray:
  """Extracts a subimage corresponding to a new FOV from an old observation.

  Args:
    image: Image (h, w, c) array.
    original_fov: FOV describing the original image.
    new_fov: FOV to use to extract a subimage.

  Returns:
    A sliced version of `image`.
  """

  if image.ndim == 2:
    image = np.expand_dims(image, -1)

  original_lower_left = np.array(original_fov.lower_left)
  new_lower_left = np.array(new_fov.lower_left)
  original_upper_right = np.array(original_fov.upper_right)
  new_upper_right = np.array(new_fov.upper_right)
  original_scale = original_upper_right - original_lower_left
  new_scale = new_upper_right - new_lower_left
  resize_factor = original_scale / new_scale

  output_shape = image.shape
  padding_shape = image.shape
  array_image_shape = np.array(output_shape)[:-1]
  array_padding_shape = np.array(padding_shape)[:-1]

  if (resize_factor != 1).any():
    new_size = np.array(image.shape[:-1]) * resize_factor
    new_size = tuple(np.round(new_size).astype(np.int32))
    resized_image = tf.image.resize(
        image,
        new_size,
        method='nearest',
    )
  else:
    resized_image = image

  padded_image = np.pad(
      resized_image,
      (
          (padding_shape[0], padding_shape[0]),
          (padding_shape[1], padding_shape[1]),
          (0, 0),
      ),
      mode='constant',
  )

  # The upper-left is actually the privileged point in images, so we find shift
  # relative to it. Since FOV is square it's just the X from LL and Y from UR.
  x_shift = new_lower_left[0] - original_lower_left[0]
  y_shift = new_upper_right[1] - original_upper_right[1]

  # In images the y-direction is the first axis, and the x-direction the second,
  # and the "y" direction is reversed.
  shift = np.array([-y_shift, x_shift])

  # Convert angstroms to pixels in the original image frame
  shift = shift * array_image_shape / new_scale

  # Start the slice at the beginning of the real image (after padding).
  slice_start = shift + array_padding_shape

  # We don't want to have a negative slice start, or a slice start that would
  # go out-of-bounds, so we clip the slice starts accordingly.
  # If a slice gets clipped, it should produce an entirely-padded image.
  slice_start[0] = np.clip(
      slice_start[0], 0, padded_image.shape[0] - output_shape[0]
  )
  slice_start[1] = np.clip(
      slice_start[1], 0, padded_image.shape[1] - output_shape[1]
  )

  slice_start = np.round(slice_start).astype(np.int32) + np.array([0, 0])
  sliced_image = padded_image[
      slice_start[0] : slice_start[0] + output_shape[0],
      slice_start[1] : slice_start[1] + output_shape[1],
  ]

  return sliced_image


class ImageAligner:
  """A wrapper that applies a pretrained image alignment model.

  Attributes:
    image_history: Deque of previous images. Created when reset is called.
    fov_history: Deque of previous FOVs. Created when reset is called.
    model_path: Path to saved model weights.
    hybrid: Whether to use a grid-based aligner as a postprocessing step.
    postprocessing_aligner: The postprocessing aligner (if in use).
    adaptive_normalization: Whether to use equalize_adaptist to smartly
      renormalize images prior to network application.
    model: The loaded model.
    history_length: Length of sequences to process. Int.
    needs_reset: Whether the model will be reset at its next application.
  """

  # TensorFlow saved model path
  model_path: epath.Path

  image_history: Deque[np.ndarray]
  fov_history: Deque[microscope_utils.MicroscopeFieldOfView]

  model: Any

  adaptive_normalization: bool = True

  hybrid: bool = False
  postprocessing_aligner: Optional['IterativeAlignmentFiltering'] = None

  needs_reset: bool = True

  history_length: int = 5

  def reset(
      self, history_length: int = 5, example_image=np.zeros((512, 512, 1))
  ):
    """Resets the internal history of the aligner.

    Args:
      history_length: The history length in use.
      example_image: An image to pad the history with.
    """
    self.image_history = collections.deque(maxlen=history_length - 1)
    self.fov_history = collections.deque(maxlen=history_length - 1)

    dummy_image = np.zeros_like(example_image)
    for _ in range(history_length):
      self.image_history.append(dummy_image)
      self.fov_history.append(
          microscope_utils.MicroscopeFieldOfView(
              geometry.Point(0, 0), geometry.Point(20, 20)
          )
      )

    if self.hybrid:
      self.postprocessing_aligner.reset()

    self.needs_reset = False

  def __init__(self, model_path: epath.Path, hybrid: bool = False):
    self.model_path = model_path
    self.hybrid = hybrid

    if self.hybrid:
      self.postprocessing_aligner = IterativeAlignmentFiltering(
          history_length=1,
          alignment_iterations=1,
          noise_scale=0.0,
          max_shift=constants.CARBON_BOND_DISTANCE_ANGSTROMS / 2,
          merge_cutoff=constants.CARBON_BOND_DISTANCE_ANGSTROMS / 2,
          accumulate_merged=False,
          clique_merging=True,
          trim=0.5,
      )

  @functools.cached_property
  def model(self) -> Any:
    return tf.saved_model.load(self.model_path)

  @classmethod
  def compute_centroids(cls, classes, class_index, erode_iters=1):
    """Finds centroids in an image based on class predictions.

    Args:
      classes: One-hot class predictions (from argmax or thresholding).
      class_index: The index of the class to select centroids for.
      erode_iters: How much erosion to do (to eliminate small blobs).

    Returns:
      List of centroids representing detected atoms.
    """
    mask = classes.copy()
    mask[classes != class_index] = 0
    mask = (mask * 255).astype(np.uint8)
    if erode_iters:
      mask = cv2.erode(mask, np.ones((2, 2)), iterations=erode_iters)
    _, contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )

    centroids = []
    for contour in contours:
      # calculate moments for each contour
      m = cv2.moments(contour)

      if m['m00'] != 0:
        c_x = int(m['m10'] / m['m00'])
        c_y = int(m['m01'] / m['m00'])
      else:
        c_x, c_y = 0, 0

      # Normalize so that 0,0 is bottom left and 1,1 is top right
      centroids.append((c_x / classes.shape[0], 1.0 - c_y / classes.shape[1]))
    return centroids

  @classmethod
  def process_detection_predictions(
      cls,
      probs: np.ndarray,
      buffer_width: float = 0.05,
  ) -> microscope_utils.AtomicGridMicroscopeFrame:
    """Processes predicted atom classes to produce an AtomicGrid.

    Args:
      probs: Predicted per-pixel probabilities of each atom class.
      buffer_width: Size of buffer around image edges to ignore when performing
        atom detection (float, in image fractions). 0.5 ignores entire image.

    Returns:
      An AtomicGrid containing the extracted atoms.
    """
    classes = np.argmax(probs, axis=-1)
    carbon_centroids = ImageAligner.compute_centroids(
        classes,
        1,
        erode_iters=1,
    )
    silicon_centroids = ImageAligner.compute_centroids(
        classes, 2, erode_iters=3
    )
    carbon_centroids = np.array(carbon_centroids)
    silicon_centroids = np.array(silicon_centroids)
    if not silicon_centroids.size:
      silicon_centroids = np.zeros((0, 2))
    if not carbon_centroids.size:
      carbon_centroids = np.zeros((0, 2))

    # Construct an array of atomic numbers for both atoms
    carbon_atomic_numbers = np.array([constants.CARBON] * len(carbon_centroids))
    silicon_atomic_numbers = np.array(
        [constants.SILICON] * len(silicon_centroids)
    )

    atom_positions = np.concatenate([carbon_centroids, silicon_centroids])
    atomic_numbers = np.concatenate(
        [carbon_atomic_numbers, silicon_atomic_numbers]
    ).astype(np.int32)

    in_bounds = (atom_positions > buffer_width).all(-1) & (
        atom_positions < (1 - buffer_width)
    ).all(-1)

    atom_positions = atom_positions[in_bounds]
    atomic_numbers = atomic_numbers[in_bounds]

    grid = microscope_utils.AtomicGrid(
        atom_positions=atom_positions, atomic_numbers=atomic_numbers
    )
    return microscope_utils.AtomicGridMicroscopeFrame(grid)

  @classmethod
  def from_url(
      cls,
      url: str = 'https://storage.googleapis.com/spr_data_bucket_public/alignment/20230403-image-aligner.zip',
      workdir: Optional[str] = None,
      reload: bool = False,
      **kwargs,
  ) -> 'ImageAligner':
    """Construct model from URL.

    Args:
      url: Model URL, expected to be a zip file.
      workdir: Optional, locatioon (e.g., temp dir) to extract weights to.
      reload: Optional, whether to force-redownload aligner.
      **kwargs: Optional arguments for the ImageAligner.

    Returns:
      ImageAligner instance.
    """
    # Model save path on disk
    if workdir is None:
      workdir = epath.resource_path('putting_dune')
    model_path = epath.Path(workdir) / 'model_weights' / 'image-alignment-model'
    if not model_path.exists() or reload:
      model_path.mkdir(parents=True, exist_ok=True)
      with urllib.request.urlopen(url) as request:
        with zipfile.ZipFile(io.BytesIO(request.read())) as model_zip:
          model_zip.extractall(model_path.parent)

    return ImageAligner(model_path=model_path, **kwargs)

  def __call__(
      self,
      image: np.ndarray,
      fov: microscope_utils.MicroscopeFieldOfView,
      grid: Optional[microscope_utils.AtomicGridMicroscopeFrame] = None,
      time_index: int = -1,
  ) -> Tuple[
      microscope_utils.AtomicGridMicroscopeFrame, Any, Any,
  ]:
    """Performs alignment and atom detection on an observation.

    Args:
      image: New image observation. Does not need to be normalized.
      fov: FOV describing the (estimated) bounds of this image on the material.
      grid: Optionally, a grid describing the atoms in the image, to replace the
        grid estimated by this class.
      time_index: Which time index to predict for. Set values other than -1 to
        perform smoothing.

    Returns:
      The AtomicGrid estimated by the detection, the estimated drift, and a
      label mask of the image.
    """
    # Perform pre-processing on the image.
    # This involves:
    #   1. Downsampling the image to 512 x 512
    #   2. Optionally performing adaptive equalization.
    #   3. Normalizing the image to be [0, 1]
    if image.ndim == 2:
      image = np.expand_dims(image, -1)

    image = image.astype(np.float32)

    if self.adaptive_normalization:
      image = exposure.equalize_adapthist(image)

    image: tf.Tensor = tf.image.resize(
        image,
        (512, 512),
        method='nearest',
    )
    image_min, image_max = tf.reduce_min(image), tf.reduce_max(image)
    image = (image - image_min) / (image_max - image_min)
    if self.needs_reset:
      self.reset(history_length=self.history_length, example_image=image)

    padded_images = [
        pad_and_crop_images_by_fov(old_image, old_fov, fov)
        for old_image, old_fov in zip(self.image_history, self.fov_history)
    ]
    padded_images.append(image)

    framestack = np.concatenate(padded_images, -1)

    # Feed image through the detection model to get the logits
    model_outputs = self.model.signatures['serving_default'](
        image=framestack,
    )
    logits = model_outputs['output_0']
    pred_drift = model_outputs['output_1'].numpy()
    pred_drift = np.reshape(pred_drift, (*pred_drift.shape[:-1], -1, 2))
    pred_drift = pred_drift[..., time_index, :]
    logits = tf.reshape(logits, (*logits.shape[:-1], -1, 3))
    logits = logits[..., time_index, :]
    probs = tf.nn.softmax(logits).numpy()
    if grid is None:
      grid = ImageAligner.process_detection_predictions(probs)

    self.image_history.append(image)
    self.fov_history.append(fov)

    if self.hybrid:
      try:
        shifted_fov = fov.shift(geometry.Point(*-pred_drift))
        material_grid = shifted_fov.microscope_frame_to_material_frame(grid)
        postprocessing_aligner = typing.cast(
            IterativeAlignmentFiltering, self.postprocessing_aligner
        )
        postprocessed_grid, postprocessed_drift = postprocessing_aligner(
            material_grid
        )
        pred_drift = pred_drift + postprocessed_drift
        shifted_fov = fov.shift(geometry.Point(*-pred_drift))
        grid = shifted_fov.material_frame_to_microscope_frame(
            postprocessed_grid
        )
      except Exception as e:  # pylint: disable=broad-except
        print('Postprocessing failed; {}'.format(str(e)))
        self.postprocessing_aligner.reset()

    return grid, pred_drift, probs


class IterativeAlignmentFiltering:
  """Class that keeps a list of recent states and aligns new states to them.

  Attributes:
    history_length: How long a history to keep (in steps).
    alignment_iterations: How many iterations to use in the closest points
      alignment method.
    noise_scale: What noise magnitude to use in the alignment method (0 to
      disable).
    max_shift: The maximum alignment shift allowed at each iterations.
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
    trim: Trim fraction to use in the mean for estimating shifts. 0.25 yields
      IQM, 0 yields standard mean, 0.5 yields median.
  """

  def __init__(
      self,
      history_length: int = 10,
      alignment_iterations: int = 20,
      noise_scale: float = 0.0,
      max_shift: float = 2.0,
      merge_cutoff: float = 1.1,
      accumulate_merged: bool = False,
      clique_merging: bool = False,
      trim: float = 0,
  ):
    self.history_length = history_length
    self.alignment_iterations = alignment_iterations
    self.noise_scale = noise_scale
    self.max_shift = max_shift
    self.merge_cutoff = merge_cutoff
    self.accumulate_merged = accumulate_merged
    self.clique_merging = clique_merging
    self.trim = trim

    self.reset()

  def reset(self):
    self.recent_observations = list()
    self.recent_classes = list()
    self.classifier = None
    self.step = 0

  def apply_shift(self, shift: np.ndarray) -> None:
    """Applies a vector shift to the aligner's state.

    The shift is assumed to be of the form (old + shift ~= new), so we convert
    this to (new ~= old + shift) to allow past shifts to be accumulated.

    Args:
      shift: (2,) ndarray containing the shift in each direction.

    Returns:
      None.
    """
    self.recent_observations = [obs + shift for obs in self.recent_observations]

  def __call__(
      self,
      new_observation: microscope_utils.AtomicGridMaterialFrame,
  ) -> tuple[microscope_utils.AtomicGridMaterialFrame, np.ndarray]:
    """Takes a new observation and reconciles it with the aligner's state.

    If no state exists, initializes it to the new observation and returns.
    If there is an alignment history, estimates an offset with iterative closest
    points and shifts the history by it to align it with the current
    observation, then creates a single merged grid by joining nearby points.

    Args:
      new_observation: AtomicGrid, the most recent observation in the chain.

    Returns:
      Postprocessed and aligned atomic grid.
      Offset by which the observation was shifted.
    """
    self.step = 1
    if not self.recent_observations:
      self.recent_observations.append(new_observation.atom_positions)
      self.classifier = get_lattice_clusterer(new_observation.atom_positions)
      self.recent_classes.append(
          classify_lattice_types(
              new_observation.atom_positions, self.classifier
          )
      )
      merged_grid = new_observation
      drift = np.zeros((2,))

    else:
      classes = classify_lattice_types(
          new_observation.atom_positions, self.classifier
      )

      drift = align_latest(
          new_observation.atom_positions,
          np.concatenate(self.recent_observations),
          classes,
          np.concatenate(self.recent_classes),
          iterations=self.alignment_iterations,
          noise_scale=self.noise_scale,
          max_shift=self.max_shift,
          mask_above=2.0,
          init_shift=np.zeros((2,)),
          trim=self.trim,
      )

      new_observation = new_observation.shift(drift)

      to_merge = list(self.recent_observations) + [
          new_observation.atom_positions
      ]
      if self.clique_merging:
        to_merge = np.concatenate(to_merge, 0)
        joined_coords, _ = clique_merge(to_merge, self.merge_cutoff)
      else:
        joined_coords, _ = naive_merge(to_merge, self.merge_cutoff)
      if self.accumulate_merged:
        self.recent_observations.append(joined_coords)
        self.recent_classes.append(
            classify_lattice_types(joined_coords, self.classifier)
        )
      else:
        self.recent_observations.append(new_observation.atom_positions)
        self.recent_classes.append(classes)

      if len(self.recent_observations) > self.history_length:
        to_cut = len(self.recent_observations) - self.history_length
        self.recent_observations = self.recent_observations[to_cut:]
        self.recent_classes = self.recent_classes[to_cut:]

      # to propagate atomic numbers
      aligned_atomic_numbers = propagate_atomic_numbers(
          new_observation.atom_positions,
          joined_coords,
          new_observation.atomic_numbers,
      )
      merged_grid = microscope_utils.AtomicGrid(
          joined_coords, aligned_atomic_numbers
      )
      merged_grid = microscope_utils.AtomicGridMaterialFrame(merged_grid)
    return merged_grid, -drift


def propagate_atomic_numbers(
    original_atom_positions: np.ndarray,
    merged_atom_positions: np.ndarray,
    original_atomic_numbers: np.ndarray,
    new_atomic_numbers: Optional[np.ndarray] = None,
    default_atomic_number: int = 6,
    threshold: float = 0.8,
) -> np.ndarray:
  """Propagates atomic numbers from one observation to another related one.

  Assigns the identity of the each atom in the original grid to its closest
  corresponding atom in the new grid, provided that this distance is below
  a certain threshold. All atoms in the new grid that do not have an analog
  in the original grid are assigned a default number, unless initial atomic
  numbers were specified.
  Args:
    original_atom_positions: (num_old_atoms, 2) array of old atom positions.
    merged_atom_positions: (num_new_atoms, 2) array of new atom positions.
    original_atomic_numbers: (num_old_atoms,) array of atomic numbers.
    new_atomic_numbers: (num_new_atoms,) array of atomic numbers or None.
    default_atomic_number: Default atomic number to assign if new_atomic_numbers
      was None. Defaults to 6 (carbon).
    threshold: Maximum distance between analogous atoms to transfer identities.

  Returns:
    (num_new_atoms,) array of updated atomic numbers.
  """
  distances = np.linalg.norm(
      original_atom_positions[:, None] - merged_atom_positions[None], axis=-1
  )

  # Find the left and right sides to transfer atomic numbers.
  closest_neighbors = distances.argmin(-1)
  original_atoms = np.arange(original_atomic_numbers.shape[0])

  # Filter out pairs that are outside our threshold
  closest_neighbors = closest_neighbors[distances.min(-1) < threshold]
  original_atoms = original_atoms[distances.min(-1) < threshold]

  if new_atomic_numbers is None:
    new_atomic_numbers = np.zeros(
        merged_atom_positions.shape[0], dtype=original_atomic_numbers.dtype
    )
    new_atomic_numbers.fill(default_atomic_number)

  new_atomic_numbers[closest_neighbors] = original_atomic_numbers[
      original_atoms
  ]
  return new_atomic_numbers


def naive_merge(
    coordinates: Sequence[np.ndarray], cutoff: float = 0.7
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


def get_lattice_clusterer(grid: np.ndarray) -> cluster.KMeans:
  """Trains a K-Means model to classify lattice atoms based on bond angles.

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
      [geometry.get_angles(x) for x in relative_neighbor_positions]
  )
  angles = np.sort(angles, axis=-1)

  clusters = cluster.KMeans(2)
  clusters.fit(angles)
  return clusters


def classify_lattice_types(
    grid: np.ndarray, clusters: cluster.KMeans
) -> np.ndarray:
  """Classifies graphene lattice atoms based on their angles to their neighbors.

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
      [geometry.get_angles(x) for x in relative_neighbor_positions]
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
