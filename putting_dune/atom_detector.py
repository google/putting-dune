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

"""Atom detection model."""
import dataclasses
import functools
import io
from typing import Any, List, Tuple
import urllib
import zipfile

import cv2
from etils import epath
import numpy as np
import numpy.typing as npt
from putting_dune import constants
from putting_dune import microscope_utils
import tensorflow as tf


def compute_centroids(
    image: npt.NDArray,
    value: int,
    threshold_value: int,
) -> List[Tuple[float, float]]:
  """Computes the centroids in an image.

  Args:
    image: Preprocessed image.
    value: Value in the image to mask out.
    threshold_value: Threshold level to apply.

  Returns:
    image: Output image where circles have been drawn over centroid locations.
    centroids: List of points denoting (x, y) centroid locations.
  """
  # Mask out value
  masked = np.zeros_like(image)
  masked[np.where(image == value)] = 1

  # Perform distance transform on the masked image
  dists = cv2.distanceTransform(masked, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
  dists = (dists * 255).astype(np.uint8)

  # Threshold the distances by the threshold value.
  _, dists = cv2.threshold(dists, threshold_value, 255, cv2.THRESH_BINARY)

  # Detect contours from the distance transform
  contours, *_ = cv2.findContours(dists, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Compute centroids
  # Repurposed from:
  # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
  centroids = []
  for contour in contours:
    # calculate moments for each contour
    m = cv2.moments(contour)

    if m["m00"] != 0:
      c_x = int(m["m10"] / m["m00"])
      c_y = int(m["m01"] / m["m00"])
    else:
      c_x, c_y = 0, 0

    # Normalize centroid values so that 0,0 is bottom left and 1,1 is top right
    centroids.append((c_x / 256, 1.0 - c_y / 256))

  return centroids


@dataclasses.dataclass(frozen=True)
class AtomDetector:
  """Atom detection model."""

  # TensorFlow saved model path
  model_path: epath.Path

  @classmethod
  def from_url(
      cls,
      url: str = "https://storage.googleapis.com/spr_data_bucket_public/detectors/atom-detection-model.zip",
  ) -> "AtomDetector":
    """Construct model from URL.

    Args:
      url: Model URL, expected to be a zip file.

    Returns:
      AtomDetector instance.
    """
    # pylint: disable=unreachable
    # Model save path on disk
    model_path = (
        epath.resource_path("putting_dune")
        / "model_weights"
        / "atom-detection-model"
    )

    if not model_path.exists():
      model_path.mkdir(parents=True)
      with urllib.request.urlopen(url) as request:
        with zipfile.ZipFile(io.BytesIO(request.read())) as model_zip:
          model_zip.extractall(model_path.parent)

    return AtomDetector(model_path=model_path)
    # pylint: enable=unreachable

  @functools.cached_property
  def model(self) -> Any:
    return tf.saved_model.load(self.model_path)

  def __call__(
      self, image: npt.NDArray
  ) -> microscope_utils.AtomicGridMicroscopeFrame:
    # Perform pre-processing on the image.
    # This involves:
    #   1. Downsampling the image to 256 x 256
    #   2. Normalizing the image to be [0, 1]
    image: tf.Tensor = tf.image.resize(
        image,
        (256, 256),
        method="nearest",
    )
    image_min, image_max = tf.reduce_min(image), tf.reduce_max(image)
    image = (image - image_min) / (image_max - image_min)

    # Feed image through the detection model to get the logits
    logits = self.model.signatures["serving_default"](
        preprocessed_inputs_0=image,
    )["output_0"]
    probs = tf.nn.softmax(logits)

    # Carbon is 1-indexed
    carbon_probs = np.asarray(probs[:, :, 1])
    # Apply a minimum threshold to the carbon layer
    _, carbon_probs = cv2.threshold(carbon_probs, 0.025, 1.0, cv2.THRESH_BINARY)

    # Dilate and erode the carbon detections to use for a mask
    dilated_carbon_probs = cv2.dilate(
        carbon_probs, np.ones((2, 2)), iterations=4
    )
    dilated_carbon_probs = cv2.erode(
        dilated_carbon_probs, np.ones((2, 2)), iterations=2
    )

    # Silicon is 2-indexed
    silicon_probs = np.asarray(probs[:, :, 2])
    # Mask the silicon segmentation with the dilated carbon segmentation.

    # Mask out any silicon that overlaps with likely carbon
    masked_silicon_probs = cv2.bitwise_xor(silicon_probs, dilated_carbon_probs)

    # Find the centroids for both the carbon and silicon layers
    carbon_centroids = compute_centroids(carbon_probs.astype(np.uint8), 1, 25)
    silicon_centroids = compute_centroids(
        masked_silicon_probs.astype(np.uint8), 1, 140
    )

    # Construct an array of atomic numbers for both atoms
    carbon_atomic_numbers = np.array([constants.CARBON] * len(carbon_centroids))
    silicon_atomic_numbers = np.array(
        [constants.SILICON] * len(silicon_centroids)
    )

    return microscope_utils.AtomicGridMicroscopeFrame(
        microscope_utils.AtomicGrid(
            atom_positions=np.concatenate(
                [carbon_centroids, silicon_centroids]
            ),
            atomic_numbers=np.concatenate(
                [carbon_atomic_numbers, silicon_atomic_numbers]
            ),
        )
    )
