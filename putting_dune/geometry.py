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
"""Geometry manipulations."""

from jax import numpy as jnp
import numpy as np
from shapely import geometry as shapely_geometry

# Since we import geometry from shapely to get a point, we create
# an alias here to avoid a name overlap between our geometry module
# and shapelys.
Point = shapely_geometry.Point


def get_angles(coordinates: np.ndarray) -> np.ndarray:
  """Gets the angles to a set of coordinates from the origin.

  The angle is calculated relative to the origin, with an angle of 0
  corresponding to the vector directly to the right (1, 0), and
  increasing angle in the counter-clockwise direction.

  Args:
    coordinates: A numpy array with shape (n, 2), where n is the number of
      coordinates to get the angle of.

  Returns:
    A numpy array of shape (n,) containing the angle of each coordinate.
  """
  angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
  return angles


def rotate_coordinates(coord: np.ndarray, theta: float):
  """Rotates coordinates by theta radians counter-clockwise.

  Args:
    coord: A numpy array with shape (n, 2).
    theta: The angle to rotate the coordinates, in radians.

  Returns:
    A numpy array with shape (n, 2) containing the rotated coordinates.
  """
  # Since we are right-multiplying, we use the transpose of the
  # common 2D rotation matrix. This saves a couple of transpositions.
  rotation_matrix = np.asarray(
      [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
  )
  return coord @ rotation_matrix


def jnp_rotate_coordinates(coord: np.ndarray, theta: float):
  """Rotates coordinates by theta radians counter-clockswise, jax-friendly.

  Args:
    coord: A numpy array with shape (n, 2).
    theta: The angle to rotate the coordinates, in radians.

  Returns:
    A numpy array with shape (n, 2) containing the rotated coordinates.
  """
  # Since we are right-multiplying, we use the transpose of the
  # common 2D rotation matrix. This saves a couple of transpositions.
  rotation_matrix = jnp.asarray(
      [[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]]
  )
  return coord @ rotation_matrix
