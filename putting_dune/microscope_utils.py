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
"""Common utilities for microscope components."""

import dataclasses
import datetime as dt
import typing
from typing import Optional, Tuple

import numpy as np
from putting_dune import geometry



@dataclasses.dataclass(frozen=True)
class AtomicGrid:
  """A grid of atoms."""

  atom_positions: np.ndarray
  atomic_numbers: np.ndarray



@dataclasses.dataclass(frozen=True)
class BeamControl:
  """Specifications to control the microscope beam for one step.

  Attributes:
    position: Point describing the beam position.
    dwell_time: Beam dwell time.
  """

  position: geometry.Point
  dwell_time: dt.timedelta



@dataclasses.dataclass(frozen=True)
class MicroscopeFieldOfView:
  """A class that tracks where the microscope is scanning at any given time.

  Agents interact with the microscope by specifying controls in the
  microscopes field of view, with (0.0, 0.0) corresponding to bottom left,
  and (1.0, 1.0) corresponding to top right. However, the underlying
  material has its own coordinate system in angstroms, since it extends
  beyond the field of view of the microscope. Thus, this class also
  contains convenience functions for moving between these coordinate spaces.
  """

  lower_left: geometry.Point
  upper_right: geometry.Point

  @property
  def offset(self) -> geometry.Point:
    return geometry.Point(
        (np.asarray(self.lower_left) + np.asarray(self.upper_right)) / 2
    )

  @property
  def width(self) -> float:
    return self.upper_right.x - self.lower_left.x

  @property
  def height(self) -> float:
    return self.upper_right.y - self.lower_left.y

  @typing.overload
  def microscope_frame_to_material_frame(self, point: np.ndarray) -> np.ndarray:
    ...

  @typing.overload
  def microscope_frame_to_material_frame(
      self, point: geometry.Point
  ) -> geometry.Point:
    ...

  @typing.overload
  def microscope_frame_to_material_frame(self, point: AtomicGrid) -> AtomicGrid:
    ...

  def microscope_frame_to_material_frame(self, point):
    """Converts a point from microscope frame to material frame."""
    lower_left = np.asarray(self.lower_left).reshape(1, 2)
    upper_right = np.asarray(self.upper_right).reshape(1, 2)
    scale = upper_right - lower_left

    if isinstance(point, AtomicGrid):
      grid = typing.cast(AtomicGrid, point)
      return AtomicGrid(
          grid.atom_positions * scale + lower_left, grid.atomic_numbers
      )
    elif isinstance(point, np.ndarray):
      point = typing.cast(np.ndarray, point)

      return_shape = (2,) if point.ndim == 1 else (-1, 2)
      return (point.reshape(-1, 2) * scale + lower_left).reshape(return_shape)
    elif isinstance(point, geometry.Point):
      point = typing.cast(geometry.Point, point)
      return geometry.Point((
          point.x * scale[0, 0] + lower_left[0, 0],
          point.y * scale[0, 1] + lower_left[0, 1],
      ))
    raise NotImplementedError(f'Point of type {type(point)} is not supported.')

  @typing.overload
  def material_frame_to_microscope_frame(self, point: np.ndarray) -> np.ndarray:
    ...

  @typing.overload
  def material_frame_to_microscope_frame(
      self, point: geometry.Point
  ) -> geometry.Point:
    ...

  @typing.overload
  def material_frame_to_microscope_frame(self, point: AtomicGrid) -> AtomicGrid:
    ...

  def material_frame_to_microscope_frame(self, point):
    """Converts a point from material frame to microscope frame."""
    lower_left = np.asarray(self.lower_left).reshape(1, 2)
    upper_right = np.asarray(self.upper_right).reshape(1, 2)
    scale = upper_right - lower_left

    if isinstance(point, AtomicGrid):
      grid = typing.cast(AtomicGrid, point)
      return AtomicGrid(
          (grid.atom_positions - lower_left) / scale, grid.atomic_numbers
      )
    elif isinstance(point, np.ndarray):
      point = typing.cast(np.ndarray, point)

      return_shape = (2,) if point.ndim == 1 else (-1, 2)
      return ((point.reshape(-1, 2) - lower_left) / scale).reshape(return_shape)
    elif isinstance(point, geometry.Point):
      point = typing.cast(geometry.Point, point)
      return geometry.Point((
          (point.x - lower_left[0, 0]) / scale[0, 0],
          (point.y - lower_left[0, 1]) / scale[0, 1],
      ))
    raise NotImplementedError(f'Point of type {type(point)} is not supported.')

  def __str__(self) -> str:
    ll = self.lower_left
    ur = self.upper_right
    return f'FOV [({ll.x:.2f}, {ll.y:.2f}), ({ur.x:.2f}, {ur.y:.2f})]'



class SimulatorObserver:
  """An observer interface for observing events in the simulator."""

  def observe_reset(self, grid: AtomicGrid, fov: MicroscopeFieldOfView) -> None:
    pass

  def observe_apply_control(
      self, start_time: dt.timedelta, control: BeamControl
  ) -> None:
    # end_time can be inferred from start_time and dwell_time.
    pass

  def observe_transition(self, time: dt.timedelta, grid: AtomicGrid) -> None:
    # Transition assumed to be instantaneous.
    pass

  def observe_take_image(
      self,
      start_time: dt.timedelta,
      end_time: dt.timedelta,
      fov: MicroscopeFieldOfView,
  ) -> None:
    pass


@dataclasses.dataclass(frozen=True)
class MicroscopeObservation:
  """An observation from interacting with a microscope."""

  grid: AtomicGrid
  fov: MicroscopeFieldOfView
  controls: Tuple[BeamControl, ...]
  elapsed_time: dt.timedelta
  image: Optional[np.ndarray] = None



@dataclasses.dataclass(frozen=True)
class Transition:
  """A single transition from a microscope.

  Attributes:
    grid_before: Observed atomic grid before the transition.
    grid_after: Observed atomic grid after the transition.
    fov_before: The FOV before the transition.
    fov_after: The FOV after the transition.
    controls: Beam controls applied during the transition.
  """

  grid_before: AtomicGrid
  grid_after: AtomicGrid
  fov_before: MicroscopeFieldOfView
  fov_after: MicroscopeFieldOfView
  controls: Tuple[BeamControl, ...]

