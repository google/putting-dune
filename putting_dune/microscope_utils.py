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

"""Common utilities for microscope components."""

import dataclasses
import datetime as dt
import typing
from typing import NewType, Tuple, Sequence, Optional

import numpy as np
from putting_dune import geometry



@dataclasses.dataclass(frozen=True)
class AtomicGrid:
  """A grid of atoms."""

  atom_positions: np.ndarray
  atomic_numbers: np.ndarray

  def shift(self, shift_vector: np.ndarray) -> 'AtomicGrid':
    # enforce broadcasting and error if shape is incorrect.
    shift_vector = shift_vector.reshape(1, 2)
    shifted_atom_positions = self.atom_positions + shift_vector
    return AtomicGrid(shifted_atom_positions, self.atomic_numbers)


AtomicGridMaterialFrame = NewType('AtomicGridMaterialFrame', AtomicGrid)
AtomicGridMicroscopeFrame = NewType('AtomicGridMicroscopeFrame', AtomicGrid)


@dataclasses.dataclass(frozen=True)
class BeamControl:
  """Specifications to control the microscope beam for one step.

  Attributes:
    position: Point describing the beam position.
    dwell_time: Beam dwell time.
  """

  position: geometry.Point
  dwell_time: dt.timedelta

  def shift(self, shift: geometry.Point) -> 'BeamControl':
    shifted_position = geometry.Point(
        self.position.x + shift.x, self.position.y + shift.y
    )
    return BeamControl(shifted_position, self.dwell_time)



BeamControlMaterialFrame = NewType('BeamControlMaterialFrame', BeamControl)
BeamControlMicroscopeFrame = NewType('BeamControlMicroscopeFrame', BeamControl)


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

  def shift(self, shift: geometry.Point) -> 'MicroscopeFieldOfView':
    new_lower_left = geometry.Point(
        self.lower_left.x + shift.x, self.lower_left.y + shift.y
    )
    new_upper_right = geometry.Point(
        self.upper_right.x + shift.x, self.upper_right.y + shift.y
    )
    return MicroscopeFieldOfView(new_lower_left, new_upper_right)

  @property
  def offset(self) -> geometry.Point:
    return geometry.Point(
        (
            np.asarray(self.lower_left.coords).reshape(-1)
            + np.asarray(self.upper_right.coords).reshape(-1)
        )
        / 2
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
  def microscope_frame_to_material_frame(
      self, point: AtomicGridMicroscopeFrame
  ) -> AtomicGridMaterialFrame:
    ...

  def microscope_frame_to_material_frame(self, point):
    """Converts a point from microscope frame to material frame."""
    lower_left = np.asarray(self.lower_left.coords)
    upper_right = np.asarray(self.upper_right.coords)
    scale = upper_right - lower_left

    if isinstance(point, AtomicGrid):
      grid = typing.cast(AtomicGrid, point)
      return AtomicGridMaterialFrame(
          AtomicGrid(
              grid.atom_positions * scale + lower_left, grid.atomic_numbers
          )
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
  def material_frame_to_microscope_frame(
      self, point: AtomicGridMaterialFrame
  ) -> AtomicGridMicroscopeFrame:
    ...

  def material_frame_to_microscope_frame(self, point):
    """Converts a point from material frame to microscope frame."""
    lower_left = np.asarray(self.lower_left.coords)
    upper_right = np.asarray(self.upper_right.coords)
    scale = upper_right - lower_left

    if isinstance(point, AtomicGrid):
      grid = typing.cast(AtomicGrid, point)
      return AtomicGridMicroscopeFrame(
          AtomicGrid(
              (grid.atom_positions - lower_left) / scale, grid.atomic_numbers
          )
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

  def get_atoms_in_bounds(
      self,
      grid: AtomicGridMaterialFrame,
  ) -> AtomicGridMaterialFrame:
    """Selects the atoms within an AtomicGrid that lie within the FOV.

    Args:
      grid: An AtomicGridMaterialFrame, to be subsetted.

    Returns:
      The observed atomic grid within the supplied bounds. Atom positions are
      left in the material frame, and can be converted by applying
      the material_frame_to_microscope_frame method.
    """
    atom_positions = grid.atom_positions
    atomic_numbers = grid.atomic_numbers
    lower_left = np.asarray(self.lower_left.coords)
    upper_right = np.asarray(self.upper_right.coords)
    indices_in_bounds = np.all(
        ((lower_left <= atom_positions) & (atom_positions <= upper_right)),
        axis=1,
    )

    selected_atom_positions = atom_positions[indices_in_bounds]
    selected_atomic_numbers = atomic_numbers[indices_in_bounds]

    return AtomicGridMaterialFrame(
        AtomicGrid(selected_atom_positions, selected_atomic_numbers)
    )



class SimulatorObserver:
  """An observer interface for observing events in the simulator."""

  def observe_reset(
      self, grid: AtomicGridMaterialFrame, fov: MicroscopeFieldOfView
  ) -> None:
    pass

  def observe_apply_control(self, control: BeamControlMaterialFrame) -> None:
    pass

  def observe_transition(
      self,
      time_since_control_was_applied: dt.timedelta,
      grid: AtomicGridMaterialFrame,
  ) -> None:
    pass

  def observe_take_image(
      self,
      duration: dt.timedelta,
      fov: MicroscopeFieldOfView,
  ) -> None:
    pass

  def observe_generated_image(
      self,
      image: np.ndarray,
  ) -> None:
    pass


@dataclasses.dataclass(frozen=True)
class MicroscopeObservation:
  """An observation from interacting with a microscope."""

  grid: AtomicGridMicroscopeFrame
  fov: MicroscopeFieldOfView
  controls: Tuple[BeamControlMicroscopeFrame, ...]
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

  grid_before: AtomicGridMicroscopeFrame
  grid_after: AtomicGridMicroscopeFrame
  fov_before: MicroscopeFieldOfView
  fov_after: MicroscopeFieldOfView
  controls: Tuple[BeamControlMicroscopeFrame, ...]



@dataclasses.dataclass(frozen=True)
class Trajectory:
  """A trajectory of observations from a microscope.

  Attributes:
    observations: Sequence of MicroscopeObservations
  """

  observations: Sequence[MicroscopeObservation]



@dataclasses.dataclass(frozen=True)
class Drift:
  """A trajectory of observations from a microscope.

  Attributes:
    drift: A shared (2,) drift vector applying to an entire material.
    jitter: A (num_atoms, 2) array of per-atom displacements.
  """

  jitter: np.ndarray
  drift: np.ndarray

  def cumulate_drift(self, drift: 'Drift') -> 'Drift':
    """Calculates a cumulative drift object from a previous drift."""

    new_drift_vector = self.drift + drift.drift
    return Drift(drift=new_drift_vector, jitter=self.jitter)

  def apply_to_observation(
      self, observation: MicroscopeObservation
  ) -> MicroscopeObservation:
    """Applies the drift to an observation, including jitter.

    Args:
      observation: The observation to be aligned.

    Returns:
      The observation with its grid shifted by `drift` and `jitter`, and fov
      and controls shifted by `drift`.
    """
    dejittered_atom_positions = observation.grid.atom_positions - self.jitter
    shifted_grid = AtomicGrid(
        dejittered_atom_positions, observation.grid.atomic_numbers
    )
    point_drift = geometry.Point(self.drift[0], self.drift[1])
    shifted_fov = observation.fov.shift(point_drift)
    shifted_observation = MicroscopeObservation(
        grid=AtomicGridMicroscopeFrame(shifted_grid),
        fov=shifted_fov,
        controls=observation.controls,
        elapsed_time=observation.elapsed_time,
        image=observation.image,
    )
    return shifted_observation



@dataclasses.dataclass(frozen=True)
class LabeledAlignmentTrajectory:
  """A trajectory of observations from a microscope, with accompanying drifts.

  Attributes:
    trajectory: a Trajectory object.
    drifts: a sequence of Drift objects.
  """

  trajectory: Trajectory
  drifts: Sequence[Drift]

