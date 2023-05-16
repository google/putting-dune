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

"""Common utilities for microscope components."""

import dataclasses
import datetime as dt
import typing
from typing import NewType, Optional, Sequence, Tuple

import numpy as np
from putting_dune import geometry
from putting_dune import putting_dune_pb2
import tensorflow as tf


def shapely_point_to_proto_point(
    point: geometry.Point,
) -> putting_dune_pb2.Point2D:
  return putting_dune_pb2.Point2D(x=point.x, y=point.y)


def proto_point_to_shapely_point(
    point: putting_dune_pb2.Point2D,
) -> geometry.Point:
  return geometry.Point((point.x, point.y))


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

  @classmethod
  def from_proto(cls, proto_grid: putting_dune_pb2.AtomicGrid) -> 'AtomicGrid':
    """Creates an AtomicGrid from a proto."""
    num_atoms = len(proto_grid.atoms)

    atom_positions = np.empty((num_atoms, 2), dtype=np.float32)
    atomic_numbers = np.empty(num_atoms, dtype=np.int32)

    for i, atom in enumerate(proto_grid.atoms):
      atom_positions[i, 0] = atom.position.x
      atom_positions[i, 1] = atom.position.y
      atomic_numbers[i] = atom.atomic_number

    return cls(atom_positions, atomic_numbers)

  def to_proto(self) -> putting_dune_pb2.AtomicGrid:
    """Creates a proto from this object."""
    num_atoms, _ = self.atom_positions.shape

    grid = putting_dune_pb2.AtomicGrid()
    for i in range(num_atoms):
      grid.atoms.append(
          putting_dune_pb2.Atom(
              atomic_number=self.atomic_numbers[i],
              position=putting_dune_pb2.Point2D(
                  x=self.atom_positions[i, 0],
                  y=self.atom_positions[i, 1],
              ),
          )
      )

    return grid


AtomicGridMaterialFrame = NewType('AtomicGridMaterialFrame', AtomicGrid)
AtomicGridMicroscopeFrame = NewType('AtomicGridMicroscopeFrame', AtomicGrid)


@dataclasses.dataclass(frozen=True)
class BeamControl:
  """Specifications to control the microscope beam for one step.

  Attributes:
    position: Point describing the beam position.
    dwell_time: Beam dwell time.
    voltage_kv: Beam voltage, in kilovolts.
    current_na: Beam current, in nanoamperes
  """

  position: geometry.Point
  dwell_time: dt.timedelta
  voltage_kv: Optional[float] = 60  # most data was gathered at 60kV.
  current_na: Optional[float] = 0.1  # typical current in real data.

  def shift(self, shift: geometry.Point) -> 'BeamControl':
    shifted_position = geometry.Point(
        self.position.x + shift.x, self.position.y + shift.y
    )
    return BeamControl(
        shifted_position, self.dwell_time, self.voltage_kv, self.current_na
    )

  @classmethod
  def from_proto(
      cls,
      control: putting_dune_pb2.BeamControl,
  ) -> 'BeamControl':
    """Creates a BeamControl object from a proto."""

    position = proto_point_to_shapely_point(control.position)
    dwell_time = dt.timedelta(seconds=control.dwell_time_seconds)

    return cls(position, dwell_time, control.voltage_kv, control.current_na)

  def to_proto(self) -> putting_dune_pb2.BeamControl:
    """Creates a proto from this object."""
    beam_position = putting_dune_pb2.BeamControl(
        position=shapely_point_to_proto_point(self.position),
        dwell_time_seconds=self.dwell_time.total_seconds(),
        voltage_kv=self.voltage_kv,
        current_na=self.current_na,
    )
    return beam_position


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

  lower_left: geometry.PointMaterialFrame
  upper_right: geometry.PointMaterialFrame

  def shift(
      self, shift: geometry.PointMaterialFrame
  ) -> 'MicroscopeFieldOfView':
    new_lower_left = geometry.PointMaterialFrame(
        geometry.Point(self.lower_left.x + shift.x, self.lower_left.y + shift.y)
    )
    new_upper_right = geometry.PointMaterialFrame(
        geometry.Point(
            self.upper_right.x + shift.x, self.upper_right.y + shift.y
        )
    )
    return MicroscopeFieldOfView(new_lower_left, new_upper_right)

  @property
  def offset(self) -> geometry.PointMaterialFrame:
    return geometry.PointMaterialFrame(
        geometry.Point(
            (
                np.asarray(self.lower_left.coords).reshape(-1)
                + np.asarray(self.upper_right.coords).reshape(-1)
            )
            / 2
        )
    )

  @property
  def width(self) -> float:
    return self.upper_right.x - self.lower_left.x

  @property
  def height(self) -> float:
    return self.upper_right.y - self.lower_left.y

  def resize(
      self, new_width: float, new_height: float
  ) -> 'MicroscopeFieldOfView':
    """Resizes the MicroscopeFieldOfView while keeping its center constant.

    Args:
      new_width: New width (in angstroms), positive float.
      new_height: New height (in angstroms), positive float.

    Returns:
      A new MicroscopeFieldOfView with the same offset as this one,
      but with different width and height.
    """
    assert new_width > 0 and new_height > 0
    new_scale_vector = np.asarray([new_width, new_height]) / 2
    centerpoint = (
        np.asarray(self.lower_left.coords).reshape(-1)
        + np.asarray(self.upper_right.coords).reshape(-1)
    ) / 2
    new_lower_left = centerpoint - new_scale_vector
    new_upper_right = centerpoint + new_scale_vector
    new_lower_left = geometry.PointMaterialFrame(
        geometry.Point(new_lower_left[0], new_lower_left[1])
    )
    new_upper_right = geometry.PointMaterialFrame(
        geometry.Point(new_upper_right[0], new_upper_right[1])
    )
    return MicroscopeFieldOfView(new_lower_left, new_upper_right)

  def zoom(self, zoom_factor: float) -> 'MicroscopeFieldOfView':
    assert zoom_factor > 0
    new_width = self.width / zoom_factor
    new_height = self.height / zoom_factor
    return self.resize(new_width, new_height)

  @typing.overload
  def microscope_frame_to_material_frame(self, point: np.ndarray) -> np.ndarray:
    ...

  @typing.overload
  def microscope_frame_to_material_frame(
      self, point: BeamControl
  ) -> BeamControl:
    ...

  @typing.overload
  def microscope_frame_to_material_frame(
      self, point: geometry.PointMicroscopeFrame
  ) -> geometry.PointMaterialFrame:
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
      return geometry.PointMaterialFrame(
          geometry.Point((
              point.x * scale[0, 0] + lower_left[0, 0],
              point.y * scale[0, 1] + lower_left[0, 1],
          ))
      )
    elif isinstance(point, BeamControl):
      position = geometry.Point((
          point.position.x * scale[0, 0] + lower_left[0, 0],
          point.position.y * scale[0, 1] + lower_left[0, 1],
      ))
      return BeamControl(
          position, point.dwell_time, point.voltage_kv, point.current_na
      )

    raise NotImplementedError(f'Point of type {type(point)} is not supported.')

  @typing.overload
  def material_frame_to_microscope_frame(self, point: np.ndarray) -> np.ndarray:
    ...

  @typing.overload
  def material_frame_to_microscope_frame(
      self, point: geometry.PointMaterialFrame
  ) -> geometry.Point:
    ...

  @typing.overload
  def material_frame_to_microscope_frame(
      self, point: AtomicGridMaterialFrame
  ) -> AtomicGridMicroscopeFrame:
    ...

  @typing.overload
  def material_frame_to_microscope_frame(
      self, point: BeamControl
  ) -> BeamControl:
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
      return geometry.PointMicroscopeFrame(
          geometry.Point((
              (point.x - lower_left[0, 0]) / scale[0, 0],
              (point.y - lower_left[0, 1]) / scale[0, 1],
          ))
      )
    elif isinstance(point, BeamControl):
      position = geometry.Point((
          (point.position.x - lower_left[0, 0]) / scale[0, 0],
          (point.position.y - lower_left[0, 1]) / scale[0, 1],
      ))
      return BeamControl(
          position,
          point.dwell_time,
          voltage_kv=point.voltage_kv,
          current_na=point.current_na,
      )

    raise NotImplementedError(f'Point of type {type(point)} is not supported.')

  def __str__(self) -> str:
    ll = self.lower_left
    ur = self.upper_right
    return f'FOV [({ll.x:.2f}, {ll.y:.2f}), ({ur.x:.2f}, {ur.y:.2f})]'

  def get_atoms_in_bounds(
      self,
      grid: AtomicGridMaterialFrame,
      tolerance: float = 0,
  ) -> AtomicGridMaterialFrame:
    """Selects the atoms within an AtomicGrid that lie within the FOV.

    Args:
      grid: An AtomicGridMaterialFrame, to be subsetted.
      tolerance: Buffer zone width (in angstroms). Positive values include atoms
        outside the FOV, while negative values include only atoms near the
        center of the FOV.

    Returns:
      The observed atomic grid within the supplied bounds. Atom positions are
      left in the material frame, and can be converted by applying
      the material_frame_to_microscope_frame method.
    """
    atom_positions = grid.atom_positions
    atomic_numbers = grid.atomic_numbers
    lower_left = np.asarray(self.lower_left.coords) - tolerance
    upper_right = np.asarray(self.upper_right.coords) + tolerance
    indices_in_bounds = np.all(
        ((lower_left <= atom_positions) & (atom_positions <= upper_right)),
        axis=1,
    )

    selected_atom_positions = atom_positions[indices_in_bounds]
    selected_atomic_numbers = atomic_numbers[indices_in_bounds]

    return AtomicGridMaterialFrame(
        AtomicGrid(selected_atom_positions, selected_atomic_numbers)
    )

  @classmethod
  def from_proto(
      cls,
      fov: putting_dune_pb2.FieldOfView,
  ) -> 'MicroscopeFieldOfView':
    return cls(
        lower_left=geometry.PointMaterialFrame(
            proto_point_to_shapely_point(fov.lower_left_angstroms)
        ),
        upper_right=geometry.PointMaterialFrame(
            proto_point_to_shapely_point(fov.upper_right_angstroms)
        ),
    )

  def to_proto(self) -> putting_dune_pb2.FieldOfView:
    return putting_dune_pb2.FieldOfView(
        lower_left_angstroms=shapely_point_to_proto_point(self.lower_left),
        upper_right_angstroms=shapely_point_to_proto_point(self.upper_right),
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
  label_image: Optional[np.ndarray] = None

  @classmethod
  def from_proto(
      cls,
      observation: putting_dune_pb2.MicroscopeObservation,
  ) -> 'MicroscopeObservation':
    """Instantiates a MicroscopeObservation from the corresponding proto.

    Args:
      observation: An observation proto.

    Returns:
      An observation as a dataclass object.
    """
    controls = tuple(
        BeamControlMicroscopeFrame(BeamControl.from_proto(control))
        for control in observation.controls
    )
    image = observation.image
    if image is not None and image.dtype != 0:
      image = tf.make_ndarray(image)
    else:
      image = None
    label_image = observation.label_image
    if label_image is not None and label_image.dtype != 0:
      label_image = tf.make_ndarray(label_image)
    else:
      label_image = None
    return cls(
        grid=AtomicGridMicroscopeFrame(AtomicGrid.from_proto(observation.grid)),
        fov=MicroscopeFieldOfView.from_proto(observation.fov),
        controls=controls,
        elapsed_time=dt.timedelta(seconds=observation.elapsed_time_seconds),
        image=image,
        label_image=label_image,
    )

  def to_proto(self) -> putting_dune_pb2.MicroscopeObservation:
    controls = [control.to_proto() for control in self.controls]
    image = tf.make_tensor_proto(self.image) if self.image is not None else None
    label_image = (
        tf.make_tensor_proto(self.label_image)
        if self.label_image is not None
        else None
    )
    return putting_dune_pb2.MicroscopeObservation(
        grid=self.grid.to_proto(),
        fov=self.fov.to_proto(),
        controls=controls,
        elapsed_time_seconds=self.elapsed_time.total_seconds(),
        image=image,
        label_image=label_image,
    )


@dataclasses.dataclass(frozen=True)
class Transition:
  """A single transition from a microscope.

  Attributes:
    grid_before: Observed atomic grid before the transition.
    grid_after: Observed atomic grid after the transition.
    fov_before: The FOV before the transition.
    fov_after: The FOV after the transition.
    controls: Beam controls applied during the transition.
    image_before: Observed image before the transition (optional).
    image_after: Observed image after the transition (optional).
    label_image_before: Labeled image before the transition (optional).
    label_image_after: Labeled image after the transition (optional).
  """

  grid_before: AtomicGridMicroscopeFrame
  grid_after: AtomicGridMicroscopeFrame
  fov_before: MicroscopeFieldOfView
  fov_after: MicroscopeFieldOfView
  controls: Tuple[BeamControlMicroscopeFrame, ...]
  image_before: Optional[np.ndarray] = None
  image_after: Optional[np.ndarray] = None
  label_image_before: Optional[np.ndarray] = None
  label_image_after: Optional[np.ndarray] = None

  @classmethod
  def from_proto(
      cls,
      transition: putting_dune_pb2.Transition,
  ) -> 'Transition':
    """Creates an AtomicGrid from a proto."""
    controls = tuple(
        BeamControlMicroscopeFrame(BeamControl.from_proto(control))
        for control in transition.controls
    )

    image_before = transition.image_before
    if image_before is not None and image_before.dtype != 0:
      image_before = tf.make_ndarray(image_before)
    else:
      image_before = None

    image_after = transition.image_after
    if image_after is not None and image_after.dtype != 0:
      image_after = tf.make_ndarray(image_after)
    else:
      image_after = None

    label_image_before = transition.label_image_before
    if label_image_before is not None and label_image_before.dtype != 0:
      label_image_before = tf.make_ndarray(label_image_before)
    else:
      label_image_before = None

    label_image_after = transition.label_image_after
    if label_image_after is not None and label_image_after.dtype != 0:
      label_image_after = tf.make_ndarray(label_image_after)
    else:
      label_image_after = None

    return cls(
        grid_before=AtomicGridMicroscopeFrame(
            AtomicGrid.from_proto(transition.grid_before)
        ),
        grid_after=AtomicGridMicroscopeFrame(
            AtomicGrid.from_proto(transition.grid_after)
        ),
        fov_before=MicroscopeFieldOfView.from_proto(transition.fov_before),
        fov_after=MicroscopeFieldOfView.from_proto(transition.fov_after),
        controls=controls,
        image_before=image_before,
        image_after=image_after,
        label_image_before=label_image_before,
        label_image_after=label_image_after,
    )

  def to_proto(self) -> putting_dune_pb2.Transition:
    """Creates a proto from this object."""
    controls = [control.to_proto() for control in self.controls]
    image_before = (
        tf.make_tensor_proto(self.image_before)
        if self.image_before is not None
        else None
    )
    image_after = (
        tf.make_tensor_proto(self.image_after)
        if self.image_after is not None
        else None
    )

    label_image_before = (
        tf.make_tensor_proto(self.label_image_before)
        if self.image_before is not None
        else None
    )
    label_image_after = (
        tf.make_tensor_proto(self.label_image_after)
        if self.image_after is not None
        else None
    )

    return putting_dune_pb2.Transition(
        grid_before=self.grid_before.to_proto(),
        grid_after=self.grid_after.to_proto(),
        fov_before=self.fov_before.to_proto(),
        fov_after=self.fov_after.to_proto(),
        controls=controls,
        image_before=image_before,
        image_after=image_after,
        label_image_before=label_image_before,
        label_image_after=label_image_after,
    )


@dataclasses.dataclass(frozen=True)
class Trajectory:
  """A trajectory of observations from a microscope.

  Attributes:
    observations: Sequence of MicroscopeObservations
  """

  observations: Sequence[MicroscopeObservation]

  @classmethod
  def from_proto(
      cls,
      trajectory: putting_dune_pb2.Trajectory,
  ) -> 'Trajectory':
    """Creates a Trajectory from a proto."""
    observations = tuple(
        MicroscopeObservation.from_proto(obs) for obs in trajectory.observations
    )
    return cls(
        observations=observations,
    )

  def to_proto(self) -> putting_dune_pb2.Trajectory:
    """Creates a proto from this object."""
    return putting_dune_pb2.Trajectory(
        observations=[obs.to_proto() for obs in self.observations],
    )


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
    point_drift = geometry.PointMaterialFrame(
        geometry.Point(self.drift[0], self.drift[1])
    )
    shifted_fov = observation.fov.shift(point_drift)
    shifted_observation = MicroscopeObservation(
        grid=AtomicGridMicroscopeFrame(shifted_grid),
        fov=shifted_fov,
        controls=observation.controls,
        elapsed_time=observation.elapsed_time,
        image=observation.image,
    )
    return shifted_observation

  @classmethod
  def from_proto(cls, proto_drift: putting_dune_pb2.Drift) -> 'Drift':
    """Creates a Drift object from a proto."""
    num_atoms = len(proto_drift.jitter)

    jitter = np.empty((num_atoms, 2), dtype=np.float32)
    drift = np.empty(2, dtype=np.int32)

    for i, atom in enumerate(proto_drift.jitter):
      jitter[i, 0] = atom.x
      jitter[i, 1] = atom.y
    drift[0] = proto_drift.drift.x
    drift[1] = proto_drift.drift.y

    return cls(jitter=jitter, drift=drift)

  def to_proto(self) -> putting_dune_pb2.Drift:
    """Creates a proto from this object."""
    num_atoms = self.jitter.shape[0]

    proto_jitter = [
        putting_dune_pb2.Point2D(x=self.jitter[i, 0], y=self.jitter[i, 1])
        for i in range(num_atoms)
    ]
    proto_drift = putting_dune_pb2.Point2D(x=self.drift[0], y=self.drift[1])
    drift = putting_dune_pb2.Drift(jitter=proto_jitter, drift=proto_drift)
    return drift


@dataclasses.dataclass(frozen=True)
class LabeledAlignmentTrajectory:
  """A trajectory of observations from a microscope, with accompanying drifts.

  Attributes:
    trajectory: a Trajectory object.
    drifts: a sequence of Drift objects.
  """

  trajectory: Trajectory
  drifts: Sequence[Drift]

  @classmethod
  def from_proto(
      cls,
      labeled_trajectory: putting_dune_pb2.LabeledAlignmentTrajectory,
  ) -> 'LabeledAlignmentTrajectory':
    """Creates a LabeledAlignmentTrajectory from a proto."""
    drifts = [Drift.from_proto(drift) for drift in labeled_trajectory.drifts]
    trajectory = Trajectory.from_proto(labeled_trajectory.trajectory)
    return cls(
        trajectory=trajectory,
        drifts=drifts,
    )

  def to_proto(self) -> putting_dune_pb2.LabeledAlignmentTrajectory:
    """Creates a proto from this object."""
    return putting_dune_pb2.LabeledAlignmentTrajectory(
        trajectory=self.trajectory.to_proto(),
        drifts=[drift.to_proto() for drift in self.drifts],
    )
