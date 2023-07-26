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

"""Tests for microscope_utils."""

import datetime as dt

from absl.testing import absltest
import numpy as np
from putting_dune import geometry
from putting_dune import microscope_utils
from putting_dune import putting_dune_pb2
import tensorflow as tf


_IMAGE = np.ones((2, 2, 1))
_IMAGE_AFTER = np.ones((2, 2, 1)) + 1
_LABEL_IMAGE = np.ones((2, 2, 3))
_LABEL_IMAGE_AFTER = np.ones((2, 2, 3)) + 1
_ATOMIC_GRID = microscope_utils.AtomicGrid(
    np.asarray([[0.0, 0.0], [1.0, 2.0]]), np.asarray([3, 4])
)
_BEAM_CONTROL = microscope_utils.BeamControl(
    geometry.Point((10.0, 15.3)),
    dt.timedelta(seconds=1.72),
    current_na=1,
    voltage_kv=2,
)
_FOV = microscope_utils.MicroscopeFieldOfView(
    lower_left=geometry.PointMaterialFrame(geometry.Point((2.1, -6.7))),
    upper_right=geometry.PointMaterialFrame(geometry.Point((3.2, -2.8))),
)
_OBSERVATION = microscope_utils.MicroscopeObservation(
    grid=microscope_utils.AtomicGridMicroscopeFrame(_ATOMIC_GRID),
    fov=_FOV,
    controls=(
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
    ),
    elapsed_time=dt.timedelta(seconds=6),
    image=_IMAGE,
    label_image=_LABEL_IMAGE,
)
_OBSERVATION_WITHOUT_IMAGE = microscope_utils.MicroscopeObservation(
    grid=microscope_utils.AtomicGridMicroscopeFrame(_ATOMIC_GRID),
    fov=_FOV,
    controls=(
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
    ),
    elapsed_time=dt.timedelta(seconds=6),
    image=None,
)
_TRANSITION = microscope_utils.Transition(
    grid_before=microscope_utils.AtomicGridMicroscopeFrame(_ATOMIC_GRID),
    grid_after=microscope_utils.AtomicGridMicroscopeFrame(_ATOMIC_GRID),
    fov_before=_FOV,
    fov_after=_FOV,
    controls=(
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
    ),
    image_before=_IMAGE,
    image_after=_IMAGE_AFTER,
    label_image_before=_LABEL_IMAGE,
    label_image_after=_LABEL_IMAGE_AFTER,
)
_TRAJECTORY = microscope_utils.Trajectory(
    observations=[_OBSERVATION, _OBSERVATION_WITHOUT_IMAGE],
)


class SimulatorUtilsTest(absltest.TestCase):

  def test_field_of_view_correctly_calculates_offset(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((0.0, 1.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((1.0, 3.0))),
    )

    offset = np.asarray(fov.offset.coords).reshape(-1)
    np.testing.assert_allclose(offset, np.asarray([0.5, 2.0]))

  def test_field_of_view_correctly_calculates_width(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((0.0, 1.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((1.0, 3.0))),
    )

    self.assertAlmostEqual(fov.width, 1.0)

  def test_field_of_view_correctly_calculates_height(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((0.0, 1.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((1.0, 3.0))),
    )

    self.assertAlmostEqual(fov.height, 2.0)

  def test_fov_to_string_formats_string_as_expected(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((0.128, -5.699))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((1.234, 8.0))),
    )

    fov_str = str(fov)

    self.assertEqual(fov_str, 'FOV [(0.13, -5.70), (1.23, 8.00)]')

  def test_fov_converts_grid_frames_of_reference(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((-5.0, 0.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((5.0, 20.0))),
    )
    microscope_grid_before = microscope_utils.AtomicGridMicroscopeFrame(
        microscope_utils.AtomicGrid(
            atom_positions=np.asarray([[0.5, 1.0], [-3.0, 2.5]]),
            atomic_numbers=np.zeros(2),  # Unimportant.
        )
    )

    material_grid = fov.microscope_frame_to_material_frame(
        microscope_grid_before
    )
    microscope_grid = fov.material_frame_to_microscope_frame(material_grid)

    with self.subTest('microscope_to_material'):
      self.assertIsInstance(material_grid, microscope_utils.AtomicGrid)
      np.testing.assert_allclose(
          material_grid.atom_positions, np.asarray([[0.0, 20.0], [-35.0, 50.0]])
      )

    with self.subTest('material_to_microscope'):
      self.assertIsInstance(microscope_grid, microscope_utils.AtomicGrid)
      np.testing.assert_allclose(
          microscope_grid.atom_positions, microscope_grid_before.atom_positions
      )

  def test_fov_resizes_correctly(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((-5.0, 0.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((5.0, 20.0))),
    )

    with self.subTest('zoom_in_2'):
      zoomed_fov = fov.zoom(2)
      zoomed_lower_left = np.asarray(zoomed_fov.lower_left.coords)
      zoomed_upper_right = np.asarray(zoomed_fov.upper_right.coords)
      np.testing.assert_allclose(zoomed_lower_left, np.array([[-2.5, 5]]))
      np.testing.assert_allclose(zoomed_upper_right, np.array([[2.5, 15]]))
    with self.subTest('zoom_out_2'):
      zoomed_fov = fov.zoom(0.5)
      zoomed_lower_left = np.asarray(zoomed_fov.lower_left.coords)
      zoomed_upper_right = np.asarray(zoomed_fov.upper_right.coords)
      np.testing.assert_allclose(zoomed_lower_left, np.array([[-10, -10]]))
      np.testing.assert_allclose(zoomed_upper_right, np.array([[10, 30]]))
    with self.subTest('zoom_identity'):
      zoomed_fov = fov.zoom(1)
      zoomed_lower_left = np.asarray(zoomed_fov.lower_left.coords)
      zoomed_upper_right = np.asarray(zoomed_fov.upper_right.coords)
      np.testing.assert_allclose(zoomed_lower_left, np.array([[-5, 0]]))
      np.testing.assert_allclose(zoomed_upper_right, np.array([[5, 20]]))

  def test_fov_converts_beamcontrolframes_of_reference(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((-5.0, 0.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((5.0, 20.0))),
    )

    point = geometry.PointMicroscopeFrame(geometry.Point((0.5, 1.0)))

    material_point = fov.microscope_frame_to_material_frame(point)
    microscope_point = fov.material_frame_to_microscope_frame(material_point)

    microscope_frame_control = microscope_utils.BeamControl(
        microscope_point, dwell_time=dt.timedelta(seconds=1)
    )
    with self.subTest('microscope_to_material'):
      material_frame_control = fov.microscope_frame_to_material_frame(
          microscope_frame_control
      )
      self.assertAlmostEqual(
          material_frame_control.position.x, material_point.x
      )
      self.assertAlmostEqual(
          material_frame_control.position.y, material_point.y
      )

    material_frame_control = microscope_utils.BeamControl(
        material_point, dwell_time=dt.timedelta(seconds=1)
    )
    with self.subTest('material_to_microscope'):
      microscope_frame_control = fov.material_frame_to_microscope_frame(
          material_frame_control
      )
      self.assertAlmostEqual(
          microscope_frame_control.position.x, microscope_point.x
      )
      self.assertAlmostEqual(
          microscope_frame_control.position.y, microscope_point.y
      )

  def test_fov_converts_ndarray_frames_of_reference(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((-5.0, 0.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((5.0, 20.0))),
    )
    points = np.asarray([[0.5, 1.0], [-3.0, 2.5]])

    material_points = fov.microscope_frame_to_material_frame(points)
    microscope_points = fov.material_frame_to_microscope_frame(material_points)

    with self.subTest('microscope_to_material'):
      self.assertIsInstance(material_points, np.ndarray)
      np.testing.assert_allclose(
          material_points, np.asarray([[0.0, 20.0], [-35.0, 50.0]])
      )
    with self.subTest('material_to_microscope'):
      self.assertIsInstance(microscope_points, np.ndarray)
      np.testing.assert_allclose(microscope_points, points)

  def test_fov_converts_point_frames_of_reference(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((-5.0, 0.0))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((5.0, 20.0))),
    )
    point = geometry.PointMicroscopeFrame(geometry.Point((0.5, 1.0)))

    material_point = fov.microscope_frame_to_material_frame(point)
    microscope_point = fov.material_frame_to_microscope_frame(material_point)

    with self.subTest('microscope_to_material'):
      self.assertIsInstance(material_point, geometry.Point)
      self.assertAlmostEqual(material_point.x, 0.0)
      self.assertAlmostEqual(material_point.y, 20.0)
    with self.subTest('material_to_microscope'):
      self.assertIsInstance(microscope_point, geometry.Point)
      self.assertAlmostEqual(microscope_point.x, point.x)
      self.assertAlmostEqual(microscope_point.y, point.y)

  def test_fov_get_atoms_in_bounds(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.PointMaterialFrame(geometry.Point((2.5, 2.5))),
        upper_right=geometry.PointMaterialFrame(geometry.Point((7.5, 7.5))),
    )

    x_positions = np.linspace(0, 10, 11)
    y_positions = np.linspace(0, 10, 11)
    atom_positions = np.stack([x_positions, y_positions], axis=-1)
    atomic_numbers = np.arange(0, 11, 1)
    grid = microscope_utils.AtomicGrid(atom_positions, atomic_numbers)
    grid = microscope_utils.AtomicGridMaterialFrame(grid)

    with self.subTest('no_tolerance'):
      target_positions = np.linspace(3, 7, 5)
      target_positions = np.stack([target_positions, target_positions], axis=-1)
      target_atomic_numbers = np.arange(3, 8, 1)
      subgrid = fov.get_atoms_in_bounds(grid, tolerance=0.0)
      np.testing.assert_allclose(subgrid.atom_positions, target_positions)
      np.testing.assert_allclose(subgrid.atomic_numbers, target_atomic_numbers)
    with self.subTest('tolerance_100'):
      subgrid = fov.get_atoms_in_bounds(grid, tolerance=100.0)
      np.testing.assert_allclose(subgrid.atom_positions, grid.atom_positions)
      np.testing.assert_allclose(subgrid.atomic_numbers, grid.atomic_numbers)
    with self.subTest('tolerance_negative_100'):
      target_positions = np.zeros((0, 2))
      target_atomic_numbers = np.zeros((0,))
      subgrid = fov.get_atoms_in_bounds(grid, tolerance=-100.0)
      np.testing.assert_allclose(subgrid.atom_positions, target_positions)
      np.testing.assert_allclose(subgrid.atomic_numbers, target_atomic_numbers)
    with self.subTest('tolerance_1'):
      target_positions = np.linspace(2, 8, 7)
      target_positions = np.stack([target_positions, target_positions], axis=-1)
      target_atomic_numbers = np.arange(2, 9, 1)
      subgrid = fov.get_atoms_in_bounds(grid, tolerance=1.0)
      np.testing.assert_allclose(subgrid.atom_positions, target_positions)
      np.testing.assert_allclose(subgrid.atomic_numbers, target_atomic_numbers)

  def test_atomic_grid_converts_to_proto(self):
    grid_proto = _ATOMIC_GRID.to_proto()

    self.assertLen(grid_proto.atoms, 2)
    self.assertEqual(grid_proto.atoms[0].atomic_number, 3)
    self.assertEqual(grid_proto.atoms[0].position.x, 0.0)
    self.assertEqual(grid_proto.atoms[0].position.y, 0.0)
    self.assertEqual(grid_proto.atoms[1].atomic_number, 4)
    self.assertEqual(grid_proto.atoms[1].position.x, 1.0)
    self.assertEqual(grid_proto.atoms[1].position.y, 2.0)

  def test_atomic_grid_can_be_created_from_proto(self):
    grid_proto = putting_dune_pb2.AtomicGrid(
        atoms=(
            putting_dune_pb2.Atom(
                atomic_number=0,
                position=putting_dune_pb2.Point2D(x=1.0, y=1.1),
            ),
            putting_dune_pb2.Atom(
                atomic_number=2,
                position=putting_dune_pb2.Point2D(x=-6.3, y=0.0),
            ),
            putting_dune_pb2.Atom(
                atomic_number=5,
                position=putting_dune_pb2.Point2D(x=12.7, y=-0.05),
            ),
        ),
    )

    atomic_grid = microscope_utils.AtomicGrid.from_proto(grid_proto)

    self.assertEqual(atomic_grid.atom_positions.shape, (3, 2))
    self.assertEqual(atomic_grid.atomic_numbers.shape, (3,))
    np.testing.assert_allclose(
        atomic_grid.atom_positions,
        np.asarray([[1.0, 1.1], [-6.3, 0.0], [12.7, -0.05]]),
    )
    np.testing.assert_array_equal(
        atomic_grid.atomic_numbers, np.asarray([0, 2, 5])
    )

  def test_atomic_grid_hashes_same_grid_to_same_hash(self):
    grid1 = _ATOMIC_GRID
    grid2 = microscope_utils.AtomicGrid(
        grid1.atom_positions, grid1.atomic_numbers
    )

    self.assertEqual(hash(grid1), hash(grid2))

  def test_atomic_grid_hashes_translated_grid_to_different_hash(self):
    grid1 = _ATOMIC_GRID
    grid2 = microscope_utils.AtomicGrid(
        grid1.atom_positions + np.asarray([[5.0, -7.325]]), grid1.atomic_numbers
    )

    self.assertNotEqual(hash(grid1), hash(grid2))

  def test_atomic_grid_hashes_same_positions_different_atoms_differently(self):
    grid1 = _ATOMIC_GRID
    atomic_numbers = grid1.atomic_numbers.copy()
    atomic_numbers[0] = atomic_numbers[0] + 1
    grid2 = microscope_utils.AtomicGrid(grid1.atom_positions, atomic_numbers)

    self.assertNotEqual(hash(grid1), hash(grid2))

  def test_atomic_grid_hashes_different_positions_differently(self):
    grid1 = _ATOMIC_GRID
    atom_positions = grid1.atom_positions.copy()
    atom_positions[0, :] = atom_positions[0, :] + np.asarray([3.0, 4.0])
    grid2 = microscope_utils.AtomicGrid(atom_positions, grid1.atomic_numbers)

    self.assertNotEqual(hash(grid1), hash(grid2))

  def test_atomic_grid_is_equal_to_same_atomic_grid(self):
    grid1 = _ATOMIC_GRID
    grid2 = microscope_utils.AtomicGrid(
        grid1.atom_positions.copy(), grid1.atomic_numbers.copy()
    )
    self.assertEqual(grid1, grid2)

  def test_atomic_grid_is_not_equal_to_grid_with_different_num_atoms(self):
    grid1 = _ATOMIC_GRID
    atom_positions = np.delete(grid1.atom_positions, 0)
    atomic_numbers = np.delete(grid1.atomic_numbers, 0)
    grid2 = microscope_utils.AtomicGrid(atom_positions, atomic_numbers)

    self.assertNotEqual(grid1, grid2)

  def test_atomic_grid_is_not_equal_to_translated_grid(self):
    grid1 = _ATOMIC_GRID
    grid2 = microscope_utils.AtomicGrid(
        grid1.atom_positions + np.asarray([[5.0, -7.325]]), grid1.atomic_numbers
    )

    self.assertNotEqual(grid1, grid2)

  def test_atomic_grid_is_not_equal_to_grid_with_different_atoms(self):
    grid1 = _ATOMIC_GRID
    atomic_numbers = grid1.atomic_numbers.copy()
    atomic_numbers[0] = atomic_numbers[0] + 1
    grid2 = microscope_utils.AtomicGrid(grid1.atom_positions, atomic_numbers)

    self.assertNotEqual(grid1, grid2)

  def test_atomic_grid_hash_is_equal_if_grids_are_equal(self):
    grid1 = _ATOMIC_GRID
    grid2 = microscope_utils.AtomicGrid(
        grid1.atom_positions.copy(), grid1.atomic_numbers.copy()
    )

    self.assertEqual(grid1, grid2)
    self.assertEqual(hash(grid1), hash(grid2))

  def test_beam_control_converts_to_proto(self):
    control_proto = _BEAM_CONTROL.to_proto()

    self.assertAlmostEqual(control_proto.position.x, 10.0, delta=1e-6)
    self.assertAlmostEqual(control_proto.position.y, 15.3, delta=1e-6)
    self.assertAlmostEqual(control_proto.dwell_time_seconds, 1.72, delta=1e-6)

  def test_beam_control_can_be_created_from_proto(self):
    control_proto = putting_dune_pb2.BeamControl(
        position=putting_dune_pb2.Point2D(x=10.0, y=15.3),
        dwell_time_seconds=1.72,
        voltage_kv=2.0,
        current_na=1.0,
    )
    control = microscope_utils.BeamControl.from_proto(control_proto)

    self.assertAlmostEqual(control.position.x, 10.0, delta=1e-6)
    self.assertAlmostEqual(control.position.y, 15.3, delta=1e-6)
    self.assertAlmostEqual(control.dwell_time.total_seconds(), 1.72, delta=1e-6)
    self.assertAlmostEqual(control.voltage_kv, 2.0, delta=1e-6)
    self.assertAlmostEqual(control.current_na, 1.0, delta=1e-6)

  def test_field_of_view_converts_to_proto(self):
    proto_fov = _FOV.to_proto()

    self.assertAlmostEqual(proto_fov.lower_left_angstroms.x, 2.1, delta=1e-6)
    self.assertAlmostEqual(proto_fov.lower_left_angstroms.y, -6.7, delta=1e-6)
    self.assertAlmostEqual(proto_fov.upper_right_angstroms.x, 3.2, delta=1e-6)
    self.assertAlmostEqual(proto_fov.upper_right_angstroms.y, -2.8, delta=1e-6)

  def test_field_of_view_can_be_created_from_proto(self):
    fov_proto = putting_dune_pb2.FieldOfView(
        lower_left_angstroms=putting_dune_pb2.Point2D(x=2.1, y=-6.7),
        upper_right_angstroms=putting_dune_pb2.Point2D(x=3.2, y=-2.8),
    )
    fov = microscope_utils.MicroscopeFieldOfView.from_proto(fov_proto)

    self.assertAlmostEqual(fov.lower_left.x, 2.1, delta=1e-6)
    self.assertAlmostEqual(fov.lower_left.y, -6.7, delta=1e-6)
    self.assertAlmostEqual(fov.upper_right.x, 3.2, delta=1e-6)
    self.assertAlmostEqual(fov.upper_right.y, -2.8, delta=1e-6)

  def test_microscope_observation_converts_grid_to_proto(self):
    grid_proto = _OBSERVATION.to_proto().grid

    self.assertLen(grid_proto.atoms, 2)

    self.assertEqual(grid_proto.atoms[0].atomic_number, 3)
    self.assertAlmostEqual(grid_proto.atoms[0].position.x, 0.0)
    self.assertAlmostEqual(grid_proto.atoms[0].position.y, 0.0)

    self.assertEqual(grid_proto.atoms[1].atomic_number, 4)
    self.assertAlmostEqual(grid_proto.atoms[1].position.x, 1.0)
    self.assertAlmostEqual(grid_proto.atoms[1].position.y, 2.0)

  def test_microscope_observation_converts_fov_to_proto(self):
    fov_proto = _OBSERVATION.to_proto().fov

    self.assertAlmostEqual(fov_proto.lower_left_angstroms.x, 2.1, delta=1e-6)
    self.assertAlmostEqual(fov_proto.lower_left_angstroms.y, -6.7, delta=1e-6)
    self.assertAlmostEqual(fov_proto.upper_right_angstroms.x, 3.2, delta=1e-6)
    self.assertAlmostEqual(fov_proto.upper_right_angstroms.y, -2.8, delta=1e-6)

  def test_microscope_observation_converts_controls_to_proto(self):
    controls_proto = _OBSERVATION.to_proto().controls

    self.assertLen(controls_proto, 2)
    self.assertAlmostEqual(controls_proto[0].position.x, 10.0, delta=1e-6)
    self.assertAlmostEqual(controls_proto[0].position.y, 15.3, delta=1e-6)
    self.assertAlmostEqual(
        controls_proto[0].dwell_time_seconds, 1.72, delta=1e-6
    )
    self.assertAlmostEqual(controls_proto[1].position.x, 10.0, delta=1e-6)
    self.assertAlmostEqual(controls_proto[1].position.y, 15.3, delta=1e-6)
    self.assertAlmostEqual(
        controls_proto[1].dwell_time_seconds, 1.72, delta=1e-6
    )

  def test_microscope_observation_converts_elapsed_time_to_proto(self):
    elapsed_time_proto = _OBSERVATION.to_proto().elapsed_time_seconds

    self.assertAlmostEqual(elapsed_time_proto, 6.0, delta=1e-6)

  def test_microscope_observation_converts_image_to_proto(self):
    image_proto = _OBSERVATION.to_proto().image
    image_tensor = tf.make_ndarray(image_proto)
    np.testing.assert_allclose(image_tensor, np.ones((2, 2, 1)))

  def test_microscope_observation_converts_null_image_to_proto(self):
    proto = _OBSERVATION_WITHOUT_IMAGE.to_proto()
    observation = microscope_utils.MicroscopeObservation.from_proto(proto)
    self.assertIsNone(observation.image)
    self.assertIsNone(observation.label_image)

  def test_microscope_observation_can_be_created_from_proto(self):
    # We already have tests that to_proto works successfully, so
    # we use it here to make this test much shorted (and easier to follow).
    observation_proto = _OBSERVATION.to_proto()
    observation = microscope_utils.MicroscopeObservation.from_proto(
        observation_proto
    )

    # Compare grids.
    with self.subTest('grids'):
      np.testing.assert_array_equal(
          observation.grid.atomic_numbers, _OBSERVATION.grid.atomic_numbers
      )
      np.testing.assert_allclose(
          observation.grid.atom_positions, _OBSERVATION.grid.atom_positions
      )

    # Compare fov.
    with self.subTest('fov'):
      np.testing.assert_allclose(
          np.asarray(observation.fov.lower_left.coords),
          np.asarray(_OBSERVATION.fov.lower_left.coords),
      )
      np.testing.assert_allclose(
          np.asarray(observation.fov.upper_right.coords),
          np.asarray(_OBSERVATION.fov.upper_right.coords),
      )

    # Compare controls.
    with self.subTest('controls'):
      self.assertLen(observation.controls, 2)
      np.testing.assert_allclose(
          np.asarray(observation.controls[0].position.coords),
          np.asarray(_OBSERVATION.controls[0].position.coords),
      )
      self.assertAlmostEqual(
          observation.controls[0].dwell_time.total_seconds(),
          _OBSERVATION.controls[0].dwell_time.total_seconds(),
          delta=1e-6,
      )

    # Compare elapsed time.
    with self.subTest('elapsed_time'):
      self.assertAlmostEqual(
          observation.elapsed_time.total_seconds(),
          _OBSERVATION.elapsed_time.total_seconds(),
          delta=1e-6,
      )

    # Compare image.
    with self.subTest('image'):
      np.testing.assert_allclose(observation.image, _OBSERVATION.image)
    with self.subTest('label_image'):
      np.testing.assert_allclose(
          observation.label_image, _OBSERVATION.label_image
      )

  def test_transition_converts_to_proto(self):
    transition_proto = _TRANSITION.to_proto()

    with self.subTest('grid_before'):
      grid_proto = transition_proto.grid_before
      self.assertLen(grid_proto.atoms, 2)

      self.assertEqual(grid_proto.atoms[0].atomic_number, 3)
      self.assertAlmostEqual(grid_proto.atoms[0].position.x, 0.0)
      self.assertAlmostEqual(grid_proto.atoms[0].position.y, 0.0)

      self.assertEqual(grid_proto.atoms[1].atomic_number, 4)
      self.assertAlmostEqual(grid_proto.atoms[1].position.x, 1.0)
      self.assertAlmostEqual(grid_proto.atoms[1].position.y, 2.0)

    with self.subTest('grid_after'):
      grid_proto = transition_proto.grid_after
      self.assertLen(grid_proto.atoms, 2)

      self.assertEqual(grid_proto.atoms[0].atomic_number, 3)
      self.assertAlmostEqual(grid_proto.atoms[0].position.x, 0.0)
      self.assertAlmostEqual(grid_proto.atoms[0].position.y, 0.0)

      self.assertEqual(grid_proto.atoms[1].atomic_number, 4)
      self.assertAlmostEqual(grid_proto.atoms[1].position.x, 1.0)
      self.assertAlmostEqual(grid_proto.atoms[1].position.y, 2.0)

    with self.subTest('fov_before'):
      fov_proto = transition_proto.fov_before

      self.assertAlmostEqual(fov_proto.lower_left_angstroms.x, 2.1, delta=1e-6)
      self.assertAlmostEqual(fov_proto.lower_left_angstroms.y, -6.7, delta=1e-6)
      self.assertAlmostEqual(fov_proto.upper_right_angstroms.x, 3.2, delta=1e-6)
      self.assertAlmostEqual(
          fov_proto.upper_right_angstroms.y, -2.8, delta=1e-6
      )

    with self.subTest('fov_after'):
      fov_proto = transition_proto.fov_after

      self.assertAlmostEqual(fov_proto.lower_left_angstroms.x, 2.1, delta=1e-6)
      self.assertAlmostEqual(fov_proto.lower_left_angstroms.y, -6.7, delta=1e-6)
      self.assertAlmostEqual(fov_proto.upper_right_angstroms.x, 3.2, delta=1e-6)
      self.assertAlmostEqual(
          fov_proto.upper_right_angstroms.y, -2.8, delta=1e-6
      )

    with self.subTest('controls'):
      controls_proto = transition_proto.controls

      self.assertLen(controls_proto, 2)
      self.assertAlmostEqual(controls_proto[0].position.x, 10.0, delta=1e-6)
      self.assertAlmostEqual(controls_proto[0].position.y, 15.3, delta=1e-6)
      self.assertAlmostEqual(
          controls_proto[0].dwell_time_seconds, 1.72, delta=1e-6
      )
      self.assertAlmostEqual(controls_proto[1].position.x, 10.0, delta=1e-6)
      self.assertAlmostEqual(controls_proto[1].position.y, 15.3, delta=1e-6)
      self.assertAlmostEqual(
          controls_proto[1].dwell_time_seconds, 1.72, delta=1e-6
      )
      self.assertAlmostEqual(controls_proto[0].voltage_kv, 2.0, delta=1e-6)
      self.assertAlmostEqual(controls_proto[0].current_na, 1.0, delta=1e-6)

  def test_transition_can_be_created_from_proto(self):
    # We already have tests that to_proto works successfully, so
    # we use it here to make this test much shorted (and easier to follow).
    transition_proto = _TRANSITION.to_proto()
    transition = microscope_utils.Transition.from_proto(transition_proto)

    # Compare grid_before.
    with self.subTest('grid_before'):
      np.testing.assert_array_equal(
          transition.grid_before.atomic_numbers,
          _TRANSITION.grid_before.atomic_numbers,
      )
      np.testing.assert_allclose(
          transition.grid_before.atom_positions,
          _TRANSITION.grid_before.atom_positions,
      )

    # Compare grid_after.
    with self.subTest('grid_before'):
      np.testing.assert_array_equal(
          transition.grid_after.atomic_numbers,
          _TRANSITION.grid_after.atomic_numbers,
      )
      np.testing.assert_allclose(
          transition.grid_after.atom_positions,
          _TRANSITION.grid_after.atom_positions,
      )

    # Compare fov_before.
    with self.subTest('fov_before'):
      np.testing.assert_allclose(
          np.asarray(transition.fov_before.lower_left.coords),
          np.asarray(_TRANSITION.fov_before.lower_left.coords),
      )
      np.testing.assert_allclose(
          np.asarray(transition.fov_before.upper_right.coords),
          np.asarray(_TRANSITION.fov_before.upper_right.coords),
      )

    # Compare fov_after.
    with self.subTest('fov_before'):
      np.testing.assert_allclose(
          np.asarray(transition.fov_after.lower_left.coords),
          np.asarray(_TRANSITION.fov_after.lower_left.coords),
      )
      np.testing.assert_allclose(
          np.asarray(transition.fov_after.upper_right.coords),
          np.asarray(_TRANSITION.fov_after.upper_right.coords),
      )

    # Compare controls.
    with self.subTest('controls'):
      self.assertLen(transition.controls, 2)
      np.testing.assert_allclose(
          np.asarray(transition.controls[0].position.coords),
          np.asarray(_TRANSITION.controls[0].position.coords),
      )
      self.assertAlmostEqual(
          transition.controls[0].dwell_time.total_seconds(),
          _TRANSITION.controls[0].dwell_time.total_seconds(),
          delta=1e-6,
      )
      self.assertAlmostEqual(
          transition.controls[0].current_na,
          _TRANSITION.controls[0].current_na,
          delta=1e-6,
      )
      self.assertAlmostEqual(
          transition.controls[0].voltage_kv,
          _TRANSITION.controls[0].voltage_kv,
          delta=1e-6,
      )

    with self.subTest('image'):
      np.testing.assert_allclose(
          transition.image_before, _TRANSITION.image_before
      )
      np.testing.assert_allclose(
          transition.image_after, _TRANSITION.image_after
      )

    with self.subTest('label_image'):
      np.testing.assert_allclose(
          transition.label_image_before, _TRANSITION.label_image_before
      )
      np.testing.assert_allclose(
          transition.label_image_after, _TRANSITION.label_image_after
      )

  def test_trajectory_can_be_created_from_proto(self):
    # We already have tests that to_proto works successfully, so
    # we use it here to make this test much shorted (and easier to follow).
    trajectory_proto = _TRAJECTORY.to_proto()
    trajectory = microscope_utils.Trajectory.from_proto(trajectory_proto)
    for observation, converted_observation in zip(
        _TRAJECTORY.observations, trajectory.observations
    ):
      # Compare grids.
      with self.subTest('grids'):
        np.testing.assert_array_equal(
            observation.grid.atomic_numbers,
            converted_observation.grid.atomic_numbers,
        )
        np.testing.assert_allclose(
            observation.grid.atom_positions,
            converted_observation.grid.atom_positions,
        )

      # Compare fov.
      with self.subTest('fov'):
        np.testing.assert_allclose(
            np.asarray(observation.fov.lower_left.coords),
            np.asarray(converted_observation.fov.lower_left.coords),
        )
        np.testing.assert_allclose(
            np.asarray(observation.fov.upper_right.coords),
            np.asarray(converted_observation.fov.upper_right.coords),
        )

      # Compare controls.
      with self.subTest('controls'):
        self.assertLen(observation.controls, 2)
        np.testing.assert_allclose(
            np.asarray(observation.controls[0].position),
            np.asarray(converted_observation.controls[0].position),
        )
        self.assertAlmostEqual(
            observation.controls[0].dwell_time.total_seconds(),
            converted_observation.controls[0].dwell_time.total_seconds(),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            observation.controls[0].current_na,
            converted_observation.controls[0].current_na,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            observation.controls[0].voltage_kv,
            converted_observation.controls[0].voltage_kv,
            delta=1e-6,
        )

      # Compare elapsed time.
      with self.subTest('elapsed_time'):
        self.assertAlmostEqual(
            observation.elapsed_time.total_seconds(),
            converted_observation.elapsed_time.total_seconds(),
            delta=1e-6,
        )

      # Compare image.
      with self.subTest('image'):
        if observation.image is None:
          self.assertIsNone(converted_observation.image)
        else:
          np.testing.assert_allclose(
              observation.image, converted_observation.image
          )

      with self.subTest('label_image'):
        if observation.label_image is None:
          self.assertIsNone(converted_observation.label_image)
        else:
          np.testing.assert_allclose(
              observation.label_image, converted_observation.label_image
          )



if __name__ == '__main__':
  absltest.main()
