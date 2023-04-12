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

"""Tests for microscope_utils."""

import datetime as dt

from absl.testing import absltest
import numpy as np
from putting_dune import geometry
from putting_dune import microscope_utils


_ATOMIC_GRID = microscope_utils.AtomicGrid(
    np.asarray([[0.0, 0.0], [1.0, 2.0]]), np.asarray([3, 4])
)
_BEAM_CONTROL = microscope_utils.BeamControl(
    geometry.Point((10.0, 15.3)),
    dt.timedelta(seconds=1.72),
)
_FOV = microscope_utils.MicroscopeFieldOfView(
    lower_left=geometry.Point((2.1, -6.7)),
    upper_right=geometry.Point((3.2, -2.8)),
)
_OBSERVATION = microscope_utils.MicroscopeObservation(
    grid=microscope_utils.AtomicGridMicroscopeFrame(_ATOMIC_GRID),
    fov=_FOV,
    controls=(
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
        microscope_utils.BeamControlMicroscopeFrame(_BEAM_CONTROL),
    ),
    elapsed_time=dt.timedelta(seconds=6),
    image=np.ones((2, 2, 1)),
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
)
_TRAJECTORY = microscope_utils.Trajectory(
    observations=[_OBSERVATION, _OBSERVATION_WITHOUT_IMAGE],
)


class SimulatorUtilsTest(absltest.TestCase):

  def test_field_of_view_correctly_calculates_offset(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point((0.0, 1.0)),
        upper_right=geometry.Point((1.0, 3.0)),
    )

    offset = np.asarray(fov.offset.coords).reshape(-1)
    np.testing.assert_allclose(offset, np.asarray([0.5, 2.0]))

  def test_field_of_view_correctly_calculates_width(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point((0.0, 1.0)),
        upper_right=geometry.Point((1.0, 3.0)),
    )

    self.assertAlmostEqual(fov.width, 1.0)

  def test_field_of_view_correctly_calculates_height(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point((0.0, 1.0)),
        upper_right=geometry.Point((1.0, 3.0)),
    )

    self.assertAlmostEqual(fov.height, 2.0)

  def test_fov_to_string_formats_string_as_expected(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point((0.128, -5.699)),
        upper_right=geometry.Point((1.234, 8.0)),
    )

    fov_str = str(fov)

    self.assertEqual(fov_str, 'FOV [(0.13, -5.70), (1.23, 8.00)]')

  def test_fov_converts_grid_frames_of_reference(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point((-5.0, 0.0)),
        upper_right=geometry.Point((5.0, 20.0)),
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
        lower_left=geometry.Point((-5.0, 0.0)),
        upper_right=geometry.Point((5.0, 20.0)),
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
        lower_left=geometry.Point((-5.0, 0.0)),
        upper_right=geometry.Point((5.0, 20.0)),
    )

    point = geometry.Point((0.5, 1.0))

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
        lower_left=geometry.Point((-5.0, 0.0)),
        upper_right=geometry.Point((5.0, 20.0)),
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
        lower_left=geometry.Point((-5.0, 0.0)),
        upper_right=geometry.Point((5.0, 20.0)),
    )
    point = geometry.Point((0.5, 1.0))

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
        lower_left=geometry.Point((2.5, 2.5)),
        upper_right=geometry.Point((7.5, 7.5)),
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



if __name__ == '__main__':
  absltest.main()
