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
    grid=_ATOMIC_GRID,
    fov=_FOV,
    controls=(_BEAM_CONTROL, _BEAM_CONTROL),
    elapsed_time=dt.timedelta(seconds=6),
)
_TRANSITION = microscope_utils.Transition(
    grid_before=_ATOMIC_GRID,
    grid_after=_ATOMIC_GRID,
    fov_before=_FOV,
    fov_after=_FOV,
    controls=(_BEAM_CONTROL, _BEAM_CONTROL),
)


class SimulatorUtilsTest(absltest.TestCase):

  def test_field_of_view_correctly_calculates_offset(self):
    fov = microscope_utils.MicroscopeFieldOfView(
        lower_left=geometry.Point((0.0, 1.0)),
        upper_right=geometry.Point((1.0, 3.0)),
    )

    offset = np.asarray(fov.offset)
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
    grid_before = microscope_utils.AtomicGrid(
        atom_positions=np.asarray([[0.5, 1.0], [-3.0, 2.5]]),
        atomic_numbers=np.zeros(2),  # Unimportant.
    )

    material_grid = fov.microscope_frame_to_material_frame(grid_before)
    microscope_grid = fov.material_frame_to_microscope_frame(material_grid)

    with self.subTest('microscope_to_material'):
      self.assertIsInstance(material_grid, microscope_utils.AtomicGrid)
      np.testing.assert_allclose(
          material_grid.atom_positions, np.asarray([[0.0, 20.0], [-35.0, 50.0]])
      )
    with self.subTest('material_to_microscope'):
      self.assertIsInstance(microscope_grid, microscope_utils.AtomicGrid)
      np.testing.assert_allclose(
          microscope_grid.atom_positions, grid_before.atom_positions
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



if __name__ == '__main__':
  absltest.main()
