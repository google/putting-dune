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
"""Tests for simulator_utils."""

import datetime

from absl.testing import absltest
import numpy as np
from putting_dune import simulator_utils
from putting_dune.google import putting_dune_pb2  # pylint: disable=g-bad-import-order
from shapely import geometry



class SimulatorUtilsTest(absltest.TestCase):

  def test_field_of_view_correctly_calculates_offset(self):
    fov = simulator_utils.SimulatorFieldOfView(
        lower_left=geometry.Point((0.0, 1.0)),
        upper_right=geometry.Point((1.0, 3.0)),
    )

    offset = np.asarray(fov.offset)
    np.testing.assert_allclose(offset, np.asarray([0.5, 2.0]))

  def test_fov_to_string_formats_string_as_expected(self):
    fov = simulator_utils.SimulatorFieldOfView(
        lower_left=geometry.Point((0.128, -5.699)),
        upper_right=geometry.Point((1.234, 8.0)),
    )

    fov_str = str(fov)

    self.assertEqual(fov_str, 'FOV [(0.13, -5.70), (1.23, 8.00)]')

  # TODO(joshgreaves): Write tests for FOV coordinate calculations.



if __name__ == '__main__':
  absltest.main()
