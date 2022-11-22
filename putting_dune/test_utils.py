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
"""Common utility functions for unit tests."""

import datetime as dt

import numpy as np
from putting_dune import graphene
from putting_dune import simulator_utils
from shapely import geometry


def create_graphene_observation_with_single_silicon_in_fov(
    rng: np.random.Generator,
) -> simulator_utils.SimulatorObservation:
  """Creates an observation of graphene with single silicon for unit tests."""
  graphene_sheet = graphene.PristineSingleDopedGraphene(rng)
  silicon_position = graphene_sheet.get_silicon_position()
  fov = simulator_utils.SimulatorFieldOfView(
      geometry.Point((silicon_position[0] - 5.0, silicon_position[1] - 5.0)),
      geometry.Point((silicon_position[0] + 5.0, silicon_position[1] + 5.0)),
  )
  observation = simulator_utils.SimulatorObservation(
      graphene_sheet.get_atoms_in_bounds(fov.lower_left, fov.upper_right),
      fov,
      None,
      dt.timedelta(seconds=0),
  )

  return observation
