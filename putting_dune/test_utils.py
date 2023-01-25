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

"""Common utility functions for unit tests."""

import datetime as dt
from typing import Any

import numpy as np
from putting_dune import geometry
from putting_dune import graphene
from putting_dune import imaging
from putting_dune import microscope_utils
from putting_dune import putting_dune_environment
from putting_dune import run_helpers
from putting_dune.experiments import registry


def create_simple_environment(
    seed: int = 0,
    **kwargs: Any,
) -> putting_dune_environment.PuttingDuneEnvironment:
  experiment = registry.create_train_experiment('relative_simple_rates')
  return run_helpers.create_putting_dune_env(
      seed,
      get_adapters_and_goal=experiment.get_adapters_and_goal,
      get_simulator_config=experiment.get_simulator_config,
      **kwargs,
  )


def create_single_silicon_observation(
    rng: np.random.Generator, return_image: bool = False
) -> microscope_utils.MicroscopeObservation:
  """Creates an observation of graphene with single silicon for unit tests."""
  graphene_sheet = graphene.PristineSingleDopedGraphene()
  graphene_sheet.reset(rng)

  silicon_position = graphene_sheet.get_silicon_position()
  fov = microscope_utils.MicroscopeFieldOfView(
      geometry.Point((silicon_position[0] - 5.0, silicon_position[1] - 5.0)),
      geometry.Point((silicon_position[0] + 5.0, silicon_position[1] + 5.0)),
  )
  grid = graphene_sheet.get_atoms_in_bounds(fov.lower_left, fov.upper_right)

  image = None
  if return_image:
    image_params = imaging.sample_image_parameters(rng)
    image = imaging.generate_stem_image(grid, fov, image_params, rng)

  observation = microscope_utils.MicroscopeObservation(
      grid=grid,
      fov=fov,
      controls=(),
      elapsed_time=dt.timedelta(seconds=1.5),
      image=image,
  )

  return observation
