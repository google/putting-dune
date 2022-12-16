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
"""Putting Dune simulator."""

import datetime as dt
from typing import Sequence

import numpy as np
from putting_dune import geometry
from putting_dune import graphene
from putting_dune import imaging
from putting_dune import microscope_utils


class PuttingDuneSimulator:
  """Putting Dune simulator."""

  def __init__(
      self,
      # TODO(joshgreaves): Expect other types of materials.
      # For now, this is easiest since FOV initialization and updating
      # assumes there is one silicon atom to follow.
      material: graphene.PristineSingleDopedGraphene,
      *,
      observers: Sequence[microscope_utils.SimulatorObserver] = (),
  ):
    """PuttingDuneSimulator Constructor.

    Note that the simulator itself is deterministic, but the objects
    it contains, such as the material, may be probabilistic. Therefore,
    to get reproducable behavior, ensure that all objects are seeded
    correctly before being passed to the simulator.

    Args:
      material: The material to simulator.
      observers: A sequence of objects that observe the inner workings of the
        simulator.
    """
    self.material = material
    self._observers = list(observers)

    # TODO(joshgreaves): Tune this to match the real image duration time.
    self._image_duration = dt.timedelta(seconds=1.5)

    # Will be instantiated on reset.
    self._has_been_reset = False
    self._fov_scale: float
    self._fov: microscope_utils.MicroscopeFieldOfView
    self._image_parameters: imaging.ImageGenerationParameters
    self.elapsed_time: dt.timedelta

  def reset(
      self,
      rng: np.random.Generator,
      return_image: bool = False,
  ) -> microscope_utils.MicroscopeObservation:
    """Reset to a plausible simulator state."""
    self._has_been_reset = True
    # Re-initialize the material.
    self.material.reset(rng)

    # It is fine to initialize the silicon to the center of the frame,
    # as that is something we are likely to do on the actual microscope.
    self._fov_scale = rng.uniform(15, 30)  # width/height of FOV.
    silicon_position = self.material.get_silicon_position()
    self._fov = microscope_utils.MicroscopeFieldOfView(
        geometry.Point(silicon_position - self._fov_scale / 2.0),
        geometry.Point(silicon_position + self._fov_scale / 2.0),
    )

    self.elapsed_time = dt.timedelta(seconds=0)
    for observer in self._observers:
      observer.observe_reset(self.material.atomic_grid, self._fov)

    observed_grid = self._get_observed_grid()

    # We always generate image parameters in case an image is
    # requested in a subsequent call to step_and_image.
    self._image_parameters = imaging.sample_image_parameters(rng)

    # Render the new image, if it is required.
    observed_image = None
    if return_image:
      observed_image = self._generate_image(observed_grid, rng)

    return microscope_utils.MicroscopeObservation(
        grid=observed_grid,
        fov=self._fov,
        controls=(),
        elapsed_time=self.elapsed_time,
        image=observed_image,
    )

  def step_and_image(
      self,
      rng: np.random.Generator,
      controls: Sequence[microscope_utils.BeamControl],
      return_image: bool = False,
  ) -> microscope_utils.MicroscopeObservation:
    """Update simulator state based on beam position delta.

    This emulates the behavior of the real STEM, where we choose a series
    of actions and then generate an image.

    Arguments:
      rng: The RNG to use during the stepping/imaging phase.
      controls: An iterable of controls to apply. This expects an iterable since
        we may want to accept a sequence of controls between generating images.
      return_image: If True, will return an image observation.

    Returns:
      An observation from the simulator.

    Raises:
      RuntimeError: If called before reset.
    """
    self._assert_has_been_reset('step_and_image')
    for control in controls:
      # Convert the control from the microscope frame to material frame.
      # TODO(joshgreaves): Control clipping to [0, 1]?
      control_position = self._fov.microscope_frame_to_material_frame(
          control.position
      )
      control = microscope_utils.BeamControl(
          control_position, control.dwell_time
      )

      for observer in self._observers:
        observer.observe_apply_control(self.elapsed_time, control)

      self.material.apply_control(
          rng, control, self.elapsed_time, self._observers
      )

      self.elapsed_time += control.dwell_time

    observed_grid = self._get_observed_grid()

    observed_silicon_positions = graphene.get_silicon_positions(observed_grid)
    assert observed_silicon_positions.shape == (1, 2)
    observed_silicon_position = observed_silicon_positions.reshape(-1)

    silicon_position_near_edge = (
        (observed_silicon_position < 0.25) | (observed_silicon_position > 0.75)
    ).any()

    if silicon_position_near_edge:
      # We cheat here a little - the real implementation would need to
      # work out the exact delta to position the atom in the middle of the
      # frame, but we can just get the coordinate of the atom directly
      # from the material.
      silicon_position = self.material.get_silicon_position()
      self._fov = microscope_utils.MicroscopeFieldOfView(
          geometry.Point(silicon_position - self._fov_scale / 2.0),
          geometry.Point(silicon_position + self._fov_scale / 2.0),
      )
      observed_grid = self._get_observed_grid()

    # Render the new image, if it is required.
    observed_image = None
    if return_image:
      observed_image = self._generate_image(observed_grid, rng)

    return microscope_utils.MicroscopeObservation(
        grid=observed_grid,
        fov=self._fov,
        controls=tuple(controls),
        elapsed_time=self.elapsed_time,
        image=observed_image,
    )

  def add_observer(self, observer: microscope_utils.SimulatorObserver) -> None:
    self._observers.append(observer)

  def remove_observer(
      self, observer: microscope_utils.SimulatorObserver
  ) -> None:
    self._observers.remove(observer)

  def _get_observed_grid(self) -> microscope_utils.AtomicGrid:
    """Gets the atomic grid of all atoms in the field of view."""
    # Note: Currently we do not expect atoms/defects to diffuse through
    # the graphene sheet during an image capture. However, if this changes,
    # we should account for it here.

    observation = self.material.get_atoms_in_bounds(
        self._fov.lower_left, self._fov.upper_right
    )

    for observer in self._observers:
      observer.observe_take_image(
          start_time=self.elapsed_time,
          end_time=self.elapsed_time + self._image_duration,
          fov=self._fov,
      )

    self.elapsed_time += self._image_duration

    return observation

  def _generate_image(
      self, observed_grid: microscope_utils.AtomicGrid, rng: np.random.Generator
  ) -> np.ndarray:
    observed_image = imaging.generate_stem_image(
        observed_grid, self._fov, self._image_parameters, rng
    )

    for observer in self._observers:
      observer.observe_generated_image(self.elapsed_time, observed_image)

    return observed_image

  def _assert_has_been_reset(self, fn_name: str) -> None:
    if not self._has_been_reset:
      raise RuntimeError(
          f'Must call reset on {self.__class__} before {fn_name}.'
      )
