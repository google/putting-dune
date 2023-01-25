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

"""A collection of action adapters for PuttingDuneEnvironment."""

import abc
import datetime as dt
from typing import List, Tuple

from dm_env import specs
import numpy as np
from putting_dune import constants
from putting_dune import geometry
from putting_dune import graphene
from putting_dune import microscope_utils


class ActionAdapter(abc.ABC):
  """Abstract base class for action adapters."""

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the action adapter.

    Called at the beginning of a new episode.
    """

  @abc.abstractmethod
  def get_action(
      self,
      previous_observation: microscope_utils.MicroscopeObservation,
      action: np.ndarray,
  ) -> List[microscope_utils.BeamControlMicroscopeFrame]:
    """Gets simulator controls from the agent action."""

  @property
  @abc.abstractmethod
  def action_spec(self) -> specs.BoundedArray:
    """Gets the action spec for the action adapter."""


class DirectActionAdapter(ActionAdapter):
  """An action adapter for specifying a point relative to the FOV."""

  def reset(self) -> None:
    """Resets the action adapter."""
    pass

  def get_action(
      self,
      previous_observation: microscope_utils.MicroscopeObservation,
      action: np.ndarray,
  ) -> List[microscope_utils.BeamControlMicroscopeFrame]:
    """Gets simulator controls from the agent action."""
    del previous_observation  # Unused.

    action = np.clip(action, 0.0, 1.0)
    # TODO(joshgreaves): Choose/parameterize dwell time.
    return [
        microscope_utils.BeamControlMicroscopeFrame(
            microscope_utils.BeamControl(
                position=geometry.Point(action),
                dwell_time=dt.timedelta(seconds=1.5),
            )
        )
    ]

  @property
  def action_spec(self) -> specs.BoundedArray:
    """Gets the action spec for the action adapter."""
    return specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=1.0
    )


class DeltaPositionActionAdapter(ActionAdapter):
  """An action adapter that moves the beam by a supplied delta.

  It uses a fixed dwell time for each action.
  """

  def __init__(self, rng: np.random.Generator):
    self.rng = rng
    self.reset()

  def reset(self):
    # Randomly place beam anywhere in [0, 1] frame.
    self.beam_pos = self.rng.uniform(0, 1, size=2)

  def get_action(
      self,
      previous_observation: microscope_utils.MicroscopeObservation,
      action: np.ndarray,
  ) -> List[microscope_utils.BeamControlMicroscopeFrame]:
    del previous_observation  # Unused.

    self.beam_pos += action
    # For now, we clip the beam to ensure it doesn't leave the field
    # of view. In the future, we may want to change this.
    self.beam_pos = np.clip(self.beam_pos, 0.0, 1.0)

    return [
        microscope_utils.BeamControlMicroscopeFrame(
            microscope_utils.BeamControl(
                geometry.Point(self.beam_pos[0], self.beam_pos[1]),
                # TODO(joshgreaves): Choose/parameterize dwell time.
                dt.timedelta(seconds=1.5),
            )
        )
    ]

  @property
  def action_spec(self) -> specs.BoundedArray:
    # x, y position of the STEM probe.
    return specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-0.1, maximum=0.1
    )


class RelativeToSiliconActionAdapter(ActionAdapter):
  """An action adapter that accepts a relative position to a silicon atom."""

  def __init__(
      self,
      *,
      dwell_time_range: Tuple[dt.timedelta, dt.timedelta] = (
          dt.timedelta(seconds=1.5),
          dt.timedelta(seconds=1.5),
      ),
  ):
    min_dwell_time, max_dwell_time = dwell_time_range
    self._fixed_dwell_time = min_dwell_time == max_dwell_time
    self._min_dwell_seconds = min_dwell_time.total_seconds()
    self._max_dwell_seconds = max_dwell_time.total_seconds()

  def reset(self):
    pass

  def get_action(
      self,
      previous_observation: microscope_utils.MicroscopeObservation,
      action: np.ndarray,
  ) -> List[microscope_utils.BeamControlMicroscopeFrame]:
    """Gets simulator controls from the agent action."""
    beam_position_action = np.clip(action[:2], -1.0, 1.0)

    silicon_position = graphene.get_silicon_positions(previous_observation.grid)

    if silicon_position.shape != (1, 2):
      raise RuntimeError(
          'Expected to find one silicon with x, y coordinates. Instead, '
          f'got {silicon_position.shape[0]} silicon atoms with '
          f'{silicon_position.shape[1]} dimensions.'
      )
    silicon_position = np.reshape(silicon_position, (2,))

    # Action is [dx, dy] in unit cell terms.
    # For generality, assume aspect ratio is not square.
    fov = previous_observation.fov
    cell_radius_x = constants.CARBON_BOND_DISTANCE_ANGSTROMS / (
        fov.upper_right.x - fov.lower_left.x
    )
    cell_radius_y = constants.CARBON_BOND_DISTANCE_ANGSTROMS / (
        fov.upper_right.y - fov.lower_left.y
    )
    cell_radius = np.asarray([cell_radius_x, cell_radius_y])
    control_position = silicon_position + (beam_position_action * cell_radius)
    control_position = np.clip(control_position, 0.0, 1.0)

    if self._fixed_dwell_time:
      dwell_time = dt.timedelta(seconds=self._min_dwell_seconds)
    else:
      dwell_time_action = np.clip(action[2], 0.0, 1.0)
      dwell_range_seconds = self._max_dwell_seconds - self._min_dwell_seconds
      dwell_time_seconds = (
          dwell_time_action * dwell_range_seconds + self._min_dwell_seconds
      )
      dwell_time = dt.timedelta(seconds=dwell_time_seconds)

    return [
        microscope_utils.BeamControlMicroscopeFrame(
            microscope_utils.BeamControl(
                geometry.Point(*control_position), dwell_time
            )
        )
    ]

  @property
  def action_spec(self) -> specs.BoundedArray:
    if self._fixed_dwell_time:
      return specs.BoundedArray(
          shape=(2,),
          dtype=np.float32,
          minimum=-1.0,
          maximum=1.0,
      )
    else:
      return specs.BoundedArray(
          shape=(3,),
          dtype=np.float32,
          minimum=np.asarray([-1.0, -1.0, 0.0]),
          maximum=np.asarray([1.0, 1.0, 1.0]),
      )
