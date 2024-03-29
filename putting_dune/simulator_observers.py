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

"""Simulator observers."""

import dataclasses
import datetime as dt
import enum
from typing import Any, Dict

import numpy as np
from putting_dune import microscope_utils


class SimulatorEventType(enum.Enum):
  RESET = enum.auto()
  TRANSITION = enum.auto()
  APPLY_CONTROL = enum.auto()
  TAKE_IMAGE = enum.auto()
  GENERATED_IMAGE = enum.auto()


@dataclasses.dataclass(frozen=True)
class SimulatorEvent:
  event_type: SimulatorEventType
  event_data: Dict[str, Any]  # Not ideal, but saves a lot of boilerplate.


class EventObserver(microscope_utils.SimulatorObserver):
  """An observer that tracks events that occur in the simulator.

  The observed events are:
    - Reset.
    - State transition.
    - Apply control.
    - Take image.
    - Generate image.
  """

  def __init__(self):
    self.grid = None
    self.events = []

  def observe_reset(
      self,
      grid: microscope_utils.AtomicGridMaterialFrame,
      fov: microscope_utils.MicroscopeFieldOfView,
  ) -> None:
    event = SimulatorEvent(
        SimulatorEventType.RESET,
        {'grid': grid, 'fov': fov},
    )
    self.events = [event]

  def observe_transition(
      self,
      time_since_control_was_applied: dt.timedelta,
      grid: microscope_utils.AtomicGridMaterialFrame,
  ) -> None:
    event = SimulatorEvent(
        SimulatorEventType.TRANSITION,
        {
            'time_since_control_was_applied': time_since_control_was_applied,
            'grid': grid,
        },
    )
    self.events.append(event)

  def observe_apply_control(
      self, control: microscope_utils.BeamControlMaterialFrame
  ) -> None:
    event = SimulatorEvent(
        SimulatorEventType.APPLY_CONTROL,
        {'dwell_time': control.dwell_time, 'position': control.position},
    )
    self.events.append(event)

  def observe_take_image(
      self,
      duration: dt.timedelta,
      fov: microscope_utils.MicroscopeFieldOfView,
  ) -> None:
    event = SimulatorEvent(
        SimulatorEventType.TAKE_IMAGE,
        {'duration': duration, 'fov': fov},
    )
    self.events.append(event)

  def observe_generated_image(
      self,
      image: np.ndarray,
  ) -> None:
    event = SimulatorEvent(SimulatorEventType.GENERATED_IMAGE, {'image': image})
    self.events.append(event)
