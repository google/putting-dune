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

"""Simulator observers."""

import dataclasses
import datetime as dt
import enum
from typing import Any

from putting_dune import simulator_utils


class SimulatorEventType(enum.Enum):
  RESET = enum.auto()
  TRANSITION = enum.auto()
  APPLY_CONTROL = enum.auto()
  TAKE_IMAGE = enum.auto()


@dataclasses.dataclass(frozen=True)
class SimulatorEvent:
  event_type: SimulatorEventType
  start_time: dt.timedelta
  end_time: dt.timedelta
  event_data: dict[str, Any]  # Not ideal, but saves a lot of boilerplate.


class EventObserver(simulator_utils.SimulatorObserver):
  """An observer that tracks events that occur in the simulator.

  The observed events are:
    - Reset.
    - State transition.
    - Apply control.
    - Take image.
  """

  def __init__(self):
    self.grid = None
    self.events = []

  def observe_reset(
      self,
      grid: simulator_utils.AtomicGrid) -> None:
    event = SimulatorEvent(
        SimulatorEventType.RESET,
        dt.timedelta(seconds=0),
        dt.timedelta(seconds=0),
        {'grid': grid})
    self.events = [event]

  def observe_transition(
      self, time: dt.timedelta, grid: simulator_utils.AtomicGrid) -> None:
    event = SimulatorEvent(
        SimulatorEventType.TRANSITION,
        time,
        time,
        {'grid': grid})
    self.events.append(event)

  def observe_apply_control(
      self,
      start_time: dt.timedelta,
      control: simulator_utils.SimulatorControl) -> None:
    event = SimulatorEvent(
        SimulatorEventType.APPLY_CONTROL,
        start_time,
        start_time + control.dwell_time,
        {'position': control.position})
    self.events.append(event)

  def observe_take_image(
      self,
      start_time: dt.timedelta,
      end_time: dt.timedelta,
      fov: simulator_utils.SimulatorFieldOfView) -> None:
    event = SimulatorEvent(
        SimulatorEventType.TAKE_IMAGE, start_time, end_time, {'fov': fov})
    self.events.append(event)
