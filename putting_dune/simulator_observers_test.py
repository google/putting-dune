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

"""Tests for simulator_observers."""

import datetime as dt

from absl.testing import absltest
import numpy as np
from putting_dune import geometry
from putting_dune import microscope_utils
from putting_dune import simulator_observers


class SimulatorObserversTest(absltest.TestCase):

  def test_event_observer_tracks_events_correctly(self):
    observer = simulator_observers.EventObserver()

    grid = microscope_utils.AtomicGridMaterialFrame(
        microscope_utils.AtomicGrid(np.zeros((5, 2)), np.zeros(5))
    )
    fov = microscope_utils.MicroscopeFieldOfView(
        geometry.PointMaterialFrame(geometry.Point((0.0, 0.0))),
        geometry.PointMaterialFrame(geometry.Point((1.0, 1.0))),
    )
    observer.observe_reset(grid, fov)
    observer.observe_apply_control(
        microscope_utils.BeamControlMaterialFrame(
            microscope_utils.BeamControl(
                geometry.Point(1.0, 1.0), dt.timedelta(seconds=1.0)
            )
        ),
    )
    observer.observe_transition(dt.timedelta(seconds=0.2), grid)
    observer.observe_transition(dt.timedelta(seconds=0.6), grid)
    observer.observe_take_image(dt.timedelta(seconds=1.5), fov)

    events = observer.events
    self.assertLen(events, 5)
    self.assertEqual(
        events[0].event_type, simulator_observers.SimulatorEventType.RESET
    )
    self.assertEqual(
        events[1].event_type,
        simulator_observers.SimulatorEventType.APPLY_CONTROL,
    )
    self.assertEqual(
        events[2].event_type, simulator_observers.SimulatorEventType.TRANSITION
    )
    self.assertEqual(
        events[3].event_type, simulator_observers.SimulatorEventType.TRANSITION
    )
    self.assertEqual(
        events[4].event_type, simulator_observers.SimulatorEventType.TAKE_IMAGE
    )

  def test_event_observer_reset_resets_events(self):
    observer = simulator_observers.EventObserver()
    grid = microscope_utils.AtomicGridMaterialFrame(
        microscope_utils.AtomicGrid(np.zeros((5, 2)), np.zeros(5))
    )
    fov = microscope_utils.MicroscopeFieldOfView(
        geometry.PointMaterialFrame(geometry.Point((0.0, 0.0))),
        geometry.PointMaterialFrame(geometry.Point((1.0, 1.0))),
    )

    observer.observe_reset(grid, fov)
    observer.observe_transition(dt.timedelta(seconds=1.0), grid)

    self.assertLen(observer.events, 2)

    observer.observe_reset(grid, fov)

    self.assertLen(observer.events, 1)


if __name__ == '__main__':
  absltest.main()
