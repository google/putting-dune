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
"""Tests for simulator."""

import datetime as dt
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from putting_dune import action_adapters
from putting_dune import graphene
from putting_dune import simulator
from putting_dune import simulator_observers
from putting_dune import simulator_utils
from shapely import geometry


_ARBITRARY_CONTROL = simulator_utils.BeamControl(
    geometry.Point(0.5, 0.7), dt.timedelta(seconds=1.0)
)
_ARBITRARY_GRID = simulator_utils.AtomicGrid(
    np.arange(10, dtype=np.float32).reshape(5, 2),
    np.asarray([14, 6, 6, 6, 6], dtype=np.float32),
)  # Si, C, C, C, C.


def _get_mock_material(
    material: graphene.Material, fov: simulator_utils.SimulatorFieldOfView
) -> mock.MagicMock:
  material = mock.create_autospec(material, spec_set=True)
  material.get_atoms_in_bounds.return_value = _ARBITRARY_GRID
  # Place the silicon in the center of the FOV.
  material.get_silicon_position.return_value = (
      np.asarray(fov.lower_left) + np.asarray(fov.upper_right)
  ) / 2
  return material


class SimulatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)
    self._material = graphene.PristineSingleDopedGraphene(self._rng)

  @parameterized.named_parameters(
      dict(
          testcase_name='unit_fov',
          control_position=_ARBITRARY_CONTROL.position,
          fov_lower_left=geometry.Point((0.0, 0.0)),
          fov_upper_right=geometry.Point((1.0, 1.0)),
          predicted_control_position=_ARBITRARY_CONTROL.position,
      ),
      dict(
          testcase_name='corner_of_fov',
          control_position=geometry.Point((0.0, 1.0)),
          fov_lower_left=geometry.Point((-5.5, -6.3)),
          fov_upper_right=geometry.Point((12.0, 9.1)),
          predicted_control_position=geometry.Point((-5.5, 9.1)),
      ),
  )
  def test_simulator_positions_probe_correctly(
      self,
      control_position: geometry.Point,
      fov_lower_left: geometry.Point,
      fov_upper_right: geometry.Point,
      predicted_control_position: geometry.Point,
  ) -> None:
    sim = simulator.PuttingDuneSimulator(self._material)
    sim.reset()
    sim._fov = simulator_utils.SimulatorFieldOfView(
        fov_lower_left, fov_upper_right
    )
    # Mock the material to inspect calls to it.
    sim.material = _get_mock_material(sim.material, sim._fov)

    control = simulator_utils.BeamControl(
        control_position, dt.timedelta(seconds=1.5)
    )
    sim.step_and_image([control])

    # Check that the probe position was the specified location for each
    # rate calculation.
    passed_probe_control = sim.material.apply_control.call_args[0][0]
    self.assertEqual(sim.material.apply_control.call_count, 1)
    np.testing.assert_allclose(
        np.asarray(passed_probe_control.position),
        np.asarray(predicted_control_position),
    )

  def test_simulator_step_takes_multiple_probe_positions(self):
    sim = simulator.PuttingDuneSimulator(self._material)
    sim.reset()
    # Mock the material to inspect calls to it.
    sim.material = _get_mock_material(sim.material, sim._fov)

    controls = [
        simulator_utils.BeamControl(
            geometry.Point(0.5, 0.7), dt.timedelta(seconds=1.0)
        ),
        simulator_utils.BeamControl(
            geometry.Point(0.6, 0.8), dt.timedelta(seconds=1.0)
        ),
    ]
    sim.step_and_image(controls)

    self.assertEqual(sim.material.apply_control.call_count, 2)

  def test_simulator_progresses_time_correctly(self):
    time_per_control = [dt.timedelta(seconds=x) for x in (1.5, 3.0, 7.23)]
    controls = [
        simulator_utils.BeamControl(geometry.Point(0.5, 0.7), x)
        for x in time_per_control
    ]
    sim = simulator.PuttingDuneSimulator(self._material)
    sim.reset()

    sim.step_and_image(controls)

    # Time of controls + 2 images, generated on reset and after the first step.
    predicted_elapsed_time = sum(time_per_control, start=dt.timedelta())
    predicted_elapsed_time += 2 * sim._image_duration

    self.assertEqual(sim.elapsed_time, predicted_elapsed_time)

  def test_simulator_behaves_deterministically_with_seeded_components(self):
    observations = []
    for _ in range(2):
      material = graphene.PristineSingleDopedGraphene(np.random.default_rng(0))
      sim = simulator.PuttingDuneSimulator(material)
      sim.reset()

      observations.append(sim.step_and_image([_ARBITRARY_CONTROL]))

    np.testing.assert_allclose(
        observations[0].grid.atom_positions, observations[1].grid.atom_positions
    )
    np.testing.assert_allclose(
        observations[0].grid.atomic_numbers, observations[1].grid.atomic_numbers
    )
    np.testing.assert_allclose(
        np.asarray(observations[0].last_probe_position),
        np.asarray(observations[1].last_probe_position),
    )
    self.assertEqual(observations[0].elapsed_time, observations[1].elapsed_time)

  def test_simulator_reset_correctly_resets_state(self):
    sim = simulator.PuttingDuneSimulator(self._material)
    sim.reset()

    sim.step_and_image([_ARBITRARY_CONTROL])

    elapsed_time_before_reset = sim.elapsed_time
    atom_positions_before_reset = np.copy(sim.material.atom_positions)
    atomic_numbers_before_reset = np.copy(sim.material.atomic_numbers)

    sim.reset()

    elapsed_time_after_reset = sim.elapsed_time
    atom_positions_after_reset = np.copy(sim.material.atom_positions)
    atomic_numbers_after_reset = np.copy(sim.material.atomic_numbers)

    # Check elapsed time was reset correctly.
    self.assertNotEqual(elapsed_time_after_reset, elapsed_time_before_reset)
    self.assertEqual(elapsed_time_after_reset, sim._image_duration)

    # Check the material was reinitialized.
    self.assertTrue(
        (atom_positions_before_reset != atom_positions_after_reset).any()
    )
    self.assertTrue(
        (atomic_numbers_before_reset != atomic_numbers_after_reset).any()
    )

  def test_simulator_calls_observers_correctly(self):
    observer = simulator_observers.EventObserver()

    sim = simulator.PuttingDuneSimulator(self._material, observers=(observer,))

    obs = sim.reset()

    # Position control near the silicon to trigger an event.
    # We can reuse the relative action adapter to make things easier.
    action_adapter = action_adapters.RelativeToSiliconActionAdapter()
    controls = action_adapter.get_action(obs.grid, np.asarray([0.5, 0.5]))
    sim.step_and_image(controls)

    events = observer.events
    # Expected events: reset, image, action, (many) transition(s), image.
    self.assertGreaterEqual(len(events), 5)
    self.assertEqual(
        events[0].event_type, simulator_observers.SimulatorEventType.RESET
    )
    self.assertEqual(
        events[1].event_type, simulator_observers.SimulatorEventType.TAKE_IMAGE
    )
    self.assertEqual(
        events[2].event_type,
        simulator_observers.SimulatorEventType.APPLY_CONTROL,
    )
    self.assertEqual(
        events[3].event_type, simulator_observers.SimulatorEventType.TRANSITION
    )
    self.assertEqual(
        events[-1].event_type, simulator_observers.SimulatorEventType.TAKE_IMAGE
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='x_too_low',
          target_percentiles=np.asarray([0.2, 0.4]),
          expect_adjustment=True,
      ),
      dict(
          testcase_name='y_too_high',
          target_percentiles=np.asarray([0.35, 0.95]),
          expect_adjustment=True,
      ),
      dict(
          testcase_name='both_just_out',
          target_percentiles=np.asarray([0.751, 0.249]),
          expect_adjustment=True,
      ),
      dict(
          testcase_name='just_in',
          target_percentiles=np.asarray([0.749, 0.250]),
          expect_adjustment=False,
      ),
  )
  def test_simulator_correctly_updates_fov(
      self, target_percentiles: np.ndarray, expect_adjustment: bool
  ):
    event_observer = simulator_observers.EventObserver()
    sim = simulator.PuttingDuneSimulator(
        self._material, observers=(event_observer,)
    )
    sim.reset()

    # Manually move the field of view so that the silicon is near the edge.
    silicon_position = sim.material.get_silicon_position()
    fov_width = 10.0
    lower_left = silicon_position - fov_width * target_percentiles
    original_fov = simulator_utils.SimulatorFieldOfView(
        lower_left=geometry.Point(lower_left),
        upper_right=geometry.Point(lower_left + fov_width),
    )
    sim._fov = original_fov

    # Apply a control for 0 seconds to ensure the silicon doesn't move.
    obs = sim.step_and_image(
        [
            simulator_utils.BeamControl(
                geometry.Point((1.0, 1.0)), dt.timedelta(seconds=0.0)
            )
        ]
    )

    # Check the returned observation is correct.
    silicon_position = graphene.get_silicon_positions(obs.grid).reshape(-1)
    self.assertLen(silicon_position, 2)  # x, y coordinates.
    if expect_adjustment:
      np.testing.assert_allclose(silicon_position, np.asarray([0.5, 0.5]))
    else:
      np.testing.assert_allclose(silicon_position, target_percentiles)

    # Check the observer got the right number of image events.
    image_events = []
    for event in event_observer.events:
      if event.event_type == simulator_observers.SimulatorEventType.TAKE_IMAGE:
        image_events.append(event)
    if expect_adjustment:
      self.assertLen(image_events, 3)  # rest, after step, after adjust.
    else:
      self.assertLen(image_events, 2)  # rest, after step.
    self.assertEqual(image_events[1].event_data['fov'], original_fov)


if __name__ == '__main__':
  absltest.main()
