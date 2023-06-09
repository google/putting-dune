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

"""Tests for action_adapters."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from putting_dune import action_adapters
from putting_dune import constants
from putting_dune import geometry
from putting_dune import graphene
from putting_dune import microscope_utils
from putting_dune import test_utils


class ActionAdaptersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)

  @parameterized.named_parameters(
      dict(
          testcase_name='move_up',
          initial_position=np.asarray([0.5, 0.5]),
          delta=np.asarray([0.0, 0.1]),
          expected_position=np.asarray([0.5, 0.6]),
      ),
      dict(
          testcase_name='move_left',
          initial_position=np.asarray([0.3, 0.78]),
          delta=np.asarray([-0.03, 0.0]),
          expected_position=np.asarray([0.27, 0.78]),
      ),
      dict(
          testcase_name='move_diagonally',
          initial_position=np.asarray([0.62, 0.73]),
          delta=np.asarray([0.1, -0.07]),
          expected_position=np.asarray([0.72, 0.66]),
      ),
      dict(
          testcase_name='move_edge',
          initial_position=np.asarray([0.95, 0.1]),
          delta=np.asarray([0.6, 0.0]),
          expected_position=np.asarray([1.0, 0.1]),
      ),
  )
  def test_delta_adapter_moves_beam_by_small_position(
      self,
      initial_position: np.ndarray,
      delta: np.ndarray,
      expected_position: np.ndarray,
  ):
    action_adapter = action_adapters.DeltaPositionActionAdapter(self.rng)
    action_adapter.reset()
    # Manually set the beam position.
    action_adapter.beam_pos = initial_position

    action_adapter.get_action(
        test_utils.create_single_silicon_observation(
            self.rng
        ),  # Observation not used.
        delta,
    )

    np.testing.assert_allclose(action_adapter.beam_pos, expected_position)

  def test_delta_adapter_has_acceptable_action_spec(self):
    action_adapter = action_adapters.DeltaPositionActionAdapter(self.rng)
    action_adapter.reset()

    sampled_action = action_adapter.action_spec.generate_value()
    simulator_controls = action_adapter.get_action(
        test_utils.create_single_silicon_observation(
            self.rng
        ),  # Observation not used.
        sampled_action,
    )

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(simulator_controls[0], microscope_utils.BeamControl)

  @parameterized.parameters(
      (np.asarray([0.0, 0.0]),),
      (np.asarray([0.5, 1.0]),),
      (np.asarray([0.78, 0.88]),),
  )
  def test_direct_adapter_passes_action_correctly(self, action):
    action_adapter = action_adapters.DirectActionAdapter()
    unused_observation = test_utils.create_single_silicon_observation(self.rng)

    controls = action_adapter.get_action(unused_observation, action)

    self.assertLen(controls, 1)
    np.testing.assert_allclose(
        np.asarray(controls[0].position.coords).reshape(-1), action
    )

  def test_direct_adapter_has_acceptable_action_spec(self):
    action_adapter = action_adapters.DirectActionAdapter()

    sampled_action = action_adapter.action_spec.generate_value()
    simulator_controls = action_adapter.get_action(
        test_utils.create_single_silicon_observation(
            self.rng
        ),  # Observation not used.
        sampled_action,
    )

    self.assertLen(simulator_controls, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name='on_silicon',
          silicon_position=np.asarray([[0.4, 0.4]]),
          delta=np.asarray([0.0, 0.0]),
          expected_position=np.asarray([0.4, 0.4]),
      ),
      dict(
          testcase_name='above_silicon',
          silicon_position=np.asarray([[0.5, 0.75]]),
          delta=np.asarray([0.0, 0.2]),
          expected_position=np.asarray([0.5, 0.7784]),
      ),
      dict(
          testcase_name='beside_silicon',
          silicon_position=np.asarray([[0.31, 0.31]]),
          delta=np.asarray([-0.1, 0.0]),
          expected_position=np.asarray([0.2958, 0.31]),
      ),
      dict(
          testcase_name='diagonal',
          silicon_position=np.asarray([[0.92, 0.11]]),
          delta=np.asarray([-1.0, 0.75]),
          expected_position=np.asarray([0.778, 0.2165]),
      ),
  )
  def test_relative_adapter_returns_position_relative_to_silicon(
      self,
      silicon_position: np.ndarray,
      delta: np.ndarray,
      expected_position: np.ndarray,
  ) -> None:
    # Shift the observed grid so the silicon is in the specified position.
    # Initially, the silicon is at (0.5, 0.5).
    observation = test_utils.create_single_silicon_observation(self.rng)
    shifted_grid = microscope_utils.AtomicGridMicroscopeFrame(
        microscope_utils.AtomicGrid(
            atom_positions=(
                observation.grid.atom_positions + silicon_position - 0.5
            ),
            atomic_numbers=observation.grid.atomic_numbers,
        )
    )
    observation = microscope_utils.MicroscopeObservation(
        grid=shifted_grid,
        fov=observation.fov,
        controls=observation.controls,
        elapsed_time=observation.elapsed_time,
    )

    action_adapter = action_adapters.RelativeToSiliconActionAdapter()

    simulator_controls = action_adapter.get_action(
        observation,
        delta,
    )

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(simulator_controls[0], microscope_utils.BeamControl)
    np.testing.assert_allclose(
        np.asarray(simulator_controls[0].position.coords).reshape(-1),
        expected_position,
    )

  def test_material_relative_adapter_has_acceptable_action_spec(self) -> None:
    action_adapter = (
        action_adapters.RelativeToSiliconMaterialFrameActionAdapter()
    )
    action_adapter.reset()

    sampled_action = action_adapter.action_spec.generate_value()
    simulator_controls = action_adapter.get_action(
        test_utils.create_single_silicon_observation(self.rng),
        sampled_action,
    )

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(simulator_controls[0], microscope_utils.BeamControl)

  def test_material_relative_adapter_changes_beam_position_in_angstroms(self):
    action_adapter = (
        action_adapters.RelativeToSiliconMaterialFrameActionAdapter()
    )
    action_adapter.reset()

    # Relative position in angstroms
    action_in_angstroms = np.asarray([1.0, 0.0], dtype=np.float32)
    observation = test_utils.create_single_silicon_observation(self.rng)

    # We have to derive the silicon position in angstroms to calculate
    # our new position
    silicon_position = graphene.get_silicon_positions(
        observation.grid
    ).reshape((2,))
    silicon_material_position = (
        observation.fov.microscope_frame_to_material_frame(silicon_position)
    )
    new_beam_position = observation.fov.material_frame_to_microscope_frame(
        silicon_material_position + action_in_angstroms
    )

    control = action_adapter.get_action(observation, action_in_angstroms)
    self.assertLen(control, 1)
    np.testing.assert_equal(np.asarray(control[0].position), new_beam_position)

  def test_relative_adapter_has_acceptable_action_spec(self) -> None:
    action_adapter = action_adapters.RelativeToSiliconActionAdapter()
    action_adapter.reset()

    sampled_action = action_adapter.action_spec.generate_value()
    simulator_controls = action_adapter.get_action(
        test_utils.create_single_silicon_observation(self.rng),
        sampled_action,
    )

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(simulator_controls[0], microscope_utils.BeamControl)

  # TODO(joshgreaves): Write tests for relative adapter dwell time range.

  def test_relative_adapter_max_distance_changes_allowed_beam_placement(self):
    unit_adapter = action_adapters.RelativeToSiliconActionAdapter()
    long_adapter = action_adapters.RelativeToSiliconActionAdapter(
        max_distance_angstroms=3.0
    )
    unit_adapter.reset()
    long_adapter.reset()

    # Action points directly to the right.
    action = np.asarray([1.0, 0.0], dtype=np.float32)
    observation = test_utils.create_single_silicon_observation(self.rng)

    unit_control = unit_adapter.get_action(observation, action)
    long_control = long_adapter.get_action(observation, action)

    si_pos = graphene.get_single_silicon_position(observation.grid)
    si_pos_material_frame = observation.fov.microscope_frame_to_material_frame(
        si_pos
    )
    unit_control_pos_material_frame = (
        observation.fov.microscope_frame_to_material_frame(
            geometry.PointMicroscopeFrame(unit_control[0].position)
        )
    )
    long_control_pos_material_frame = (
        observation.fov.microscope_frame_to_material_frame(
            geometry.PointMicroscopeFrame(long_control[0].position)
        )
    )

    unit_control_distance = np.linalg.norm(
        np.asarray(unit_control_pos_material_frame.coords)
        - si_pos_material_frame
    )
    long_control_distance = np.linalg.norm(
        np.asarray(long_control_pos_material_frame.coords)
        - si_pos_material_frame
    )

    self.assertAlmostEqual(
        unit_control_distance, constants.CARBON_BOND_DISTANCE_ANGSTROMS
    )
    self.assertAlmostEqual(long_control_distance, 3.0)


if __name__ == '__main__':
  absltest.main()
