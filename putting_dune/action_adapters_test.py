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
"""Tests for action_adapters."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from putting_dune import action_adapters
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
    shifted_grid = microscope_utils.AtomicGrid(
        atom_positions=observation.grid.atom_positions + silicon_position - 0.5,
        atomic_numbers=observation.grid.atomic_numbers,
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


if __name__ == '__main__':
  absltest.main()
