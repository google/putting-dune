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
from putting_dune import graphene
from putting_dune import simulator_utils


_EMPTY_GRID = simulator_utils.AtomicGrid(np.zeros(2), np.asarray([6]))


def _make_unit_hexagonal_grid(
    rng: np.random.Generator,
) -> simulator_utils.AtomicGrid:
  material = graphene.PristineSingleDopedGraphene(rng)
  atom_positions = (
      material.atom_positions / graphene.CARBON_BOND_DISTANCE_ANGSTROMS
  )
  return simulator_utils.AtomicGrid(atom_positions, material.atomic_numbers)


class ActionAdaptersTest(parameterized.TestCase):

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
    rng = np.random.default_rng(0)

    action_adapter = action_adapters.DeltaPositionActionAdapter(rng)
    action_adapter.reset()
    # Manually set the beam position.
    action_adapter.beam_pos = initial_position

    action_adapter.get_action(
        _EMPTY_GRID,  # Grid not used.
        delta,
    )

    np.testing.assert_allclose(action_adapter.beam_pos, expected_position)

  def test_delta_adapter_has_acceptable_action_spec(self):
    rng = np.random.default_rng(0)

    action_adapter = action_adapters.DeltaPositionActionAdapter(rng)
    action_adapter.reset()

    sampled_action = action_adapter.action_spec.generate_value()
    simulator_controls = action_adapter.get_action(
        _EMPTY_GRID,  # Grid not used.
        sampled_action,
    )

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(
        simulator_controls[0], simulator_utils.BeamControl
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='above_silicon',
          silicon_position=np.asarray([[0.5, 0.75]]),
          delta=np.asarray([0.0, 0.035]),
          expected_position=np.asarray([0.5, 0.785]),
      ),
      dict(
          testcase_name='beside_silicon',
          silicon_position=np.asarray([[0.31, 0.31]]),
          delta=np.asarray([-0.1, 0.0]),
          expected_position=np.asarray([0.21, 0.31]),
      ),
      dict(
          testcase_name='diagonal',
          silicon_position=np.asarray([[0.92, 0.11]]),
          delta=np.asarray([-0.02, 0.1]),
          expected_position=np.asarray([0.9, 0.21]),
      ),
  )
  def test_relative_adapter_returns_position_relative_to_silicon(
      self,
      silicon_position: np.ndarray,
      delta: np.ndarray,
      expected_position: np.ndarray,
  ) -> None:
    grid = _make_unit_hexagonal_grid(np.random.default_rng(0))
    # Shift the whole grid so the silicon is in the specified position.
    current_silicon_position = grid.atom_positions[
        grid.atomic_numbers == graphene.SILICON
    ]
    grid.atom_positions[:, :] += silicon_position - current_silicon_position

    action_adapter = action_adapters.RelativeToSiliconActionAdapter()

    simulator_controls = action_adapter.get_action(
        grid,
        delta,
    )

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(
        simulator_controls[0], simulator_utils.BeamControl
    )
    np.testing.assert_allclose(
        np.asarray(simulator_controls[0].position), expected_position
    )

  @parameterized.parameters(0.01, 0.1, 0.8, 2.0, 200.0)
  def test_relative_adapter_adapts_to_grid_scale(
      self,
      grid_scale: float,
  ) -> None:
    grid = _make_unit_hexagonal_grid(np.random.default_rng(0))
    # Shift the whole grid so the silicon is at the origin.
    current_silicon_position = grid.atom_positions[
        grid.atomic_numbers == graphene.SILICON
    ]
    grid.atom_positions[:, :] -= current_silicon_position
    grid.atom_positions[:, :] *= grid_scale  # Scale the grid as specified.

    action_adapter = action_adapters.RelativeToSiliconActionAdapter()
    delta = np.asarray([0.0, 1.0])

    simulator_controls = action_adapter.get_action(grid, delta)

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(
        simulator_controls[0], simulator_utils.BeamControl
    )
    np.testing.assert_allclose(
        np.asarray(simulator_controls[0].position), delta * grid_scale
    )

  def test_relative_adapter_has_acceptable_action_spec(self) -> None:
    material = graphene.PristineSingleDopedGraphene(np.random.default_rng(0))
    grid = simulator_utils.AtomicGrid(
        material.atom_positions, material.atomic_numbers
    )
    action_adapter = action_adapters.RelativeToSiliconActionAdapter()
    action_adapter.reset()

    sampled_action = action_adapter.action_spec.generate_value()
    simulator_controls = action_adapter.get_action(
        grid,
        sampled_action,
    )

    self.assertLen(simulator_controls, 1)
    self.assertIsInstance(
        simulator_controls[0], simulator_utils.BeamControl
    )


if __name__ == '__main__':
  absltest.main()
