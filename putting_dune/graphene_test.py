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

"""Tests for graphene."""

import datetime as dt
from unittest import mock

from absl.testing import absltest
import numpy as np
from putting_dune import constants
from putting_dune import geometry
from putting_dune import graphene
from putting_dune import microscope_utils
from sklearn import neighbors

_ARBITRARY_CONTROL = microscope_utils.BeamControlMaterialFrame(
    microscope_utils.BeamControl(
        geometry.Point(0.5, 0.7), dt.timedelta(seconds=1.0)
    )
)


class GrapheneTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)

  def test_generate_pristine_graphene_has_correct_scale(self):
    positions = graphene.generate_pristine_graphene(self.rng)

    # Choose a position near the middle of the sheet.
    sq_distances = np.sum(positions**2, axis=1)
    middle_idx = np.argmin(sq_distances)
    middle_position = positions[middle_idx]

    # Get the neighbors of that position.
    neighbor_distances, _ = (
        neighbors.NearestNeighbors(
            n_neighbors=1 + 3,
            metric='l2',
            algorithm='brute',
        )
        .fit(positions)
        .kneighbors(middle_position.reshape(1, 2))
    )
    neighbor_distances = neighbor_distances.reshape(-1)

    self.assertEqual(neighbor_distances[0], 0.0)  # Self-distance.
    self.assertAlmostEqual(
        neighbor_distances[1], constants.CARBON_BOND_DISTANCE_ANGSTROMS
    )
    self.assertAlmostEqual(
        neighbor_distances[2], constants.CARBON_BOND_DISTANCE_ANGSTROMS
    )
    self.assertAlmostEqual(
        neighbor_distances[3], constants.CARBON_BOND_DISTANCE_ANGSTROMS
    )

  def test_atoms_in_graphene_are_a_fixed_distance_apart(self):
    material = graphene.PristineSingleDopedGraphene()
    material.reset(self.rng)

    num_atoms = material.atom_positions.shape[0]
    chosen_idx = self.rng.choice(num_atoms)
    chosen_atom_position = material.atom_positions[chosen_idx].reshape(1, 2)
    neighbor_distances, _ = material.nearest_neighbors.kneighbors(
        chosen_atom_position
    )
    neighbor_distances = neighbor_distances.reshape(-1)

    self.assertEqual(neighbor_distances[0], 0.0)  # Self-distance.
    self.assertAlmostEqual(
        neighbor_distances[1], constants.CARBON_BOND_DISTANCE_ANGSTROMS
    )
    self.assertAlmostEqual(
        neighbor_distances[2], constants.CARBON_BOND_DISTANCE_ANGSTROMS
    )
    self.assertAlmostEqual(
        neighbor_distances[3], constants.CARBON_BOND_DISTANCE_ANGSTROMS
    )

  def test_different_seeds_give_different_orientations_of_graphene(self):
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    material1 = graphene.PristineSingleDopedGraphene()
    material2 = graphene.PristineSingleDopedGraphene()
    material1.reset(rng1)
    material2.reset(rng2)

    # Pick an atom at random from both sheets.
    num_atoms1 = material1.atom_positions.shape[0]
    num_atoms2 = material2.atom_positions.shape[0]
    chosen_idx1 = self.rng.choice(num_atoms1)
    chosen_idx2 = self.rng.choice(num_atoms2)
    chosen_atom_position1 = material1.atom_positions[chosen_idx1].reshape(1, 2)
    chosen_atom_position2 = material2.atom_positions[chosen_idx2].reshape(1, 2)

    # Get the nearest neighbors for both atoms.
    _, nearest_neighbors1 = material1.nearest_neighbors.kneighbors(
        chosen_atom_position1
    )
    _, nearest_neighbors2 = material1.nearest_neighbors.kneighbors(
        chosen_atom_position1
    )

    # Remove the self atom in the nearest neighbors.
    nearest_neighbors1 = nearest_neighbors1.reshape(-1)
    nearest_neighbors2 = nearest_neighbors2.reshape(-1)
    nearest_neighbors1 = material1.atom_positions[nearest_neighbors1][1:]
    nearest_neighbors2 = material2.atom_positions[nearest_neighbors2][1:]

    # Get the angle to the nearest neighbor that is closest to directly up.
    up_vector = np.asarray([0.0, 1.0])

    vecs1 = nearest_neighbors1 - chosen_atom_position1
    vecs1 = vecs1 / np.linalg.norm(vecs1, axis=1, keepdims=True)
    angle1 = np.min(np.arccos(vecs1 @ up_vector))

    vecs2 = nearest_neighbors2 - chosen_atom_position2
    vecs2 = vecs2 / np.linalg.norm(vecs2, axis=1, keepdims=True)
    angle2 = np.min(np.arccos(vecs2 @ up_vector))

    # The way angles are calculated here ignore signs, but it is
    # unlikely that two random rotations would lead to the same
    # rotation but in a different direction.
    self.assertNotAlmostEqual(angle1, angle2)

  def test_graphene_only_contains_a_single_dopant(self):
    material = graphene.PristineSingleDopedGraphene()
    material.reset(self.rng)

    num_silicon = np.sum(material.atomic_numbers == constants.SILICON)
    self.assertEqual(num_silicon, 1)

  def test_get_atoms_in_bounds_gets_correct_atoms(self):
    material = graphene.PristineSingleDopedGraphene()
    material.reset(self.rng)

    lower_left = geometry.Point((0.0, 0.0))
    upper_right = geometry.Point((
        2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
    ))
    grid = material.get_atoms_in_bounds(lower_left, upper_right)

    # We are getting a 2 unit cell by 2 unit cell distance.
    # First, check the number of atoms we get is in the right ballpark.
    # We probably expect 4 atoms, so between 2 and 6 gives some wiggle room.
    num_atoms = grid.atom_positions.shape[0]
    self.assertBetween(num_atoms, 2, 6)
    for atom_position in grid.atom_positions:
      self.assertBetween(
          atom_position[0],  # Check x is in bounds.
          0.0,
          2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
      )
      self.assertBetween(
          atom_position[1],  # Check y is in bounds.
          0.0,
          2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
      )

  def test_get_atoms_in_bounds_correctly_normalizes_points_in_unit_square(self):
    material = graphene.PristineSingleDopedGraphene()
    material.reset(self.rng)

    lower_left = geometry.Point((-10.0, -10.0))
    upper_right = geometry.Point((
        -10.0 + 5 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        -10.0 + 5 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
    ))
    grid = material.get_atoms_in_bounds(lower_left, upper_right)

    self.assertTrue((grid.atom_positions >= 0.0).all())
    self.assertTrue((grid.atom_positions <= 1.0).all())

  @mock.patch.object(
      graphene,
      'simple_transition_rates',
      autospec=True,
      # About 15 events should occur per second on average.
      return_value=np.full((3,), 5.0, dtype=np.float32),
  )
  def test_graphene_transitions_without_affecting_structure(self, mock_rate_fn):
    material = graphene.PristineSingleDopedGraphene(predict_rates=mock_rate_fn)
    material.reset(self.rng)

    atom_positions_before = np.copy(material.atom_positions)
    atomic_numbers_before = np.copy(material.atomic_numbers)

    material.apply_control(self.rng, _ARBITRARY_CONTROL)
    atom_positions_after = np.copy(material.atom_positions)
    atomic_numbers_after = np.copy(material.atomic_numbers)

    # Check the structure hasn't changed, but the silicon has moved.
    np.testing.assert_array_equal(atom_positions_before, atom_positions_after)
    self.assertTrue((atomic_numbers_before != atomic_numbers_after).any())

    num_silicon = np.sum(material.atomic_numbers == constants.SILICON)
    self.assertEqual(num_silicon, 1)

  @mock.patch.object(
      graphene,
      'simple_transition_rates',
      autospec=True,
      # About 15 events should occur per second on average.
      return_value=np.full((3,), 5.0, dtype=np.float32),
  )
  def test_simulator_allows_multiple_transitions_when_rates_are_high(
      self, mock_rate_fn
  ):
    material = graphene.PristineSingleDopedGraphene(predict_rates=mock_rate_fn)
    material.reset(self.rng)
    material.apply_control(self.rng, _ARBITRARY_CONTROL)

    self.assertGreater(mock_rate_fn.call_count, 1)

  def test_simulator_moves_atoms_more_frequently_with_higher_rates(self):
    num_trials = 10
    high_rate_transitioned_more_than_low_rate = []
    for _ in range(num_trials):
      # Get the number of transitions with a high rate.
      with mock.patch.object(graphene, 'simple_transition_rates') as rate_fn:
        # About 0.3 events should occur per second on average.
        rate_fn.return_value = np.full((3,), 0.1, dtype=np.float32)
        material = graphene.PristineSingleDopedGraphene(predict_rates=rate_fn)
        material.reset(self.rng)
        material.apply_control(self.rng, _ARBITRARY_CONTROL)
        low_rate_transitions = rate_fn.call_count

      # Get the number of transitions with a low rate.
      with mock.patch.object(graphene, 'simple_transition_rates') as rate_fn:
        # About 6 events should occur per second on average.
        rate_fn.return_value = np.full((3,), 2.0, dtype=np.float32)
        material = graphene.PristineSingleDopedGraphene(predict_rates=rate_fn)
        material.reset(self.rng)
        material.apply_control(self.rng, _ARBITRARY_CONTROL)
        high_rate_transitions = rate_fn.call_count

      high_rate_transitioned_more_than_low_rate.append(
          high_rate_transitions > low_rate_transitions
      )

    # Check the high rate function transitions more frequently 90% of the time.
    p = sum(high_rate_transitioned_more_than_low_rate) / num_trials
    self.assertGreaterEqual(p, 0.9)

  def test_graphene_initializes_silicon_away_from_edge(self):
    # This simply tests that the silicon isn't initialized right on the edge.
    # In fact, we should test that it isn't initialized _close_ to and edge,
    # but we will skim over this for now 🤷‍♂️.
    material = graphene.PristineSingleDopedGraphene(grid_columns=10)

    # Reset many times to check each initialization is not near an edge.
    for _ in range(100):
      material.reset(self.rng)

      neighbor_distances, _ = material.nearest_neighbors.kneighbors(
          material.get_silicon_position().reshape(1, 2)
      )
      self.assertLessEqual(
          neighbor_distances[0, -1],
          constants.CARBON_BOND_DISTANCE_ANGSTROMS + 1e-3,
      )

  def test_human_prior_rates(self):
    transition_model = graphene.HumanPriorRatePredictor()
    material = graphene.PristineSingleDopedGraphene(
        predict_rates=transition_model.predict
    )
    material.reset(self.rng)

    material.apply_control(self.rng, _ARBITRARY_CONTROL)


if __name__ == '__main__':
  absltest.main()
