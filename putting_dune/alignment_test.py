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

"""Tests for alignment.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from putting_dune import alignment
from putting_dune import graphene
from putting_dune import microscope_utils


class AlignmentUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)
    self.material = graphene.PristineSingleDopedGraphene()
    self.material.reset(self.rng)

  def test_atom_classifer(self):
    coordinates = self.material.atom_positions
    classifier = alignment.get_lattice_clusterer(coordinates)

    classes = alignment.classify_lattice_types(coordinates, classifier)
    distances = np.linalg.norm(
        coordinates[None] - coordinates[:, None], axis=-1
    )
    nearest_neighbors = distances.argsort(-1)

    neighbor_classes = classes[nearest_neighbors[:, 1]]

    np.testing.assert_allclose(classes, 1 - neighbor_classes, rtol=1e-6)

  def test_offset_calculation(self):
    coordinates = self.material.atom_positions
    shift = np.array((0.5, 0.5))
    shifted_coordinates = coordinates + shift
    offset = alignment.get_offsets(coordinates, shifted_coordinates)
    np.testing.assert_allclose(offset.mean(0), shift, rtol=1e-6)


class GrapheneScaleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)

  @parameterized.named_parameters(
      dict(
          testcase_name='small_default',
          grid_columns=10,
          scale=1,
      ),
      dict(
          testcase_name='large_default',
          grid_columns=30,
          scale=1,
      ),
      dict(
          testcase_name='small_microscope_scale',
          grid_columns=10,
          scale=28,
      ),
      dict(
          testcase_name='large_microscope_scale',
          grid_columns=30,
          scale=28,
      ),
  )
  def test_scale_estimation(self, scale, grid_columns):
    material = graphene.PristineSingleDopedGraphene(grid_columns=grid_columns)
    material.reset(self.rng)
    coordinates = material.atom_positions

    est_scale = alignment.get_graphene_scale_factor(coordinates * scale)

    np.testing.assert_allclose(np.array(est_scale), np.array(scale), rtol=1e-3)


class IteratedAlignmentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)

  @parameterized.named_parameters(
      dict(
          testcase_name='small_defaults',
          grid_columns=10,
          accumulate_merged=False,
          clique_merging=False,
      ),
      dict(
          testcase_name='large_defaults',
          grid_columns=30,
          accumulate_merged=False,
          clique_merging=False,
      ),
      dict(
          testcase_name='small_pairwise',
          grid_columns=10,
          accumulate_merged=False,
          clique_merging=True,
      ),
      dict(
          testcase_name='small_accumulate_merged',
          grid_columns=10,
          accumulate_merged=True,
          clique_merging=False,
      ),
  )
  def test_iterative_alignment(
      self, accumulate_merged, clique_merging, grid_columns
  ):
    material = graphene.PristineSingleDopedGraphene(grid_columns=grid_columns)
    material.reset(self.rng)
    aligner = alignment.IterativeAlignmentFiltering(
        accumulate_merged=accumulate_merged,
        clique_merging=clique_merging,
    )

    observation = microscope_utils.AtomicGrid(
        material.atom_positions, material.atomic_numbers
    )

    drift = np.array([0.1, 0.1])
    shifts = np.stack([drift] * 10, 0).cumsum(0)
    aligner.align(observation)
    cumulative_shift = np.zeros_like(drift)

    for shift in shifts:
      shifted_observation = observation.shift(-shift)
      shifted_observation = shifted_observation.shift(cumulative_shift)
      joined_observation, assessed_shift = aligner.align(shifted_observation)
      cumulative_shift += assessed_shift
      reference_coordinates = aligner.recent_observations[0]
      np.testing.assert_allclose(
          np.sort((joined_observation.atom_positions).reshape(-1)),
          np.sort(reference_coordinates.reshape(-1)),
          atol=1e-3,
      )
      np.testing.assert_allclose(assessed_shift, drift, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
