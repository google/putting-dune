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
"""Tests for data utils."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from putting_dune import alignment_utils
from putting_dune import graphene


class AlignmentUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)

  def test_atom_classifer(self):
    material = graphene.PristineSingleDopedGraphene(self.rng)
    coordinates = material.atom_positions
    classifier = alignment_utils.get_carbon_clusterer(coordinates)

    classes = alignment_utils.classify_carbon_types(coordinates, classifier)
    distances = np.linalg.norm(
        coordinates[None] - coordinates[:, None], axis=-1
    )
    nearest_neighbors = distances.argsort(-1)

    neighbor_classes = classes[nearest_neighbors[:, 1]]

    np.testing.assert_allclose(classes, 1 - neighbor_classes, rtol=1e-6)

  def test_offset_calculation(self):
    material = graphene.PristineSingleDopedGraphene(self.rng)
    coordinates = material.atom_positions
    shift = np.array((0.5, 0.5))
    shifted_coordinates = coordinates + shift
    offset = alignment_utils.get_offsets(coordinates, shifted_coordinates)
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
  def test_scale_estimation(
      self, scale, grid_columns
  ):
    material = graphene.PristineSingleDopedGraphene(
        self.rng, grid_columns=grid_columns
    )
    coordinates = material.atom_positions

    est_scale = alignment_utils.get_graphene_scale_factor(coordinates*scale)

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
    material = graphene.PristineSingleDopedGraphene(
        self.rng, grid_columns=grid_columns
    )
    aligner = alignment_utils.IterativeAlignmentFiltering(
        accumulate_merged=accumulate_merged,
        clique_merging=clique_merging,
    )
    coordinates = material.atom_positions

    drift = np.array([0.1, 0.1])
    shifts = np.stack([drift]*10, 0).cumsum(0)
    aligner.align(coordinates)

    for shift in shifts:
      shifted_coordinates = coordinates - shift
      joined_coordinates, assessed_shift = aligner.align(shifted_coordinates)
      reference_coordinates = aligner.recent_observations[0]
      np.testing.assert_allclose(
          np.sort((joined_coordinates).reshape(-1)),
          np.sort(reference_coordinates.reshape(-1)),
          atol=1e-3,
      )
      np.testing.assert_allclose(assessed_shift,
                                 drift, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
