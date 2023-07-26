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

r"""A script for aligning trajectories iteratively.

"""

import dataclasses
import tempfile

from absl import app
from etils import eapp
from etils import epath
from etils import etqdm
import numpy as np
from putting_dune import alignment
from putting_dune import geometry
from putting_dune import io as pdio
from putting_dune import microscope_utils


@dataclasses.dataclass
class Args:
  """Command line arguments."""

  source_path: epath.Path
  target_path: epath.Path
  aligner_url: str
  history_length: int = 5
  alignment_iterations: int = 1
  base_step_size: float = 1
  hybrid: bool = False
  relabel: bool = False


def do_alignment(
    trajectory: microscope_utils.Trajectory,
    args: Args,
    aligner: alignment.ImageAligner,
) -> microscope_utils.Trajectory:
  """Run the iterative aligner on a trajectory.

  Args:
    trajectory: Trajectory (Python object, not pb2 format).
    args: Command line arguments.
    aligner: Aligner to use.

  Returns:
    Aligned trajectory.
  """
  n_iters = args.alignment_iterations
  for i in range(1, n_iters + 1):
    aligned_observations = []
    cumulative_shift = np.zeros((2,))

    step_size = args.base_step_size + (1 - args.base_step_size) * i / n_iters
    aligner.reset(args.history_length)

    for observation in trajectory.observations:
      fov = observation.fov
      shifted_fov = fov.shift(
          shift=geometry.PointMaterialFrame(geometry.Point(-cumulative_shift))
      )
      extracted_grid, new_shift, _ = aligner(observation.image, shifted_fov)
      cumulative_shift = cumulative_shift + new_shift * step_size
      shifted_fov = observation.fov.shift(
          shift=geometry.PointMaterialFrame(geometry.Point(-cumulative_shift))
      )
      aligned_observation = microscope_utils.MicroscopeObservation(
          extracted_grid if args.relabel else observation.grid,
          shifted_fov,
          observation.controls,
          observation.elapsed_time,
          observation.image,
          observation.label_image,
      )
      aligned_observations.append(aligned_observation)

    trajectory = microscope_utils.Trajectory(aligned_observations)
  return trajectory


def main(args: Args) -> None:
  trajectories = []
  if args.source_path.is_dir():
    files = list(args.source_path.glob('*.recordio'))
  else:
    files = [args.source_path]
  for file in files:
    trajectories.extend(pdio.read_records(file, microscope_utils.Trajectory))

  with tempfile.TemporaryDirectory() as tmpdir:
    aligner = alignment.ImageAligner.from_url(
        args.aligner_url, workdir=tmpdir, hybrid=args.hybrid
    )

    aligned_trajectories = []
    for trajectory in etqdm.tqdm(trajectories):
      aligned_trajectory = do_alignment(trajectory, args, aligner)
      aligned_trajectories.append(aligned_trajectory)

  pdio.write_records(args.target_path, aligned_trajectories)


if __name__ == '__main__':
  app.run(main, flags_parser=eapp.make_flags_parser(Args))
