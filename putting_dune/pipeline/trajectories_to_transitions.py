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

r"""A script for converting observations to transitions.

"""

import dataclasses

from absl import app
from etils import eapp
from putting_dune import io as pdio
from putting_dune import microscope_utils


@dataclasses.dataclass
class Args:
  """Command line arguments."""

  source_path: str
  target_path: str


def trajectories_to_transitions(
    trajectories: list[microscope_utils.Trajectory],
    *,
    previous_controls_at_current_timestep: bool = False,
) -> list[microscope_utils.Transition]:
  """Extracts adjacent observations from trajectories to form transitions.

  Args:
    trajectories: List of trajectories (as Python objects, not protobuf format).
    previous_controls_at_current_timestep: This flag determines whether an
      observation is labelled as (s_t, a_t) or (s_t, a_{t-1}). If
      `previous_controls_at_current_timestep` is true this means that the
      trajectory is in (s_t, a_{t-1}) form. The main place to watch out for this
      is when collecting data from the simulator which is in (s_t, a_{t-1})
      versus data from the real microscope which is (s_t, a_t).

  Returns:
    List of extracted transitions.
  """

  transitions = []
  for trajectory in trajectories:
    grid_before = None
    grid_after = None
    fov_before = None
    fov_after = None
    image_before = None
    image_after = None
    label_image_before = None
    label_image_after = None
    controls = None
    controls_before = None

    for observation in trajectory.observations:
      grid_after = observation.grid
      fov_after = observation.fov
      controls = observation.controls
      image_after = observation.image
      label_image_after = observation.label_image

      if grid_before is not None:
        transitions.append(
            microscope_utils.Transition(
                grid_before=grid_before,
                grid_after=grid_after,
                fov_before=fov_before,
                fov_after=fov_after,
                controls=controls_before
                if not previous_controls_at_current_timestep
                else controls,
                image_before=image_before,
                image_after=image_after,
                label_image_before=label_image_before,
                label_image_after=label_image_after,
            )
        )

      grid_before = grid_after
      fov_before = fov_after
      image_before = image_after
      label_image_before = label_image_after
      controls_before = controls

  return transitions


def main(args: Args) -> None:
  trajectories = list(
      pdio.read_records(
          args.source_path,
          microscope_utils.Trajectory,
      )
  )
  transitions = trajectories_to_transitions(trajectories)
  pdio.write_records(args.target_path, transitions)


if __name__ == '__main__':
  app.run(main, flags_parser=eapp.make_flags_parser(Args))
