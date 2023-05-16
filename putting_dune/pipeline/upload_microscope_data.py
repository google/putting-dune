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

"""Upload microscope data."""

import dataclasses
import datetime as dt
import io
import json
import pathlib
import sys
from typing import Any

from absl import app
from etils import eapp
from google.cloud import storage
import inflection
import numpy as np
from putting_dune import geometry
from putting_dune import microscope_utils
import riegeli
import simple_parsing as sp
import termcolor
import tqdm


@dataclasses.dataclass(frozen=True, kw_only=True)
class Args:
  directory: pathlib.Path = sp.field(positional=True)
  bucket: str = 'putting-dune'


def load_dataset(
    dataset_path: pathlib.Path,
) -> tuple[np.ndarray, dict[str, Any], dict[Any, dict[str, Any]]]:
  """Loads data from the microscope as numpy arrays and dictionaries.

  Extracts data from the format it is saved in by ORNL on disk, doing limited
  initial processing.
  The structures returned are based on those originally used by ORNL
  and are intended to be transitional to the protobuf formats used internally.

  Arguments:
    dataset_path: path to dataset on disk.

  Returns:
    raw_image_stack: (N, H, W, C) stack of raw images
    coordinates: Mapping of timestep to coordinate arrays.
    parameters: Mapping of timestep to scan parameters and metadata.
  """
  adf_filename = dataset_path / 'ADF-stack-000.ndata1'
  label_filename = dataset_path / 'Label-stack-000.ndata1'
  raw_image_stack = np.load(adf_filename)['data']
  label_stack = np.load(label_filename)['data']
  metadata = json.loads(np.load(label_filename)['metadata.json'])
  stacklength = len(metadata['metadata'])

  # truncate arrays; it seems that in some cases they may be padded to a default
  # length higher than that of the collected trajectory.
  raw_image_stack = raw_image_stack[:stacklength]
  label_stack = label_stack[:stacklength]

  # Find scaling between label image and raw image
  # Coordinates are taken from the labels, so must be rescaled.
  downscale_factor = raw_image_stack.shape[1] / label_stack.shape[1]

  coordinates = {}
  parameters = {}

  ## Constants ##
  readout_time = None
  for _, v in metadata['metadata'].items():
    if 'ADF readouttime' in v:
      readout_time = v['ADF readouttime']
      break
  if readout_time is None:
    raise ValueError('Required ADF readouttime information not present in data')

  for frame, val in metadata['metadata'].items():
    coordinates[frame] = {}
    parameters[frame] = {}
    coordinates[frame]['label'] = np.asarray(val['All coordinates']['0'])
    coordinates[frame]['image'] = downscale_factor * coordinates[frame]['label']
    parameters[frame]['readout_time'] = readout_time
    parameters[frame]['FOV'] = val['All parameters']['image_parameters'][0]
    # If no dopants were detected or present, populate the dict with NaN
    try:
      coordinates[frame]['dopant'] = downscale_factor * np.asarray(
          val['Blast coordinates']['Dopants']
      )
      coordinates[frame]['beam loc'] = downscale_factor * np.asarray(
          val['Beam location']
      ).reshape(-1, 2)
      parameters[frame]['beam dwelltime'] = val['Beam dwelltime']
      parameters[frame]['ADFreadout'] = np.asarray(val['ADF intensities'])
      parameters[frame]['pixelshifts'] = np.asarray(val['Pixelshifts'])
    except KeyError:
      coordinates[frame]['dopant'] = np.asarray([np.nan, np.nan]).reshape(-1, 2)
      coordinates[frame]['beam loc'] = np.asarray([np.nan, np.nan]).reshape(
          -1, 2
      )
      parameters[frame]['beam dwelltime'] = np.nan
      parameters[frame]['ADFreadout'] = np.asarray([np.nan, np.nan]).reshape(
          -1, 2
      )
      parameters[frame]['pixelshifts'] = np.asarray([0, 0])

  return raw_image_stack, coordinates, parameters


def convert_dataset_to_proto(
    raw_image_stack: np.ndarray,
    coordinate_dict: dict[str, np.ndarray],
    parameters: dict[str, dict[str, np.ndarray]],
) -> microscope_utils.Trajectory:
  """Converts a dataset in the format provided by ORNL into a Trajectory proto.

  Args:
    raw_image_stack: Images, of format (N, H, W, C)
    coordinate_dict: Dictionary of timesteps to coordinates of various objects.
    parameters: Dictionary of timesteps to metadata.

  Returns:
    Trajectory object containing extracted observations.
  """

  length = raw_image_stack.shape[0]
  fov = parameters[str(0)]['FOV']
  corners = np.stack([[0, 0], [10 * fov, 10 * fov]], axis=0)
  observations = []
  for t in range(length):
    fov = parameters[str(t)]['FOV']
    rescale_factor = np.array(raw_image_stack.shape[1:])
    grid = coordinate_dict[str(t)]['image'][:, :2] / rescale_factor
    atomic_numbers = np.zeros(grid.shape[0], dtype=np.int32) + 6
    if len(coordinate_dict) <= 3:
      break
    dopant_position = coordinate_dict[str(t)]['dopant'] / rescale_factor
    for dopant in dopant_position:
      dists = np.linalg.norm(grid - dopant, axis=-1)
      matches = dists < 1e-6
      atomic_numbers[matches] = 14

    shift = 10 * fov * parameters[str(t)]['pixelshifts'] / rescale_factor
    corners = corners + shift
    grid = microscope_utils.AtomicGridMicroscopeFrame(
        microscope_utils.AtomicGrid(grid, atomic_numbers)
    )
    fov = microscope_utils.MicroscopeFieldOfView(
        geometry.PointMaterialFrame(geometry.Point(corners[0])),
        geometry.PointMaterialFrame(geometry.Point(corners[1])),
    )

    if np.isnan(parameters[str(t)]['beam dwelltime']):
      beam_control = ()
      elapsed_time = dt.timedelta(seconds=2.0)
    else:
      beam_loc = coordinate_dict[str(t)]['beam loc'][0] / rescale_factor
      beam_loc = geometry.PointMicroscopeFrame(
          geometry.Point(beam_loc[0], beam_loc[1])
      )
      beam_loc = fov.microscope_frame_to_material_frame(beam_loc)
      beam_control = (
          microscope_utils.BeamControl(
              beam_loc,
              dt.timedelta(seconds=float(parameters[str(t)]['beam dwelltime'])),
          ),
      )
      elapsed_time = dt.timedelta(
          seconds=float(parameters[str(t)]['beam dwelltime']) + 2.0
      )
    obs = microscope_utils.MicroscopeObservation(
        grid,
        fov,
        beam_control,
        elapsed_time,
        raw_image_stack[t],
    )
    observations.append(obs)
  return microscope_utils.Trajectory(observations)


def main(args: Args) -> None:
  # Create our GCS storage client
  client = storage.Client()
  bucket = client.get_bucket(args.bucket)

  # Search for all dataset paths, i.e., paths that have contain a chunk of data
  paths = [
      leaf.parent for leaf in args.directory.rglob('**/*/ADF-stack-000.ndata1')
  ]
  if not paths:
    print(
        termcolor.colored(
            f'No microscope datasets found in directory {args.directory}',
            'red',
        ),
        file=sys.stderr,
    )
    sys.exit(1)

  for dataset_path in (pbar := tqdm.tqdm(paths)):
    # Create a filename by converting each part of the path to be
    # "underscored" and then split path parts with a dash.
    filename = pathlib.Path(
        '-'.join(
            map(
                inflection.underscore,
                dataset_path.relative_to(args.directory).parts,
            )
        )
    ).with_suffix('.riegeli')
    pbar.set_postfix_str(filename)

    # Load dataset
    try:
      dataset = load_dataset(dataset_path)
    except ValueError as e:
      print(
          termcolor.colored(
              f'Failed to load dataset {dataset_path}: {e}', 'red'
          ),
          file=sys.stderr,
      )
      continue

    # Convet dataset to a protobuf
    trajectory = convert_dataset_to_proto(*dataset)
    trajectory_proto = trajectory.to_proto()

    # Write record to a memory backed buffer
    fp = io.BytesIO()
    with riegeli.RecordWriter(fp) as writer:
      writer.write_record(trajectory_proto.SerializeToString())

    # Seek to beginning of the buffer before writting
    fp.seek(0)

    # Create a blob and upload from our in-memory buffer
    blob = bucket.blob(filename.as_posix())
    blob.upload_from_file(fp)


if __name__ == '__main__':
  app.run(main, flags_parser=eapp.make_flags_parser(Args))
