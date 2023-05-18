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

"""Useful utility functions for plotting."""

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from putting_dune import constants
from putting_dune import geometry
from putting_dune import graphene
from putting_dune import microscope_utils
from putting_dune import simulator_observers
import seaborn as sns

_SimulatorEventType = simulator_observers.SimulatorEventType


def format_timedelta(delta: dt.timedelta) -> str:
  total_seconds = delta.total_seconds()
  minutes = int(total_seconds) // 60
  seconds = int(total_seconds) % 60
  remainder = round((total_seconds - int(total_seconds)) * 100)
  return f'{minutes:02d}:{seconds:02d}:{remainder:02d}'


def _plot_atomic_grid(
    ax: plt.Axes,
    grid: microscope_utils.AtomicGrid,
    goal_position: Optional[np.ndarray] = None,
    control_position: Optional[np.ndarray] = None,
    timedelta: Optional[dt.timedelta] = None,
    *,
    carbon_size: float = 6.0,
    silicon_size: float = 8.0,
    goal_size: float = 15.0,
    control_size: float = 10.0,
) -> None:
  """Utility function for common plotting."""
  carbon_atoms = grid.atom_positions[grid.atomic_numbers == constants.CARBON]
  silicon_atoms = grid.atom_positions[grid.atomic_numbers == constants.SILICON]

  # Plot the carbon atoms.
  ax.plot(
      carbon_atoms[:, 0],
      carbon_atoms[:, 1],
      'o',
      markersize=carbon_size,
      alpha=0.5,
  )

  # Plot the silicon atoms.
  ax.plot(
      silicon_atoms[:, 0], silicon_atoms[:, 1], 'ro', markersize=silicon_size
  )

  # Plot the goal
  if goal_position is not None:
    ax.plot(goal_position[0], goal_position[1], 'gx', markersize=goal_size)

  # Plot the control position.
  if control_position is not None:
    ax.plot(
        control_position[0], control_position[1], 'k.', markersize=control_size
    )

  # Plot the current time.
  if timedelta is not None:
    lower_left = np.min(grid.atom_positions, axis=0)
    ax.text(
        lower_left[0],
        lower_left[1],
        format_timedelta(timedelta),
        fontsize='x-large',
    )


def plot_microscope_frame(
    ax: plt.Axes,
    grid: microscope_utils.AtomicGrid,
    goal_position: Optional[np.ndarray] = None,
    control_position: Optional[np.ndarray] = None,
    timedelta: Optional[dt.timedelta] = None,
) -> None:
  """Plots the frame in supplied axis, with coordinates in microscope frame."""
  _plot_atomic_grid(ax, grid, goal_position, control_position, timedelta)

  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim((0, 1))
  ax.set_ylim((0, 1))


def plot_material_frame(
    ax: plt.Axes,
    grid: microscope_utils.AtomicGrid,
    goal_position: Optional[np.ndarray] = None,
    control_position: Optional[np.ndarray] = None,
    timedelta: Optional[dt.timedelta] = None,
    fov: Optional[microscope_utils.MicroscopeFieldOfView] = None,
) -> None:
  """Plots the frame in supplied axis, with coordinates in material frame."""
  _plot_atomic_grid(
      ax=ax,
      grid=grid,
      goal_position=goal_position,
      control_position=control_position,
      timedelta=timedelta,
      carbon_size=1.0,
      silicon_size=2.0,
      goal_size=8.0,
      control_size=2.0,
  )

  if fov is not None:
    fov_bounds = [
        (fov.lower_left.x, fov.lower_left.y),
        (fov.upper_right.x, fov.lower_left.y),
        (fov.upper_right.x, fov.upper_right.y),
        (fov.lower_left.x, fov.upper_right.y),
        (fov.lower_left.x, fov.lower_left.y),
    ]
    ax.plot([x for x, _ in fov_bounds], [y for _, y in fov_bounds], color='red')

  ax.set_xticks([])
  ax.set_yticks([])

  min_x = np.min(grid.atom_positions[:, 0])
  max_x = np.max(grid.atom_positions[:, 0])
  min_y = np.min(grid.atom_positions[:, 1])
  max_y = np.max(grid.atom_positions[:, 1])
  width = max_x - min_x
  height = max_y - min_y
  width_padding = width * 0.05
  height_padding = height * 0.05
  ax.set_xlim((min_x - width_padding, max_x + width_padding))
  ax.set_ylim((min_y - height_padding, max_y + height_padding))


def generate_video_from_simulator_events(
    events: Sequence[simulator_observers.SimulatorEvent],
    goal_position: np.ndarray,
) -> animation.Animation:
  """Generates a video for the set of events."""
  grid: microscope_utils.AtomicGrid = None
  fov: microscope_utils.MicroscopeFieldOfView = None
  control_position: np.ndarray = None
  image: np.ndarray = None
  frames: List[Dict[str, Any]] = []

  # Check if any of the events have images in them. If they do, plot them.
  events_contain_images = False
  for event in events:
    if event.event_type == _SimulatorEventType.GENERATED_IMAGE:
      events_contain_images = True
      break

  if events_contain_images:
    fig = plt.figure(figsize=(12, 4))
    axes = fig.subplots(1, 3)
  else:
    fig = plt.figure(figsize=(8, 4))
    axes = fig.subplots(1, 2)

  def plot_frame(args: Dict[str, Any]) -> None:
    for ax in axes:
      ax.clear()

    plot_material_frame(
        ax=axes[0],
        grid=args['grid'],
        goal_position=goal_position,
        control_position=args['control_position'],
        timedelta=args['timedelta'],
        fov=args['fov'],
    )

    # Convert items to microscope frame.
    microscope_grid = args['fov'].material_frame_to_microscope_frame(
        args['grid']
    )
    # We make a grid containing just the goal and control to make it easy
    # to convert them to the microscope frame.
    material_frame_data = microscope_utils.AtomicGrid(
        atom_positions=np.stack([goal_position, args['control_position']]),
        atomic_numbers=np.asarray(()),  # Unused.
    )
    microscope_frame_data = args['fov'].material_frame_to_microscope_frame(
        material_frame_data
    )
    plot_microscope_frame(
        ax=axes[1],
        grid=microscope_grid,
        goal_position=microscope_frame_data.atom_positions[0],
        control_position=microscope_frame_data.atom_positions[1],
    )

    if events_contain_images and args['image'] is not None:
      axes[2].imshow(args['image'], cmap='gray')
      axes[2].set_xticks([])
      axes[2].set_yticks([])

  elapsed_time = dt.timedelta(seconds=0)
  for event in events:
    if event.event_type == _SimulatorEventType.RESET:
      grid = event.event_data['grid']
      fov = event.event_data['fov']
    if event.event_type == _SimulatorEventType.APPLY_CONTROL:
      control_position = np.asarray(
          event.event_data['position'].coords
      ).reshape(-1)

      # We plot after taking the control, since it is the most appealing
      # visual. We want to see the state of the system before the control,
      # rather than after the control.
      frames.append({
          'grid': grid,
          'fov': fov,
          'control_position': control_position,
          'timedelta': elapsed_time,
          'image': image,
      })

      # Tick the clock.
      elapsed_time += event.event_data['dwell_time']
    if event.event_type == _SimulatorEventType.TRANSITION:
      grid = event.event_data['grid']
    if event.event_type == _SimulatorEventType.TAKE_IMAGE:
      fov = event.event_data['fov']
      elapsed_time += event.event_data['duration']
    if event.event_type == _SimulatorEventType.GENERATED_IMAGE:
      image = event.event_data['image']

  # Append one last event to make the plotting nicer.
  frames.append({
      'grid': grid,
      'fov': fov,
      'control_position': control_position,
      'timedelta': elapsed_time,
      'image': image,
  })

  anim = animation.FuncAnimation(fig, plot_frame, frames)
  return anim


def _center_grid_on_single_silicon(
    grid: microscope_utils.AtomicGridMaterialFrame,
) -> microscope_utils.AtomicGridMaterialFrame:
  try:
    si_pos = graphene.get_single_silicon_position(grid)
    grid = microscope_utils.AtomicGrid(
        grid.atom_positions - si_pos.reshape(1, 2), grid.atomic_numbers
    )
    return microscope_utils.AtomicGridMaterialFrame(grid)
  except graphene.SiliconNotFoundError as e:
    raise ValueError('Grid does not contain single silicon.') from e


def plot_rate_function3(
    ax: plt.Axes,
    rate_function: graphene.RateFunction,
    grid: microscope_utils.AtomicGridMaterialFrame,
    *,
    extent: Tuple[float, float] = (
        -3 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        3 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
    ),
    num_raster_points: int = 50,
):
  """Plots a rate function.

  Note: This expects a 3-fold single silicon in the grid.

  Args:
    ax: The axis to plot the rate function on.
    rate_function: The rate function to plot.
    grid: The atomic grid to use for plotting.
    extent: The extent to plot. The silicon atom will be centered in this
      extent.
    num_raster_points: The number of x and y points to use, spaced equally
      across the extent.

  Raises:
    ValueError: If grid doesn't contain a single silicon atom.
  """
  grid = _center_grid_on_single_silicon(grid)

  data = {
      'x': [],
      'y': [],
      'z': [],
      'next_si_pos': [],
  }

  for x in np.linspace(extent[0], extent[1], num_raster_points):
    for y in np.linspace(extent[0], extent[1], num_raster_points):
      beam_pos = geometry.PointMaterialFrame(geometry.Point((x, y)))
      rates = rate_function(grid, beam_pos)

      for ss in rates.successor_states:
        si_pos = tuple(graphene.get_single_silicon_position(ss.grid))

        data['x'].append(x)
        data['y'].append(y)
        data['z'].append(ss.rate)
        data['next_si_pos'].append(si_pos)

  df = pd.DataFrame(data)

  next_si_positions = set(df['next_si_pos'])
  cmaps = ['Blues', 'Oranges', 'Greens']

  # Plot the contours.
  for next_si_pos, cmap in zip(next_si_positions, cmaps):
    sns.kdeplot(
        data=df[df['next_si_pos'] == next_si_pos],
        ax=ax,
        x='x',
        y='y',
        weights='z',
        cmap=cmap,
        fill=True,
        alpha=0.2,
        levels=10,
    )

  # Plot all the atoms in extent (faintly)
  ax.scatter(
      grid.atom_positions[:, 0],
      grid.atom_positions[:, 1],
      c='black',
      alpha=0.05,
  )
  # Plot the neighboring atoms in their associated colors.
  ax.scatter(
      [x[0] for x in next_si_positions],
      [x[1] for x in next_si_positions],
      c=[matplotlib.colormaps[cmap](1.0) for cmap in cmaps],
  )
  # Plot the silicon atom.
  ax.scatter([0], [0], c='black')

  ax.set_xlim(extent[0], extent[1])
  ax.set_ylim(extent[0], extent[1])


def plot_rate_along_neighbor_vector3(
    ax: plt.Axes,
    rate_function: graphene.RateFunction,
    grid: microscope_utils.AtomicGridMaterialFrame,
    *,
    extent: Tuple[float, float] = (-5.0, 10.0),
    num_points: int = 250,
):
  """Plots a rate function along the silicon-to-neighbor vector.

  Note: This expects a 3-fold single silicon in the grid.

  Args:
    ax: The axis to plot the rate function on.
    rate_function: The rate function to plot.
    grid: The atomic grid to use for plotting.
    extent: The extent to plot. This is normalized so that 0.0 corresponds to
      the silicon atom and 1.0 corresponds to the neighbor.
    num_points: The number of points to plot, spaced equally in extent.

  Raises:
    ValueError: If grid doesn't contain a single silicon atom.
  """
  grid = _center_grid_on_single_silicon(grid)

  neighbor_indices = geometry.nearest_neighbors3(
      grid.atom_positions, np.asarray((0.0, 0.0))
  ).neighbor_indices
  neighbor_vec = grid.atom_positions[neighbor_indices][0]

  data = {
      'x': [],
      'y': [],
  }

  # Scans lower left to top right.
  for x in np.linspace(extent[0], extent[1], num_points):
    beam_pos = geometry.PointMaterialFrame(geometry.Point(neighbor_vec * x))
    rates = rate_function(grid, beam_pos)

    for ss in rates.successor_states:
      si_pos = graphene.get_single_silicon_position(ss.grid)

      # Only get the rates for the neighbor of interest.
      if np.linalg.norm(si_pos - neighbor_vec) < 0.01:
        data['x'].append(x)
        data['y'].append(ss.rate)

  sns.lineplot(data=pd.DataFrame(data), ax=ax, x='x', y='y')

  # Add vertical lines where the atoms are.
  ax.vlines(
      [0.0, 1.0], ymin=0.0, ymax=max(data['y']) * 1.1, linestyles='dashed'
  )

  ax.set_title('Rate along vector from silicon to neighbor')
  ax.set_xlabel('alpha')
  ax.set_ylabel('rate')


def plot_rate_along_arc3(
    ax: plt.Axes,
    rate_function: graphene.RateFunction,
    grid: microscope_utils.AtomicGridMaterialFrame,
    *,
    normalized_radius: float = 1.0,
    num_points: int = 250,
):
  """Plots a rate function in a circle at a fixed radius from a silicon atom.

  Note: This expects a 3-fold single silicon in the grid.

  Args:
    ax: The axis to plot the rate function on.
    rate_function: The rate function to plot.
    grid: The atomic grid to use for plotting.
    normalized_radius: The radius to use when calculating the rate function.
      This is normalized so that 0.0 corresponds to on the silicon atom, and 1.0
      is the mean neighbor distance.
    num_points: The number of angles to use in calculating the rate function,
      spaced evenly between 0 and 2 * pi radians.

  Raises:
    ValueError: If grid doesn't contain a single silicon atom.
  """
  grid = _center_grid_on_single_silicon(grid)

  neighbor_indices = geometry.nearest_neighbors3(
      grid.atom_positions, np.asarray((0.0, 0.0))
  ).neighbor_indices

  mean_neighbor_distance = np.mean(
      np.linalg.norm(
          grid.atom_positions[neighbor_indices],
          axis=-1,
      )
  )
  radius = normalized_radius * mean_neighbor_distance

  data = {'angle (radians)': [], 'rate': [], 'next_state': []}
  next_state_labels = {
      tuple(pos): i + 1
      for i, pos in enumerate(grid.atom_positions[neighbor_indices])
  }

  for angle in np.linspace(0.0, 2 * np.pi, num_points):
    beam_pos = np.asarray([np.sin(angle) * radius, np.cos(angle) * radius])
    beam_pos = geometry.PointMaterialFrame(geometry.Point(beam_pos))
    rates = rate_function(grid, beam_pos)

    data['angle (radians)'].append(angle)
    data['rate'].append(rates.total_rate)
    data['next_state'].append('Total rate')

    for ss in rates.successor_states:
      si_pos = graphene.get_single_silicon_position(ss.grid)
      next_state_label = next_state_labels[tuple(si_pos)]

      data['angle (radians)'].append(angle)
      data['rate'].append(ss.rate)
      data['next_state'].append(next_state_label)

  df = pd.DataFrame(data)
  sns.lineplot(data=df, ax=ax, x='angle (radians)', y='rate', hue='next_state')
  ax.set_title('Rate along angle')
