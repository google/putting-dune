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
"""Useful utility functions for plotting."""

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from putting_dune import constants
from putting_dune import microscope_utils
from putting_dune import simulator_observers

_SimulatorEventType = simulator_observers.SimulatorEventType


def format_timedelta(delta: dt.timedelta) -> str:
  total_seconds = delta.total_seconds()
  minutes = int(total_seconds) // 60
  seconds = int(total_seconds) % 60
  remainder = round((total_seconds - int(total_seconds)) * 100)
  return f'{minutes:02d}:{seconds:02d}:{remainder:02d}'


def _plot(
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
    ax.text(0.01, 0.01, format_timedelta(timedelta), fontsize='x-large')


def plot_microscope_frame(
    ax: plt.Axes,
    grid: microscope_utils.AtomicGrid,
    goal_position: Optional[np.ndarray] = None,
    control_position: Optional[np.ndarray] = None,
    timedelta: Optional[dt.timedelta] = None,
) -> None:
  """Plots the frame in supplied axis, with coordinates in microscope frame."""
  _plot(ax, grid, goal_position, control_position, timedelta)

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
  _plot(
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

  for event in events:
    if event.event_type == _SimulatorEventType.RESET:
      grid = event.event_data['grid']
      fov = event.event_data['fov']
    if event.event_type == _SimulatorEventType.APPLY_CONTROL:
      control_position = np.asarray(event.event_data['position'])

      # We plot after taking the control, since it is the most appealing
      # visual. We want to see the state of the system before the control,
      # rather than after the control.
      frames.append({
          'grid': grid,
          'fov': fov,
          'control_position': control_position,
          'timedelta': event.start_time,
          'image': image,
      })
    if event.event_type == _SimulatorEventType.TRANSITION:
      grid = event.event_data['grid']
    if event.event_type == _SimulatorEventType.TAKE_IMAGE:
      fov = event.event_data['fov']
    if event.event_type == _SimulatorEventType.GENERATED_IMAGE:
      image = event.event_data['image']

  # Append one last event to make the plotting nicer.
  frames.append({
      'grid': grid,
      'fov': fov,
      'control_position': control_position,
      'timedelta': events[-1].end_time,
      'image': image,
  })

  anim = animation.FuncAnimation(fig, plot_frame, frames)
  return anim
