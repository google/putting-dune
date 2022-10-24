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

"""Useful utility functions for plotting."""

import datetime as dt
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
from putting_dune import graphene
from putting_dune import simulator_utils


def format_timedelta(delta: dt.timedelta) -> str:
  total_seconds = delta.total_seconds()
  minutes = int(total_seconds) // 60
  seconds = int(total_seconds) % 60
  remainder = round((total_seconds - int(total_seconds)) * 100)
  return f'{minutes:02d}:{seconds:02d}:{remainder:02d}'


def plot_frame(
    ax: plt.Axes,
    grid: simulator_utils.AtomicGrid,
    goal_position: Optional[np.ndarray] = None,
    control_position: Optional[np.ndarray] = None,
    timedelta: Optional[dt.timedelta] = None) -> None:
  """Plots the frame in supplied axis."""
  ax.clear()

  carbon_atoms = grid.atom_positions[grid.atomic_numbers == graphene.CARBON]
  silicon_atoms = grid.atom_positions[grid.atomic_numbers == graphene.SILICON]

  # Plot the carbon atoms.
  ax.plot(
      carbon_atoms[:, 0],
      carbon_atoms[:, 1],
      'o',
      markersize=6,
      alpha=0.5)

  # Plot the silicon atoms.
  ax.plot(silicon_atoms[:, 0], silicon_atoms[:, 1], 'ro', markersize=8,)

  # Plot the goal
  if goal_position is not None:
    ax.plot(goal_position[0], goal_position[1], 'gx', markersize=15)

  # Plot the control position.
  if control_position is not None:
    ax.plot(control_position[0], control_position[1], 'k.', markersize=10)

  # Plot the current time.
  if timedelta is not None:
    ax.text(0.01, 0.01, format_timedelta(timedelta), fontsize='x-large')

  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim((0, 1))
  ax.set_ylim((0, 1))
