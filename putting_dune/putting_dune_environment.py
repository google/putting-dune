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
"""Putting Dune Environment for use with RL agents."""

import datetime as dt
import typing
from typing import Optional

import dm_env
from dm_env import specs
import matplotlib.pyplot as plt
import numpy as np
from putting_dune import action_adapters
from putting_dune import feature_constructors
from putting_dune import geometry
from putting_dune import goals
from putting_dune import graphene
from putting_dune import microscope_utils
from putting_dune import plotting_utils
from putting_dune import simulator


class PuttingDuneEnvironment(dm_env.Environment):
  """Putting Dune Environment."""

  def __init__(
      self,
      material: graphene.Material,
      action_adapter: action_adapters.ActionAdapter,
      feature_constructor: feature_constructors.FeatureConstructor,
      goal: goals.Goal,
  ):
    self._rng = np.random.default_rng()

    # Create objects that persist across episodes, but may be reset.
    self._material = material
    self.sim = simulator.PuttingDuneSimulator(self._material)
    # TODO(joshgreaves): Make the action adapter configurable.
    self._action_adapter = action_adapter
    self._feature_constructor = feature_constructor
    self.goal = goal

    # Variables that will be set on reset.
    self._last_microscope_observation = microscope_utils.MicroscopeObservation(
        microscope_utils.AtomicGrid(np.zeros((1, 2)), np.asarray([14])),
        microscope_utils.MicroscopeFieldOfView(
            geometry.Point((0.0, 0.0)), geometry.Point((1.0, 1.0))
        ),
        (),
        dt.timedelta(seconds=0),
    )

    # We need to reset if:
    #   1. We have just made the environment and reset has not been called yet.
    #   2. The last step was terminal.
    # See dm_env semantics:
    # https://github.com/deepmind/dm_env/blob/master/docs/index.md
    self._requires_reset = True

  def seed(self, seed: Optional[int]) -> None:
    self._rng = np.random.default_rng(seed)

    # Replace the rng in child objects.
    # TODO(joshgreaves): This isn't robust to changes deeper in the stack.
    if hasattr(self._action_adapter, 'rng'):
      self._action_adapter.rng = self._rng

  def reset(self) -> dm_env.TimeStep:
    # We won't need to reset immediately.
    self._requires_reset = False

    # Generate a realistic doped graphene configuration.
    self._last_microscope_observation = self.sim.reset(self._rng)
    self._action_adapter.reset()
    self._feature_constructor.reset()
    self.goal.reset(self._rng, self._last_microscope_observation)

    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=0.0,
        discount=0.99,
        observation=self._feature_constructor.get_features(
            self._last_microscope_observation, self.goal
        ),
    )

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    if self._requires_reset:
      return self.reset()

    # 1. Convert action with ActionAdapter.
    #    This allows us to experiment with different action spaces.
    # TODO(joshgreaves): Cache last probe position, and use it wherever
    # beam_position is required.
    simulator_controls = self._action_adapter.get_action(
        self._last_microscope_observation, action
    )

    # 2. Step the simulator with the action returned from ActionAdapter.
    self._last_microscope_observation = self.sim.step_and_image(
        self._rng, simulator_controls
    )

    # 3. Create an observation with ObservationConstructor, using
    #    the new state returned from the simulator.
    observation = self._feature_constructor.get_features(
        self._last_microscope_observation, self.goal
    )

    # 4. Calculate the reward (and terminal?) using RewardFunction.
    # Perhaps never terminate to teach the agent to keep the silicon at goal?
    goal_return = self.goal.calculate_reward_and_terminal(
        self._last_microscope_observation
    )

    # TODO(joshgreaves): Make discount configurable.
    discount = 0.99
    if goal_return.is_terminal:
      self._requires_reset = True
      return dm_env.termination(goal_return.reward, observation)
    elif goal_return.is_truncated:
      self._requires_reset = True
      return dm_env.truncation(goal_return.reward, observation, discount)

    return dm_env.transition(goal_return.reward, observation, discount)

  def action_spec(self) -> specs.BoundedArray:
    return self._action_adapter.action_spec

  def observation_spec(self) -> feature_constructors.NestedObservationSpec:
    return self._feature_constructor.observation_spec()

  def render(self):
    fig = plt.figure(figsize=[5, 5])
    ax = fig.subplots()

    # TODO(joshgreaves): beam_position lags by 1 timestep.
    previous_controls = self._last_microscope_observation.controls
    if previous_controls:
      beam_position = np.asarray(previous_controls[-1].position)
    else:
      beam_position = None

    goal_position = None
    if isinstance(self.goal, goals.SingleSiliconGoalReaching):
      goal = typing.cast(goals.SingleSiliconGoalReaching, self.goal)
      fov = self._last_microscope_observation.fov
      goal_position = fov.material_frame_to_microscope_frame(
          goal.goal_position_material_frame
      )

    plotting_utils.plot_microscope_frame(
        ax,
        self._last_microscope_observation.grid,
        goal_position,
        beam_position,
        self.sim.elapsed_time,
    )

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img
