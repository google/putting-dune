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

import dataclasses
import datetime as dt
import enum
from typing import Optional

import dm_env
from dm_env import specs
import matplotlib.pyplot as plt
import numpy as np
from putting_dune import action_adapters
from putting_dune import graphene
from putting_dune import plotting_utils
from putting_dune import simulator
from putting_dune import simulator_utils
from shapely import geometry


class RatePredictorType(str, enum.Enum):
  PRIOR = 'prior'
  SIMPLE = 'simple'


@dataclasses.dataclass(frozen=True)
class GoalReturn:
  reward: float
  is_terminal: bool
  is_truncated: bool


# TODO(joshgreaves): Eventually add interface for this.
class SingleSiliconGoalReaching:
  """A single silicon goal-reaching goal."""

  def __init__(self):
    # For now, require only that we reach the goal. This makes
    # the problem much less sparse, especially under the
    # relative-to-silicon action adapter.
    self._required_consecutive_goal_steps_for_termination = 1

    # Will be set on reset.
    self.goal_position_material_frame = np.zeros((2,), dtype=np.float32)
    self._consecutive_goal_steps = 0

  def reset(
      self,
      rng: np.random.Generator,
      initial_observation: simulator_utils.SimulatorObservation,
      sim: simulator.PuttingDuneSimulator,
  ):
    """Resets the goal, picking a new position.

    Args:
      rng: The RNG to use for sampling a new goal.
      initial_observation: The initial simulator observation.
      sim: The simulator in use.
    """
    del initial_observation  # Unused.

    self.goal_position_material_frame, _ = graphene.sample_point_away_from_edge(
        rng, sim.material.atom_positions, sim.material.nearest_neighbors
    )
    self._consecutive_goal_steps = 0

  def caluclate_reward_and_terminal(
      self,
      observation: simulator_utils.SimulatorObservation,
      sim: simulator.PuttingDuneSimulator,
  ) -> GoalReturn:
    """Calculates the reward and terminal signals for the goal.

    Note: we assume that this is called once per simulator/agent step.

    Args:
      observation: The last observation from the simulator.
      sim: The simulator being used.

    Returns:
      The reward, and whether the episode is terminal or should be
        truncated. Truncation happens when atoms get to the edge of
        the material and we can no longer simulate.
    """
    del observation  # Unused.

    # Calculate the reward.
    silicon_position_material_frame = sim.material.get_silicon_position()
    cost = np.linalg.norm(
        silicon_position_material_frame - self.goal_position_material_frame
    )

    # Update whether the silicon is near the goal.
    goal_radius = graphene.CARBON_BOND_DISTANCE_ANGSTROMS * 0.5
    silicon_position_material_frame = sim.material.get_silicon_position()
    goal_distance = np.linalg.norm(
        silicon_position_material_frame - self.goal_position_material_frame
    )
    if goal_distance < goal_radius:
      self._consecutive_goal_steps += 1
    else:
      self._consecutive_goal_steps = 0

    # Calculate whether it is a terminal state.
    is_terminal = (
        self._consecutive_goal_steps
        >= self._required_consecutive_goal_steps_for_termination
    )

    # Truncate if near the graphene edge.
    si_neighbor_distances, _ = sim.material.nearest_neighbors.kneighbors(
        silicon_position_material_frame.reshape(1, 2)
    )

    # If any of the neighbors are much greater than the expected bond distance,
    # then there aren't three neighbors and we're at the edge.
    # Since the neighbors are sorted, just look at the furthest neighbor.
    is_truncation = (
        si_neighbor_distances[0, -1]
        > graphene.CARBON_BOND_DISTANCE_ANGSTROMS * 1.1
    )

    return GoalReturn(-cost, is_terminal, is_truncation)


class PuttingDuneEnvironment(dm_env.Environment):
  """Putting Dune Environment."""

  def __init__(
      self, rate_predictor_type: RatePredictorType = RatePredictorType.PRIOR
  ):
    self._rng = np.random.default_rng()

    # Create objects that persist across episodes, but may be reset.
    # TODO(joshgreaves): Make the material configurable.
    if rate_predictor_type == RatePredictorType.PRIOR:
      self.rate_predictor = graphene.HumanPriorRatePredictor().predict
    else:
      self.rate_predictor = graphene.simple_transition_rates

    self._material = graphene.PristineSingleDopedGraphene(
        self._rng, predict_rates=self.rate_predictor
    )
    self.sim = simulator.PuttingDuneSimulator(self._material)
    # TODO(joshgreaves): Make the action adapter configurable.
    self._action_adapter = action_adapters.RelativeToSiliconActionAdapter()
    self.goal = SingleSiliconGoalReaching()

    # Variables that will be set on reset.
    self._last_simulator_observation = simulator_utils.SimulatorObservation(
        simulator_utils.AtomicGrid(np.zeros((1, 2)), np.asarray([14])),
        None,
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
    self._material.rng = self._rng
    if hasattr(self._action_adapter, 'rng'):
      self._action_adapter.rng = self._rng

  # TODO(joshgreaves): Abstract into ObservationConstructor.
  def _make_observation(self):
    silicon_position = graphene.get_silicon_positions(
        self._last_simulator_observation.grid
    ).reshape(2)

    if self._last_simulator_observation.last_probe_position is None:
      probe_position = getattr(self._action_adapter, 'beam_pos', np.zeros(2))
    else:
      probe_position = np.asarray(
          self._last_simulator_observation.last_probe_position
      )

    # TODO(joshgreaves): Remove tight coupling to SinglSiliconGoalReaching.
    # This won't be a problem once we abstract this into a separate
    # class, since we can allow a suite of tightly-coupled items.
    silicon_position_material_frame = self.sim.material.get_silicon_position()
    goal_delta_material_frame = (
        self.goal.goal_position_material_frame - silicon_position_material_frame
    )
    # TODO(joshgreaves): Maybe clip, or saturate in some way.
    goal_delta_microscope_frame = np.asarray(
        self.sim.convert_point_to_microscope_frame(
            geometry.Point(goal_delta_material_frame)
        )
    )

    obs = np.concatenate(
        [silicon_position, probe_position, goal_delta_microscope_frame]
    )

    return obs.astype(np.float32)

  def reset(self) -> dm_env.TimeStep:
    # We won't need to reset immediately.
    self._requires_reset = False

    # Generate a realistic doped graphene configuration.
    self._last_simulator_observation = self.sim.reset()
    self._action_adapter.reset()
    self.goal.reset(self._rng, self._last_simulator_observation, self.sim)

    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=0.0,
        discount=0.99,
        observation=self._make_observation(),
    )

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    if self._requires_reset:
      return self.reset()

    # 1. Convert action with ActionAdapter.
    #    This allows us to experiment with different action spaces.
    # TODO(joshgreaves): Cache last probe position, and use it wherever
    # beam_position is required.
    simulator_control = self._action_adapter.get_action(
        self._last_simulator_observation.grid, action
    )

    # 2. Step the simulator with the action returned from ActionAdapter.
    self._last_simulator_observation = self.sim.step_and_image(
        simulator_control
    )

    # 3. Create an observation with ObservationConstructor, using
    #    the new state returned from the simulator.
    observation = self._make_observation()

    # 4. Calculate the reward (and terminal?) using RewardFunction.
    # Perhaps never terminate to teach the agent to keep the silicon at goal?
    goal_return = self.goal.caluclate_reward_and_terminal(
        self._last_simulator_observation, self.sim
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

  def observation_spec(self) -> specs.Array:
    obs_shape = self._make_observation().shape
    return specs.Array(obs_shape, np.float32)

  def render(self):
    fig = plt.figure(figsize=[5, 5])
    ax = fig.subplots()

    # TODO(joshgreaves): beam_position lags by 1 timestep.
    beam_position = self._last_simulator_observation.last_probe_position
    if beam_position is not None:
      beam_position = np.asarray(beam_position)
    goal_position = np.asarray(
        self.sim.convert_point_to_microscope_frame(
            geometry.Point(self.goal.goal_position_material_frame)
        )
    )

    plotting_utils.plot_microscope_frame(
        ax,
        self._last_simulator_observation.grid,
        goal_position,
        beam_position,
        self.sim.elapsed_time,
    )

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img
