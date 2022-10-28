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
from sklearn import neighbors


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


class SingleSiliconPristineGraphineFeatureConstuctor:
  """A feature constructor assuming pristine graphene with single dopant."""

  def reset(self) -> None:
    # We include this because other feature constructors may be stateful.
    # However, this one isn't.
    pass

  def get_features(
      self,
      observation: simulator_utils.SimulatorObservation,
      goal: SingleSiliconGoalReaching,
  ) -> np.ndarray:
    """Gets features for an agent based on the osbervation and goal."""
    silicon_position = graphene.get_silicon_positions(observation.grid).reshape(
        2
    )

    # Get the vectors to the nearest neighbors.
    nearest_neighbors = neighbors.NearestNeighbors(
        n_neighbors=1 + 3,
        metric='l2',
        algorithm='brute',
    ).fit(observation.grid.atom_positions)
    neighbor_distances, neighbor_indices = nearest_neighbors.kneighbors(
        silicon_position.reshape(1, 2)
    )
    neighbor_positions = observation.grid.atom_positions[
        neighbor_indices[0, 1:]
    ]
    neighbor_deltas = neighbor_positions - silicon_position.reshape(1, 2)
    neighbor_distances = neighbor_distances[0, 1:].reshape(-1, 1)
    normalized_deltas = neighbor_deltas / neighbor_distances

    material_frame_grid = observation.fov.microscope_grid_to_material_grid(
        observation.grid
    )
    silicon_position_material_frame = graphene.get_silicon_positions(
        material_frame_grid
    ).reshape(2)
    goal_delta_material_frame = (
        goal.goal_position_material_frame - silicon_position_material_frame
    )

    obs = np.concatenate([
        silicon_position,
        normalized_deltas.reshape(-1),
        goal_delta_material_frame,
    ])

    return obs.astype(np.float32)

  def observation_spec(self) -> specs.Array:
    # 2 for silicon position.
    # 6 for 3 nearest neighbor delta vectors.
    # 2 for goal delta.
    return specs.Array((2 + 6 + 2,), np.float32)


class PuttingDuneEnvironment(dm_env.Environment):
  """Putting Dune Environment."""

  def __init__(
      self
  ):
    self._rng = np.random.default_rng()

    # Create objects that persist across episodes, but may be reset.
    rate_predictor = graphene.simple_transition_rates
    self._material = graphene.PristineSingleDopedGraphene(
        self._rng, predict_rates=rate_predictor
    )
    self.sim = simulator.PuttingDuneSimulator(self._material)
    # TODO(joshgreaves): Make the action adapter configurable.
    self._action_adapter = action_adapters.RelativeToSiliconActionAdapter()
    self._feature_constructor = SingleSiliconPristineGraphineFeatureConstuctor()
    self.goal = SingleSiliconGoalReaching()

    # Variables that will be set on reset.
    self._last_simulator_observation = simulator_utils.SimulatorObservation(
        simulator_utils.AtomicGrid(np.zeros((1, 2)), np.asarray([14])),
        simulator_utils.SimulatorFieldOfView(
            geometry.Point((0.0, 0.0)), geometry.Point((1.0, 1.0))
        ),
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

  def reset(self) -> dm_env.TimeStep:
    # We won't need to reset immediately.
    self._requires_reset = False

    # Generate a realistic doped graphene configuration.
    self._last_simulator_observation = self.sim.reset()
    self._action_adapter.reset()
    self._feature_constructor.reset()
    self.goal.reset(self._rng, self._last_simulator_observation, self.sim)

    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=0.0,
        discount=0.99,
        observation=self._feature_constructor.get_features(
            self._last_simulator_observation, self.goal
        ),
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
    observation = self._feature_constructor.get_features(
        self._last_simulator_observation, self.goal
    )

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
    return self._feature_constructor.observation_spec()

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
