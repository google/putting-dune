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

"""An agent for interfacing directly with a microscope."""

import csv
import datetime as dt
import os
import types
import typing
from typing import List, Type, TypedDict

import dm_env
from etils import epath
import numpy as np
from putting_dune import constants
from putting_dune import geometry
from putting_dune import goals
from putting_dune import graphene
from putting_dune import io as pdio
from putting_dune import microscope_utils
from putting_dune.experiments import experiments
import wrapt


class MicroscopeAgent:
  """An agent for interfacing directly with a microscope."""

  def __init__(
      self,
      rng: np.random.Generator,
      experiment: experiments.MicroscopeExperiment,
  ):
    adapters_and_goal = experiment.get_adapters_and_goal()
    self.agent = experiment.get_agent(rng, adapters_and_goal)
    self.action_adapter = adapters_and_goal.action_adapter
    self.feature_constructor = adapters_and_goal.feature_constructor
    self.goal = adapters_and_goal.goal
    self._is_first_step = True

  def reset(
      self,
      rng: np.random.Generator,
      observation: microscope_utils.MicroscopeObservation,
  ) -> None:
    """Resets the agent."""
    self.feature_constructor.reset()
    self.goal.reset(rng, observation)
    self.action_adapter.reset()

    self._is_first_step = True

  def step(
      self,
      observation: microscope_utils.MicroscopeObservation,
  ) -> List[microscope_utils.BeamControlMicroscopeFrame]:
    """Steps the agent."""
    try:
      features = self.feature_constructor.get_features(observation, self.goal)
      goal_return = self.goal.calculate_reward_and_terminal(observation)
    except graphene.SiliconNotFoundError:
      # TODO(joshgreaves): This is desirable for now, but we will need to
      # rethink it in the future.
      # If we couldn't find a silicon, rescan.
      return [
          microscope_utils.BeamControlMicroscopeFrame(
              microscope_utils.BeamControl(
                  position=geometry.Point((0.0, 0.0)),
                  dwell_time=dt.timedelta(seconds=0),
              )
          )
      ]

    elapsed_seconds = observation.elapsed_time.total_seconds()
    discount = constants.GAMMA_PER_SECOND**elapsed_seconds

    if goal_return.is_terminal:
      time_step = dm_env.termination(goal_return.reward, features)
    elif goal_return.is_truncated:
      time_step = dm_env.truncation(goal_return.reward, features, discount)
    elif self._is_first_step:
      time_step = dm_env.restart(features)
    else:
      time_step = dm_env.transition(goal_return.reward, features, discount)

    action = self.agent.step(time_step)

    beam_control = self.action_adapter.get_action(observation, action)

    # First step is only after immediately calling reset.
    self._is_first_step = False

    return beam_control


class StepRecord(TypedDict):
  episode: int
  episode_step: int
  reward: float
  elapsed_seconds: float
  terminal: bool


class EpisodeRecord(TypedDict):
  episode: int
  episode_steps: int
  episode_return: float
  # Episode goal point in material frame if it's single silicon goal reaching
  # TODO(jfarebro): NotRequire on py311
  episode_goal: tuple[float, float] | None


class MicroscopeAgentLogger(wrapt.ObjectProxy):
  """A wrapper to log the microscope agent's data."""

  __wrapped__: MicroscopeAgent

  def __init__(
      self,
      agent: MicroscopeAgent,
      *,
      logdir: epath.Path | str | os.PathLike[str]
  ) -> None:
    super().__init__(agent)
    self._episode = 0
    self._episode_return = 0
    self._episode_step = 0

    self._logdir = epath.Path(logdir)
    self._current_trajectory: list[microscope_utils.MicroscopeObservation] = []
    self._trajectories: list[microscope_utils.Trajectory] = []
    self._step_records: list[StepRecord] = []
    self._episode_records: list[EpisodeRecord] = []

  def _make_episode_record(self) -> EpisodeRecord:
    record = EpisodeRecord(
        episode=self._episode,
        episode_steps=self._episode_step,
        episode_return=self._episode_return,
        episode_goal=None,
    )

    if isinstance(self.__wrapped__.goal, goals.SingleSiliconGoalReaching):
      record |= {
          'episode_goal': (
              self.__wrapped__.goal.current_goal.x,
              self.__wrapped__.goal.current_goal.y,
          )
      }
    return typing.cast(EpisodeRecord, record)

  def _make_step_record(
      self,
      observation: microscope_utils.MicroscopeObservation,
      goal_return: goals.GoalReturn,
  ) -> StepRecord:
    return StepRecord(
        episode=self._episode,
        episode_step=self._episode_step,
        reward=goal_return.reward,
        elapsed_seconds=observation.elapsed_time.total_seconds(),
        terminal=goal_return.is_terminal,
    )

  def __enter__(self) -> 'MicroscopeAgent':
    return self

  def __exit__(
      self,
      exc_type: Type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: types.TracebackType | None,
  ) -> None:
    del exc_val, exc_tb
    # If no exception was raised we'll flush
    if exc_type is None:
      self.flush()

  def flush(self) -> None:
    if self._current_trajectory:
      self._trajectories.append(
          microscope_utils.Trajectory(self._current_trajectory)
      )
    if self._episode_step != 0:
      self._episode_records.append(self._make_episode_record())

    # Write the records
    pdio.write_records(
        self._logdir / 'trajectories.tfrecords',
        self._trajectories,
    )

    # Write steps
    with (self._logdir / 'steps.csv').open('w') as fp:
      writer = csv.DictWriter(
          fp, fieldnames=typing.get_type_hints(StepRecord).keys()
      )
      writer.writeheader()
      for record in self._step_records:
        writer.writerow(record)

    # Write episodes
    with (self._logdir / 'episodes.csv').open('w') as fp:
      writer = csv.DictWriter(
          fp, fieldnames=typing.get_type_hints(EpisodeRecord).keys()
      )
      writer.writeheader()
      for record in self._episode_records:
        writer.writerow(record)

  def reset(
      self,
      rng: np.random.Generator,
      observation: microscope_utils.MicroscopeObservation,
  ) -> None:
    if self._episode_step > 0:
      self._episode += 1
      self._episode_records.append(self._make_episode_record())
    if self._current_trajectory:
      self._trajectories.append(
          microscope_utils.Trajectory(self._current_trajectory)
      )

    self._episode_step = 0
    self._episode_return = 0.0
    self._current_trajectory = []

    return self.__wrapped__.reset(rng, observation)

  def step(
      self,
      observation: microscope_utils.MicroscopeObservation,
  ) -> List[microscope_utils.BeamControlMicroscopeFrame]:
    beam_control = self.__wrapped__.step(observation)
    goal_return = self.__wrapped__.goal.calculate_reward_and_terminal(
        observation
    )
    self._episode_step += 1
    self._episode_return += goal_return.reward

    # Log data
    self._current_trajectory.append(observation)
    self._step_records.append(self._make_step_record(observation, goal_return))

    return beam_control
