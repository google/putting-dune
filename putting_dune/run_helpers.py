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

"""Shared helpers for RL experiments."""

import typing
from typing import Optional, Sequence

import dm_env
import numpy as np
from putting_dune import microscope_utils
from putting_dune import putting_dune_environment
from putting_dune.experiments import experiments


def create_putting_dune_env(
    seed: int,
    *,
    get_adapters_and_goal: experiments.AdaptersAndGoalConstructor,
    get_simulator_config: experiments.SimulatorConfigConstructor,
    simulator_observers: Sequence[microscope_utils.SimulatorObserver] = (),
    # 30 minutes, based on current exposure/image capturing times.
    step_limit: int = 600,
) -> putting_dune_environment.PuttingDuneEnvironment:
  """Creates Putting Dune environment for RL experiments."""
  adapters_and_goal = get_adapters_and_goal()
  simulator_config = get_simulator_config()
  env = putting_dune_environment.PuttingDuneEnvironment(
      material=simulator_config.material,
      action_adapter=adapters_and_goal.action_adapter,
      feature_constructor=adapters_and_goal.feature_constructor,
      goal=adapters_and_goal.goal,
      image_duration=simulator_config.image_duration,
  )
  env = StepLimitWrapper(env, step_limit=step_limit)

  # Before returning, cast the environment back to PuttingDuneEnvironment.
  # While not strictly true, it has the same public interface.
  env = typing.cast(putting_dune_environment.PuttingDuneEnvironment, env)
  env.seed(seed)

  for observer in simulator_observers:
    env.sim.add_observer(observer)

  return env


##################################################################
# ENVIRONMENT WRAPPER - copied from Acme to avoid dependency.
##################################################################


class EnvironmentWrapper(dm_env.Environment):
  """Environment that wraps another environment.

  This exposes the wrapped environment with the `.environment` property and also
  defines `__getattr__` so that attributes are invisibly forwarded to the
  wrapped environment (and hence enabling duck-typing).
  """

  _environment: dm_env.Environment

  def __init__(self, environment: dm_env.Environment):
    self._environment = environment

  def __getattr__(self, name):
    if name.startswith("__"):
      raise AttributeError(
          "attempted to get missing private attribute '{}'".format(name)
      )
    return getattr(self._environment, name)

  @property
  def environment(self) -> dm_env.Environment:
    return self._environment

  # The following lines are necessary because methods defined in
  # `dm_env.Environment` are not delegated through `__getattr__`, which would
  # only be used to expose methods or properties that are not defined in the
  # base `dm_env.Environment` class.

  def step(self, action) -> dm_env.TimeStep:
    return self._environment.step(action)

  def reset(self) -> dm_env.TimeStep:
    return self._environment.reset()

  def action_spec(self):
    return self._environment.action_spec()

  def discount_spec(self):
    return self._environment.discount_spec()

  def observation_spec(self):
    return self._environment.observation_spec()

  def reward_spec(self):
    return self._environment.reward_spec()

  def close(self):
    return self._environment.close()


##################################################################
# STEP LIMIT WRAPPER - copied from Acme to avoid dependency.
##################################################################


class StepLimitWrapper(EnvironmentWrapper):
  """A wrapper which truncates episodes at the specified step limit."""

  def __init__(
      self, environment: dm_env.Environment, step_limit: Optional[int] = None
  ):
    super().__init__(environment)
    self._step_limit = step_limit
    self._elapsed_steps = 0

  def reset(self) -> dm_env.TimeStep:
    self._elapsed_steps = 0
    return self._environment.reset()

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    if self._elapsed_steps == -1:
      # The previous episode was truncated by the wrapper, so start a new one.
      timestep = self._environment.reset()
    else:
      timestep = self._environment.step(action)
    # If this is the first timestep, then this `step()` call was done on a new,
    # terminated or truncated environment instance without calling `reset()`
    # first. In this case this `step()` call should be treated as `reset()`,
    # so should not increment step count.
    if timestep.first():
      self._elapsed_steps = 0
      return timestep
    self._elapsed_steps += 1
    if self._step_limit is not None and self._elapsed_steps >= self._step_limit:
      self._elapsed_steps = -1
      return dm_env.truncation(
          timestep.reward, timestep.observation, timestep.discount
      )
    return timestep
