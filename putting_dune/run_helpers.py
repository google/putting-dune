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
"""Shared helpers for RL experiments."""

import typing
from typing import Sequence

from acme import wrappers
from acme.utils.google import experiment_utils
from putting_dune import microscope_utils
from putting_dune import putting_dune_environment
from putting_dune.experiments import experiments

make_logger = experiment_utils.make_experiment_logger


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
  )
  env = wrappers.SinglePrecisionWrapper(env)
  env = wrappers.StepLimitWrapper(env, step_limit=step_limit)

  # Before returning, cast the environment back to PuttingDuneEnvironment.
  # While not strictly true, it has the same public interface.
  env = typing.cast(putting_dune_environment.PuttingDuneEnvironment, env)
  env.seed(seed)

  for observer in simulator_observers:
    env.sim.add_observer(observer)

  return env
