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
from typing import Callable, Sequence

from acme import wrappers
from acme.utils.google import experiment_utils
from putting_dune import action_adapters
from putting_dune import feature_constructors
from putting_dune import goals
from putting_dune import microscope_utils
from putting_dune import putting_dune_environment

make_logger = experiment_utils.make_experiment_logger


def _default_env_config_factory() -> (
    putting_dune_environment.EnvironmentConfiguration
):
  return putting_dune_environment.EnvironmentConfiguration(
      action_adapter=action_adapters.RelativeToSiliconActionAdapter(),
      feature_constructor=(
          feature_constructors.SingleSiliconPristineGraphineFeatureConstuctor()
      ),
      goal=goals.SingleSiliconGoalReaching(),
  )


def create_putting_dune_env(
    seed: int,
    *,
    env_config_factory: Callable[
        [], putting_dune_environment.EnvironmentConfiguration
    ] = _default_env_config_factory,
    simulator_observers: Sequence[microscope_utils.SimulatorObserver] = (),
    # 30 minutes, based on current exposure/image capturing times.
    step_limit: int = 600,
) -> putting_dune_environment.PuttingDuneEnvironment:
  """Creates Putting Dune environment for RL experiments."""
  env_config = env_config_factory()
  env = putting_dune_environment.PuttingDuneEnvironment(
      action_adapter=env_config.action_adapter,
      feature_constructor=env_config.feature_constructor,
      goal=env_config.goal,
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
