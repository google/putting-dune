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
"""Useful structures for experiment definitions."""

import dataclasses
from typing import Callable

import numpy as np
from putting_dune import action_adapters
from putting_dune import agent_lib
from putting_dune import feature_constructors
from putting_dune import goals
from putting_dune import graphene


@dataclasses.dataclass(frozen=True)
class AdaptersAndGoal:
  action_adapter: action_adapters.ActionAdapter
  feature_constructor: feature_constructors.FeatureConstructor
  goal: goals.Goal


@dataclasses.dataclass(frozen=True)
class SimulatorConfig:
  material: graphene.Material


AgentConstructor = Callable[
    [np.random.Generator, AdaptersAndGoal], agent_lib.Agent
]
AdaptersAndGoalConstructor = Callable[[], AdaptersAndGoal]
SimulatorConfigConstructor = Callable[[], SimulatorConfig]


@dataclasses.dataclass(frozen=True)
class MicroscopeExperiment:
  get_agent: AgentConstructor
  get_adapters_and_goal: AdaptersAndGoalConstructor


@dataclasses.dataclass(frozen=True)
class TrainExperiment:
  get_adapters_and_goal: AdaptersAndGoalConstructor
  get_simulator_config: SimulatorConfigConstructor


@dataclasses.dataclass(frozen=True)
class EvalExperiment:
  get_agent: AgentConstructor
  get_adapters_and_goal: AdaptersAndGoalConstructor
  get_simulator_config: SimulatorConfigConstructor