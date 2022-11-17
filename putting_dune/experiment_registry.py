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
"""A collection of experiments."""

import dataclasses
from typing import Callable

import frozendict
import numpy as np
from putting_dune import action_adapters
from putting_dune import agent_lib
from putting_dune import goals
from putting_dune import putting_dune_environment



@dataclasses.dataclass
class Experiment:
  agent: agent_lib.Agent
  action_adapter: action_adapters.ActionAdapter
  # TODO(joshgreaves): Use abstract base class as the type hint.
  feature_constructor: putting_dune_environment.SingleSiliconPristineGraphineFeatureConstuctor
  goal: goals.Goal


def _random_experiment(rng: np.random.Generator) -> Experiment:
  action_adapter = action_adapters.RelativeToSiliconActionAdapter()

  return Experiment(
      agent=agent_lib.UniformRandomAgent(
          rng,
          action_adapter.action_spec.minimum,
          action_adapter.action_spec.maximum,
          action_adapter.action_spec.shape,
      ),
      action_adapter=action_adapter,
      feature_constructor=putting_dune_environment.SingleSiliconPristineGraphineFeatureConstuctor(),
      goal=goals.SingleSiliconGoalReaching(),
  )




_EXPERIMENT_CONSTRUCTORS = frozendict.frozendict(
    {
        'relative_random': _random_experiment,
    }
)


def create_experiment(name: str, rng: np.random.Generator) -> Experiment:
  if name not in _EXPERIMENT_CONSTRUCTORS:
    raise ValueError(f'Unknown experiment {name}.')
  return _EXPERIMENT_CONSTRUCTORS[name](rng)
