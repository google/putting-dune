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

"""Functions for evaluating an agent."""

import dataclasses
import os
from typing import Optional

from absl import logging
import dm_env
from putting_dune import agent_lib
from putting_dune import plotting_utils
from putting_dune import putting_dune_environment
from putting_dune import simulator_observers


@dataclasses.dataclass(frozen=True)
class EvalSuite:
  seeds: tuple[int, ...]


@dataclasses.dataclass
class EvalResult:
  seed: int
  reached_goal: bool
  num_actions_taken: int
  seconds_to_goal: float
  total_reward: float


def evaluate(
    agent: agent_lib.Agent,
    env: putting_dune_environment.PuttingDuneEnvironment,
    eval_suite: EvalSuite,
    *,
    video_save_dir: Optional[str] = None) -> list[EvalResult]:
  """Evaluates an agent on the specified environment and evaluation suite.

  Args:
    agent: The agent to evluate.
    env: The PuttingDuneEnvironment to evaluate on.
    eval_suite: The evaluation suite to run.
    video_save_dir: A directory to save videos of the evaluation runs at.
      If set to None, then videos won't be generated. Generating videos
      significantly slows down evaluation time.

  Returns:
    A list containing eval results, one for each seed in the eval_suite.
  """
  agent.set_mode(agent_lib.AgentMode.EVAL)
  results = []
  observers = {}

  if video_save_dir is not None:
    observers['event_observer'] = simulator_observers.EventObserver()

  for observer in observers.values():
    env.sim.add_observer(observer)

  for seed in eval_suite.seeds:
    logging.info('Evaluating seed %d', seed)
    num_actions_taken = 0
    total_reward = 0.0

    env.seed(seed)
    time_step = env.reset()
    action = agent.step(time_step)

    while time_step.step_type != dm_env.StepType.LAST:
      time_step = env.step(action)
      action = agent.step(time_step)

      num_actions_taken += 1
      total_reward += time_step.reward

    reached_goal = (time_step.step_type == dm_env.StepType.LAST
                    and time_step.discount == 0.0)

    seconds_to_goal = env.sim.elapsed_time.total_seconds()
    if not reached_goal:
      seconds_to_goal = float('nan')

    eval_result = EvalResult(
        seed=seed,
        reached_goal=reached_goal,
        num_actions_taken=num_actions_taken,
        seconds_to_goal=seconds_to_goal,
        total_reward=total_reward)
    results.append(eval_result)

    if video_save_dir is not None:
      anim = plotting_utils.generate_video_from_simulator_events(
          observers['event_observer'].events,
          env._goal_pos_material_frame)  # pylint: disable=protected-access
      anim.save(os.path.join(video_save_dir, f'{seed}.gif'))

  for observer in observers.values():
    env.sim.remove_observer(observer)

  return results
