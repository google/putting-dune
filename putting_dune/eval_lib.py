# Copyright 2023 The Putting Dune Authors.
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
import datetime as dt
import shutil
import tempfile
import time
from typing import List, Optional, Sequence, Tuple

from absl import logging
import dm_env
from etils import epath
import frozendict
from putting_dune import plotting_utils
from putting_dune import putting_dune_environment
from putting_dune import simulator_observers
from putting_dune.agents import agent_lib


@dataclasses.dataclass(frozen=True)
class EvalSuite:
  seeds: Tuple[int, ...]


EVAL_SUITES = frozendict.frozendict({
    'tiny_eval': EvalSuite(tuple(range(10))),
    'small_eval': EvalSuite(tuple(range(100))),
    'medium_eval': EvalSuite(tuple(range(1_000))),
    'big_eval': EvalSuite(tuple(range(10_000))),
})


@dataclasses.dataclass(frozen=True)
class EvalResult:
  seed: int
  reached_goal: bool
  num_actions_taken: int
  agent_seconds_to_goal: float
  environment_seconds_to_goal: float
  total_reward: float

  @property
  def seconds_to_goal(self) -> float:
    return self.agent_seconds_to_goal + self.environment_seconds_to_goal


@dataclasses.dataclass(frozen=True)
class AggregateEvalResults:
  average_num_times_reached_goal: float
  average_num_actions_taken: float
  average_agent_seconds_to_goal: float
  average_environment_seconds_to_goal: float
  average_total_reward: float

  @property
  def average_seconds_to_goal(self) -> float:
    return (
        self.average_agent_seconds_to_goal
        + self.average_environment_seconds_to_goal
    )


def evaluate(
    agent: agent_lib.Agent,
    env: putting_dune_environment.PuttingDuneEnvironment,
    eval_suite: EvalSuite,
    *,
    timeout: dt.timedelta = dt.timedelta(minutes=10),
    video_save_dir: Optional[str] = None,
) -> List[EvalResult]:
  """Evaluates an agent on the specified environment and evaluation suite.

  Args:
    agent: The agent to evaluate.
    env: The PuttingDuneEnvironment to evaluate on.
    eval_suite: The evaluation suite to run.
    timeout: A timeout to impose on evaluation. This timeout includes both the
      simulated time and time the agent spends computing actions.
    video_save_dir: A directory to save videos of the evaluation runs at. If set
      to None, then videos won't be generated. Generating videos significantly
      slows down evaluation time.

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

    # We keep track of environment time and agent time separately, since
    # the environment time is the elapsed simulated time, but the agent
    # time is actually wall clock clock time ⏰.
    agent_elapsed_time = dt.timedelta(seconds=0)
    environment_elapsed_time = dt.timedelta(seconds=0)

    env.seed(seed)
    time_step = env.reset()

    environment_elapsed_time += env.last_microscope_observation.elapsed_time

    # Ideally, the timeout should be enforced by the environment wrapper,
    # but it doesn't know how long the agent spends computing.
    while agent_elapsed_time + environment_elapsed_time < timeout:
      # Step the agent.
      agent_start_time = time.perf_counter()
      action = agent.step(time_step)
      agent_delta_seconds = time.perf_counter() - agent_start_time

      # Step the environment.
      time_step = env.step(action)

      # Update metrics.
      agent_elapsed_time += dt.timedelta(seconds=agent_delta_seconds)
      environment_elapsed_time += env.last_microscope_observation.elapsed_time
      num_actions_taken += 1
      total_reward += time_step.reward

      if time_step.last():
        break

    reached_goal = (
        time_step.step_type == dm_env.StepType.LAST
        and time_step.discount == 0.0
    )

    agent_seconds_to_goal = agent_elapsed_time.total_seconds()
    environment_seconds_to_goal = environment_elapsed_time.total_seconds()
    if not reached_goal:
      agent_seconds_to_goal = float('nan')
      environment_seconds_to_goal = float('nan')

    eval_result = EvalResult(
        seed=seed,
        reached_goal=reached_goal,
        num_actions_taken=num_actions_taken,
        agent_seconds_to_goal=agent_seconds_to_goal,
        environment_seconds_to_goal=environment_seconds_to_goal,
        total_reward=total_reward,
    )
    results.append(eval_result)

    if video_save_dir is not None:
      epath.Path(video_save_dir).mkdir(parents=True, exist_ok=True)
      # NOTE: This will not work on Windows, since Windows won't allow us
      # to open the temp file twice.
      with tempfile.NamedTemporaryFile(suffix='.gif') as src_f:
        anim = plotting_utils.generate_video_from_simulator_events(
            observers['event_observer'].events,
            env.goal.goal_position_material_frame,  # pylint: disable=protected-access
        )
        anim.save(src_f.name)

        with (epath.Path(video_save_dir) / f'{seed}.gif').open('wb') as dest_f:
          shutil.copyfileobj(src_f, dest_f)

  for observer in observers.values():
    env.sim.remove_observer(observer)

  return results


def aggregate_results(results: Sequence[EvalResult]) -> AggregateEvalResults:
  """Aggregates a sequence of eval results."""
  num_times_reached_goal = 0
  num_actions_taken = 0
  agent_seconds_to_goal = 0.0
  environment_seconds_to_goal = 0.0
  total_reward = 0.0

  for result in results:
    num_times_reached_goal += int(result.reached_goal)

    if result.reached_goal:
      num_actions_taken += result.num_actions_taken
      agent_seconds_to_goal += result.agent_seconds_to_goal
      environment_seconds_to_goal += result.environment_seconds_to_goal
      total_reward += result.total_reward

  denominator = max(num_times_reached_goal, 1)

  return AggregateEvalResults(
      average_num_times_reached_goal=num_times_reached_goal / len(results),
      average_num_actions_taken=num_actions_taken / denominator,
      average_agent_seconds_to_goal=agent_seconds_to_goal / denominator,
      average_environment_seconds_to_goal=(
          environment_seconds_to_goal / denominator
      ),
      average_total_reward=total_reward / denominator,
  )
