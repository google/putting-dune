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

r"""Entry point for standalone evaluation.

"""

import dataclasses
from typing import Optional

from absl import app
from absl import logging
from etils import eapp
import numpy as np
from putting_dune import eval_lib
from putting_dune import run_helpers
from putting_dune.experiments import registry


@dataclasses.dataclass
class Args:
  """Command line arguments."""

  experiment_name: str
  eval_suite: str
  video_save_dir: Optional[str] = None


def main(args: Args) -> None:
  # TODO(joshgreaves): Pass seed in for agent.
  rng = np.random.default_rng(0)
  experiment = registry.create_eval_experiment(args.experiment_name)

  adapters_and_goal = experiment.get_adapters_and_goal()
  agent = experiment.get_agent(rng, adapters_and_goal)

  # Set up the environment.
  # Note: We pass in an arbitrary seed here, since evaluate will
  # pass in a specific seed for each episode.
  env = run_helpers.create_putting_dune_env(
      seed=0,
      get_adapters_and_goal=experiment.get_adapters_and_goal,
      get_simulator_config=experiment.get_simulator_config,
  )

  # Set up the eval suite.
  eval_suite = eval_lib.EVAL_SUITES[args.eval_suite]

  eval_results = eval_lib.evaluate(
      agent, env, eval_suite, video_save_dir=args.video_save_dir
  )

  aggregate_results = eval_lib.aggregate_results(eval_results)
  logging.info('Finished evaluation for experiment %s', args.experiment_name)
  logging.info(
      'Proportion successful runs: %.2f',
      aggregate_results.average_num_times_reached_goal,
  )
  logging.info(
      'Average number of actions taken: %.2f',
      aggregate_results.average_num_actions_taken,
  )
  logging.info(
      'Average seconds to goal: %.2f', aggregate_results.average_seconds_to_goal
  )
  logging.info(
      'Average agent seconds to goal: %.2f',
      aggregate_results.average_agent_seconds_to_goal,
  )
  logging.info(
      'Average environment seconds to goal: %.2f',
      aggregate_results.average_environment_seconds_to_goal,
  )
  logging.info(
      'Average total reward: %.2f', aggregate_results.average_total_reward
  )


if __name__ == '__main__':
  app.run(main, flags_parser=eapp.make_flags_parser(Args))
