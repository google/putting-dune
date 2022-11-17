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
r"""Entry point for standalone evaluation.

"""

import dataclasses
from typing import Optional

from absl import app
from etils import eapp
import numpy as np
from putting_dune import eval_lib
from putting_dune import experiment_registry
from putting_dune import run_helpers


@dataclasses.dataclass
class Args:
  """Command line arguments."""

  experiment_name: str
  video_save_dir: Optional[str] = None


def main(args: Args) -> None:
  # TODO(joshgreaves): Pass seed in for agent.
  rng = np.random.default_rng(0)
  experiment = experiment_registry.create_experiment(args.experiment_name, rng)
  agent = experiment.agent

  # Set up the environment.
  # Note: We pass in an arbitrary seed here, since evaluate will
  # pass in a specific seed for each episode.
  # TODO(joshgreaves): Pass the relative objects from the experiment into
  # the environment.
  env = run_helpers.create_putting_dune_env(seed=0)

  # Set up the eval suite.
  # TODO(joshgreaves): Specify specific eval suites.
  eval_suite = eval_lib.EvalSuite(tuple(range(10)))

  eval_lib.evaluate(agent, env, eval_suite, video_save_dir=args.video_save_dir)


if __name__ == '__main__':
  app.run(main, flags_parser=eapp.make_flags_parser(Args))
