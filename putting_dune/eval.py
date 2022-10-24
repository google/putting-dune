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

"""Entry point for standalone evaluation."""

from collections.abc import Sequence

from absl import app
import numpy as np
from putting_dune import agent_lib
from putting_dune import eval_lib
from putting_dune import run_helpers


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Set up the environment.
  # Note: We pass in an arbitrary seed here, since evaluate will
  # pass in a specific seed for each episode.
  env = run_helpers.create_putting_dune_env(seed=0)

  # Set up the agent.
  # TODO(joshgreaves): Pass seed in for agent.
  rng = np.random.default_rng(0)

  # For now we assume that the action_spec min and max are floats.
  # However, this may change.
  action_spec = env.action_spec()
  # These are actually np arrays with a single value, so unpack the float.
  action_minimum = action_spec.minimum.item()
  action_maximum = action_spec.maximum.item()
  assert isinstance(action_minimum, float)
  assert isinstance(action_maximum, float)

  agent = agent_lib.UniformRandomAgent(
      rng, action_minimum, action_maximum, action_spec.shape)

  # Set up the eval suite.
  # TODO(joshgreaves): Specify specific eval suites.
  eval_suite = eval_lib.EvalSuite(tuple(range(10)))

  eval_lib.evaluate(agent, env, eval_suite)


if __name__ == '__main__':
  app.run(main)
