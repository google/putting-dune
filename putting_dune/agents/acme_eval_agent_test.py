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
"""Unit tests for acme_eval_agent."""

from absl.testing import absltest
from acme.tf import savers
from putting_dune.agents import acme_eval_agent
from putting_dune.agents import agent_lib
from putting_dune.agents import ppo
from putting_dune.experiments import registry


class EvalAgentTest(absltest.TestCase):

  def test_acme_acme_eval_agent_can_be_constructed(self):
    # Create an Acme checkpoint.
    checkpoint_dir = self.create_tempdir()

    train_experiment = registry.create_train_experiment('relative_simple_rates')
    ppo_config = ppo.build_experiment_config(
        seed=0,
        experiment=train_experiment,
        num_steps=100,
        checkpoint_dir=checkpoint_dir.full_path,
    )
    eval_builder = acme_eval_agent._AcmeEvalAgentBuilder(ppo_config)
    # learner = typing.cast(core.Saveable, eval_builder.learner)
    learner = eval_builder.learner

    checkpointer = savers.Checkpointer(
        {'learner': learner},
        directory=checkpoint_dir.full_path,
        subdirectory='learner',
        add_uid=False,
    )
    checkpointer.save(force=True)

    agent = acme_eval_agent.AcmeEvalAgent(
        ppo.build_experiment_config(
            seed=0, experiment=train_experiment, num_steps=100_000_000
        ),
        checkpoint_dir.full_path,
    )
    self.assertIsInstance(agent, agent_lib.Agent)

  def test_acme_acme_eval_agent_raises_error_if_not_a_valid_directory(self):
    train_experiment = registry.create_train_experiment('relative_simple_rates')
    with self.assertRaises(FileNotFoundError):
      acme_eval_agent.AcmeEvalAgent(
          ppo.build_experiment_config(
              seed=0, experiment=train_experiment, num_steps=100_000_000
          ),
          'not_a_valid_dir',
      )

  def test_acme_acme_eval_agent_raises_error_if_no_checkpoint_found(self):
    empty_dir = self.create_tempdir()
    train_experiment = registry.create_train_experiment('relative_simple_rates')

    with self.assertRaises(FileNotFoundError):
      acme_eval_agent.AcmeEvalAgent(
          ppo.build_experiment_config(
              seed=0, experiment=train_experiment, num_steps=100_000_000
          ),
          empty_dir.full_path,
      )


if __name__ == '__main__':
  absltest.main()
