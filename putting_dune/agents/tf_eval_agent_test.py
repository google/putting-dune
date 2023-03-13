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

"""Tests for tf_eval_agent."""

from absl.testing import absltest
import dm_env
import numpy as np
from putting_dune.agents import agent_lib
from putting_dune.agents import tf_eval_agent
import tensorflow as tf


class TfEvalAgentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Create a tf module to load.
    self._tf_dir = self.create_tempdir()
    model = tf.Module()
    model.__call__ = tf.function(
        lambda x: x,
        input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)],
        autograph=False,
    )
    tf.saved_model.save(model, self._tf_dir.full_path)

  def test_agent_loads_model_correctly(self):
    agent = tf_eval_agent.TfEvalAgent(self._tf_dir.full_path)
    self.assertIsNotNone(agent._model)

  def test_agent_step_returns_a_valid_value(self):
    agent = tf_eval_agent.TfEvalAgent(self._tf_dir.full_path)

    observation = np.asarray([1.0]).astype(np.float32)
    timestep = dm_env.transition(
        reward=0.0, observation=observation, discount=1.0
    )
    action = agent.step(timestep)

    self.assertIsInstance(action, np.ndarray)
    np.testing.assert_array_equal(action, observation)  # By construction.

  def test_set_mode_is_callable(self):
    agent = tf_eval_agent.TfEvalAgent(self._tf_dir.full_path)
    agent.set_mode(agent_lib.AgentMode.TRAIN)
    agent.set_mode(agent_lib.AgentMode.EVAL)


if __name__ == '__main__':
  absltest.main()
