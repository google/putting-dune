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

"""An evaluation-only agent using a tf saved model."""

import dm_env
import numpy as np
from putting_dune.agents import agent_lib
import tensorflow as tf


class TfEvalAgent(agent_lib.Agent):

  def __init__(self, path: str):
    self._model = tf.saved_model.load(path)

  def step(self, time_step: dm_env.TimeStep) -> np.ndarray:
    return self._model(time_step.observation).numpy()

  def set_mode(self, mode: agent_lib.AgentMode) -> None:
    pass  # No action required.
