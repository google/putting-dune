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
"""Tests for plotting_utils."""

from absl.testing import absltest

from matplotlib import animation
import numpy as np
from putting_dune import agent_lib
from putting_dune import plotting_utils
from putting_dune import run_helpers
from putting_dune import simulator_observers


def _generate_events_with_random_policy() -> (
    list[simulator_observers.SimulatorEvent]
):
  env = run_helpers.create_putting_dune_env(seed=0, step_limit=5)
  rng = np.random.default_rng(0)

  action_spec = env.action_spec()
  # These are actually np arrays with a single value, so unpack the float.
  action_minimum = action_spec.minimum.item()
  action_maximum = action_spec.maximum.item()
  assert isinstance(action_minimum, float)
  assert isinstance(action_maximum, float)

  agent = agent_lib.UniformRandomAgent(
      rng, action_minimum, action_maximum, action_spec.shape
  )

  event_observer = simulator_observers.EventObserver()
  env.sim.add_observer(event_observer)

  timestep = env.reset()
  while not timestep.last():
    action = agent.step(timestep)
    timestep = env.step(action)

  return event_observer.events


class EvalLibTest(absltest.TestCase):

  def test_something(self):
    events = _generate_events_with_random_policy()
    anim = plotting_utils.generate_video_from_simulator_events(
        events, goal_position=np.asarray([0.0, 0.0])
    )

    self.assertIsInstance(anim, animation.Animation)


if __name__ == '__main__':
  absltest.main()
