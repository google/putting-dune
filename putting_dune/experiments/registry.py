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
from typing import Callable, Optional
import urllib.request
import zipfile

from absl import logging
from etils import epath
import frozendict
import numpy as np
from putting_dune import action_adapters
from putting_dune import feature_constructors
from putting_dune import goals
from putting_dune import graphene
from putting_dune.agents import agent_lib
from putting_dune.agents import tf_eval_agent
from putting_dune.experiments import experiments



# -------------------- AGENTS --------------------


def _get_relative_random_agent(
    rng: np.random.Generator, adapters_and_goal: experiments.AdaptersAndGoal
) -> agent_lib.UniformRandomAgent:
  return agent_lib.UniformRandomAgent(
      rng,
      adapters_and_goal.action_adapter.action_spec.minimum,
      adapters_and_goal.action_adapter.action_spec.maximum,
      adapters_and_goal.action_adapter.action_spec.shape,
  )


@dataclasses.dataclass(frozen=True)
class _TfAgentCreator:
  """Gets a tf eval agent, loading from the specified path."""

  model_name: str
  download_url: str

  def __call__(
      self,
      rng: np.random.Generator,
      adapters_and_goal: experiments.AdaptersAndGoal,
  ) -> tf_eval_agent.TfEvalAgent:
    del rng  # Unused.
    del adapters_and_goal  # Unused.


    logging.info('Loading %s', self.model_name)
    experiments_path = epath.Path(__file__).parent.resolve()
    model_weights_path = experiments_path / 'model_weights'
    model_weights_path.mkdir(parents=False, exist_ok=True)
    model_path = model_weights_path / self.model_name

    # Download the file if necessary.
    if not model_path.exists():
      logging.info("Couldn't find agent checkpointing. Downloading...")
      zip_path = model_weights_path / 'tmp.zip'
      zip_path_str = str(zip_path)
      urllib.request.urlretrieve(
          self.download_url,
          zip_path_str,
      )

      logging.info('Unzipping agent checkpoint...')
      with zipfile.ZipFile(zip_path_str, mode='r') as zf:
        zf.extractall(model_weights_path)

      # Delete the zip file.
      zip_path.unlink()

    agent = tf_eval_agent.TfEvalAgent(str(model_path))
    logging.info('Agent loaded!')
    return agent


_GET_PPO_SIMPLE_IMAGES_TF = _TfAgentCreator(
    model_name='ppo_simple_images_tf',
    download_url=(
        'https://storage.googleapis.com/spr_data_bucket_public/'
        'ppo_simple_images_tf.zip'
    ),
)




# -------------------- AdaptersAndGoal --------------------


def _get_single_silicon_goal_reaching_adapters() -> experiments.AdaptersAndGoal:
  return experiments.AdaptersAndGoal(
      action_adapter=action_adapters.RelativeToSiliconActionAdapter(),
      feature_constructor=feature_constructors.SingleSiliconPristineGraphineFeatureConstuctor(),
      goal=goals.SingleSiliconGoalReaching(),
  )


def _get_single_silicon_goal_reaching_from_pixels() -> (
    experiments.AdaptersAndGoal
):
  return experiments.AdaptersAndGoal(
      action_adapter=action_adapters.RelativeToSiliconActionAdapter(),
      feature_constructor=feature_constructors.ImageFeatureConstructor(),
      goal=goals.SingleSiliconGoalReaching(),
  )


def _get_direct_goal_reaching_from_pixels() -> experiments.AdaptersAndGoal:
  return experiments.AdaptersAndGoal(
      action_adapter=action_adapters.DirectActionAdapter(),
      feature_constructor=feature_constructors.ImageFeatureConstructor(),
      goal=goals.SingleSiliconGoalReaching(),
  )


# -------------------- SimulatorConfigs --------------------


def _get_simple_rates_config() -> experiments.SimulatorConfig:
  return experiments.SimulatorConfig(
      material=graphene.PristineSingleDopedGraphene(
          predict_rates=graphene.simple_transition_rates
      )
  )


def _get_human_prior_rates_config() -> experiments.SimulatorConfig:
  return experiments.SimulatorConfig(
      material=graphene.PristineSingleDopedGraphene(
          predict_rates=graphene.HumanPriorRatePredictor().predict
      )
  )


_MICROSCOPE_EXPERIMENTS = frozendict.frozendict({
    'relative_random': experiments.MicroscopeExperiment(
        get_agent=_get_relative_random_agent,
        get_adapters_and_goal=_get_single_silicon_goal_reaching_adapters,
    ),
    'ppo_simple_images_tf': experiments.MicroscopeExperiment(
        get_agent=_GET_PPO_SIMPLE_IMAGES_TF,
        get_adapters_and_goal=_get_single_silicon_goal_reaching_from_pixels,
    ),
})

_TRAIN_EXPERIMENTS = frozendict.frozendict({
    'relative_simple_rates': experiments.TrainExperiment(
        get_adapters_and_goal=_get_single_silicon_goal_reaching_adapters,
        get_simulator_config=_get_simple_rates_config,
    ),
    'relative_prior_rates': experiments.TrainExperiment(
        get_adapters_and_goal=_get_single_silicon_goal_reaching_adapters,
        get_simulator_config=_get_human_prior_rates_config,
    ),
    'relative_simple_rates_from_images': experiments.TrainExperiment(
        get_adapters_and_goal=_get_single_silicon_goal_reaching_from_pixels,
        get_simulator_config=_get_simple_rates_config,
    ),
    'direct_simple_rates_from_images': experiments.TrainExperiment(
        get_adapters_and_goal=_get_direct_goal_reaching_from_pixels,
        get_simulator_config=_get_simple_rates_config,
    ),
})

_EVAL_EXPERIMENTS = frozendict.frozendict(
    {
        'relative_random_simple': experiments.EvalExperiment(
            get_agent=_get_relative_random_agent,
            get_adapters_and_goal=_get_single_silicon_goal_reaching_adapters,
            get_simulator_config=_get_simple_rates_config,
        ),
        'relative_random_prior_rates': experiments.EvalExperiment(
            get_agent=_get_relative_random_agent,
            get_adapters_and_goal=_get_single_silicon_goal_reaching_adapters,
            get_simulator_config=_get_human_prior_rates_config,
        ),
        'ppo_simple_images_tf': experiments.EvalExperiment(
            get_agent=_GET_PPO_SIMPLE_IMAGES_TF,
            get_adapters_and_goal=_get_single_silicon_goal_reaching_from_pixels,
            get_simulator_config=_get_simple_rates_config,
        ),
    }
)


def create_microscope_experiment(name: str) -> experiments.MicroscopeExperiment:
  if name not in _MICROSCOPE_EXPERIMENTS:
    raise ValueError(f'Unknown micorscope experiment {name}.')
  return _MICROSCOPE_EXPERIMENTS[name]


def create_train_experiment(name: str) -> experiments.TrainExperiment:
  if name not in _TRAIN_EXPERIMENTS:
    raise ValueError(f'Unknown train experiment {name}.')
  return _TRAIN_EXPERIMENTS[name]


def create_eval_experiment(name: str) -> experiments.EvalExperiment:
  if name not in _EVAL_EXPERIMENTS:
    raise ValueError(f'Unknown eval experiment {name}.')
  return _EVAL_EXPERIMENTS[name]
