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

"""A collection of experiments."""

import dataclasses
import datetime as dt
import functools
from typing import Callable, Optional, Tuple
import urllib.request
import zipfile

from absl import logging
from etils import epath
import frozendict
import numpy as np
from putting_dune import action_adapters
from putting_dune import constants
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


def _get_greedy_agent(
    rng: np.random.Generator,
    adapters_and_goal: experiments.AdaptersAndGoal,
    argmax=np.asarray([1.42, 0.0]),
    transition_function=None,
    fixed_offset=np.zeros(
        2,
    ),
) -> agent_lib.GreedyAgent:
  return agent_lib.GreedyAgent(
      rng=rng,
      argmax=argmax,
      transition_function=transition_function,
      fixed_offset=fixed_offset,
      low=adapters_and_goal.action_adapter.action_spec.minimum,
      high=adapters_and_goal.action_adapter.action_spec.maximum,
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

_GET_PPO_LEARNED_TF_2S = _TfAgentCreator(
    model_name='230127_from_state_2s',
    download_url=(
        'https://storage.googleapis.com/spr_data_bucket_public/'
        '230127_from_state_2s.zip'
    ),
)
_GET_PPO_LEARNED_TF_3S = _TfAgentCreator(
    model_name='230127_from_state_3s',
    download_url=(
        'https://storage.googleapis.com/spr_data_bucket_public/'
        '230127_from_state_3s.zip'
    ),
)
_GET_PPO_LEARNED_TF_4S = _TfAgentCreator(
    model_name='230127_from_state_4s',
    download_url=(
        'https://storage.googleapis.com/spr_data_bucket_public/'
        '230127_from_state_4s.zip'
    ),
)
_PPO_V3_TF_2S = _TfAgentCreator(
    model_name='230422_ppo_v3_2s',
    download_url=(
        'https://storage.googleapis.com/spr_data_bucket_public/'
        '230422_ppo_v3_2s.zip'
    ),
)
_PPO_V3_TF_3S = _TfAgentCreator(
    model_name='230422_ppo_v3_3s',
    download_url=(
        'https://storage.googleapis.com/spr_data_bucket_public/'
        '230422_ppo_v3_3s.zip'
    ),
)
_PPO_V3_TF_4S = _TfAgentCreator(
    model_name='230422_ppo_v3_4s',
    download_url=(
        'https://storage.googleapis.com/spr_data_bucket_public/'
        '230422_ppo_v3_4s.zip'
    ),
)




# -------------------- AdaptersAndGoal --------------------


@dataclasses.dataclass(frozen=True)
class _SingleSiliconGoalReaching:
  dwell_time_range: Tuple[dt.timedelta, dt.timedelta] = (
      dt.timedelta(seconds=1.5),
      dt.timedelta(seconds=1.5),
  )
  max_distance_angstroms: float = constants.CARBON_BOND_DISTANCE_ANGSTROMS

  def __call__(self) -> experiments.AdaptersAndGoal:
    return experiments.AdaptersAndGoal(
        action_adapter=action_adapters.RelativeToSiliconActionAdapter(
            dwell_time_range=self.dwell_time_range,
            max_distance_angstroms=self.max_distance_angstroms,
        ),
        feature_constructor=feature_constructors.SingleSiliconPristineGrapheneFeatureConstuctor(),
        goal=goals.SingleSiliconGoalReaching(),
    )


@dataclasses.dataclass(frozen=True)
class _SingleSiliconGoalReachingMaterialFrame:
  dwell_time_range: Tuple[dt.timedelta, dt.timedelta] = (
      dt.timedelta(seconds=1.5),
      dt.timedelta(seconds=1.5),
  )
  max_distance_angstroms: float = constants.CARBON_BOND_DISTANCE_ANGSTROMS * 2.0

  def __call__(self) -> experiments.AdaptersAndGoal:
    return experiments.AdaptersAndGoal(
        action_adapter=action_adapters.RelativeToSiliconMaterialFrameActionAdapter(
            dwell_time_range=self.dwell_time_range,
            max_distance_angstroms=self.max_distance_angstroms,
        ),
        feature_constructor=feature_constructors.SingleSiliconMaterialFrameFeatureConstructor(),
        goal=goals.SingleSiliconGoalReaching(),
    )


@dataclasses.dataclass(frozen=True)
class _SingleSiliconGoalReachingFromPixels:
  dwell_time_range: Tuple[dt.timedelta, dt.timedelta] = (
      dt.timedelta(seconds=1.5),
      dt.timedelta(seconds=1.5),
  )

  def __call__(self) -> experiments.AdaptersAndGoal:
    return experiments.AdaptersAndGoal(
        action_adapter=action_adapters.RelativeToSiliconActionAdapter(
            dwell_time_range=self.dwell_time_range
        ),
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
          rate_function=graphene.PristineSingleSiGrRatePredictor(
              canonical_rate_prediction_fn=graphene.simple_canonical_rate_function,
          ),
      ),
      image_duration=dt.timedelta(seconds=2.0),
  )


def _get_human_prior_rates_config() -> experiments.SimulatorConfig:
  return experiments.SimulatorConfig(
      material=graphene.PristineSingleDopedGraphene(
          rate_function=graphene.PristineSingleSiGrRatePredictor(
              canonical_rate_prediction_fn=graphene.HumanPriorRatePredictor().predict,
          ),
      ),
      image_duration=dt.timedelta(seconds=2.0),
  )




_MICROSCOPE_EXPERIMENTS = frozendict.frozendict({
    'relative_random': experiments.MicroscopeExperiment(
        get_agent=_get_relative_random_agent,
        get_adapters_and_goal=_SingleSiliconGoalReaching(),
    ),
    'relative_random_long': experiments.MicroscopeExperiment(
        get_agent=_get_relative_random_agent,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.0),
                dt.timedelta(seconds=5.0),
            ),
            max_distance_angstroms=2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        ),
    ),
    'relative_random_extra_long': experiments.MicroscopeExperiment(
        get_agent=_get_relative_random_agent,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.0),
                dt.timedelta(seconds=5.0),
            ),
            max_distance_angstroms=3 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        ),
    ),
    'greedy_on_neighbor': experiments.MicroscopeExperiment(
        get_agent=functools.partial(
            _get_greedy_agent, argmax=np.array([1.42, 0.0])
        ),
        get_adapters_and_goal=_SingleSiliconGoalReachingMaterialFrame(
            dwell_time_range=(
                dt.timedelta(seconds=5.0),
                dt.timedelta(seconds=5.0),
            ),
            max_distance_angstroms=2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        ),
    ),
    'greedy_short_of_neighbor': experiments.MicroscopeExperiment(
        get_agent=functools.partial(
            _get_greedy_agent, argmax=np.array([0.58, 0.0])
        ),
        get_adapters_and_goal=_SingleSiliconGoalReachingMaterialFrame(
            dwell_time_range=(
                dt.timedelta(seconds=5.0),
                dt.timedelta(seconds=5.0),
            ),
            max_distance_angstroms=2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        ),
    ),
    'greedy_on_neighbor_offset_horizontally': experiments.MicroscopeExperiment(
        get_agent=functools.partial(
            _get_greedy_agent, argmax=np.array([1.42, 0.42])
        ),
        get_adapters_and_goal=_SingleSiliconGoalReachingMaterialFrame(
            dwell_time_range=(
                dt.timedelta(seconds=5.0),
                dt.timedelta(seconds=5.0),
            ),
            max_distance_angstroms=2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        ),
    ),
    'greedy_from_learned_rates_v3': experiments.MicroscopeExperiment(
        get_agent=functools.partial(
            _get_greedy_agent, argmax=np.array([1.8686869, 0.0])
        ),
        get_adapters_and_goal=_SingleSiliconGoalReachingMaterialFrame(
            dwell_time_range=(
                dt.timedelta(seconds=5.0),
                dt.timedelta(seconds=5.0),
            ),
            max_distance_angstroms=2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        ),
    ),
    'greedy_from_learned_rates_v5': experiments.MicroscopeExperiment(
        get_agent=functools.partial(
            _get_greedy_agent,
            argmax=np.array([2.1717172, -0.15151516]),
        ),
        get_adapters_and_goal=_SingleSiliconGoalReachingMaterialFrame(
            dwell_time_range=(
                dt.timedelta(seconds=5.0),
                dt.timedelta(seconds=5.0),
            ),
            max_distance_angstroms=2 * constants.CARBON_BOND_DISTANCE_ANGSTROMS,
        ),
    ),
    'ppo_simple_images_tf': experiments.MicroscopeExperiment(
        get_agent=_GET_PPO_SIMPLE_IMAGES_TF,
        get_adapters_and_goal=_SingleSiliconGoalReachingFromPixels(),
    ),
    'ppo_learned_tf_2s': experiments.MicroscopeExperiment(
        get_agent=_GET_PPO_LEARNED_TF_2S,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.0),
                dt.timedelta(seconds=10.0),
            )
        ),
    ),
    'ppo_learned_tf_3s': experiments.MicroscopeExperiment(
        get_agent=_GET_PPO_LEARNED_TF_3S,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.0),
                dt.timedelta(seconds=10.0),
            )
        ),
    ),
    'ppo_learned_tf_4s': experiments.MicroscopeExperiment(
        get_agent=_GET_PPO_LEARNED_TF_4S,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.0),
                dt.timedelta(seconds=10.0),
            )
        ),
    ),
    # V3 experiments from 04/19/23.
    'ppo_v3_2s': experiments.MicroscopeExperiment(
        get_agent=_PPO_V3_TF_2S,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.5),
                dt.timedelta(seconds=20.0),
            ),
            max_distance_angstroms=(
                constants.CARBON_BOND_DISTANCE_ANGSTROMS * 3
            ),
        ),
    ),
    'ppo_v3_3s': experiments.MicroscopeExperiment(
        get_agent=_PPO_V3_TF_3S,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.5),
                dt.timedelta(seconds=20.0),
            ),
            max_distance_angstroms=(
                constants.CARBON_BOND_DISTANCE_ANGSTROMS * 3
            ),
        ),
    ),
    'ppo_v3_4s': experiments.MicroscopeExperiment(
        get_agent=_PPO_V3_TF_4S,
        get_adapters_and_goal=_SingleSiliconGoalReaching(
            dwell_time_range=(
                dt.timedelta(seconds=1.5),
                dt.timedelta(seconds=20.0),
            ),
            max_distance_angstroms=(
                constants.CARBON_BOND_DISTANCE_ANGSTROMS * 3
            ),
        ),
    ),
})

_TRAIN_EXPERIMENTS = frozendict.frozendict(
    {
        'relative_simple_rates': experiments.TrainExperiment(
            get_adapters_and_goal=_SingleSiliconGoalReaching(),
            get_simulator_config=_get_simple_rates_config,
        ),
        'relative_prior_rates': experiments.TrainExperiment(
            get_adapters_and_goal=_SingleSiliconGoalReaching(),
            get_simulator_config=_get_human_prior_rates_config,
        ),
        'relative_simple_rates_from_images': experiments.TrainExperiment(
            get_adapters_and_goal=_SingleSiliconGoalReachingFromPixels(),
            get_simulator_config=_get_simple_rates_config,
        ),
        'relative_simple_rates_from_images_variable_time': (
            experiments.TrainExperiment(
                get_adapters_and_goal=_SingleSiliconGoalReachingFromPixels(
                    dwell_time_range=(
                        dt.timedelta(seconds=1.0),
                        dt.timedelta(seconds=10.0),
                    )
                ),
                get_simulator_config=_get_simple_rates_config,
            )
        ),
        'direct_simple_rates_from_images': experiments.TrainExperiment(
            get_adapters_and_goal=_get_direct_goal_reaching_from_pixels,
            get_simulator_config=_get_simple_rates_config,
        ),
    }
)

_EVAL_EXPERIMENTS = frozendict.frozendict(
    {
        'relative_random_simple': experiments.EvalExperiment(
            get_agent=_get_relative_random_agent,
            get_adapters_and_goal=_SingleSiliconGoalReaching(),
            get_simulator_config=_get_simple_rates_config,
        ),
        'relative_random_prior_rates': experiments.EvalExperiment(
            get_agent=_get_relative_random_agent,
            get_adapters_and_goal=_SingleSiliconGoalReaching(),
            get_simulator_config=_get_human_prior_rates_config,
        ),
        'ppo_simple_images_tf': experiments.EvalExperiment(
            get_agent=_GET_PPO_SIMPLE_IMAGES_TF,
            get_adapters_and_goal=_SingleSiliconGoalReachingFromPixels(),
            get_simulator_config=_get_simple_rates_config,
        ),
    }
)


def register_eval_experiment(
    name: str, eval_experiment: experiments.EvalExperiment
):
  global _EVAL_EXPERIMENTS
  if name not in _EVAL_EXPERIMENTS:
    _EVAL_EXPERIMENTS = frozendict.frozendict(
        {name: eval_experiment, **_EVAL_EXPERIMENTS}
    )


def create_microscope_experiment(name: str) -> experiments.MicroscopeExperiment:
  if name not in _MICROSCOPE_EXPERIMENTS:
    raise ValueError(f'Unknown microscope experiment {name}.')
  return _MICROSCOPE_EXPERIMENTS[name]


def create_train_experiment(name: str) -> experiments.TrainExperiment:
  if name not in _TRAIN_EXPERIMENTS:
    raise ValueError(f'Unknown train experiment {name}.')
  return _TRAIN_EXPERIMENTS[name]


def create_eval_experiment(name: str) -> experiments.EvalExperiment:
  if name not in _EVAL_EXPERIMENTS:
    raise ValueError(f'Unknown eval experiment {name}.')
  return _EVAL_EXPERIMENTS[name]
