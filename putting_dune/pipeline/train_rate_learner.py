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

r"""A script for training rates from real data for use in KMC.

"""

import collections.abc
import dataclasses
import enum
import io
import os
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union

from absl import app
from etils import eapp
from etils import epath
import jax
from jax import numpy as jnp
# TODO(jfarebro) change sklearn serialization to skops if it becomes available.
import joblib
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
from putting_dune import constants
from putting_dune import io as pdio
from putting_dune import microscope_utils
from putting_dune.pipeline import trajectories_to_transitions
from putting_dune.rate_learning import data_utils
from putting_dune.rate_learning import learn_rates
from sklearn import ensemble as skensemble
from sklearn import gaussian_process as skgp
from sklearn import neighbors as skneighbors
from sklearn import neural_network as sknn
from sklearn import pipeline as skpipeline
from sklearn import preprocessing as skpreprocessing
from sklearn import svm as sksvm


class LearnerType(str, enum.Enum):
  RATE_NETWORK = 'rate_network'
  CLASSIFICATION_NETWORK = 'classification_network'
  RANDOM_FOREST = 'random_forest'
  SVM = 'svm'
  KNN = 'knn'
  GP = 'gp'
  SKLEARN_MLP = 'sklearn_mlp'


@dataclasses.dataclass
class Args:
  """Command line arguments."""

  source_path: epath.Path
  workdir: epath.Path
  log_metrics: bool = True
  plot_metrics: bool = True
  visualize_rates: bool = True
  plot_transitions: bool = False
  batch_size: int = 256
  epochs: int = 500
  num_models: int = 100
  bootstrap: bool = True
  hidden_dimensions: Iterable[int] = (128, 128)
  weight_decay: float = 1e-1
  learning_rate: float = 1e-3
  val_frac: float = 0
  neighbor_distance_cutoff: float = constants.CARBON_BOND_DISTANCE_ANGSTROMS / 2
  seed: int = 42
  distill: bool = True
  augment_data: bool = True
  use_voltage: bool = True
  use_current: bool = True
  batchnorm: bool = True
  dropout_rate: float = 0.0
  class_loss_weight: float = 0.1
  rate_loss_weight: float = 1.0

  learner_type: str = 'rate_network'


class DataPoint(TypedDict):
  next_state: int
  beam_pos: np.ndarray
  seconds_between: float
  current: Optional[float]
  voltage: Optional[float]


class Dataset(TypedDict):
  next_state: np.ndarray
  dt: np.ndarray
  position: np.ndarray
  context: Optional[np.ndarray]
  rates: np.ndarray


def get_sklearn_classifier(
    learner_type: Union[str, LearnerType],
    config: config_dict.FrozenConfigDict,
    standardize: bool = True,
    **kwargs,
):
  """Create a scikit-learn classifier based on a type and optional kwargs.

  Args:
    learner_type: Type of model to use.
    config: Standard argument dictionary.
    standardize: Include a scikit-learn StandardScaler to rescale model inputs.
    **kwargs: Any additional arguments for the model.

  Returns:
    An untrained scikit-learn classifier.

  Raises:
    ValueError if learner type is invalid.
  """
  if learner_type == LearnerType.SKLEARN_MLP:
    model = sknn.MLPClassifier(
        alpha=config.weight_decay,
        max_iter=config.epochs,
        learning_rate=config.learning_rate,
        **kwargs,
    )
  elif learner_type == LearnerType.KNN:
    model = skneighbors.KNeighborsClassifier(25, **kwargs)
  elif learner_type == LearnerType.GP:
    model = skgp.GaussianProcessClassifier(
        1.0 * skgp.kernels.RBF(1.0), **kwargs
    )
  elif learner_type == LearnerType.RANDOM_FOREST:
    model = skensemble.RandomForestClassifier(
        max_depth=5, n_estimators=10, **kwargs
    )
  elif learner_type == LearnerType.SVM:
    model = sksvm.SVC(gamma=2, C=1, **kwargs)
  else:
    raise ValueError('Invalid learner type specified.')

  if standardize:
    model = skpipeline.make_pipeline(skpreprocessing.StandardScaler(), model)

  return model


def train_sklearn_classifier(
    model: Any,
    training_data: Mapping[str, np.ndarray],
    testing_data: Mapping[str, np.ndarray],
) -> Tuple[Any, np.ndarray, np.ndarray]:
  """Train a sklearn classifier and evaluate it on a test set.

  Args:
    model: A scikit-learn classifier.
    training_data: Training dataset, map from string to array.
    testing_data: Testing dataset, map from string to array.

  Returns:
    Trained classifier, training accuracy, and testing accuracy.
  """
  train_x = training_data['context']
  train_y = training_data['next_state']
  test_x = testing_data['context']
  test_y = testing_data['next_state']

  model.fit(train_x, train_y)
  test_accuracy = model.score(test_x, test_y)
  train_accuracy = model.score(train_x, train_y)
  return model, train_accuracy, test_accuracy


def plot_transition(
    transition,
    neighbor_positions,
    neighbor_order,
    silicon_position,
    silicon_position_after,
    next_state,
    save_path: Optional[epath.Path] = None,
    show: bool = True,
):
  """Generate and optionally save or show a plot of a transition.

  Args:
    transition: Transition to visualize.
    neighbor_positions: Positions of the neighbors of the target atom.
    neighbor_order: Order of those neighbors in the standardized frame.
    silicon_position: Silicon position in material frame.
    silicon_position_after: Silicon position after transition in material frame.
    next_state: State the silicon is assessed as occupying in next observation.
    save_path: Path to save plot to. Does not save if None.
    show: Whether to show the plot.
  """
  fov_before = transition.fov_before
  fov_after = transition.fov_after
  plt.figure(figsize=(10, 10))
  if transition.image_before is None:
    image = np.zeros((512, 512, 1), dtype=np.uint8)
  else:
    image = transition.image_before
  imsize = image.shape[0]
  plt.imshow(image, cmap='gray', alpha=0.5)
  plt.scatter(
      transition.grid_before.atom_positions[:, 0] * imsize,
      imsize - transition.grid_before.atom_positions[:, 1] * imsize,
      c='k',
      marker='o',
      label='Atoms before transition',
  )

  material_frame_grid_after = fov_after.microscope_frame_to_material_frame(
      transition.grid_after
  )
  before_frame_grid_after = fov_before.material_frame_to_microscope_frame(
      material_frame_grid_after
  )
  plt.scatter(
      before_frame_grid_after.atom_positions[:, 0] * imsize,
      imsize - before_frame_grid_after.atom_positions[:, 1] * imsize,
      marker='d',
      c='k',
      label='Atoms after transition',
  )

  beam_position = transition.controls[0].position

  plt.scatter(
      beam_position.x * imsize,
      imsize - beam_position.y * imsize,
      marker='o',
      label='Beam position',
  )
  silicon_position = fov_before.material_frame_to_microscope_frame(
      silicon_position
  )
  silicon_position_after = fov_before.material_frame_to_microscope_frame(
      silicon_position_after
  )
  plt.scatter(
      imsize * silicon_position[0, 0],
      imsize - imsize * silicon_position[0, 1],
      label='Si position (before)',
  )
  plt.scatter(
      imsize * silicon_position_after[0, 0],
      imsize - imsize * silicon_position_after[0, 1],
      label='Si position (after)',
  )

  for i in range(0, 3):
    position = neighbor_positions[neighbor_order[i] + 1]
    position = fov_before.material_frame_to_microscope_frame(position)
    plt.scatter(
        imsize[0] * position[0],
        imsize[1] - imsize[1] * position[1],
        marker='x' if i + 1 == next_state else '+',
        label='Neighbor {}'.format(i + 1),
    )
  plt.legend()

  if save_path:
    transition_save_path = epath.Path(str(save_path) + '_transition.png')
    with transition_save_path.open('wb') as f:
      plt.savefig(f, format='png', bbox_inches='tight')

  if show:
    plt.show()
  plt.close()

  if transition.label_image_before is not None:
    plt.figure(figsize=(10, 10))
    plt.imshow(transition.label_image_before)
    if save_path:
      label_save_path = epath.Path(str(save_path) + '_label.png')
      with label_save_path.open('wb') as f:
        plt.savefig(f, format='png', bbox_inches='tight')
    if show:
      plt.show()
    plt.close()


def transitions_to_datapoints(
    transitions: List[microscope_utils.Transition],
    args: Args,
    plot_transitions: bool = False,
    save_path: Optional[str] = None,
    show_plots: bool = False,
) -> List[DataPoint]:
  """Process transitions to datapoints usable for rate learning.

  Args:
    transitions: List of transitions
    args: Command line arguments
    plot_transitions: Whether to generate an image of each transition.
    save_path: Where to save plots, if generated.
    show_plots: Whether to show plots, if generated.

  Returns:
    List of of dictionaries, each containing one usable datapoint.
  """

  wrong_number_of_silicons_count = 0
  no_controls_count = 0
  no_silicon_neighbor_count = 0
  wrong_number_of_neighbors_count = 0
  wrong_number_of_next_step_neighbors_count = 0
  neighbors_too_distant_count = 0
  is_fourfold_count = 0
  not_threefold_count = 0

  data = []

  if save_path is not None and save_path:
    save_path = epath.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

  for transition in transitions:
    if len(transition.controls) != 1:
      no_controls_count += 1
      continue

    # For now, assume only one control applied.
    control = transition.controls[0]

    grid_before = transition.grid_before
    grid_after = transition.grid_after

    grid_before = transition.fov_before.microscope_frame_to_material_frame(
        grid_before
    )
    grid_after = transition.fov_after.microscope_frame_to_material_frame(
        grid_after
    )
    control = transition.fov_before.microscope_frame_to_material_frame(control)

    # Get the silicon position in the grid before.
    silicon_position = grid_before.atom_positions[
        grid_before.atomic_numbers == constants.SILICON
    ]
    if silicon_position.shape != (1, 2):
      wrong_number_of_silicons_count += 1
      continue

    # Get the neighbors of the grid before.
    _, neighbor_indices_before = (
        skneighbors.NearestNeighbors(
            n_neighbors=1 + 9,
            metric='l2',
            algorithm='brute',
        )
        .fit(grid_before.atom_positions)
        .kneighbors(silicon_position)
    )
    neighbor_indices_before = neighbor_indices_before.reshape(-1)
    neighbor_positions_before = grid_before.atom_positions[
        neighbor_indices_before
    ]
    if neighbor_positions_before.shape != (10, 2):
      wrong_number_of_neighbors_count += 1
      continue
    if neighbor_indices_before.shape != (10,):
      wrong_number_of_neighbors_count += 1
      continue

    distances_from_silicon = np.linalg.norm(
        silicon_position - neighbor_positions_before, axis=-1
    )
    is_fourfold = (
        np.abs(distances_from_silicon[1] - distances_from_silicon[4]) < 0.5
    )
    is_threefold = (distances_from_silicon[1:4] < 2.0).all() and (
        distances_from_silicon[4:] > 2.0
    ).all()
    if is_fourfold:
      is_fourfold_count += 1
      continue
    if not is_threefold and not is_fourfold:
      not_threefold_count += 1
      continue

    neighbor_positions_before = neighbor_positions_before[:4]
    # Get the neighbors of the grid after.
    _, neighbor_indices_after = (
        skneighbors.NearestNeighbors(
            n_neighbors=1 + 3,
            metric='l2',
            algorithm='brute',
        )
        .fit(grid_after.atom_positions)
        .kneighbors(silicon_position)
    )
    neighbor_indices_after = neighbor_indices_after.reshape(-1)
    neighbor_positions_after = grid_after.atom_positions[neighbor_indices_after]
    neighbor_atomic_numbers_after = grid_after.atomic_numbers[
        neighbor_indices_after
    ]
    silicon_position_after = grid_after.atom_positions[
        grid_after.atomic_numbers == constants.SILICON
    ]
    if silicon_position_after.shape != (1, 2):
      wrong_number_of_silicons_count += 1
      continue
    if not (neighbor_atomic_numbers_after == constants.SILICON).any():
      # There is a chance that the silicon moved more than one position.
      # For now, we skip this edge case.
      no_silicon_neighbor_count += 1
      continue
    if neighbor_positions_after.shape != (4, 2):
      wrong_number_of_next_step_neighbors_count += 1
      continue
    if neighbor_indices_after.shape != (4,):
      wrong_number_of_next_step_neighbors_count += 1
      continue
    if neighbor_atomic_numbers_after.shape != (4,):
      wrong_number_of_next_step_neighbors_count += 1
      continue

    # Get the distances of all neighbors before from all neighbors after.
    # We have to do this, since the neighbor order may not be consistent.
    neighbor_distances = np.linalg.norm(
        neighbor_positions_before.reshape(4, 1, 2)
        - neighbor_positions_after.reshape(1, 4, 2),
        axis=-1,
    )
    neighbor_distances = np.min(neighbor_distances, axis=-1)
    if neighbor_distances.mean() > args.neighbor_distance_cutoff:
      neighbors_too_distant_count += 1
      continue

    # Get the canonical representation for the grid before.
    control_position = np.asarray(control.position)
    control_delta = control_position - silicon_position.reshape(-1)
    neighbor_before_deltas = neighbor_positions_before - silicon_position
    (
        standardized_beam_position_before,
        _,
        neighbor_order_before,
    ) = data_utils.standardize_beam_and_neighbors(
        control_delta, neighbor_before_deltas[1:]
    )

    seconds_between = control.dwell_time.total_seconds()
    next_state = 0
    silicon_transition_state = np.argmin(
        np.linalg.norm(
            neighbor_positions_before - silicon_position_after, axis=-1
        )
    )
    if silicon_transition_state > 0:
      next_state = (
          np.argsort(neighbor_order_before)[silicon_transition_state - 1] + 1
      )
    else:
      next_state = 0

    data.append(
        DataPoint(
            next_state=next_state,
            beam_pos=standardized_beam_position_before,
            seconds_between=seconds_between,
            current=control.current_na,
            voltage=control.voltage_kv,
        )
    )

    if plot_transitions:
      if save_path is not None:
        class_directory = save_path / f'class_{next_state}'
        class_directory.mkdir(parents=True, exist_ok=True)
        img_save_path = class_directory / f'transition_{len(data)}'
      else:
        img_save_path = None

      plot_transition(
          transition=transition,
          neighbor_positions=neighbor_before_deltas + silicon_position,
          neighbor_order=neighbor_order_before,
          silicon_position=silicon_position,
          silicon_position_after=silicon_position_after,
          next_state=next_state,
          show=show_plots,
          save_path=img_save_path,
      )

  print('Initial number of transitions: {}'.format(len(transitions)))

  print('Number of points excluded for given reasons:')
  print('No controls: {}'.format(no_controls_count))
  print('Wrong number of silicons: {}'.format(wrong_number_of_silicons_count))
  print('No Si neighbor: {}'.format(no_silicon_neighbor_count))
  print('Fourfold: {}'.format(is_fourfold_count))
  print('Not threefold: {}'.format(not_threefold_count))
  print('Wrong number of neighbors: {}'.format(wrong_number_of_neighbors_count))
  print(
      'Wrong number of neighbors at next step: {}'.format(
          wrong_number_of_next_step_neighbors_count
      )
  )

  print('Neighbors too distant: {}'.format(neighbors_too_distant_count))
  print('Number of transitions remaining after filtering: {}'.format(len(data)))

  return data


def stack_data(
    data: List[DataPoint],
    use_current: bool = False,
    use_voltage: bool = False,
    dwell_time_in_context: bool = False,
    *,
    num_neighbors: int = 3,
) -> Dataset:
  """Stack a list of datapoints into a single dataset for rate learning.

  Args:
    data: List of DataPoints.
    use_current: Bool, whether currents should be added to the context vector.
    use_voltage: Bool, whether voltage should be added to the context vector.
    dwell_time_in_context: Bool, whether dwell time should be added to context.
    num_neighbors: int, the number of neighbors, default 3 for 3-fold.

  Returns:
    Single Dataset that can be passed to a rate learner.
  """
  beam_positions = np.stack([d['beam_pos'] for d in data])
  next_states = np.stack([d['next_state'] for d in data])
  dts = np.stack([d['seconds_between'] for d in data])
  rates = np.zeros((next_states.shape[0], num_neighbors))

  context = []
  if use_current:
    currents = np.stack([d['current'] for d in data])
    context.append(currents)
  if use_voltage:
    voltages = np.stack([d['voltage'] for d in data])
    context.append(voltages)
  if dwell_time_in_context:
    context.append(dts)

  if context:
    context = np.stack(context, axis=-1)
  else:
    context = None

  dataset = Dataset(
      next_state=next_states,
      position=beam_positions,
      dt=dts,
      rates=rates,
      context=context,
  )
  return dataset


def visualize_data(
    next_states: np.ndarray,
    positions: np.ndarray,
    dwell_times: np.ndarray,
    num_states: int = 3,
    save_path: Optional[str] = None,
) -> None:
  """Plot three-fold Si transition data in a human-readable form.

  Args:
    next_states: Array of next state indices.
    positions: Array of beam positions.
    dwell_times: Array of dwell times (used to set intensity).
    num_states: How many states are in the data.
    save_path: Path to save plot to.
  """
  plt.figure(figsize=(10, 10))

  for i in range(num_states + 1):
    mask = next_states == i
    local_positions = positions[mask]
    if local_positions.size == 0:
      continue

    label: str = ''
    if i == 0:
      label = 'No movement'
    elif i == 1:
      label = 'Moved right'
    elif i == 2:
      label = 'Moved up-left'
    elif i == 3:
      label = 'Moved down-left'
    assert label, f'Invalid label index {i}'

    plt.scatter(
        local_positions[:, 0],
        local_positions[:, 1],
        label=label,
        alpha=dwell_times[mask] / np.max(dwell_times),
    )

  plt.scatter(0, 0, label='Silicon position')

  plt.hlines(0, -5, 5)
  plt.vlines(0, -5, 5)

  plt.xlim(-5, 5)
  plt.ylim(-5, 5)

  plt.xlabel('Beam position (x-displacement)')
  plt.ylabel('Beam position (y-displacement)')

  plt.legend()

  if save_path is not None:
    with epath.Path(save_path).open('wb') as f:
      plt.savefig(f, bbox_inches='tight')


def load_trajectories_from_records(
    path: epath.Path | Sequence[epath.Path],
) -> List[microscope_utils.Trajectory]:
  """Load trajectories from a path.

  If it is a directory, load its children.

  Args:
    path: Path to trajectories (single records file or directory of records).

  Returns:
    List of loaded trajectories.
  """
  trajectories = []
  if isinstance(path, collections.abc.Sequence):
    files = path
  elif path.is_dir():
    files = path.iterdir()
  else:
    files = [path]

  for file in files:
    trajectories.extend(pdio.read_records(file, microscope_utils.Trajectory))
  return trajectories


def main(args: Args) -> None:
  trajectories = load_trajectories_from_records(args.source_path)
  transitions = trajectories_to_transitions.trajectories_to_transitions(
      trajectories
  )
  processed_transitions = transitions_to_datapoints(
      transitions,
      args,
      args.plot_transitions,
      save_path=os.path.join(args.workdir, 'data_plots/'),
      show_plots=False,
  )

  counts = {0: 0, 1: 0, 2: 0, 3: 0}
  for t in processed_transitions:
    counts[t['next_state']] += 1
  print(counts, flush=True)

  stacked_data = stack_data(
      processed_transitions,
      use_current=args.use_current,
      use_voltage=args.use_voltage,
      dwell_time_in_context=args.learner_type != LearnerType.RATE_NETWORK,
  )

  rng_key = jax.random.PRNGKey(args.seed)

  if args.learner_type != LearnerType.RATE_NETWORK:
    # for any model implementing rate-based transitions, setting the dt field
    # to a constant 1 will cause a shift to regular classification.
    stacked_data['dt'].fill(1)

  visualize_data(
      stacked_data['next_state'],
      stacked_data['position'],
      stacked_data['dt'],
      num_states=3,
      save_path=os.path.join(args.workdir, 'raw_data.png'),
  )
  augmented_data = data_utils.augment_data(**stacked_data)  # pytype: disable=wrong-arg-types  # jax-ndarray
  visualize_data(
      augmented_data['next_state'],
      augmented_data['position'],
      augmented_data['dt'],
      num_states=3,
      save_path=os.path.join(
          args.workdir, os.path.join(args.workdir, 'augmented_data.png')
      ),
  )

  frozen_config = config_dict.FrozenConfigDict(dataclasses.asdict(args))

  if (
      args.learner_type == LearnerType.RATE_NETWORK
      or args.learner_type == LearnerType.CLASSIFICATION_NETWORK
  ):
    rate_predictor_keys = jax.random.split(rng_key)
    rate_predictor = learn_rates.LearnedTransitionRatePredictor(
        num_states=3,
        init_key=rate_predictor_keys[0],
        config=frozen_config,
    )

    training_metrics = rate_predictor.train(
        {k: jnp.asarray(v) for k, v in stacked_data.items()},
        rate_predictor_keys[1],
        bootstrap=args.bootstrap,
    )

    if args.log_metrics:
      path = epath.Path(os.path.join(args.workdir, 'metrics.npz'))
      with path.open('wb') as file:
        metrics_buffer = io.BytesIO()
        np.savez_compressed(metrics_buffer, **training_metrics)
        file.write(metrics_buffer.getvalue())

    if args.plot_metrics:
      for k, v in training_metrics.items():
        plt.figure()
        for i in range(v.shape[0]):
          plt.plot(v[i][0:])

        best_iter = v.mean(0)[0:].argmin()
        plt.axvline(best_iter, label='Best iteration')
        plt.plot(v.mean(0)[0:], label='Average', linewidth=4)
        plt.yscale('log')
        title = k.replace('_', ' ').title()
        plt.legend()
        plt.title(title)
        plot_path = os.path.join(args.workdir, f'{k}.png')
        with epath.Path(plot_path).open('wb') as f:
          plt.savefig(f, bbox_inches='tight')
        plt.clf()

    if args.distill:
      distillation_defaults = config_dict.FrozenConfigDict({
          'batch_size': 4096,
          'epochs': 10000,
          'batches_per_epoch': 10,
      })

      rate_predictor.distill(augmented_data, config=distillation_defaults)

    avg_context = np.median(stacked_data['context'], axis=0)
    rate_predictor.save(args.workdir.as_posix(), fixed_context=avg_context)
    rate_prediction_function = rate_predictor.apply_model

  # Using a scikit-learn classifier
  else:
    train_datasets, test_datasets = learn_rates.create_dataset_splits(
        {k: jnp.asarray(v) for k, v in stacked_data.items()},
        num_splits=args.num_models,
        key=rng_key,
        bootstrap=args.bootstrap,
        augment_data=args.augment_data,
        test_fraction=args.val_frac,
    )

    models = []
    train_accuracies = []
    test_accuracies = []
    for i in range(args.num_models):
      train_data = {k: np.asarray(v[i]) for k, v in train_datasets.items()}
      test_data = {k: np.asarray(v[i]) for k, v in test_datasets.items()}

      model = get_sklearn_classifier(
          args.learner_type,
          config=frozen_config,
          standardize=True,
      )
      model, train_accuracy, test_accuracy = train_sklearn_classifier(
          model, train_data, test_data
      )

      save_path = epath.Path(
          os.path.join(args.workdir, f'{args.learner_type}_{i}.joblib')
      )
      with save_path.open('wb') as f:
        joblib.dump(model, f)

      models.append(model)
      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)

    training_metrics = {
        'train_accuracy': np.stack([train_accuracies], axis=0),
        'test_accuracy': np.stack([test_accuracies], axis=0),
    }
    ensemble = skensemble.VotingClassifier(
        [(f'{args.learner_type}_{i}', models[i]) for i in range(len(models))],
        voting='soft',
    )
    # SKLearn doesn't recognize that the models are already trained by default.
    ensemble.estimators_ = models
    save_path = epath.Path(
        os.path.join(args.workdir, f'{args.learner_type}_ensemble.joblib')
    )
    with save_path.open('wb') as f:
      joblib.dump(ensemble, f)

    if args.log_metrics:
      path = epath.Path(os.path.join(args.workdir, 'metrics.npz'))
      with path.open('wb') as file:
        np.savez_compressed(file, **training_metrics)

    rate_prediction_function = lambda x: ensemble.predict_proba(x)[..., 1:]

  if stacked_data['context'] is not None:
    min_context = stacked_data['context'].min(axis=0)
    max_context = stacked_data['context'].max(axis=0)
    avg_context = np.median(stacked_data['context'], axis=0)
  else:
    min_context = max_context = avg_context = None

  if args.visualize_rates:
    learn_rates.visualize_rates(
        os.path.join(args.workdir, 'max_context_rates.png'),
        rate_prediction_function,
        grid_range=10.0,
        num_points=300**2,
        fixed_context=max_context,
    )
    learn_rates.visualize_rates(
        os.path.join(args.workdir, 'min_context_rates.png'),
        rate_prediction_function,
        grid_range=10.0,
        num_points=300**2,
        fixed_context=min_context,
    )
    learn_rates.visualize_rates(
        os.path.join(args.workdir, 'avg_context_rates.png'),
        rate_prediction_function,
        grid_range=10.0,
        num_points=300**2,
        fixed_context=avg_context,
    )


if __name__ == '__main__':
  app.run(main, flags_parser=eapp.make_flags_parser(Args))
