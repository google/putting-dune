# Putting Dune

[![Unittests](https://github.com/google/putting-dune/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google/putting-dune/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/putting_dune.svg)](https://badge.fury.io/py/putting_dune)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10626248.svg)](https://doi.org/10.5281/zenodo.10626248)

This repository contains the code of the paper ["Learning and Controlling Silicon Dopant Transitions in Graphene using Scanning Transmission Electron Microscopy"](https://arxiv.org/abs/2311.17894).
The so-called Putting Dune, provides the simulator and methods to learn transition rates of 3-fold silicon-doped graphene.

<img alt="Method Overview" src="/.github/assets/method-overview.png" />

## Data Representation

The protocol buffer data representation used in Putting Dune can be found in
`putting_dune/putting_dune.proto`. You can find a one-to-one correspondence
of the protocol buffer messages and a Python dataclass in
`putting_dune/microscope_utils.py`.

When performing alignment and rate learning we expect a sequence of serialized
`Trajectory` objects. As an example, we can serialize trajectories as follows:

```py
from putting_dune import io as pdio
from putting_dune import microscope_utils

trajectories = [
  microscope_utils.Trajectory(
    observations=[
      microscope_utils.Observation(...),
      ...,
    ],
  ),
  ...,
]

pdio.write_records("my-recorded-trajectories.tfrecords", trajectories)
```

## Image Aligner

The first step in our pipeline is to perform image alignment.
To train the image alignment model you can follow the steps in
`putting_dune/image_alignment/train.py`.

Once the image aligner is trained you can perform image alignment on the
recorded trajectories via the script `putting_dune/pipeline/align_trajectories.py`.
For example,

```sh
python -m putting_dune.pipeline.align_trajectories \
  --source_path my-recorded-trajectories.tfrecords \
  --target_path my-aligned-recorded-trajectories.tfrecords \
  --aligner_path my_trained_aligner \
  --alignment_iterations 5
```

## Rate Learner

Once the trajectories have been aligned you can now train the rate model.
This can be done with `putting_dune/pipeline/train_rate_learner.py`.
For example,

```sh
python -m putting_dune.pipeline.train_rate_learner \
  --source_path my-aligned-recorded-trajectories.tfrecords \
  --workdir my-rate-model
```

Once training is complete there will be various plots and checkpoints
that are saved to the working directory. This model can then be used
to derive a greedy controller or predict learned rates.

## Citation

```bib
@article{schwarzer23stem,
  author = {
    Max Schwarzer and
    Jesse Farebrother and
    Joshua Greaves and
    Ekin Dogus Cubuk and
    Rishabh Agarwal and
    Aaron Courville and
    Marc G. Bellemare and
    Sergei Kalinin and
    Igor Mordatch and
    Pablo Samuel Castro and
    Kevin M. Roccapriore
  },
  title = {Learning and Controlling Silicon Dopant Transitions in Graphene
           using Scanning Transmission Electron Microscopy},
  journal = {CoRR},
  volume = {abs/2311.17894},
  year = {2023},
}
```

## Note

*This is not an officially supported Google product.*
