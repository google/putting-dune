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

"""U-net model for semantic segmentation."""
import functools
from typing import Callable, Sequence, Tuple

from flax import linen as nn
from flax.linen import kw_only_dataclasses
import jax
import jax.numpy as jnp

Array = jax.Array

Conv3x3 = functools.partial(
    nn.Conv,
    kernel_size=(3, 3),
    padding='SAME',
)
ConvTranspose3x3 = functools.partial(
    nn.ConvTranspose,
    kernel_size=(3, 3),
    padding='SAME',
)


class GlobalLocalUNet(nn.Module):
  """Hybrid U-net model that produces both local and global outputs."""

  # Number of output dimensions
  local_output_size: int = kw_only_dataclasses.field(default=3, kw_only=True)
  global_output_size: int = kw_only_dataclasses.field(default=2, kw_only=True)

  activation: Callable[[Array], Array] = nn.gelu
  norm: Callable[..., Callable[[Array], Array]] = nn.normalization.LayerNorm

  # Features per layer
  features: Sequence[int] = (64, 128, 256, 512, 1024)

  def recurse(self, x: Array, *, depth: int = 0) -> Tuple[Array, Array]:
    x = Conv3x3(
        features=self.features[depth],
        name=f'ConvDown_{depth}',
    )(x)
    x = self.norm()(x)
    x = self.activation(x)

    if len(self.features) > depth + 1:
      x_down = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
      x_down, x_bottom = self.recurse(x_down, depth=depth + 1)
      x_up = ConvTranspose3x3(
          features=self.features[depth],
          strides=(2, 2),
          name=f'ConvTranspose_{depth}',
      )(x_down)

      x = jnp.concatenate((x_up, x), axis=-1)
      x = Conv3x3(
          features=self.features[depth],
          name=f'ConvUp_{depth}',
      )(x)
      x = self.norm()(x)
      x = self.activation(x)
    else:
      x_bottom = x

    return x, x_bottom

  @nn.compact
  def __call__(self, x: Array) -> Tuple[Array, Array]:
    x = nn.Conv(
        features=self.features[0],
        kernel_size=(7, 7),
        name='ConvInput',
    )(x)
    x = self.norm()(x)
    x = self.activation(x)
    x, x_bottom = self.recurse(x)
    segmentation_output = nn.Conv(
        features=self.local_output_size,
        kernel_size=(7, 7),
        name='ConvOutput',
    )(x)
    global_output = nn.Conv(
        features=256,
        kernel_size=(1, 1),
        name='global_output',
    )(x_bottom)
    global_output = self.norm()(global_output)
    global_output = self.activation(global_output)
    global_output = global_output.mean((-2, -3))
    global_output = nn.Dense(self.global_output_size)(global_output)

    return segmentation_output, global_output
