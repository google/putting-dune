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

"""Graph neural network model for atomic grid alignment."""

import functools
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax_md import nn
from jax_md import partition
from jax_md import space
import jraph
import numpy as np


PyTree = Any
Array = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat


Array = jax.Array


class GraphNetEncoder(hk.Module):
  """Implements a Graph Neural Network to encode a graph.

  Adapted from JaxMD, with support for normalization layers introduced to
  counter divergence when using many recurrences.

  Attributes:
    n_recurrences: How many graph connection recurrences to apply.
    mlp_sizes: Tuple of sizes defining the depth and width of MLPs connecting
      nodes and encoding states.
    mlp_kwargs: Keyword arguments to pass to Haiku MLPs.
    norm_type: Optionally a Haiku normalization class to apply between
      recurrences. Choices are GroupNorm, LayerNorm, and None.
  """

  def __init__(
      self,
      n_recurrences: int,
      mlp_sizes: Tuple[int, ...],
      mlp_kwargs: Optional[Dict[str, Any]] = None,
      norm_type: Optional[Any] = hk.GroupNorm,
      name: str = 'GraphNetEncoder',
  ):
    super(GraphNetEncoder, self).__init__(name=name)

    if mlp_kwargs is None:
      mlp_kwargs = {}

    self._n_recurrences = n_recurrences

    if norm_type is None:
      self._norm = lambda: lambda x: x
    elif norm_type == hk.LayerNorm:
      self._norm = functools.partial(
          hk.LayerNorm, axis=-1, create_scale=True, create_offset=True
      )
    elif norm_type == hk.GroupNorm:
      self._norm = functools.partial(
          hk.GroupNorm, groups=8, axis=-1, create_scale=True, create_offset=True
      )
    else:
      self._norm = norm_type

    def embedding_fn(name, x):
      mlp_output = hk.nets.MLP(
          output_sizes=mlp_sizes, activate_final=True, name=name, **mlp_kwargs
      )(x)
      mlp_output = self._norm()(mlp_output)
      return mlp_output

    def model_fn(name, *args):
      mlp_output = hk.nets.MLP(
          output_sizes=mlp_sizes, activate_final=True, name=name, **mlp_kwargs
      )(jnp.concatenate(args, axis=-1))
      mlp_output = self._norm()(mlp_output)
      return mlp_output

    self._encoder = jraph.GraphMapFeatures(
        functools.partial(embedding_fn, 'EdgeEncoder'),
        functools.partial(embedding_fn, 'NodeEncoder'),
        functools.partial(embedding_fn, 'GlobalEncoder'),
    )
    self._propagation_network = functools.partial(
        jraph.GraphNetwork,
        functools.partial(model_fn, 'EdgeFunction'),
        functools.partial(model_fn, 'NodeFunction'),
        functools.partial(model_fn, 'GlobalFunction'),
    )

  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    encoded = self._encoder(graph)
    outputs = encoded
    for _ in range(self._n_recurrences):
      inputs = nn.concatenate_graph_features((outputs, encoded))
      outputs = self._propagation_network()(inputs)

    return outputs


class AlignmentGraphNetwork(hk.Module):
  """Implements a Graph Neural Network for iterative alignment.


  This model uses a GraphNetEmbedding combined with a decoder applied
  to the global state.
  """

  def __init__(
      self,
      n_recurrences: int,
      mlp_sizes: Tuple[int, ...],
      mlp_kwargs: Optional[Dict[str, Any]] = None,
      partition_format: Any = partition.Sparse,
      sequence_length: int = 5,
      output_dims: int = 2,
      name: str = 'Alignment',
  ):
    super(AlignmentGraphNetwork, self).__init__(name=name)

    self.sequence_length = sequence_length

    if mlp_kwargs is None:
      mlp_kwargs = {
          'w_init': hk.initializers.VarianceScaling(),
          'b_init': hk.initializers.VarianceScaling(0.1),
          'activation': jax.nn.gelu,
      }
    self._format = partition_format
    self._graph_net = GraphNetEncoder(
        n_recurrences,
        mlp_sizes,
        mlp_kwargs,
    )
    self.global_output_shape = (sequence_length, output_dims)
    self._global_decoder = hk.nets.MLP(
        output_sizes=mlp_sizes + (np.prod(self.global_output_shape),),
        activate_final=False,
        name='GlobalDecoder',
        **mlp_kwargs,
    )

    self._local_decoder = hk.nets.MLP(
        output_sizes=mlp_sizes + (output_dims,),
        activate_final=False,
        name='LocalDecoder',
        **mlp_kwargs,
    )

    self._pooling_decoder = hk.nets.MLP(
        output_sizes=mlp_sizes + (output_dims,),
        activate_final=False,
        name='PoolingDecoder',
        **mlp_kwargs,
    )

  def __call__(
      self,
      graph: jraph.GraphsTuple,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    output = self._graph_net(graph)
    node_latents = jnp.array(output.nodes)
    global_latents = output.globals
    if self._format is partition.Sparse:
      global_latents = global_latents[0]
      node_latents = node_latents[..., :-1, :]

    node_latents = node_latents.reshape(
        *node_latents.shape[:-2],
        self.sequence_length,
        -1,
        node_latents.shape[-1],
    )

    global_output = self._global_decoder(global_latents)
    global_output = global_output.reshape(*self.global_output_shape)

    local_output = self._local_decoder(node_latents)
    pooling_output = self._pooling_decoder(node_latents)
    return global_output + pooling_output.mean(-2), local_output


def create_sparse_neighbor_list(
    capacity,
    initial_position,
    update_fn,
    cell_list_capacity,
):
  """Creates a static JaxMD neighbor list of given size, skipping allocate.

  Args:
    capacity: Total number of bonds to allow.
    initial_position: Starting configuration of the graph, array of shape
      (N_atoms, n_dim).
    update_fn: Update function, e.g., as returned by JaxMD's neighbor_list.
    cell_list_capacity: Cell list capacity (only applies if using cell lists).

  Returns:
    A JaxMD NeighborList that can be updated. Must call update() at least once
    to initialize edges.
  """
  idx_matrix = jnp.zeros((2, capacity), jnp.int32)
  overflow = jnp.zeros((), dtype=jnp.bool_)
  neighbor_list = partition.NeighborList(  # pytype: disable=wrong-arg-count
      idx_matrix,
      initial_position,
      overflow,
      cell_list_capacity,
      capacity,
      partition.Sparse,
      update_fn,
  )
  return neighbor_list


def graph_network_neighbor_list(
    displacement_fn: DisplacementFn,
    box_size: Box,
    r_cutoff: float,
    dr_threshold: float,
    n_recurrences: int = 2,
    output_dims: int = 2,
    sequence_length: int = 5,
    mlp_sizes: Tuple[int, ...] = (64, 64),
    mlp_kwargs: Optional[Dict[str, Any]] = None,
    fractional_coordinates: bool = False,
    partition_format: partition.NeighborListFormat = partition.Sparse,
    **neighbor_kwargs,
) -> Tuple[
    NeighborFn,
    Callable[..., Mapping[str, Mapping[str, jax._src.basearray.Array]]],
    nn.CallFn,
]:
  """Convenience wrapper around AlignmentGraphNetwork that uses neighbor lists.

  Args:
    displacement_fn: Function to compute displacement between two positions.
    box_size: The size of the simulation volume, used to construct neighbor
      list.
    r_cutoff: A floating point cutoff; Edges will be added to the graph for
      pairs of particles whose separation is smaller than the cutoff.
    dr_threshold: A floating point number specifying a "halo" radius that we use
      for neighbor list construction. See `neighbor_list` for details.
    n_recurrences: The number of steps of message passing in the graph network.
    output_dims: Number of output dimensions of the network.
    sequence_length: Length of input sequences being passed.
    mlp_sizes: A tuple specifying the layer-widths for the fully-connected
      networks used to update the states in the graph network.
    mlp_kwargs: A dict specifying args for the fully-connected networks used to
      update the states in the graph network.
    fractional_coordinates: A boolean specifying whether or not the coordinates
      will be in the unit cube.
    partition_format: The format of the neighbor list. See
      `partition.NeighborListFormat` for details. Only `Dense` and `Sparse`
      formats are accepted. If the `Dense` format is used, then the graph
      network is constructed using the JAX MD backend, otherwise Jraph is used.
    **neighbor_kwargs: Extra keyword arguments are passed to the neighbor list.

  Returns:
    A pair of functions. An `params = init_fn(key, R)` that instantiates the
    model parameters and an `E = apply_fn(params, R)` that computes the energy
    for a particular state.
  """

  def create_graph_fn(
      positions,
      neighbor,
      nodes,
      displacement_fn=displacement_fn,
      r_cutoff=r_cutoff,
      dr_threshold=dr_threshold,
      **kwargs,
  ):
    num_points = np.prod(positions.shape[:-1])
    positions = jnp.reshape(positions, (num_points, positions.shape[-1]))
    nodes = jnp.reshape(nodes, (num_points, nodes.shape[-1]))
    num_points = positions.shape[0]
    displacement_fn = functools.partial(displacement_fn, **kwargs)  # pytype: disable=name-error

    global_inputs = jnp.zeros((1,), positions.dtype)

    if partition_format is partition.Dense:
      displacement_fn = space.map_neighbor(displacement_fn)
      neighbor_positions = positions[neighbor.idx]
      displacements = displacement_fn(positions, neighbor_positions)

      dr_2 = space.square_distance(displacements)
      edge_idx = jnp.where(dr_2 < r_cutoff**2, neighbor.idx, num_points)
      graph = nn.GraphsTuple(nodes, displacements, global_inputs, edge_idx)  # pytype: disable=wrong-arg-count
    else:
      displacement_fn = space.map_bond(displacement_fn)
      displacements = displacement_fn(
          positions[neighbor.idx[0]], positions[neighbor.idx[1]]
      )
      if dr_threshold > 0.0:
        dr_2 = space.square_distance(displacements)
        mask = dr_2 < r_cutoff**2 + 1e-5
        graph = partition.to_jraph(neighbor, mask)
        displacements = displacement_fn(
            positions[graph.receivers], positions[graph.senders]
        )
      else:
        graph = partition.to_jraph(neighbor)

      graph = graph._replace(
          nodes=jnp.concatenate(
              (nodes, jnp.zeros((1,) + nodes.shape[1:], positions.dtype)),
              axis=0,
          ),
          edges=displacements,
          globals=jnp.broadcast_to(global_inputs[:, None], (2, 1)),
      )
      return graph

  @hk.without_apply_rng
  @hk.transform
  def model(positions, neighbor, nodes, **kwargs):
    graph = create_graph_fn(positions, neighbor, nodes, **kwargs)

    net = AlignmentGraphNetwork(
        n_recurrences,
        mlp_sizes,
        mlp_kwargs,
        partition_format,
        sequence_length=sequence_length,
        output_dims=output_dims,
    )
    return net(graph)  # pytype: disable=wrong-arg-count

  neighbor_fn = partition.neighbor_list(
      displacement_fn,
      box_size,
      r_cutoff,
      dr_threshold,
      mask_self=False,
      fractional_coordinates=fractional_coordinates,
      format=partition_format,
      **neighbor_kwargs,
  )
  init_fn, apply_fn = model.init, model.apply

  return neighbor_fn, init_fn, apply_fn
