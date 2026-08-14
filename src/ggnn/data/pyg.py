import jax.numpy as jnp
import torch

from ggnn.data.types import Graph

def to_jax_padded(xs, pad_len, pad_value=0, dtype=jnp.float32):
    return jnp.array([
        torch.nn.functional.pad(x, (0, pad_len - x.shape[-1]), value=pad_value)
        for x in xs
    ], dtype=dtype)

def pyg_to_jax(
    pyg_graphs, f_node_concat=None, dummy_node_feature=False,
    max_nodes=None, max_edges=None
):
    n = len(pyg_graphs)
    if max_nodes is None:
        max_nodes = max(graph.num_nodes for graph in pyg_graphs)
    if max_edges is None:
        max_edges = max(graph.num_edges for graph in pyg_graphs)
    if f_node_concat is None:
        f_node_concat = []
    if len(f_node_concat) > 0 or pyg_graphs[0].x is not None:
        f_node = [
            torch.cat(
                (
                    [graph.x] if graph.x is not None else []
                ) + [
                    graph[feat]
                    for feat in f_node_concat
                ],
                dim=1
            ).T
            for graph in pyg_graphs
        ]
        f_node = to_jax_padded(
            f_node,
            max_nodes+1,
        ).transpose(0, 2, 1)
    else:
        if dummy_node_feature:
            f_node = jnp.ones((n, max_nodes+1, 1), dtype=jnp.float32)
        else:
            f_node = jnp.zeros((n, max_nodes+1, 0), dtype=jnp.float32)
    if pyg_graphs[0].edge_attr is not None:
        f_edge = to_jax_padded(
            [
                graph.edge_attr.reshape(graph.edge_attr.shape[0], -1).T
                for graph in pyg_graphs
            ],
            max_edges
        ).transpose(0, 2, 1)
    else:
        f_edge = jnp.zeros((n, max_edges, 1), dtype=jnp.float32)
    return Graph(
        n_nodes=jnp.array([graph.num_nodes for graph in pyg_graphs]),
        n_edges=jnp.array([graph.num_edges for graph in pyg_graphs]),
        f_node=f_node,
        f_edge=f_edge,
        f_graph=jnp.array([graph.y.squeeze() for graph in pyg_graphs]),
        edges=to_jax_padded(
            [graph.edge_index for graph in pyg_graphs],
            max_edges,
            max_nodes,
            dtype=jnp.int32
        ).transpose(0, 2, 1)
    )
