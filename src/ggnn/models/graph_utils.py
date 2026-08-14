import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ggnn.data.types import Graph

# Graphs must contain all self edges exactly once
## Graph has to be directed
def get_degrees(g: Graph) -> Float[Array, 'n']:
    return jnp.maximum(jax.ops.segment_sum(
        (jnp.ones(g.edges.shape[0]) < g.n_edges).astype(jnp.int32),
        g.edges[:, 0],
        num_segments=g.f_node.shape[0]
    ), 1).astype(jnp.float32)

def norm_sym(g: Graph) -> Graph:
    deg = get_degrees(g)
    val = g.f_edge
    if val is None:
        val = jnp.ones(g.edges.shape[0], dtype=jnp.float32)
    weight = jax.lax.rsqrt(deg)
    val = weight[g.edges[:, 0]] * val * weight[g.edges[:, 1]]
    return eqx.tree_at(lambda g: g.f_edge, g, val)

def norm_row(g: Graph) -> Graph:
    deg = get_degrees(g)
    val = g.f_edge
    if val is None:
        val = jnp.ones(g.edges.shape[0], dtype=jnp.float32)
    val = val / deg[g.edges[:, 0]]
    return eqx.tree_at(lambda g: g.f_edge, g, val)

def top_k_graph(
    scores: Float[Array, 'n_nodes'],
    graph: Graph,
    k: int,
    aug_adj: bool=False
) -> Tuple[Graph, Int[Array, 'c_nodes']]:
    assert len(scores.shape) == 1
    n = scores.shape[0]
    m = graph.edges.shape[0]
    values, idx = jax.lax.top_k(scores, max(2, k))

    new_h = graph.f_node[idx] * values[:, None]

    edge_mask = jnp.arange(graph.edges.shape[0]) < graph.n_edges
    if aug_adj:
        edges = jax.vmap(jnp.where, in_axes=(None, 1, None), out_axes=1)(
            edge_mask, graph.edges, n
        )
        A = jnp.zeros((n+1, n+1), dtype=jnp.float32).at[*edges.T].set(jnp.where(
            edge_mask,
            1,
            0
        ))
        A2 = A @ A > 0
        A2 = A2[jnp.ix_(idx, idx)]
        new_edges = jnp.argwhere(A2, size=k*k, fill_value=0)
        new_g = Graph(
            n_nodes=jnp.array(k),
            n_edges=A2.sum(),
            f_node=new_h,
            f_edge=None,
            f_graph=graph.f_graph,
            edges=new_edges
        )
    else:
        is_top = jnp.zeros(n, dtype=bool).at[idx].set(True)
        keep_edge = is_top[graph.edges[:, 0]] & is_top[graph.edges[:, 1]]
        keep_edge &= edge_mask
        new_m = min(m, k*k)
        edge_idx = jnp.where(keep_edge, size=new_m, fill_value=0)
        new_g = Graph(
            n_nodes=jnp.array(k),
            n_edges=keep_edge.sum(),
            f_node=new_h,
            f_edge=None if graph.f_edge is None else graph.f_edge[edge_idx],
            f_graph=graph.f_graph,
            edges=graph.edges[edge_idx]
        )
    return norm_row(new_g), idx

## Graph has to be undirected
def und_to_dir(g: Graph) -> Graph:
    m = g.edges.shape[0]
    self_edges = g.edges[:, 0] == g.edges[:, 1]
    true_edges = jnp.arange(m) < g.n_edges
    half_edges = jnp.where(true_edges & ~self_edges, size=m, fill_value=0)
    new_edges = jnp.concat([
        g.edges,
        jnp.fliplr(g.edges[half_edges])
    ])
    new_f_edge = None
    if g.f_edge is not None:
        # Breaks norm_row!!!
        new_f_edge = jnp.concat([
            g.f_edge,
            g.f_edge[half_edges]
        ])
    return Graph(
        n_nodes=g.n_nodes,
        n_edges=2*g.n_edges-g.n_nodes,
        f_node=g.f_node,
        f_edge=new_f_edge,
        f_graph=g.f_graph,
        edges=new_edges
    )

def und_get_degrees(g: Graph) -> Float[Array, 'n']:
    return get_degrees(und_to_dir(g))

def und_norm_sym(g: Graph) -> Graph:
    deg = und_get_degrees(g)
    val = g.f_edge
    if val is None:
        val = jnp.ones(g.edges.shape[0], dtype=jnp.float32)
    weight = jax.lax.rsqrt(deg)
    val = weight[g.edges[:, 0]] * val * weight[g.edges[:, 1]]
    return eqx.tree_at(lambda g: g.f_edge, g, val)

class EdgeDrop(eqx.Module):
    dropout: eqx.nn.Dropout
    renorm: str=eqx.field(static=True)

    def __init__(self, dropout: float=0.2, renorm: str='sym'):
        self.dropout = eqx.nn.Dropout(dropout)
        self.renorm = 'sym'

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        inference: bool=False,
    ) -> Graph:
        m = graph.edges.shape[0]

        if inference or self.dropout.p == 0:
            if graph.f_edge is None:
                graph = eqx.tree_at(lambda g: g.f_edge, graph, jnp.ones(m))
            return graph

        true_edges = jnp.arange(m) < graph.n_edges
        self_edges = graph.edges[:, 0] == graph.edges[:, 1]
        key, subkey = jax.random.split(key)
        edge_mask = self.dropout(
            true_edges, key=subkey, inference=inference
        ) > 0
        edge_mask |= self_edges
        edge_idx = jnp.where(edge_mask, size=m, fill_value=0)
        graph = eqx.tree_at(lambda g: g.n_edges, graph, edge_mask.sum())
        graph = eqx.tree_at(lambda g: g.edges, graph, graph.edges[edge_idx])
        if graph.f_edge is not None:
            graph = eqx.tree_at(lambda g: g.f_edge, graph, graph.f_edge)
        match self.renorm:
            case 'sym':
                return und_norm_sym(graph)
            case 'row':
                raise NotImplementedError
            case _:
                raise ValueError(f'Unknown renorm {self.renorm}')
