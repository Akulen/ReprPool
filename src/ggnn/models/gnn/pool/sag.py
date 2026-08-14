import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.gnn.types import ClusteringLayer
from ggnn.models.gnn.clustering import normalize_mask

# ---------------------------------------------------------------------------
# SAGPool  (Self-Attention Graph Pooling)
# ---------------------------------------------------------------------------
# Like TopKPool but the scoring function is a single GNN message-passing
# step: each node aggregates its neighbours' features before projecting to a
# scalar.  This makes the score topology-aware.
# Reference: Lee et al., "Self-Attention Graph Pooling", ICML 2019.
# ---------------------------------------------------------------------------

class SAGPool(ClusteringLayer):
    k: int = eqx.field(static=True)
    proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int = 64,
        dropout: float = 0.0,
        k: int = 8,
    ):
        self.k = k
        self.proj = eqx.nn.Linear(d_model, 'scalar', key=key)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool = False,
    ) -> Tuple[Graph, eqx.nn.State]:
        n = graph.f_node.shape[0]
        m = graph.f_edge.shape[0]
        k = self.k

        key, subkey = jax.random.split(key)
        h = self.dropout(graph.f_node, key=subkey, inference=inference)

        # One GCN-style aggregation step: h_i ← h_i + mean_{j∈N(i)} h_j
        norm_mask = normalize_mask(graph, no_self_edge=True)
        src, dst = graph.edges[:, 0], graph.edges[:, 1]

        agg = jax.ops.segment_sum(
            jnp.where(norm_mask[:, None], h[src], 0.0),
            dst,
            num_segments=n,
        )                                                    # (n, d)
        deg = jax.ops.segment_sum(
            norm_mask.astype(jnp.float32),
            dst,
            num_segments=n,
        )                                                    # (n,)
        h_agg = h + agg / jnp.maximum(deg[:, None], 1.0)   # (n, d)

        # Score using aggregated features; keep -inf only for argsort
        scores_raw = jax.vmap(self.proj)(h_agg)             # (n,)
        scores = jax.nn.tanh(scores_raw)

        real_mask = jnp.arange(n) < graph.n_nodes
        scores_for_sort = jnp.where(real_mask, scores, -jnp.inf)
        topk_idx = jnp.argsort(-scores_for_sort)[:k]        # (k,)

        # Gate original features by score; no -inf in gradients
        gated = jnp.where(real_mask[:, None], graph.f_node * scores[:, None], 0.0)  # (n, d)
        f_node_new = jnp.zeros((k + 1, graph.f_node.shape[-1]))
        f_node_new = f_node_new.at[:k].set(gated[topk_idx])

        # Clamp to actual graph size; zero out slots beyond n_nodes_new
        n_nodes_new = jnp.minimum(graph.n_nodes, k)
        slot_real = jnp.arange(k) < n_nodes_new             # (k,)
        f_node_new = f_node_new.at[:k].set(
            jnp.where(slot_real[:, None], f_node_new[:k], 0.0)
        )

        # Map old node indices to new slots; unmapped and excess → padding (k)
        old_to_new = jnp.full(n, k, dtype=jnp.int32)        # (n,)
        old_to_new = old_to_new.at[topk_idx].set(
            jnp.where(slot_real, jnp.arange(k), k)
        )

        # Re-index edges; drop any edge whose endpoint didn't survive
        src_new = old_to_new[src]                            # (m,)
        dst_new = old_to_new[dst]                            # (m,)

        edge_valid = norm_mask & (src_new < k) & (dst_new < k)
        edges_new = jnp.stack([
            jnp.where(edge_valid, src_new, k),
            jnp.where(edge_valid, dst_new, k),
        ], axis=-1)                                          # (m, 2)
        n_edges_new = jnp.sum(edge_valid)

        return Graph(
            n_nodes=n_nodes_new,
            n_edges=n_edges_new,
            f_node=f_node_new,                              # (k+1, d)
            f_edge=graph.f_edge,
            f_graph=graph.f_graph,
            edges=edges_new,
        ), state
