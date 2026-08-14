import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.gnn.types import ClusteringLayer
from ggnn.models.gnn.clustering import normalize_mask

# ---------------------------------------------------------------------------
# DiffPool
# ---------------------------------------------------------------------------
# Two parallel GNN branches: one produces embeddings Z, the other produces
# the soft assignment matrix S.  The coarsened graph is then:
#   X_new = S^T Z,   A_new = S^T A S
# Two auxiliary losses (link prediction + entropy) should be added to your
# training objective — they are stored in f_graph[-2] and f_graph[-1].
# Reference: Ying et al., "Hierarchical Graph Representation Learning with
# Differentiable Pooling", NeurIPS 2018.
# ---------------------------------------------------------------------------

class DiffPool(ClusteringLayer):
    k: int = eqx.field(static=True)
    # Embedding GNN (one message-passing step + linear)
    embed_msg: eqx.nn.Linear
    embed_upd: eqx.nn.Linear
    # Assignment GNN (same structure, outputs k logits)
    assign_msg: eqx.nn.Linear
    assign_upd: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int = 64,
        dropout: float = 0.0,
        k: int = 8,
    ):
        self.k = k
        k1, k2, k3, k4 = jax.random.split(key, 4)
        # Each GNN: message linear maps concatenated [h_i, h_j] → d_model,
        # update linear maps aggregated message → d_model (embed) or k (assign)
        self.embed_msg = eqx.nn.Linear(2 * d_model, d_model, key=k1)
        self.embed_upd = eqx.nn.Linear(2 * d_model, d_model, key=k2)
        self.assign_msg = eqx.nn.Linear(2 * d_model, d_model, key=k3)
        self.assign_upd = eqx.nn.Linear(2 * d_model, k,       key=k4)
        self.dropout = eqx.nn.Dropout(dropout)

    def _gnn_step(
        self,
        h: Float[Array, 'n d'],
        edges: Int[Array, 'm 2'],
        norm_mask: Bool[Array, 'm'],
        msg_lin: eqx.nn.Linear,
        upd_lin: eqx.nn.Linear,
        out_dim: int,
    ) -> Float[Array, 'n out_dim']:
        """One message-passing step: aggregate then update."""
        n = h.shape[0]
        src, dst = edges[:, 0], edges[:, 1]

        # Messages: concat sender and receiver features
        msgs = jnp.concatenate([h[src], h[dst]], axis=-1)   # (m, 2d)
        msgs = jax.vmap(msg_lin)(msgs)                       # (m, d)
        msgs = jax.nn.relu(msgs)
        msgs = jnp.where(norm_mask[:, None], msgs, 0.0)

        agg = jax.ops.segment_sum(msgs, dst, num_segments=n) # (n, d)

        # Update: concat node feature with aggregated message
        upd_in = jnp.concatenate([h, agg], axis=-1)         # (n, 2d)
        out = jax.vmap(upd_lin)(upd_in)                      # (n, out_dim)
        return jax.nn.relu(out)

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

        norm_mask = normalize_mask(graph, no_self_edge=True)
        src, dst = graph.edges[:, 0], graph.edges[:, 1]

        # Embedding branch → Z  (n, d)
        Z = self._gnn_step(
            h, graph.edges, norm_mask,
            self.embed_msg, self.embed_upd, h.shape[-1],
        )

        # Assignment branch → S  (n, k)
        S_logits = self._gnn_step(
            h, graph.edges, norm_mask,
            self.assign_msg, self.assign_upd, k,
        )
        S = jax.nn.softmax(S_logits, axis=-1)

        # Zero out padding nodes
        real_mask = (jnp.arange(n) < graph.n_nodes)[:, None]
        S = jnp.where(real_mask, S, 0.0)

        # Dense adjacency
        A = jnp.zeros((n, n)).at[src, dst].set(
            jnp.where(norm_mask, 1.0, 0.0)
        )

        # Coarsened graph
        f_node_new = S.T @ Z                                 # (k, d)
        A_new = S.T @ A @ S
        A_new = jnp.fill_diagonal(A_new, 0.0, inplace=False)

        # ---- Auxiliary losses ----
        # Link prediction loss: || A - S S^T ||_F
        # loss_lp = jnp.linalg.norm(A - S @ S.T) / (n * n)

        # Entropy loss: encourages crisp (low-entropy) assignments
        # eps = 1e-8
        # entropy = -jnp.sum(S * jnp.log(S + eps), axis=-1)   # (n,)
        # real_nodes = (jnp.arange(n) < graph.n_nodes).astype(jnp.float32)
        # loss_ent = jnp.sum(entropy * real_nodes) / (graph.n_nodes + eps)

        # f_graph_new = graph.f_graph.at[-2].set(loss_lp).at[-1].set(loss_ent)

        # Convert A_new → edge list (same as MinCutPool)
        rows, cols = jnp.meshgrid(jnp.arange(k), jnp.arange(k), indexing='ij')
        flat_src = rows.ravel()                                      # (k*k,)
        flat_dst = cols.ravel()                                      # (k*k,)
        flat_w   = A_new.ravel()                                     # (k*k,)

        # Real edges are those with positive weight and not self-loops
        edge_valid = (flat_w > 0.0) & (flat_src != flat_dst)
        n_edges_new = jnp.sum(edge_valid)

        pad_node = k
        edges_new = jnp.stack([
            jnp.where(edge_valid, flat_src, pad_node),
            jnp.where(edge_valid, flat_dst, pad_node),
        ], axis=-1)                                                  # (k*k, 2)

        f_node_new = jnp.concatenate([
            S.T @ Z,
            jnp.zeros((1, Z.shape[-1]))
        ], axis=0)

        return Graph(
            n_nodes=jnp.array(k),
            n_edges=n_edges_new,
            f_node=f_node_new,
            f_edge=jnp.zeros((k * k, graph.f_edge.shape[-1])), #graph.f_edge,
            f_graph=graph.f_graph, #f_graph_new,
            edges=edges_new,
        ), state

