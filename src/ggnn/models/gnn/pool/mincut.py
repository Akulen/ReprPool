import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.gnn.types import ClusteringLayer
from ggnn.models.gnn.clustering import normalize_mask

# ---------------------------------------------------------------------------
# MinCutPool
# ---------------------------------------------------------------------------
# Soft node clustering via a learned assignment matrix S ∈ R^{n×k}.
# The mincut loss encourages the clusters to respect graph cuts; the
# orthogonality loss prevents degenerate (all-same) assignments.
# Both losses should be added to your training objective.
#
# Because S is dense (n×k) this is O(n·k) memory — fine for moderate n.
# Reference: Bianchi et al., "Spectral Clustering with Graph Neural
# Networks for Graph Pooling", ICML 2020.
# ---------------------------------------------------------------------------

class MinCutPool(ClusteringLayer):
    k: int = eqx.field(static=True)
    mlp_1: eqx.nn.Linear
    mlp_2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int = 64,
        dropout: float = 0.0,
        k: int = 8,
    ):
        self.k = k
        k1, k2 = jax.random.split(key)
        self.mlp_1 = eqx.nn.Linear(d_model, d_model, key=k1)
        self.mlp_2 = eqx.nn.Linear(d_model, k, key=k2)
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

        # Assignment matrix S: (n, k)  — softmax over clusters per node
        s_logits = jax.vmap(self.mlp_2)(jax.nn.selu(jax.vmap(self.mlp_1)(h)))
        S = jax.nn.softmax(s_logits, axis=-1)               # (n, k)

        # Mask padding nodes: zero their assignment rows so they don't
        # contribute to any cluster
        real_mask = (jnp.arange(n) < graph.n_nodes)[:, None]
        S = jnp.where(real_mask, S, 0.0)

        # Build dense adjacency (only real, non-self edges)
        norm_mask = normalize_mask(graph, no_self_edge=True)
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        A = jnp.zeros((n, n)).at[src, dst].set(
            jnp.where(norm_mask, 1.0, 0.0)
        )

        # Pooled features:  X_new = S^T X   shape (k, d)
        f_node_pool = S.T @ graph.f_node                     # (k, d)

        # Pooled adjacency: A_new = S^T A S  shape (k, k), zero diagonal
        A_new = S.T @ A @ S
        A_new = jnp.fill_diagonal(A_new, 0.0, inplace=False)

        # ---- MinCut loss (detached from graph output, add to your loss) ----
        # L_mc = -Tr(S^T A S) / Tr(S^T D S)
        # D_diag = A.sum(axis=-1)                              # (n,)
        # num = jnp.trace(S.T @ A @ S)
        # denom = jnp.trace(S.T @ (D_diag[:, None] * S))
        # loss_mincut = -num / (denom + 1e-8)

        # # Orthogonality loss: || S^T S / ||S^T S||_F  - I/k ||_F
        # StS = S.T @ S
        # StS_norm = StS / (jnp.linalg.norm(StS) + 1e-8)
        # loss_ortho = jnp.linalg.norm(StS_norm - jnp.eye(k) / k)

        # Store losses in f_graph so the caller can retrieve them.
        # Convention: f_graph[-2] = loss_mincut, f_graph[-1] = loss_ortho
        # (adjust to however you pass auxiliary losses in your codebase)
        # f_graph_new = graph.f_graph.at[-2].set(loss_mincut).at[-1].set(loss_ortho)

        # Convert A_new to edge-list format (reuse existing edge slots)
        # We keep at most m edges; take nonzero entries of A_new
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
            f_node_pool,
            jnp.zeros((1, graph.f_node.shape[-1]))
        ], axis=0)                                           # (k+1, d)

        return Graph(
            n_nodes=jnp.array(k),          # all k clusters are real
            n_edges=n_edges_new,
            f_node=f_node_new,
            f_edge=jnp.zeros((k * k, graph.f_edge.shape[-1])), #graph.f_edge,
            f_graph=graph.f_graph, #f_graph_new,
            edges=edges_new,
        ), state

