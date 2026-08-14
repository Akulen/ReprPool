import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.gnn.types import ClusteringLayer
from ggnn.models.gnn.uf import count_clusters, UFCluster, GNCluster

def normalize_mask(
    graph: Graph,
    no_self_edge: bool=True
) -> Bool[Array, 'm']:
    m = graph.f_edge.shape[0]
    mask = jnp.arange(m) < graph.n_edges
    if no_self_edge:
        mask &= graph.edges[:, 0] != graph.edges[:, 1]
    _, unique_idx = jnp.unique(graph.edges, size=m, axis=0, return_index=True)
    mask &= jnp.full(m, False).at[unique_idx].set(True)
    return mask

class ClusterPooling(ClusteringLayer):
    lin: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        dropout: float=0.0
    ):
        self.lin = eqx.nn.Linear(
            in_features=2*d_model,
            out_features='scalar',
            key=key,
        )
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        # We assume that each non self-edge is present twice, once in each
        # orientation 
        e = jnp.concatenate([
            graph.f_node[graph.edges[:, 0]],
            graph.f_node[graph.edges[:, 1]]
        ], axis=-1)
        e = jax.vmap(self.lin)(e)
        key, subkey = jax.random.split(key)
        e = self.dropout(e, key=subkey, inference=inference)
        e = jax.nn.tanh(e)

        return self.merge_edges(e, graph), state

    def connected_components(
        self,
        graph: Graph,
        edge_mask: Bool[Array, 'n_edge']
    ) -> Tuple[Int[Array, 'n'], Int[Array, 'n']]:
        n = graph.f_node.shape[0]
        labels = jnp.arange(n)

        edge_u = jnp.where(
            edge_mask,
            graph.edges[:, 0],
            n-1
        )
        edge_v = jnp.where(
            edge_mask,
            graph.edges[:, 1],
            n-1
        )
        edge_u, edge_v = (
            jnp.concatenate([edge_u, edge_v]),
            jnp.concatenate([edge_v, edge_u])
        )
        # Add Self Edges
        edge_u = jnp.concatenate([
            jnp.arange(n-1),
            edge_u
        ])
        edge_v = jnp.concatenate([
            jnp.arange(n-1),
            edge_v
        ])

        def flood(
            carry: Tuple[Int[Array, 'm'], Int[Array, 'm']]
        ) -> Tuple[Int[Array, 'm'], Int[Array, 'm']]:
            _, lbl = carry
            new_lbl = jax.ops.segment_min(
                lbl[edge_v],
                edge_u,
                n
            )
            return lbl, new_lbl
        _, labels = jax.lax.while_loop(
            lambda c: (c[0] != c[1]).any(),
            flood,
            (jnp.zeros_like(labels), labels)
        )
        labels = jnp.where(jnp.arange(n) < graph.n_nodes, labels, n-1)
        n_clusters = count_clusters(labels)

        _, lbl_cluster = jnp.unique(
            labels, return_inverse=True, size=n, fill_value=n-1
        )
        return n_clusters, lbl_cluster

    def merge_edges(
        self,
        e: Float[Array, 'n_edge'],
        graph: Graph
    ) -> Graph:
        assert(len(graph.n_nodes.shape) == 0)
        n = graph.f_node.shape[0]
        m = graph.f_edge.shape[0]
        mask_treshold = e > 0
        mask_norm = normalize_mask(graph, no_self_edge=True)
        mask = mask_treshold & mask_norm

        n_clusters, cluster_index = self.connected_components(graph, mask)
        # C = jax.nn.one_hot(cluster_index, n)
        # Z = jnp.zeros((n, n), dtype=jnp.float32)
        # A = Z.at[graph.edges[:, 0], graph.edges[:, 1]].set(jnp.where(
        #     mask_norm,
        #     1,
        #     0
        # ))
        # S = Z.at[graph.edges[:, 0], graph.edges[:, 1]].set(jnp.where(
        #     mask_norm,
        #     e,
#     0
        # ))

        # A_contract = Z.at[graph.edges[:, 0], graph.edges[:, 1]].set(jnp.where(
        #     mask,
        #     1,
        #     0
        # ))
        # nodes_single = (A_contract.sum(axis=-1) + A_contract.sum(axis=-2)) == 0
        # S = S.at[jnp.diag_indices(n)].set(
        #     jnp.where(nodes_single, 1, S.at[jnp.diag_indices(n)].get())
        # )

        # X_new = (S @ C).T @ graph.f_node

        # Paper says mask, code says mask_norm
        clean_e = jnp.where(mask_norm, e, 0) 
        node_message = jax.ops.segment_sum(
            graph.f_node[graph.edges[:, 0]] * clean_e[:, None],
            graph.edges[:, 1],
            num_segments=n
        )
        cluster_sizes = jax.ops.segment_sum(
            (jnp.ones(n, dtype=jnp.int32) < graph.n_nodes).astype(jnp.int32),
            cluster_index,
            num_segments=n
        )
        node_messages = jnp.where(
            (cluster_sizes[cluster_index] == 1)[:, None],
            graph.f_node + node_message,
            node_message
        )
        f_cluster = jax.ops.segment_sum(
            node_messages,
            cluster_index,
            num_segments=n
        )
        # A_new = jnp.fill_diagonal(C.T @ A @ C, 0, inplace=False)
        # edge_index_new = jnp.argwhere(A_new, size=m, fill_value=n-1)

        return Graph(
            n_nodes=n_clusters,
            n_edges=graph.n_edges,
            f_node=f_cluster,
            f_edge=graph.f_edge,
            f_graph=graph.f_graph,
            edges=cluster_index[graph.edges]
        )

class UFPooling(ClusteringLayer):
    pool: UFCluster

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        dropout: float=0.0,
        cluster_activation: str='sigmoid-off'
    ):
        self.pool = UFCluster(
            key, d_model, dropout=dropout,
            cluster_activation=cluster_activation
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        graph, repr_cluster, scores, state =  self.pool(
            key, state, graph, inference
        )
        return graph, state

class GNPooling(ClusteringLayer):
    pool: GNCluster

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        dropout: float=0.0,
        cluster_activation: str='sigmoid-off',
        edge_mask: bool=True
    ):
        self.pool = GNCluster(
            key, d_model, dropout=dropout,
            cluster_activation=cluster_activation, edge_mask=edge_mask
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        graph, repr_cluster, node_score, state =  self.pool(
            key, state, graph, inference
        )
        return graph, state
