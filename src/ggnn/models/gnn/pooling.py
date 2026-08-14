import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.gnn.types import PoolLayer
from ggnn.models.gnn.uf import GNCluster

class GlobalAddPool(PoolLayer):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64
    ):
        del key, d_model

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        mask: Optional[Bool[Array, "n"]]=None,
        inference: bool=False
    ) -> Float[Array, "d_model"]:
        # Make sure graph only contains a single graph
        assert len(graph.f_node.shape) == 2

        if mask is None:
            mask = jnp.ones_like(graph.f_node)
        return (graph.f_node * mask[:, None]).sum(axis=0)

class GlobalMeanPool(PoolLayer):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64
    ):
        del key, d_model

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        mask: Optional[Bool[Array, "n"]]=None,
        inference: bool=False
    ) -> Float[Array, "d_model"]:
        # Make sure graph only contains a single graph
        assert len(graph.f_node.shape) == 2

        if mask is None:
            mask = jnp.ones_like(graph.f_node)
        return (graph.f_node * mask[:, None]).sum(axis=0) / mask.sum()

class GlobalReprPool(PoolLayer):
    clustering: GNCluster
    up_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        dropout: float=0.0,
        edge_mask: bool=True,
        # cc_op: str='att'
    ):
        self.clustering = GNCluster(
            key=key,
            d_model=d_model,
            dropout=dropout,
            cluster_activation='tanh',
            edge_mask=edge_mask
        )
        self.up_norm = eqx.nn.LayerNorm(d_model)

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        mask: Optional[Bool[Array, "n"]]=None,
        inference: bool=False
    ) -> Float[Array, "d_model"]:
        max_depth = 5 # 10 # 20
        reduction_factor = 10 # 2

        def fully_clustered(graph):
            edges = graph.edges
            mask = jnp.arange(edges.shape[0]) < graph.n_edges
            # Check if all edges are self-edges (fully compressed)
            return jnp.logical_or(
                ~mask, edges[:, 0] == edges[:, 1]
            ).all()

        def up_step_fn(carry, depth):
            cur_graph, state, key = carry

            def clustering_fn(carry):
                clustered_graph, clustering, _, state = self.clustering(*carry)
                # clustered_graph = eqx.tree_at(
                #     lambda g: g.f_node,
                #     clustered_graph,
                #     jax.vmap(self.up_norm)(clustered_graph.f_node)
                # )
                return clustered_graph, clustering, state

            key, subkey = jax.random.split(key)
            clustered_graph, clustering, state = jax.lax.cond(
                fully_clustered(cur_graph),
                lambda c: (
                    c[2],
                    -jnp.ones(cur_graph.f_node.shape[0], dtype=jnp.int32),
                    c[1]
                ),
                clustering_fn,
                (
                    subkey, state, cur_graph, inference,
                    (1 / reduction_factor) ** depth - 1
                )
            )

            return (
                (clustered_graph, state, key),
                (clustered_graph, clustering)
            )

        (_, _, key), (clustered_graphs, clusterings) = jax.lax.scan(
            up_step_fn,
            (graph, None, key),
            jnp.arange(max_depth)
        )
        graphs = jax.tree.map(
            lambda x, xs: jnp.concatenate([x[None], xs], axis=0),
            graph,
            clustered_graphs
        )

        top_depth = jnp.argmin(jnp.concatenate([
            clusterings[:,0] > -1,
            jnp.array([False])
        ]))

        x = graphs[top_depth].f_node
        mask = jnp.arange(x.shape[0]) < graphs[top_depth].n_nodes
        assert mask is not None
        return (x * mask[:, None]).sum(axis=0)
