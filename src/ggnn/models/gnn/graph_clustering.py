import jax
import equinox as eqx
from typing import List, Tuple
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.gnn.types import GNNLayer
from ggnn.models.gnn.uf import GNCluster
from ggnn.models.gnn.gin import GINLayer

class GNClustering(eqx.Module):
    linear: eqx.nn.Linear
    gnn_layers: List[GNNLayer]
    pooling: GNCluster
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int,

        d_model: int=128,
        n_gnn_layers: int=0,
        dropout: float=0.0,
        dropout_pool: float=0.0,

        cluster_activation: str='tanh',
        edge_mask: bool=True
    ):
        del d_in_edge, n_classes
        key_lin, key_gnn, key_pool = jax.random.split(key, 3)
        self.linear = eqx.nn.Linear(
            in_features=d_in,
            out_features=d_model,
            key=key_lin
        )
        keys = jax.random.split(key_gnn, n_gnn_layers)
        self.gnn_layers = [
            GINLayer(
                key=subkey,
                d_model=d_model,
                dropout=dropout,
                batch_norm=True,
                residual=True,
                learn_eps=True
            )
            for subkey in keys
        ]
        self.pooling = GNCluster(
            key_pool,
            d_model=d_model,
            dropout=dropout_pool,
            cluster_activation=cluster_activation,
            edge_mask=edge_mask
        )
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Int[Array, 'n'], Float[Array, 'n'], eqx.nn.State]:
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            jax.vmap(self.linear)(graph.f_node)
        )
        for layer in self.gnn_layers:
            key, subkey = jax.random.split(key)
            graph, state = layer(subkey, state, graph, inference)
        graph, cluster, score, state = self.pooling(
            key, state, graph, inference
        )
        return cluster, score, state
