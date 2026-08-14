import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List, Tuple, Optional
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.utils import GlorotLinear, get_activation
from ggnn.models.graph_utils import norm_row, und_to_dir, EdgeDrop, top_k_graph
from ggnn.models.gnn.types import ClusteringLayer, GNNLayer, PoolLayer
from ggnn.models.gnn.gcn import GCNLayer
from ggnn.models.gnn.mlp import MLP

class gPool(eqx.Module):
    k: int=eqx.field(static=True)
    proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int,
        dropout: float=0.92,
        k: int=-1,
        aug_adj: bool=False
    ):
        assert k > 0
        self.k = k
        self.proj = eqx.nn.Linear(d_model, 'scalar', key=key)
        # self.proj = GlorotLinear(d_model, 'scalar', key=key)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, Int[Array, 'c_nodes'], eqx.nn.State]:
        key, subkey = jax.random.split(key)
        Z = self.dropout(graph.f_node, key=subkey, inference=inference)
        graph = eqx.tree_at(lambda g: g.f_node, graph, Z)
        weights = jax.vmap(self.proj)(Z)
        scores = jax.nn.sigmoid(weights)

        return *top_k_graph(scores, graph, self.k), state

class gUnpool(eqx.Module):
    def __init__(self):
        pass

    def __call__(
        self,
        graph: Graph,
        clustered_graph: Graph,
        idx: Int[Array, 'c_nodes']
    ) -> Graph:
        h = jnp.zeros_like(graph.f_node)
        h = h.at[idx].set(clustered_graph.f_node)
        return eqx.tree_at(lambda g: g.f_node, graph, h)

class UNetGCNLayer(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_out: int,
    ):
        key, subkey = jax.random.split(key)
        self.linear = eqx.nn.Linear(
            in_features=d_in,
            out_features=d_out,
            key=subkey,
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False,
    ) -> Tuple[Graph, eqx.nn.State]:
        x = jax.vmap(self.linear)(graph.f_node)
        messages = x[graph.edges[:, 0]]
        assert graph.f_edge is not None
        messages *= graph.f_edge[:, None]
        y = jax.ops.segment_sum(
            messages,
            graph.edges[:, 1],
            num_segments=x.shape[0]
        )
        return eqx.tree_at(lambda g: g.f_node, graph, y), state

# GCNLayer = GUNetGCNLayer

class GraphUNet(eqx.Module):
    bottom_gcn: UNetGCNLayer
    down_gcns: List[UNetGCNLayer]
    up_gcns: List[UNetGCNLayer]
    pools: List[gPool]
    unpools: List[gUnpool]
    activation: Callable
    dropout: eqx.nn.Dropout
    
    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int,
        ks: List[int]=[2000, 1000, 500, 200],
        dropout: float=0.92,
        activation: str='elu',
        aug_adj: bool=False
    ):
        key_down, key_up, key_bot = jax.random.split(key, 3)
        self.bottom_gcn = UNetGCNLayer(key_bot, d_model, d_model)
        key_gcn, key_pool = jax.random.split(key_down, 2)
        keys = jax.random.split(key_gcn, len(ks))
        self.down_gcns = [
            UNetGCNLayer(subkey, d_model, d_model)
            for subkey in keys
        ]
        keys = jax.random.split(key_up, len(ks))
        self.up_gcns = [
            UNetGCNLayer(subkey, d_model, d_model)
            for subkey in keys
        ]
        keys = jax.random.split(key_pool, len(ks))
        self.pools = [
            gPool(subkey, d_model, dropout=dropout, k=k, aug_adj=aug_adj)
            for k, subkey in zip(ks, keys)
        ]
        self.unpools = [gUnpool() for _ in ks]

        self.activation = get_activation(activation)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool = False
    ) -> Tuple[List[Float[Array, 'n d']], eqx.nn.State]:
        graphs = []
        indices_list = []
        hs = []
        h0 = graph.f_node
        
        # Encoder
        for gcn, pool in zip(self.down_gcns, self.pools):
            key, subkey = jax.random.split(key)
            graph = eqx.tree_at(
                lambda g: g.f_node,
                graph,
                self.dropout(graph.f_node, key=subkey, inference=inference)
            )
            key, subkey = jax.random.split(key)
            graph, state = gcn(subkey, state, graph, inference)
            graph = eqx.tree_at(
                lambda g: g.f_node, graph, self.activation(graph.f_node)
            )

            graphs.append(graph)

            key, subkey = jax.random.split(key)
            graph, idx, state = pool(subkey, state, graph, inference)
            indices_list.append(idx)
            
        # Bottom
        key, subkey = jax.random.split(key)
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            self.dropout(graph.f_node, key=subkey, inference=inference)
        )
        key, subkey = jax.random.split(key)
        graph, state = self.bottom_gcn(subkey, state, graph, inference)
        graph = eqx.tree_at(
            lambda g: g.f_node, graph, self.activation(graph.f_node)
        )
        
        # Decoder
        I = range(len(self.up_gcns)-1, -1, -1)
        for i, unpool, gcn in zip(I, self.unpools, self.up_gcns):
            graph = unpool(graphs[i], graph, indices_list[i])

            key, subkey = jax.random.split(key)
            graph = eqx.tree_at(
                lambda g: g.f_node,
                graph,
                self.dropout(graph.f_node, key=subkey, inference=inference)
            )
            key, subkey = jax.random.split(key)
            graph, state = gcn(subkey, state, graph, inference)
            graph = eqx.tree_at(
                lambda g: g.f_node, graph, self.activation(graph.f_node)
            )
            
            h = graph.f_node + graphs[i].f_node
            graph = eqx.tree_at(lambda g: g.f_node, graph, h)
            hs.append(h)
            
        hs.append(graph.f_node + h0)
        
        return hs, state

class GraphUNetNodeClassifier(eqx.Module):
    pre_gcn: UNetGCNLayer
    unet: GraphUNet
    post_gcn: UNetGCNLayer
    activation: Callable
    edge_drop: EdgeDrop
    dropout: eqx.nn.Dropout
    improved: bool=eqx.field(static=False)
    
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int,
        
        d_model: int=48, # Maybe? Unspecified in paper
        dropout_feat: float=0.92,
        dropout_adj: float=0.2, # Unused in GUNet paper???
        ks: List[int]=[2000, 1000, 500, 200],
        activation: str='elu',
        improved: bool=False,
        aug_adj: bool=False
    ):
        key_pre, key_unet, key_post = jax.random.split(key, 3)
        self.pre_gcn = UNetGCNLayer(key_pre, d_in, d_model)
        self.unet = GraphUNet(
            key_unet, d_model, ks, dropout=dropout_feat, activation=activation,
            aug_adj=aug_adj
        )
        self.post_gcn = UNetGCNLayer(key_post, d_model, n_classes)
        
        self.activation = get_activation(activation)
        self.edge_drop = EdgeDrop(dropout_adj)
        self.dropout = eqx.nn.Dropout(dropout_feat)
        self.improved = improved
        
    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, 'n n_class'], eqx.nn.State]:
        if self.improved:
            graph = eqx.tree_at(
                lambda g: g.n_edges, graph, graph.n_edges + graph.n_nodes
            )
            graph = eqx.tree_at(
                lambda g: g.edges, graph, jnp.concat([
                    jnp.tile(jnp.arange(graph.f_node.shape[0]), (2, 1)).T,
                    graph.edges
                ])
            )
            if graph.f_edge is not None:
                raise ValueError('f_edge must not be defined when improving the graph')

        key, subkey = jax.random.split(key)
        graph = self.edge_drop(subkey, graph, inference=inference)

        # We make the graph directed
        graph = und_to_dir(graph)
        graph = norm_row(graph)

        key, subkey = jax.random.split(key)
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            self.dropout(graph.f_node, key=subkey, inference=inference)
        )

        key, subkey = jax.random.split(key)
        graph, state = self.pre_gcn(subkey, state, graph, inference)
        graph = eqx.tree_at(
            lambda g: g.f_node, graph, self.activation(graph.f_node)
        )

        key, subkey = jax.random.split(key)
        hs, state = self.unet(subkey, graph, state, inference)

        graph = eqx.tree_at(
            lambda g: g.f_node, graph, hs[-1]
        )

        key, subkey = jax.random.split(key)
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            self.dropout(graph.f_node, key=subkey, inference=inference)
        )

        key, subkey = jax.random.split(key)
        graph, state = self.post_gcn(subkey, state, graph, inference)

        return graph.f_node, state
