import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type
from jaxtyping import Array, Float, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.utils import AtomEncoder, BondEncoder, BatchNorm
from ggnn.models.gnn.types import GNNLayer, PoolLayer, ClusteringLayer
from ggnn.models.gnn.pooling import GlobalAddPool, GlobalMeanPool
from ggnn.models.gnn.gcn import GCNLayer
from ggnn.models.gnn.clustering import ClusterPooling, UFPooling, GNPooling
from ggnn.models.gnn.pool import TopKPool, SAGPool, MinCutPool, DiffPool

class GNNClassifier(eqx.Module):
    atom_encoder: AtomEncoder
    # bond_encoder: BondEncoder
    gnn_layers: List[GNNLayer]
    bns_node: List[BatchNorm | eqx.nn.LayerNorm]
    # bns_edge: List[BatchNorm]
    dropout: eqx.nn.Dropout
    pool: PoolLayer
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

        # key: PRNGKeyArray,
        # d_model: int=64,
        # mlp_depth: int=2,
        # self_edge: bool=True
    def __init__(
        self,
        key: PRNGKeyArray,
        gnn_layer_gen: Callable[[PRNGKeyArray], GNNLayer], #Literal['GIN', 'UF']='GIN',
        # n_class: int=2,
        d_model: int=64,
        n_gnn_layers: int=2,
        pooling: Literal['add']='add',
        dropout: float=0.5,
        use_ln: bool=False
    ):
        key_enc_node, key_gnn, key_pool, key_lin1, key_lin2 = (
            jax.random.split(key, 5)
        )
        self.atom_encoder = AtomEncoder(key_enc_node, d_model)
        # TODO
        # self.bond_encoder = BondEncoder(edge_key, d_model)
        keys = jax.random.split(key_gnn, n_gnn_layers)
        self.gnn_layers = [
            gnn_layer_gen(key)
            for key in keys
        ]
        if use_ln:
            self.bns_node = [
                eqx.nn.LayerNorm(d_model)
                for _ in range(n_gnn_layers-1)
            ]
        else:
            self.bns_node = [
                BatchNorm(
                    input_size=d_model,
                    axis_name='batch',
                    momentum=0.9,
                    mode='batch'
                )
                for _ in range(n_gnn_layers-1)
            ]
        # TODO
        # self.bns_edge = [
        #     BatchNorm(
        #         input_size=d_model,
        #         axis_name=('devices', 'batch'),
        #         momentum=0.9,
        #         mode='batch'
        #     )
        #     for _ in range(n_gnn_layers-1)
        # ]
        self.dropout = eqx.nn.Dropout(dropout)
        if pooling == 'add':
            poolingLayer = GlobalAddPool
        # elif pooling == 'attention':
        #     poolingLayer = AttentionPooling
        else:
            raise NotImplementedError
        self.pool = poolingLayer(key_pool, d_model)
        self.linear1 = eqx.nn.Linear(d_model, d_model, key=key_lin1)
        self.linear2 = eqx.nn.Linear(d_model, 1, key=key_lin2)

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool,
    ) -> Tuple[
        Float[Array, 'n_class'],
        eqx.nn.State
    ]:
        # Make sure graph only contains a single graph
        assert len(graph.f_node.shape) == 2

        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            jax.vmap(self.atom_encoder)(graph.f_node)
        )
        mask_node = jnp.arange(graph.f_node.shape[0]) < graph.n_nodes
        # mask_edge = jnp.repeat(
        #     jnp.array([1, 0]),
        #     graph.n_edge,
        #     axis=0,
        #     total_repeat_length=cast(Array, graph.edges).shape[0]
        # )
        for i, layer in enumerate(self.gnn_layers):
            key, subkey = jax.random.split(key, 2)
            graph, state = layer(subkey, state, graph, inference)
            if i < len(self.gnn_layers) - 1:
                x = graph.f_node
                bn = self.bns_node[i]
                if isinstance(bn, eqx.nn.LayerNorm):
                    x = jax.vmap(bn)(x)
                else:
                    x, state = bn(
                        x.T,
                        state,
                        mask=mask_node,
                        inference=inference
                    )
                    x = x.T
                x = jax.nn.relu(x)
                key, subkey = jax.random.split(key, 2)
                x = self.dropout(x, inference=inference, key=subkey)
        #         g, state = self.bns_edge[i](
        #             g.T,
        #             state,
        #             mask=mask_edge,
        #             inference=not training
        #         )
        #         g = jax.nn.relu(g.T)
                graph = eqx.tree_at(lambda g: g.f_node, graph, x)

        key, subkey = jax.random.split(key)
        graph_features = self.pool(
            subkey,
            graph,
            mask=mask_node,
            inference=inference
        )
        key, subkey = jax.random.split(key)
        graph_features = self.dropout(
            graph_features,
            inference=inference,
            key=subkey
        )
        graph_features = jax.nn.relu(
            self.linear1(graph_features)
        )
        graph_features = self.dropout(
            graph_features,
            inference=inference,
            key=key
        )
        return self.linear2(graph_features).squeeze(), state

class PoolingClassifier(eqx.Module):
    gnn_layers: List[GNNLayer]
    pool_freq: int=eqx.field(static=True)
    pool_layers: List[ClusteringLayer]
    pool: PoolLayer
    head_layers: List[eqx.nn.Linear]
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,

        clustering: Type[ClusteringLayer]=ClusterPooling,
        clustering_params: Optional[List[Dict[str, Any]]]=None
    ):
        del d_in_edge
        if n_classes == 2:
            n_classes = 'scalar'
        key_gnn, key_pool, key_gpool, key_head = jax.random.split(key, 4)
        keys = jax.random.split(key_gnn, n_gnn_layers)
        self.gnn_layers = [
            GCNLayer(
                key=subkey,
                d_in=d_model if i > 0 else d_in,
                d_out=d_model,
            )
            for i, subkey in enumerate(keys)
        ]
        self.pool_freq = pooling_freq
        n_pool_layers = (n_gnn_layers - 1) // pooling_freq
        keys = jax.random.split(key_pool, n_pool_layers)
        if clustering_params is None:
            clustering_params = [{} for _ in keys]
        assert(len(clustering_params) == n_pool_layers)
        self.pool_layers = [
            clustering(
                key=subkey,
                d_model=d_model,
                dropout=dropout_pool,
                **params
            )
            for subkey, params in zip(keys, clustering_params)
        ]
        self.pool = GlobalMeanPool(key_gpool)
        keys = jax.random.split(key_head, head_n_linear)
        self.head_layers = [
            eqx.nn.Linear(
                in_features=d_model,
                out_features=d_model if i < head_n_linear - 1 else n_classes,
                key=subkey
            )
            for i, subkey in enumerate(keys)
        ]
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool=False,
        return_cluster_sizes: bool=False
    ) -> Tuple[
        Float[Array, 'n_class'],
        eqx.nn.State
    ]:
        sizes = [graph.n_nodes]
        for i, layer in enumerate(self.gnn_layers):
            if i > 0 and i % self.pool_freq == 0:
                key, subkey = jax.random.split(key)
                graph, state = self.pool_layers[i//self.pool_freq-1](
                    subkey, state, graph, inference
                )
                if return_cluster_sizes:
                    sizes.append(graph.n_nodes)
            key, subkey = jax.random.split(key)
            graph, state = layer(subkey, state, graph, inference)
            key, subkey = jax.random.split(key)
            graph = eqx.tree_at(
                lambda g: g.f_node,
                graph,
                self.dropout(
                    jax.nn.relu(graph.f_node),
                    inference=inference,
                    key=subkey
                )
            )
        if return_cluster_sizes:
            return sizes # type: ignore

        mask = jnp.arange(graph.f_node.shape[0]) < graph.n_nodes
        key, subkey = jax.random.split(key)
        x = self.pool(subkey, graph, mask, inference)

        for i, layer in enumerate(self.head_layers):
            x = layer(x)
            if i < len(self.head_layers) - 1:
                x = jax.nn.relu(x)
                key, subkey = jax.random.split(key, 2)
                x = self.dropout(x, inference=inference, key=subkey)

        # x = jax.nn.sigmoid(x)
        return x, state

class ClusterPoolingClassifier(PoolingClassifier):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,
    ):
        super().__init__(
            key,
            d_in,
            d_in_edge,
            n_classes,

            d_model,
            n_gnn_layers,
            pooling_freq,
            dropout,
            dropout_pool,
            head_n_linear,

            ClusterPooling
        )

class UFPoolingClassifier(PoolingClassifier):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,

        cluster_activation: str='sigmoid-off'
    ):
        super().__init__(
            key,
            d_in,
            d_in_edge,
            n_classes,

            d_model,
            n_gnn_layers,
            pooling_freq,
            dropout,
            dropout_pool,
            head_n_linear,

            eqx.Partial(UFPooling, cluster_activation=cluster_activation)
        )

class GNPoolingClassifier(PoolingClassifier):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,

        cluster_activation: str='sigmoid-off',
        edge_mask: bool=True
    ):
        super().__init__(
            key,
            d_in,
            d_in_edge,
            n_classes,

            d_model,
            n_gnn_layers,
            pooling_freq,
            dropout,
            dropout_pool,
            head_n_linear,

            eqx.Partial(
                GNPooling, cluster_activation=cluster_activation,
                edge_mask=edge_mask
            )
        )

class TopKPoolClassifier(PoolingClassifier):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,

        ks: List[int]=[],

        cluster_activation: str='sigmoid-off',
        edge_mask: bool=True
    ):
        assert(len(ks) == (n_gnn_layers - 1) // pooling_freq)
        super().__init__(
            key,
            d_in,
            d_in_edge,
            n_classes,

            d_model,
            n_gnn_layers,
            pooling_freq,
            dropout,
            dropout_pool,
            head_n_linear,

            TopKPool,
            clustering_params=[{'k': k} for k in ks]
        )

class SAGPoolClassifier(PoolingClassifier):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,

        ks: List[int]=[],

        cluster_activation: str='sigmoid-off',
        edge_mask: bool=True
    ):
        assert(len(ks) == (n_gnn_layers - 1) // pooling_freq)
        super().__init__(
            key,
            d_in,
            d_in_edge,
            n_classes,

            d_model,
            n_gnn_layers,
            pooling_freq,
            dropout,
            dropout_pool,
            head_n_linear,

            SAGPool,
            clustering_params=[{'k': k} for k in ks]
        )

class MinCutPoolClassifier(PoolingClassifier):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,

        ks: List[int]=[],

        cluster_activation: str='sigmoid-off',
        edge_mask: bool=True
    ):
        assert(len(ks) == (n_gnn_layers - 1) // pooling_freq)
        super().__init__(
            key,
            d_in,
            d_in_edge,
            n_classes,

            d_model,
            n_gnn_layers,
            pooling_freq,
            dropout,
            dropout_pool,
            head_n_linear,

            MinCutPool,
            clustering_params=[{'k': k} for k in ks]
        )

class DiffPoolClassifier(PoolingClassifier):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int | Literal['scalar'],

        d_model: int=128,
        n_gnn_layers: int=5,
        pooling_freq: int=2,
        dropout: float=0.0,
        dropout_pool: float=0.0,
        head_n_linear: int=2,

        ks: List[int]=[],

        cluster_activation: str='sigmoid-off',
        edge_mask: bool=True
    ):
        assert(len(ks) == (n_gnn_layers - 1) // pooling_freq)
        super().__init__(
            key,
            d_in,
            d_in_edge,
            n_classes,

            d_model,
            n_gnn_layers,
            pooling_freq,
            dropout,
            dropout_pool,
            head_n_linear,

            DiffPool,
            clustering_params=[{'k': k} for k in ks]
        )
