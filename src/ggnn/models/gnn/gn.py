import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List, Literal, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.utils import BatchNorm, get_activation
from ggnn.models.graph_utils import und_to_dir
from ggnn.models.gnn.types import GNNLayer, PoolLayer
from ggnn.models.gnn.gin import GINLayer
from ggnn.models.gnn.gcn import GCNLayer, GatedGCNLayer
from ggnn.models.gnn.pooling import GlobalAddPool, GlobalMeanPool, GlobalReprPool
from ggnn.models.gnn.mlp import MLP
from ggnn.models.gnn.uf import GNHook
from ggnn.models.gnn.unet import EdgeDrop

class GNLayer(GNNLayer):
    gn_layer: GNHook
    nn: MLP
    epsilon: Float[Array, '']
    self_edge: float=eqx.field(static=True)
    aggr_type: Literal['sum', 'mean', 'max']=eqx.field(static=True)
    bn: Optional[BatchNorm | eqx.nn.LayerNorm]
    residual: bool=eqx.field(static=True)
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        layer_hook: str="DefaultHook",
        hook_args: dict={},

        mlp_depth: int=2,

        aggr_type: str='sum',
        dropout: float=0.0,
        batch_norm: bool=False,
        residual: bool=False,
        learn_eps: bool=False,

        self_edge: bool=True,
        fixed_mlp: bool=False
    ):
        use_ln = layer_hook == 'UF'
        key, subkey = jax.random.split(key)
        super().__init__(subkey, d_model, "DefaultHook", {})
        key, subkey = jax.random.split(key)
        self.gn_layer = GNHook(subkey, d_model, **hook_args)
        self.nn = MLP(
            key=key,
            dims=[d_model for _ in range(mlp_depth + 1)],
            v2=use_ln#fixed_mlp
        )
        self.epsilon = jnp.array(0.0, dtype=jnp.float32)
        if not learn_eps:
            self.epsilon = jax.lax.stop_gradient(self.epsilon)
        # if not(aggr_type == 'mean' or aggr_type == 'max' or aggr_type == 'sum'):
        if not aggr_type == 'sum':
            raise ValueError(f'Aggregation type {aggr_type} not supported')
        self.aggr_type = aggr_type
        self.self_edge = 1 if self_edge else 0

        if batch_norm:
            if use_ln:
                self.bn = eqx.nn.LayerNorm(d_model)
            else:
                self.bn = BatchNorm(
                    input_size=d_model,
                    axis_name=('batch'),
                    momentum=0.9,
                    mode='batch'
                )
        else:
            self.bn = None
        self.residual = residual
        self.dropout = eqx.nn.Dropout(dropout)

    def forward(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        # Make sure graph only contains a single graph
        assert len(graph.f_node.shape) == 2

        x = graph.f_node
        key, subkey = jax.random.split(key)
        y, state = self.gn_layer.stack_cluster(
            subkey, state, graph, inference, max_depth=10
        )
        x = x * (self.self_edge + self.epsilon) + y.f_node
        mask = jnp.arange(graph.f_node.shape[0]) < graph.n_nodes
        key, subkey = jax.random.split(key)
        x, state = self.nn(x, subkey, state, mask=mask, inference=inference)

        if isinstance(self.bn, BatchNorm):
            x, state = self.bn(x.T, state, mask=mask, inference=inference)
            x = x.T
        elif isinstance(self.bn, eqx.nn.LayerNorm):
            x = jax.vmap(self.bn)(x)

        x = jax.nn.relu(x)

        if self.residual:
            x = graph.f_node + x

        x = self.dropout(x, inference=inference, key=key)

        return eqx.tree_at(lambda g: g.f_node, graph, x), state

class GNGINNet(eqx.Module):
    n_classes: int=eqx.field(static=True)
    embedding_h: eqx.nn.Linear
    gin_layers: List[GINLayer]
    linears_prediction: List[eqx.nn.Linear]
    pool: PoolLayer

    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int,

        d_model: int=64,
        dropout: float=0.0,
        n_layers: int=2,
        # layer_hook: str="DefaultHook",
        hook_args: dict={},

        n_mlp_layers: int=2,
        learn_eps: bool=True,
        neighbor_aggr_type: str='sum',
        readout: str='sum',
        batch_norm: bool=True,
        residual: bool=True
    ):
        del d_in_edge
        self.n_classes = n_classes
        key_emb, key_gin, key_lin = jax.random.split(key, 3)
        self.embedding_h = eqx.nn.Linear(
            in_features=d_in,
            out_features=d_model,
            key=key_emb
        )
        keys_gin = jax.random.split(key_gin, n_layers)
        self.gin_layers = [
            GINLayer(
                key=subkey,
                d_model=d_model,
                layer_hook='DefaultHook' if i == 0 else 'GN',
                hook_args={} if i == 0 else hook_args,
                mlp_depth=n_mlp_layers,
                aggr_type=neighbor_aggr_type,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual,
                learn_eps=learn_eps
            )
            for i, subkey in enumerate(keys_gin)
        ]

        keys_lin = jax.random.split(key_lin, n_layers + 1)
        self.linears_prediction = [
            eqx.nn.Linear(
                in_features=d_model,
                out_features=n_classes,
                key=subkey
            )
            for subkey in keys_lin
        ]

        match readout:
            case 'sum':
                self.pool = GlobalAddPool(key)
            case 'repr':
                self.pool = GlobalReprPool(key, d_model)
            case _:
                raise ValueError(f'Pool type {readout} not supported')

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[
        Float[Array, 'n_class'],
        eqx.nn.State
    ]:
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            jax.vmap(self.embedding_h)(graph.f_node)
        )

        hidden_rep = [graph.f_node]

        for layer in self.gin_layers:
            key, subkey = jax.random.split(key)
            graph, state = layer(subkey, state, graph, inference)
            hidden_rep.append(graph.f_node)

        score_over_layer = jnp.zeros(self.n_classes)

        mask = jnp.arange(graph.f_node.shape[0]) < graph.n_nodes
        for i, h in enumerate(hidden_rep):
            key, subkey = jax.random.split(key)
            pooled_h = self.pool(
                subkey,
                eqx.tree_at(lambda g: g.f_node, graph, h),
                mask,
                inference
            )
            score_over_layer += self.linears_prediction[i](pooled_h)

        return score_over_layer, state

class GNNet(eqx.Module):
    n_classes: int=eqx.field(static=True)
    embedding_h: eqx.nn.Linear
    embedding_e: Optional[eqx.nn.Linear]
    gcn_layers: List[GCNLayer | GatedGCNLayer]
    pool: PoolLayer
    mlp: MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int,

        d_model: int=64,
        dropout: float=0.0,
        n_layers: int=2,
        # layer_hook: str="DefaultHook",
        hook_args: dict={},

        edge_feat: bool=True,
        readout: str='mean',
        batch_norm: bool=True,
        use_ln: bool=False,
        residual: bool=True
    ):
        self.n_classes = n_classes
        key_embh, key_embe, key_gcn, key_mlp = jax.random.split(key, 4)
        self.embedding_h = eqx.nn.Linear(
            in_features=d_in,
            out_features=d_model,
            key=key_embh
        )
        if edge_feat:
            self.embedding_e = eqx.nn.Linear(
                in_features=d_in_edge,
                out_features=d_model,
                key=key_embe
            )
        else:
            self.embedding_e = None
        keys_gcn = jax.random.split(key_gcn, n_layers)
        if not edge_feat:
            self.gcn_layers = [
                GCNLayer(
                    key=subkey,
                    d_in=d_model,
                    d_out=d_model,
                    layer_hook='DefaultHook' if i == 0 else 'GN',
                    hook_args={} if i == 0 else hook_args,
                )
                for i, subkey in enumerate(keys_gcn)
            ]
        else:
            self.gcn_layers = [
                GatedGCNLayer(
                    key=subkey,
                    d_model=d_model,
                    layer_hook='DefaultHook' if i == 0 else 'GN',
                    hook_args={} if i == 0 else hook_args,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    use_ln=use_ln,
                    residual=residual,
                    edge_feat=edge_feat
                )
                for i, subkey in enumerate(keys_gcn)
            ] # Benchmarking parametrise the last layer output dim

        match readout:
            case 'sum':
                self.pool = GlobalAddPool(key)
            case 'mean':
                self.pool = GlobalMeanPool(key)
            case 'repr':
                self.pool = GlobalReprPool(key, d_model)
            case _:
                raise ValueError(f'Unknown readout {readout}')

        self.mlp = MLP(
            key=key_mlp,
            dims=[d_model, d_model // 2, d_model // 4, n_classes],
            p=0.0,
            batch_norm=False
        )

    def __call__(
        self, 
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[
        Float[Array, 'n_class'],
        eqx.nn.State
    ]:
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            jax.vmap(self.embedding_h)(graph.f_node)
        )
        if self.embedding_e is not None:
            graph = eqx.tree_at(
                lambda g: g.f_edge,
                graph,
                jax.vmap(self.embedding_e)(graph.f_edge)
            )

        for layer in self.gcn_layers:
            key, subkey = jax.random.split(key)
            graph, state = layer(subkey, state, graph, inference)

        key, subkey = jax.random.split(key)
        h = self.pool(
            key,
            graph,
            jnp.arange(graph.f_node.shape[0]) < graph.n_nodes,
            inference=inference
        )

        h, state = self.mlp(h[None], key, state)
        return h[0], state

class GNNet2(eqx.Module):
    n_classes: int=eqx.field(static=True)
    embedding_h: eqx.nn.Linear
    embedding_e: Optional[eqx.nn.Linear]
    layers: List[GNLayer]
    pool: PoolLayer
    mlp: MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_in_edge: int,
        n_classes: int,

        d_model: int=64,
        dropout: float=0.0,
        n_layers: int=2,
        # layer_hook: str="DefaultHook",
        hook_args: dict={},

        edge_feat: bool=True,
        readout: str='mean',
        batch_norm: bool=True,
        n_mlp_layers: int=2,
        learn_eps: bool=True,
        neighbor_aggr_type: str='sum',
        residual: bool=True
    ):
        self.n_classes = n_classes
        key_embh, key_embe, key_layers, key_mlp = jax.random.split(key, 4)
        self.embedding_h = eqx.nn.Linear(
            in_features=d_in,
            out_features=d_model,
            key=key_embh
        )
        if edge_feat:
            self.embedding_e = eqx.nn.Linear(
                in_features=d_in_edge,
                out_features=d_model,
                key=key_embe
            )
        else:
            self.embedding_e = None
        keys_layers = jax.random.split(key_layers, n_layers)
        self.layers = [
            GNLayer(
                key=subkey,
                d_model=d_model,
                layer_hook='DefaultHook',
                hook_args=hook_args,
                mlp_depth=n_mlp_layers,
                aggr_type=neighbor_aggr_type,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual,
                learn_eps=learn_eps
            )
            for subkey in keys_layers
        ] # Benchmarking parametrise the last layer output dim

        match readout:
            case 'mean':
                self.pool = GlobalMeanPool(key)
            case 'repr':
                self.pool = GlobalReprPool(key, d_model)
            case _:
                raise ValueError(f'Unknown readout {readout}')

        self.mlp = MLP(
            key=key_mlp,
            dims=[d_model, d_model // 2, d_model // 4, n_classes],
            p=0.0,
            batch_norm=False
        )

    def __call__(
        self, 
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[
        Float[Array, 'n_class'],
        eqx.nn.State
    ]:
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            jax.vmap(self.embedding_h)(graph.f_node)
        )
        if self.embedding_e is not None:
            assert graph.f_edge is not None
            graph = eqx.tree_at(
                lambda g: g.f_edge,
                graph,
                jax.vmap(self.embedding_e)(graph.f_edge)
            )

        for layer in self.layers:
            key, subkey = jax.random.split(key)
            graph, state = layer(subkey, state, graph, inference)

        key, subkey = jax.random.split(key)
        h = self.pool(
            key,
            graph,
            jnp.arange(graph.f_node.shape[0]) < graph.n_nodes,
            inference=inference
        )

        h, state = self.mlp(h[None], key, state)
        return h[0], state

class ReGNet(eqx.Module):
    gn: GNHook
    max_depth: int=eqx.field(static=True)
    reduction_factor: float=eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int,
        num_heads: int=2,
        dropout: float=0.1,
        edge_mask: bool=True,
        # activation: str,
        max_depth: int=10,
        reduction_factor: float=10
    ):
        self.gn = GNHook(
            key,
            d_model,
            num_heads,
            dropout,
            edge_mask,
        )
        self.max_depth = max_depth
        self.reduction_factor = reduction_factor

    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        return self.gn.stack_cluster(
            key, state, graph, inference, max_depth=self.max_depth,
            reduction_factor=self.reduction_factor
        )


class ReGNodeClassifier(eqx.Module):
    pre_gcn: GCNLayer
    regnet: ReGNet
    post_gcn: GCNLayer
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

        d_model: int=48,
        dropout_feat: float=0.92,
        dropout_adj: float=0.2,
        max_depth: int=10,
        reduction_factor: float=10,
        activation: str='elu',
        improved: bool=False
    ):
        key_pre, key_regnet, key_post = jax.random.split(key, 3)
        self.pre_gcn = GCNLayer(
            key_pre, d_in, d_model, dropout=dropout_feat
        )
        self.regnet = ReGNet(
            key_regnet, d_model, dropout=dropout_feat, # activation=activation,
            max_depth=max_depth, reduction_factor=reduction_factor
        )
        self.post_gcn = GCNLayer(
            key_post, d_model, n_classes, dropout=dropout_feat
        )

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

        key, subkey = jax.random.split(key)
        graph, state = self.pre_gcn(subkey, state, graph, inference)
        graph = eqx.tree_at(
            lambda g: g.f_node, graph, self.activation(graph.f_node)
        )

        key, subkey = jax.random.split(key)
        graph, state = self.regnet(subkey, graph, state, inference)

        key, subkey = jax.random.split(key)
        graph, state = self.post_gcn(subkey, state, graph, inference)
        return graph.f_node, state
