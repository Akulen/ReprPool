import equinox as eqx
import jax
import jax.numpy as jnp
import logging
from typing import List, Literal, Optional, Tuple
from jaxtyping import Array, Float, Int, PRNGKeyArray

logger = logging.getLogger(__name__)

from ggnn.data.types import Graph
from ggnn.models.utils import BatchNorm
from ggnn.models.gnn.types import GNNLayer, PoolLayer
# from ggnn.models.gnn.classifier import GNNClassifier
from ggnn.models.gnn.pooling import GlobalAddPool, GlobalMaxPool, GlobalReprPool
from ggnn.models.gnn.mlp import MLP

class GINLayer(GNNLayer):
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
        super().__init__(subkey, d_model, layer_hook, hook_args)
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
        x = x * (self.self_edge + self.epsilon) + jax.ops.segment_sum(
            x[graph.edges[:, 0]],
            graph.edges[:, 1],
            num_segments=x.shape[0]
        )
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

# class GINClassifier(GNNClassifier):
#     def __init__(
#         self,
#         key: PRNGKeyArray,
#         d_model: int=64,
#         n_gnn_layers: int=2,
#         pooling: Literal['add']='add',
#         dropout: float=0.5,
#         mlp_depth: int=2,
#         self_edge: bool=True,
#         fixed_mlp: bool=False,
#         use_ln: bool=False
#     ):
#         raise NotImplementedError
#         super().__init__(
#             key,
#             lambda k: GINLayer(
#                 k, d_model, mlp_depth, self_edge=self_edge, fixed_mlp=fixed_mlp
#             ),
#             d_model,
#             n_gnn_layers,
#             pooling,
#             dropout,
#             use_ln
#         )

class GINNet(eqx.Module):
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
        n_layers: int=4,
        layer_hook: str="DefaultHook",
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
                layer_hook='DefaultHook' if i == 0 else layer_hook,
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
            case 'max':
                self.pool = GlobalMaxPool(key)
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
