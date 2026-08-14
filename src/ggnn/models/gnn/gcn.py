import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List,Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.utils import BatchNorm, GlorotLinear
from ggnn.models.gnn.types import GNNLayer, PoolLayer
from ggnn.models.gnn.pooling import GlobalMeanPool, GlobalReprPool
from ggnn.models.gnn.mlp import MLP

class GCNLayer(GNNLayer):
    lin: eqx.nn.Linear
    bias: Float[Array, 'd_out']
    dropout: eqx.nn.Dropout
    double_edge: bool=eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        d_in: int,
        d_out: int,
        layer_hook: str="DefaultHook",
        hook_args: dict={},
        dropout: float=0.0,
        double_edge: bool=False
    ):
        key, subkey = jax.random.split(key)
        super().__init__(subkey, d_in, layer_hook, hook_args)
        key, subkey = jax.random.split(key)
        self.lin = eqx.nn.Linear(
        # self.lin = GlorotLinear(
            in_features=d_in,
            out_features=d_out,
            key=subkey,
            use_bias=False
        ) # TODO: Should be glorot init
        self.bias = jnp.zeros(d_out, dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(dropout)
        self.double_edge = double_edge

    def forward(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False,
    ) -> Tuple[Graph, eqx.nn.State]:
        key, subkey = jax.random.split(key)
        x = self.dropout(graph.f_node, key=subkey, inference=inference)
        x = jax.vmap(self.lin)(x)

        weight = jnp.ones(graph.f_edge.shape[0], dtype=jnp.float32)
        # Add self-loops to the degree calculation to ensure signal persistence
        deg = jax.ops.segment_sum(
            weight,
            graph.edges[:, 0],
            num_segments=x.shape[0]
        ) + 1.0 
        
        deg_inv_sqrt = jnp.where(
            deg > 0,
            1.0 / jnp.sqrt(deg),
            0.0
        )
        
        # Scale messages by deg_inv_sqrt of both ends
        weight_scaled = (
            deg_inv_sqrt[graph.edges[:, 0]] *
            weight * 
            deg_inv_sqrt[graph.edges[:, 1]]
        )
        
        # Aggregate messages from neighbors
        x_agg = jax.ops.segment_sum(
            x[graph.edges[:, 0]] * weight_scaled[:, None],
            graph.edges[:, 1],
            num_segments=x.shape[0]
        )
        
        # Add self-loop contribution: x[i] / deg_i
        scale = 1
        if self.double_edge:
            scale = 2
        x = x_agg + scale * x * (deg_inv_sqrt[:, None] ** 2)
        x = x + self.bias
        graph = eqx.tree_at(
            lambda g: g.f_node,
            graph,
            x
        )
        return graph, state

class GatedGCNLayer(GNNLayer):
    linear_A: eqx.nn.Linear
    linear_B: eqx.nn.Linear
    linear_C: Optional[eqx.nn.Linear]
    linear_u: eqx.nn.Linear
    linear_v: eqx.nn.Linear
    bne: Optional[BatchNorm | eqx.nn.LayerNorm]
    bnn: Optional[BatchNorm | eqx.nn.LayerNorm]
    eps: float=eqx.field(static=True)
    residual: bool=eqx.field(static=True)
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        layer_hook: str="DefaultHook",
        hook_args: dict={},

        dropout: float=0.0,
        batch_norm: bool=True,
        use_ln: bool=False,
        residual: bool=True,
        edge_feat: bool=True
    ):
        key, subkey = jax.random.split(key)
        super().__init__(subkey, d_model, layer_hook, hook_args)
        (
            key_lina, key_linb, key_linc, key_linu, key_linv
        ) = jax.random.split(key, 5)
        self.linear_A = eqx.nn.Linear(
            in_features=d_model,
            out_features=d_model,
            # use_bias=False,
            key=key_lina
        )
        self.linear_B = eqx.nn.Linear(
            in_features=d_model,
            out_features=d_model,
            # use_bias=False,
            key=key_linb
        )
        if edge_feat:
            self.linear_C = eqx.nn.Linear(
                in_features=d_model,
                out_features=d_model,
                # use_bias=False,
                key=key_linc
            )
        else:
            self.linear_C = None
        self.linear_u = eqx.nn.Linear(
            in_features=d_model,
            out_features=d_model,
            # use_bias=False,
            key=key_linu
        )
        self.linear_v = eqx.nn.Linear(
            in_features=d_model,
            out_features=d_model,
            # use_bias=False,
            key=key_linv
        )
        if batch_norm:
            if use_ln:
                self.bne = eqx.nn.LayerNorm(d_model)
                self.bnn = eqx.nn.LayerNorm(d_model)
            else:
                self.bne = BatchNorm(
                    input_size=d_model,
                    axis_name=('batch'),
                    momentum=0.9,
                    mode='batch'
                )
                self.bnn = BatchNorm(
                    input_size=d_model,
                    axis_name=('batch'),
                    momentum=0.9,
                    mode='batch'
                )
        else:
            self.bne = None
            self.bnn = None
        self.eps = 1e-6
        self.residual = residual
        self.dropout = eqx.nn.Dropout(dropout)

    def forward(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        x = graph.f_node
        y = graph.f_edge
        mask_n = jnp.arange(graph.f_node.shape[0]) < graph.n_nodes
        mask_e = jnp.arange(graph.edges.shape[0]) < graph.n_edges

        e = (
            jax.vmap(self.linear_A)(x)[graph.edges[:, 0]] +
            jax.vmap(self.linear_B)(x)[graph.edges[:, 1]]
        )
        if self.linear_C is not None:
            assert y is not None
            e += jax.vmap(self.linear_C)(y)
        g = jax.nn.sigmoid(e) # Bug, should be y?
        m = jax.ops.segment_sum(
            g * jax.vmap(self.linear_v)(x)[graph.edges[:, 0]],
            graph.edges[:, 1],
            num_segments=x.shape[0]
        )
        norm = self.eps + jax.ops.segment_sum(
            g,
            graph.edges[:, 1],
            num_segments=x.shape[0]
        )
        h = jax.vmap(self.linear_u)(x) + m / norm

        if self.bnn is not None:
            if isinstance(self.bnn, BatchNorm):
                h, state = self.bnn(h.T, state, mask=mask_n, inference=inference)
                h = h.T
            else:
                h = jax.vmap(self.bnn)(h)
        if self.bne is not None:
            if isinstance(self.bne, BatchNorm):
                e, state = self.bne(e.T, state, mask=mask_e, inference=inference)
                e = e.T
            else:
                e = jax.vmap(self.bne)(e)

        h = jax.nn.relu(h)
        e = jax.nn.relu(e)

        if self.residual:
            h = h + x
            if y is not None:
                e = e + y

        key1, key2 = jax.random.split(key)
        h = self.dropout(h, inference=inference, key=key1)
        e = self.dropout(e, inference=inference, key=key2)

        graph = eqx.tree_at(lambda g: g.f_node, graph, h)
        graph = eqx.tree_at(lambda g: g.f_edge, graph, e)
        return graph, state

class GatedGCNNet(eqx.Module):
    n_classes: int=eqx.field(static=True)
    embedding_h: eqx.nn.Linear
    embedding_e: Optional[eqx.nn.Linear]
    gcn_layers: List[GatedGCNLayer]
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
        n_layers: int=4,
        layer_hook: str="DefaultHook",
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
        self.gcn_layers = [
            GatedGCNLayer(
                key=subkey,
                d_model=d_model,
                layer_hook=layer_hook,
                hook_args=hook_args,
                dropout=dropout,
                batch_norm=batch_norm,
                use_ln=use_ln,
                residual=residual,
                edge_feat=edge_feat
            )
            for subkey in keys_gcn
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
        else:
            graph = eqx.tree_at(
                lambda g: g.f_edge,
                graph,
                None
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
