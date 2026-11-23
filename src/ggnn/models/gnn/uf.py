import jax
import jax.numpy as jnp
import equinox as eqx
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from jaxtyping import Array, Float, Int, Shaped, PRNGKeyArray

if TYPE_CHECKING:
    from ggnn.models.gnn.types import GNNLayer
from ggnn.models.gnn.hooks import DefaultHook
from ggnn.data.types import Graph
# from ggnn.models.gnn.types import PoolLayer
# from ggnn.models.utils import layer_norm, BatchNorm, masked_mean
# from ggnn.models.gnn.classifier import GNNClassifier
# from ggnn.models.gnn.gin import GINLayer
# from ggnn.models.gnn.pooling import GlobalAddPool
from utils.jaxtools import cond_print, count_distinct

def segment_argmax(
    x: Shaped[Array, "n"],
    segment_ids: Int[Array, "n"],
    num_segments: int
):
    max_per_seg = jax.ops.segment_max(x, segment_ids, num_segments)
    is_max = x == max_per_seg[segment_ids]

    idx = jnp.arange(x.shape[0])
    masked_idx = jnp.where(is_max, idx, jnp.iinfo(idx.dtype).max)

    return jax.ops.segment_min(masked_idx, segment_ids, num_segments)

def segment_softmax(
    x: Float[Array, "n"],
    segment_ids: Int[Array, "n"],
    num_segments: int
) -> Float[Array, "n"]:
    maxs = jax.ops.segment_max(x, segment_ids, num_segments)
    logits = x - maxs[segment_ids]
    logits = jnp.exp(logits)
    normalizers = jax.ops.segment_sum(logits, segment_ids, num_segments)
    normalizers = normalizers[segment_ids]

    return logits / normalizers

def count_unique(
    x: Int[Array, 'n']
) -> Int[Array, '']:
    x = jnp.sort(x)
    return 1 + (x[1:] != x[:-1]).sum()

def count_clusters(
    repr: Int[Array, 'n']
) -> Int[Array, '']:
    n = repr.shape[0]
    is_real = repr != (n - 1)

    cluster_sizes = jax.ops.segment_sum(
        is_real.astype(jnp.int32),
        repr,
        num_segments=n
    )

    return jnp.sum(cluster_sizes > 0)

def fully_clustered(graph, debug=0):
    edges = graph.edges
    mask = jnp.arange(edges.shape[0]) < graph.n_edges
    jax.lax.cond(
        debug > 0,
        lambda x, y: jax.debug.print('Depth {} => {}', x-1, y),
        lambda x, y: None,
        debug,
        (mask, edges[:, 0] == edges[:, 1])
    )
    # Check if all edges are self-edges (fully compressed)
    return jnp.logical_or(
        ~mask, edges[:, 0] == edges[:, 1]
    ).all()

# Greatest Neighbor Clustering
class GNCluster(eqx.Module):
    score_node: eqx.nn.Linear
    score_edge: Optional[eqx.nn.Linear]
    dropout: eqx.nn.Dropout
    activation: Any
    edge_feat: bool=eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        dropout: float=0.0,
        cluster_activation: str='sigmoid-off',
        edge_mask: bool=False,
        edge_feat: bool=False
    ):
        key_node, key_edge = jax.random.split(key, 2)
        self.score_node = eqx.nn.Linear(
            in_features=d_model,
            out_features='scalar',
            key=key_node
        )
        if edge_mask:
            self.score_edge = eqx.nn.Linear(
                in_features=d_model if edge_feat else 2*d_model,
                out_features='scalar',
                key=key_edge
            )
        else:
            self.score_edge = None
        self.dropout = eqx.nn.Dropout(dropout)
        self.edge_feat = edge_feat
        match cluster_activation:
            case 'sigmoid':
                self.activation = jax.nn.sigmoid
            case 'sigmoid-off':
                self.activation = lambda x: 0.5 + jax.nn.sigmoid(x)
            case 'tanh':
                self.activation = jax.nn.tanh
            case _:
                raise ValueError(f'Unknown activation {cluster_activation}')

    def find_clusters(
        self,
        edge_up: Int[Array, 'n'],
    ) -> Int[Array, 'n']:
        n = edge_up.shape[0]
        repr = jnp.arange(n)
        def find(
            carry: Tuple[Int[Array, "n"], Int[Array, "n"]]
        ) -> Tuple[Int[Array, "n"], Int[Array, "n"]]:
            _, repr = carry
            new_repr = repr[repr]
            return repr, new_repr
        _, repr = jax.lax.while_loop(
            lambda c: (c[0] != c[1]).any(),
            find,
            (repr, edge_up)
        )
        return repr

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False,
        edge_threshold: float=0.,
    ) -> Tuple[Graph, Int[Array, 'n'], Float[Array, 'n'], eqx.nn.State]:
        n = graph.f_node.shape[0]
        m = graph.edges.shape[0]
        mask_node = jnp.arange(n) < graph.n_nodes
        mask_edge = jnp.arange(m) < graph.n_edges
        mask_edge = jnp.concatenate([
            mask_edge,
            mask_edge
        ])

        # We assumed the initial graph is undirected: each edge is present in
        # the opposite direction
        # UPDATE: This assumption is wrong
        edges = jnp.concatenate([
            graph.edges,
            graph.edges[:, ::-1]
        ])
        edge_h = None
        if graph.f_edge is not None:
            edge_h = jnp.concatenate([
                graph.f_edge,
                graph.f_edge
            ], axis=0)

        # Compute Scores
        node_score = jax.vmap(self.score_node)(graph.f_node)
        key, subkey = jax.random.split(key, 2)
        g = -jnp.log(-jnp.log(jax.random.uniform(subkey, node_score.shape)))
        node_score = node_score + 1e-4 * g
        key, subkey = jax.random.split(key, 2)
        node_score = self.dropout(node_score, inference=inference, key=subkey)

        edge_score = 2 * mask_edge.astype(jnp.float32) - 1
        if self.score_edge is not None:
            if not self.edge_feat:
                edge_h = jnp.concatenate([
                    graph.f_node[edges[:, 0]],
                    graph.f_node[edges[:, 1]]
                ], axis=-1)
            assert edge_h is not None
            edge_score = jax.vmap(self.score_edge)(edge_h)
            key, subkey = jax.random.split(key, 2)
            edge_score = self.dropout(
                edge_score, inference=inference, key=subkey
            )
            edge_score = self.activation(edge_score)

        mask_norm = mask_edge
        _, unique_idx = jnp.unique(
            edges, size=edges.shape[0], axis=0, return_index=True
        )
        mask_norm &= jnp.full(edges.shape[0], False).at[unique_idx].set(True)
        # Add Self Edges
        edge_u = jnp.concatenate([
            jnp.arange(n),
            edges[:, 0]
        ])
        edge_v = jnp.concatenate([
            jnp.arange(n),
            edges[:, 1]
        ])
        edge_score_self = jnp.concatenate([
            jnp.ones(n),
            edge_score
        ])

        # Find best node in Neighborhood
        # TODO: Think about the correct direction when directed graph
        edge_up = segment_argmax(
            jnp.where(
                edge_score_self > edge_threshold,
                node_score[edge_v],
                -jnp.inf
            ),
            edge_u,
            n
        )
        edge_up = edge_v[edge_up]

        repr = self.find_clusters(edge_up)
        repr = jnp.where(mask_node, repr, n-1)

        is_leader = repr == jnp.arange(n)
        repr_cluster = (jnp.cumsum(is_leader) - 1)[repr]
        cluster_repr = jnp.nonzero(is_leader, size=n, fill_value=n-1)[0]
        n_clusters = jnp.sum(is_leader & mask_node)

        # n_clusters = count_clusters(repr)
        # cluster_repr, repr_cluster = jnp.unique(
        #     repr, return_inverse=True, size=n, fill_value=n-1
        # )

        # TODO try different score regularizations, like softmax
        # Maybe should be + 1 / degree instead of + 0.5?
        # scores = jax.nn.sigmoid(scores) + 0.5
        # scores = jax.nn.tanh(scores)
        node_score = self.activation(node_score)
        if self.score_edge is None:
            edge_score = mask_norm.astype(jnp.float32)
        else:
            edge_score: Float[Array, 'm'] = jnp.where(mask_norm, edge_score, 0)

        node_message = jax.ops.segment_sum(
            graph.f_node[edges[:, 0]] * edge_score[:, None],
            edges[:, 1],
            num_segments=n
        )
        # cluster_sizes = jax.ops.segment_sum(
        #     (jnp.ones(n, dtype=jnp.int32) < graph.n_nodes).astype(jnp.int32),
        #     repr_cluster,
        #     num_segments=n
        # )
        # node_messages = jnp.where(
        #     (cluster_sizes[repr_cluster] == 1)[:, None],
        #     graph.f_node + node_message,
        #     node_message
        # )
        node_normalizer = jax.ops.segment_sum(
            edge_score,
            edges[:, 1],
            num_segments=n
        )
        normalizer = jax.ops.segment_sum(
            node_normalizer * node_score,
            repr_cluster,
            num_segments=n
        )
        f_cluster = graph.f_node[cluster_repr] + jax.ops.segment_sum(
            node_message * node_score[:, None],
            repr_cluster,
            num_segments=n
        ) / (jax.nn.softplus(normalizer) + jax.nn.softplus(-normalizer))[:,None]

        return Graph(
            n_nodes=n_clusters,
            n_edges=graph.n_edges,
            f_node=f_cluster,
            # TODO: How to aggregate edges?
            f_edge=None if graph.f_edge is None else graph.f_edge,
            f_graph=graph.f_graph,
            edges=repr_cluster[graph.edges],
        ), repr_cluster, node_score, state


class UFCluster(eqx.Module):
    node_score: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: Any
    noise: float=eqx.field(static=True)

    neg_scores: bool=eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        neg_scores: bool=False,
        noise: float=1e-4,
        dropout: float=0.0,
        cluster_activation: str='sigmoid-off'
    ):
        self.node_score = eqx.nn.Linear(
            in_features=d_model,
            out_features='scalar',
            key=key
        )
        self.dropout = eqx.nn.Dropout(dropout)
        self.noise = noise
        self.neg_scores = neg_scores
        match cluster_activation:
            case 'sigmoid':
                self.activation = jax.nn.sigmoid
            case 'sigmoid-off':
                self.activation = lambda x: 0.5 + jax.nn.sigmoid(x)
            case 'tanh':
                self.activation = jax.nn.tanh
            case _:
                raise ValueError(f'Unknown activation {cluster_activation}')

    def find_clusters(
        self,
        edge_up: Int[Array, 'n'],
    ) -> Int[Array, 'n']:
        n = edge_up.shape[0]
        repr = jnp.arange(n)
        def find(
            carry: Tuple[Int[Array, "n"], Int[Array, "n"]]
        ) -> Tuple[Int[Array, "n"], Int[Array, "n"]]:
            _, repr = carry
            new_repr = repr[repr]
            return repr, new_repr
        _, repr = jax.lax.while_loop(
            lambda c: (c[0] != c[1]).any(),
            find,
            (repr, edge_up)
        )
        return repr

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, Int[Array, 'n'], Float[Array, 'n'], eqx.nn.State]:
        n = graph.f_node.shape[0]
        mask = jnp.arange(n) < graph.n_nodes

        # Compute Node Scores
        scores = jax.vmap(self.node_score)(graph.f_node)
        g = -jnp.log(-jnp.log(jax.random.uniform(key, scores.shape)))
        scores = scores + self.noise * g
        key, subkey = jax.random.split(key, 2)
        scores = self.dropout(scores, inference=inference, key=subkey)

        # Add Self Edges
        edge_u = jnp.concatenate([
            jnp.arange(n),
            graph.edges[:, 0]
        ])
        edge_v = jnp.concatenate([
            jnp.arange(n),
            graph.edges[:, 1]
        ])

        # Find best node in Neighborhood
        # TODO: Think about the correct direction when directed graph
        edge_up = segment_argmax(scores[edge_v], edge_u, n)
        edge_up = edge_v[edge_up]

        repr = self.find_clusters(edge_up)
        repr = jnp.where(mask, repr, n-1)
        n_clusters = count_clusters(repr)

        _, repr_cluster = jnp.unique(
            repr, return_inverse=True, size=n, fill_value=n-1
        )

        if self.neg_scores: # Do we reward cluster centers or extremities?
            scores = -scores
        # TODO try different score regularizations, like softmax
        # Maybe should be + 1 / degree instead of + 0.5?
        # scores = jax.nn.sigmoid(scores) + 0.5
        # scores = jax.nn.tanh(scores)
        scores = self.activation(scores)

        # Compute Cluster Embeddings
        # 1. Neighborhood aggregation (weighted mean)
        neighbor_nodes_idx = graph.edges[:, 0]
        neighbor_target_idx = graph.edges[:, 1]

        weighted_neighbor_features = jax.ops.segment_sum(
            scores[neighbor_nodes_idx, None] * graph.f_node[neighbor_nodes_idx],
            neighbor_target_idx,
            num_segments=n
        )
        neighbor_score_abs_sum = jax.ops.segment_sum(
            jnp.abs(scores[neighbor_nodes_idx]),
            neighbor_target_idx,
            num_segments=n
        )

        total_abs_score = jnp.abs(scores) + neighbor_score_abs_sum
        messages = (scores[:, None] * graph.f_node + weighted_neighbor_features) / (total_abs_score[:, None] + 1e-8)

        # 2. Cluster aggregation (weighted sum)
        # Note: Summing into clusters allows the signal magnitude to reflect
        # cluster size, similar to ClusterPooling and GlobalAddPool.
        f_cluster = jax.ops.segment_sum(
            scores[:, None] * messages,
            repr_cluster,
            num_segments=n
        )

        return Graph(
            n_nodes=n_clusters,
            n_edges=graph.n_edges,
            f_node=f_cluster,
            f_edge=graph.f_edge, # TODO: How to aggregate edges?
            f_graph=graph.f_graph,
            edges=repr_cluster[graph.edges],
        ), repr_cluster, scores, state

class UFHook(DefaultHook):
    clustering: UFCluster | List[UFCluster]
    cluster_proj: eqx.nn.Linear | List[eqx.nn.Linear]
    up_norm: eqx.nn.LayerNorm
    max_depth: int=eqx.field(static=True)
    cc_norm: eqx.nn.LayerNorm
    cc_op: eqx.nn.MultiheadAttention
    dropout: eqx.nn.Dropout
    reduce_weight: eqx.nn.Linear | List[eqx.nn.Linear]
    reduce_cluster: eqx.nn.Linear | List[eqx.nn.Linear]
    down_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        max_depth: int=1,
        same_weights: bool=True,
        num_heads: int=2,
        dropout: float=0.1,
        # mlp_depth: int=2,
        # self_edge: bool=True,
        # fixed_mlp: bool=False,
        # neg_scores: bool=False,
        # cc_op: str='att'
    ):
        (
            key_clust, key_cl_lin, key_cc, key_redw, key_redc
        ) = jax.random.split(key, 5)
        if same_weights:
            self.clustering = UFCluster(
                key=key_clust,
                d_model=d_model,
                # neg_scores: bool=False,
                # noise: float=1e-4
            )
            self.cluster_proj = eqx.nn.Linear(
                in_features=d_model,
                out_features=d_model,
                key=key_cl_lin
            )
        else:
            keys = jax.random.split(key_clust, max_depth)
            self.clustering = [
                UFCluster(
                    key=subkey,
                    d_model=d_model,
                ) for subkey in keys
            ]
            keys = jax.random.split(key_cl_lin, max_depth)
            self.cluster_proj = [
                eqx.nn.Linear(
                    in_features=d_model,
                    out_features=d_model,
                    key=subkey
                ) for subkey in keys
            ]
        self.up_norm = eqx.nn.LayerNorm(d_model)
            # BatchNorm(
            #     input_size=d_model,
            #     axis_name=('batch'),
            #     momentum=0.9,
            #     mode='batch'
            # )
        #     for _ in range(max_depth)
        # ]
        self.max_depth = max_depth
        self.cc_norm = eqx.nn.LayerNorm(d_model)
        # self.cc_norm = BatchNorm(
        #     input_size=d_model,
        #     axis_name=('batch'),
        #     momentum=0.9,
        #     mode='batch'
        # )
        self.cc_op = eqx.nn.MultiheadAttention(
                num_heads=num_heads,
                query_size=d_model,
                use_query_bias=True,
                use_key_bias=True,
                use_value_bias=True,
                use_output_bias=True,
                dropout_p=dropout,
                key=key_cc
        )
        self.dropout = eqx.nn.Dropout(dropout)
        if same_weights:
            self.reduce_weight = eqx.nn.Linear(
                in_features=2*d_model,
                out_features='scalar',
                key=key_redw
            )
            self.reduce_cluster = eqx.nn.Linear(
                in_features=d_model,
                out_features=d_model,
                key=key_redc
            )
        else:
            keys = jax.random.split(key_redw, max_depth)
            self.reduce_weight = [
                eqx.nn.Linear(
                    in_features=2*d_model,
                    out_features='scalar',
                    key=subkey
                ) for subkey in keys
            ]
            keys = jax.random.split(key_redc, max_depth)
            self.reduce_cluster = [
                eqx.nn.Linear(
                    in_features=d_model,
                    out_features=d_model,
                    key=subkey
                ) for subkey in keys
            ]
        self.down_norm = eqx.nn.LayerNorm(d_model)
            # BatchNorm(
            #     input_size=d_model,
            #     axis_name=('batch'),
            #     momentum=0.9,
            #     mode='batch'
            # )
        #     for _ in range(max_depth)
        # ]

    def __call__(
        self,
        layer: "GNNLayer",
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        n = graph.f_node.shape[0]
        graphs = [graph]
        clusterings = []
        scores = []

        # Hierarchical Clustering
        key, subkey = jax.random.split(key, 2)
        keys = jax.random.split(subkey, self.max_depth)
        for i, subkey in enumerate(keys):
            key_cluster, key_do = jax.random.split(subkey)
            clustering = self.clustering
            if isinstance(clustering, list):
                clustering = clustering[i]
            graph, cluster, score, state = clustering(
                key_cluster,
                state,
                graphs[-1],
                inference=inference
            )
            cluster_proj = self.cluster_proj
            if isinstance(cluster_proj, list):
                cluster_proj = cluster_proj[i]
            f_cluster = jax.vmap(cluster_proj)(graph.f_node)
            f_cluster = jax.vmap(self.up_norm)(f_cluster)
            # f_cluster, state = self.up_norms[i](
            #     f_cluster.T,
            #     state,
            #     mask=jnp.arange(n) < graph.n_nodes,
            #     inference=inference
            # )
            # f_cluster = f_cluster.T
            f_cluster = jax.nn.relu(f_cluster)

            f_cluster = self.dropout(f_cluster, inference=inference, key=key_do)

            graphs.append(eqx.tree_at(
                lambda g: g.f_node,
                graph,
                f_cluster
            ))
            clusterings.append(cluster)
            scores.append(score)

        # Global Layer
        f_cc = graphs[-1].f_node
        mask = jnp.arange(n) < graphs[-1].n_nodes
        f_cc = jax.vmap(self.cc_norm)(f_cc)
        # f_cc, state = self.cc_norm(
        #     f_cc.T, state, mask=mask, inference=inference
        # )
        # f_cc = f_cc.T
        key, subkey = jax.random.split(key, 2)
        f_cc = self.cc_op(
            query=f_cc,
            key_=f_cc,
            value=f_cc,
            mask=jnp.outer(mask, mask),
            inference=inference,
            key=subkey
        )
        key, subkey = jax.random.split(key, 2)
        f_cc = self.dropout(f_cc, inference=inference, key=subkey)
        graphs[-1] = eqx.tree_at(
            lambda g: g.f_node,
            graphs[-1],
            graphs[-1].f_node + f_cc
        )

        # Downstream Update
        for i in range(len(graphs)-2, -1, -1):
            x = graphs[i].f_node
            x_c = graphs[i+1].f_node[clusterings[i]]
            reduce_weight = self.reduce_weight
            if isinstance(reduce_weight, list):
                reduce_weight = reduce_weight[i]
            weight = jax.vmap(reduce_weight)(jnp.concatenate([
                x,
                x_c,
            ], axis=1))
            weight = jax.nn.sigmoid(weight) / scores[i]
            reduce_cluster = self.reduce_cluster
            if isinstance(reduce_cluster, list):
                reduce_cluster = reduce_cluster[i]
            upd = weight[:, None] * jax.vmap(reduce_cluster)(x_c)
            mask = jnp.arange(graphs[i].f_node.shape[0]) < graphs[i].n_nodes
            upd = jax.vmap(self.down_norm)(upd)
            # upd, state = self.down_norms[i](
            #     upd.T, state, mask=mask, inference=inference
            # )
            # upd = upd.T
            upd = jax.nn.relu(upd)
            key, subkey = jax.random.split(key, 2)
            upd = self.dropout(upd, inference=inference, key=subkey)
            x = x + upd
            key, subkey = jax.random.split(key, 2)
            graphs[i], state = layer.forward(
                subkey,
                state,
                eqx.tree_at(
                    lambda g: g.f_node,
                    graphs[i],
                    x
                ),
                inference
            )
        return graphs[0], state

# class UFLayer(GNNLayer):
#     clusterers: List[UFCluster]
#     cc_norm: BatchNorm
#     cc_op: None | eqx.nn.MultiheadAttention | eqx.nn.MLP
#     reduce_weight: List[eqx.nn.Linear]
#     reduce_cluster: List[eqx.nn.Linear]
#     bns: List[BatchNorm]
#     dropout: eqx.nn.Dropout
#
#     gnns: List[GNNLayer]
#
#     def __init__(
#         self,
#         key: PRNGKeyArray,
#         d_model: int=64,
#         max_depth: int=1,
#         num_heads: int=2,
#         dropout_p: float=0.1,
#         mlp_depth: int=2,
#         aggr_type: str='sum',
#         dropout_gin: float=0.0,
#         batch_norm: bool=False,
#         residual: bool=False,
#         learn_eps: bool=False,
#         self_edge: bool=True,
#         fixed_mlp: bool=True,
#         neg_scores: bool=False,
#         cc_op: str='att'
#     ):
#         (
#             key_cluster, key_cc, key_reduce, key_mlp
#         ) = jax.random.split(key, 4)
#         keys = jax.random.split(key_cluster, max_depth)
#         self.clusterers = [
#             UFCluster(
#                 key=subkey,
#                 d_model=d_model,
#                 dropout_p=dropout_p,
#                 neg_scores=neg_scores
#             )
#             for subkey in keys
#         ]
#         self.cc_norm = BatchNorm(
#             input_size=d_model,
#             axis_name=('batch'),
#             momentum=0.9,
#             mode='batch'
#         )
#         if cc_op == 'att':
#             self.cc_op = eqx.nn.MultiheadAttention(
#                 num_heads=num_heads,
#                 query_size=d_model,
#                 use_query_bias=True,
#                 use_key_bias=True,
#                 use_value_bias=True,
#                 use_output_bias=True,
#                 dropout_p=dropout_p,
#                 key=key_cc
#             )
#         elif cc_op == 'none':
#             self.cc_op = None
#         elif cc_op == 'mlp':
#             self.cc_op = eqx.nn.MLP(
#                 in_size=d_model,
#                 out_size=d_model,
#                 width_size=d_model,
#                 depth=mlp_depth-1,
#                 key=key_cc
#             )
#         else:
#             raise ValueError(f'Unknown cc_op: {cc_op}')
#         key_red_weight, key_red_cluster = jax.random.split(key_reduce, 2)
#         keys = jax.random.split(key_red_weight, max_depth)
#         self.reduce_weight = [
#             eqx.nn.Linear(
#                 in_features=2*d_model,
#                 out_features='scalar',
#                 key=subkey
#             )
#             for subkey in keys
#         ]
#         keys = jax.random.split(key_red_cluster, max_depth)
#         self.reduce_cluster = [
#             eqx.nn.Linear(
#                 in_features=d_model,
#                 out_features=d_model,
#                 key=subkey
#             )
#             for subkey in keys
#         ]
#         self.bns = [
#             BatchNorm(
#                 input_size=d_model,
#                 axis_name='batch',
#                 momentum=0.9,
#                 mode='batch',
#             )
#             for _ in range(max_depth)
#         ]
#         self.dropout = eqx.nn.Dropout(dropout_p)
#
#         keys = jax.random.split(key_mlp, max_depth)
#         self.gnns = [
#             GINLayer(
#                 key=subkey,
#                 d_model=d_model,
#                 mlp_depth=mlp_depth,
#                 aggr_type=aggr_type,
#                 dropout=dropout_gin,
#                 batch_norm=batch_norm,
#                 residual=residual,
#                 learn_eps=learn_eps,
#                 self_edge=self_edge,
#                 fixed_mlp=fixed_mlp
#             )
#             for subkey in keys
#         ]
#
#     def __call__(
#         self,
#         key: PRNGKeyArray,
#         state: eqx.nn.State,
#         graph: Graph,
#         inference: bool=False
#     ) -> Tuple[Graph, eqx.nn.State]:
#         n = graph.f_node.shape[0]
#         graphs = [graph]
#         clusters = []
#         scores = []
#         # Merge up
#         for clusterer in self.clusterers:
#             key, subkey = jax.random.split(key, 2)
#             graph, cluster, score, state = clusterer(
#                 subkey, state, graphs[-1], inference=inference
#             )
#             graphs.append(graph)
#             clusters.append(cluster)
#             scores.append(score)
#
#         # TODO: detect when stable (all edges are self_edges), and zero out
#         # after
#         # for layer in self.layers:
#             # x_0 = x
#             # x = jax.vmap(utils.layer_norm)(x)
#             # key, key_layer, key_dropout = jax.random.split(key, 3)
#             # x = layer(key_layer, x, *args, inference=inference, **kwargs)
#             # x = x_0 + self.dropout(x, inference=inference, key=key_dropout)
#
#         # Attention between connected components
#         f_cc = graphs[-1].f_node
#         mask = jnp.arange(n) < graphs[-1].n_nodes
#         f_cc, state = self.cc_norm(f_cc.T, state, mask=mask, inference=inference)
#         f_cc = f_cc.T
#         key, subkey = jax.random.split(key, 2)
#         if isinstance(self.cc_op, eqx.nn.MultiheadAttention):
#             f_cc = self.cc_op(
#                 query=f_cc,
#                 key_=f_cc,
#                 value=f_cc,
#                 mask=jnp.outer(mask, mask),
#                 inference=inference,
#                 key=subkey
#             )
#         elif isinstance(self.cc_op, eqx.nn.MLP):
#             f_cc = jax.vmap(self.cc_op)(f_cc)
#             f_cc = jax.vmap(masked_mean, in_axes=(1, None))(f_cc, mask)
#             f_cc = f_cc[None]
#         elif self.cc_op is None:
#             f_cc = 0 * f_cc
#         key, subkey = jax.random.split(key, 2)
#         f_cc = graphs[-1].f_node + self.dropout(
#             f_cc, inference=inference, key=subkey
#         )
#         graphs[-1] = eqx.tree_at(
#             lambda g: g.f_node,
#             graphs[-1],
#             f_cc
#         )
#
#         # Expand down
#         for i in range(len(graphs)-2, -1, -1):
#             x = graphs[i].f_node
#             x_c = graphs[i+1].f_node[clusters[i]]
#             weight = jax.vmap(self.reduce_weight[i])(jnp.concatenate([
#                 x,
#                 x_c,
#             ], axis=1))
#             weight = jax.nn.sigmoid(weight) / scores[i]
#             x = x + weight[:, None] * jax.vmap(self.reduce_cluster[i])(x_c)
#             mask = jnp.arange(graphs[i].f_node.shape[0]) < graphs[i].n_nodes
#             x, state = self.bns[i](x.T, state, mask=mask, inference=inference)
#             x = x.T
#             # x = jax.nn.relu(x.T)
#             # key, subkey = jax.random.split(key, 2)
#             # x = self.dropout(x, inference=inference, key=subkey)
#             # TODO: generic GNN layer here
#             key, subkey = jax.random.split(key, 2)
#             graphs[i], state = self.gnns[i](
#                 subkey,
#                 state,
#                 eqx.tree_at(
#                     lambda g: g.f_node,
#                     graphs[i],
#                     x
#                 ),
#                 inference
#             )
#
#         return graphs[0], state

# class UFClassifier(GNNClassifier):
#     def __init__(
#         self,
#         key: PRNGKeyArray,
#         d_model: int=64,
#         n_gnn_layers: int=2,
#         pooling: Literal['add']='add',
#         dropout: float=0.5,
#         max_depth: int=1,
#         num_heads: int=2,
#         mlp_depth: int=2,
#         self_edge: bool=True,
#         fixed_mlp: bool=False,
#         neg_scores: bool=False,
#         cc_op: str='att'
#     ):
#         raise NotImplementedError
#         super().__init__(
#             key,
#             lambda k: UFLayer(
#                 k, d_model, max_depth, num_heads, dropout, mlp_depth,
#                 self_edge=self_edge, fixed_mlp=fixed_mlp,
#                 neg_scores=neg_scores, cc_op=cc_op
#             ),
#             d_model,
#             n_gnn_layers,
#             pooling,
#             dropout
#         )

class UFNet(eqx.Module):
    pass
    # n_classes: int=eqx.field(static=True)
    # embedding_h: eqx.nn.Linear
    # uf_layers: List[UFLayer]
    # linears_prediction: List[eqx.nn.Linear]
    # pool: PoolLayer

    # def __init__(
    #     self,
    #     key: PRNGKeyArray,
    #     d_in: int,
    #     d_in_edge,
    #     n_classes: int,
    #     d_model: int=64,
    #     max_depth: int=1,
    #     num_heads: int=2,
    #     dropout_p: float=0.1,
    #     dropout_gin: float=0.0,
    #     n_layers: int=4,
    #     n_mlp_layers: int=2,
    #     learn_eps: bool=True,
    #     neighbor_aggr_type: str='sum',
    #     readout: str='sum',
    #     batch_norm: bool=True,
    #     residual: bool=True
    # ):
    #     raise NotImplementedError
    #     del d_in_edge
    #     self.n_classes = n_classes
    #     key_emb, key_gin, key_lin = jax.random.split(key, 3)
    #     self.embedding_h = eqx.nn.Linear(
    #         in_features=d_in,
    #         out_features=d_model,
    #         key=key_emb
    #     )
    #     keys_gin = jax.random.split(key_gin, n_layers)
    #     self.uf_layers = [
    #         UFLayer(
    #             key=subkey,
    #             d_model=d_model,
    #             max_depth=max_depth,
    #             num_heads=num_heads,
    #             dropout_p=dropout_p,
    #             mlp_depth=n_mlp_layers,
    #             aggr_type=neighbor_aggr_type,
    #             dropout_gin=dropout_gin,
    #             batch_norm=batch_norm,
    #             residual=residual,
    #             learn_eps=learn_eps,
    #             self_edge=True,
    #             fixed_mlp=False
    #         )
    #         for subkey in keys_gin
    #     ]

    #     keys_lin = jax.random.split(key_lin, n_layers + 1)
    #     self.linears_prediction = [
    #         eqx.nn.Linear(
    #             in_features=d_model,
    #             out_features=n_classes,
    #             key=subkey
    #         )
    #         for subkey in keys_lin
    #     ]

    #     match readout:
    #         case 'sum':
    #             self.pool = GlobalAddPool(key)
    #         case _:
    #             raise ValueError(f'Pool type {readout} not supported')

    # def __call__(
    #     self,
    #     key: PRNGKeyArray,
    #     graph: Graph,
    #     state: eqx.nn.State,
    #     inference: bool=False
    # ) -> Tuple[
    #     Float[Array, 'n_class'],
    #     eqx.nn.State
    # ]:
    #     raise NotImplementedError
    #     graph = eqx.tree_at(
    #         lambda g: g.f_node,
    #         graph,
    #         jax.vmap(self.embedding_h)(graph.f_node)
    #     )

    #     hidden_rep = [graph.f_node]

    #     for layer in self.uf_layers:
    #         key, subkey = jax.random.split(key)
    #         graph, state = layer(subkey, state, graph, inference)
    #         hidden_rep.append(graph.f_node)

    #     score_over_layer = jnp.zeros(self.n_classes)

    #     mask = jnp.arange(graph.f_node.shape[0]) < graph.n_nodes
    #     for i, h in enumerate(hidden_rep):
    #         pooled_h = self.pool(
    #             eqx.tree_at(lambda g: g.f_node, graph, h), mask
    #         )
    #         score_over_layer += self.linears_prediction[i](pooled_h)

    #     return score_over_layer, state

class GNHook(DefaultHook):
    clustering: GNCluster
    # cluster_proj: eqx.nn.Linear | List[eqx.nn.Linear]
    up_norm: eqx.nn.LayerNorm
    cc_norm: eqx.nn.LayerNorm
    cc_op: eqx.nn.MultiheadAttention
    dropout: eqx.nn.Dropout
    reduce_weight: eqx.nn.Linear
    reduce_cluster: eqx.nn.Linear
    down_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        num_heads: int=2,
        dropout: float=0.1,
        edge_mask: bool=True,
        # cc_op: str='att'
        edge_feat: bool=False
    ):
        (
            key_clust, key_cl_lin, key_cc, key_redw, key_redc
        ) = jax.random.split(key, 5)
        self.clustering = GNCluster(
            key=key_clust,
            d_model=d_model,
            dropout=dropout,
            cluster_activation='tanh',
            edge_mask=edge_mask,
            edge_feat=edge_feat
        )
        # self.cluster_proj = eqx.nn.Linear(
        #     in_features=d_model,
        #     out_features=d_model,
        #     key=key_cl_lin
        # )
        self.up_norm = eqx.nn.LayerNorm(d_model)
            # BatchNorm(
            #     input_size=d_model,
            #     axis_name=('batch'),
            #     momentum=0.9,
            #     mode='batch'
            # )
        #     for _ in range(max_depth)
        # ]
        self.cc_norm = eqx.nn.LayerNorm(d_model)
        # self.cc_norm = BatchNorm(
        #     input_size=d_model,
        #     axis_name=('batch'),
        #     momentum=0.9,
        #     mode='batch'
        # )
        self.cc_op = eqx.nn.MultiheadAttention(
                num_heads=num_heads,
                query_size=d_model,
                use_query_bias=True,
                use_key_bias=True,
                use_value_bias=True,
                use_output_bias=True,
                dropout_p=dropout,
                key=key_cc
        )
        self.dropout = eqx.nn.Dropout(dropout)
        self.reduce_weight = eqx.nn.Linear(
            in_features=2*d_model,
            out_features='scalar',
            key=key_redw
        )
        # Initialize bias to negative to start with small hierarchical contribution
        self.reduce_weight = eqx.tree_at(
            lambda l: l.bias,
            self.reduce_weight,
            jnp.array([-2.0])
        )
        self.reduce_cluster = eqx.nn.Linear(
            in_features=d_model,
            out_features=d_model,
            key=key_redc
        )
        self.down_norm = eqx.nn.LayerNorm(d_model)
            # BatchNorm(
            #     input_size=d_model,
            #     axis_name=('batch'),
            #     momentum=0.9,
            #     mode='batch'
            # )
        #     for _ in range(max_depth)
        # ]

    def cc_step(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        inference: bool=False,
    ) -> Graph:
        x = graph.f_node
        mask = jnp.arange(x.shape[0]) < graph.n_nodes
        x_norm = jax.vmap(self.cc_norm)(x)
        key_att, key_do = jax.random.split(key)

        att_mask = jnp.outer(mask, mask)

        x_att = self.cc_op(
            query=x_norm,
            key_=x_norm,
            value=x_norm,
            mask=att_mask,
            inference=inference,
            key=key_att
        )
        x_att = self.dropout(x_att, inference=inference, key=key_do)

        # Residual connection
        x = x + x_att

        return eqx.tree_at(lambda g: g.f_node, graph, x)

    def up_step_fn(
        self,
        carry: Tuple[Graph, eqx.nn.State, PRNGKeyArray],
        depth: int | Int[Array, ""],
        inference: bool=False,
        reduction_factor: float=100,
    ) -> Tuple[
        Tuple[Graph, eqx.nn.State, PRNGKeyArray],
        Tuple[Graph, Int[Array, "n"]]
    ]:
        cur_graph, state, key = carry

        def clustering_fn(args):
            clustered_graph, clustering, _, state = self.clustering(
                args[0],
                args[1],
                args[2],
                inference,
                args[-1],
            )
            # clustered_graph = eqx.tree_at(
            #     lambda g: g.f_node,
            #     clustered_graph,
            #     jax.vmap(self.up_norm)(clustered_graph.f_node)
            # )
            return clustered_graph, clustering, state

        key, subkey = jax.random.split(key)

        def noop(c):
            return (
                c[2],
                -jnp.ones(cur_graph.f_node.shape[0], dtype=jnp.int32),
                c[1]
            )

        clustered_graph, clustering, state = jax.lax.cond(
            fully_clustered(cur_graph),
            noop,
            clustering_fn,
            (
                subkey, state, cur_graph,
                -2 #(1 / reduction_factor) ** depth - 1
            )
        )

        return (
            (clustered_graph, state, key),
            (clustered_graph, clustering)
        )

    def stack_cluster(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False,
        max_depth: int=20,
        reduction_factor: float=100
    ) -> Tuple[Graph, eqx.nn.State]:
        (_, state, key), (clustered_graphs, clusterings) = jax.lax.scan(
            lambda carry, depth: self.up_step_fn(
                carry, depth, inference=inference, reduction_factor=reduction_factor
            ),
            (graph, state, key),
            jnp.arange(max_depth)
        )
        def residual_stats(g):
            e, n = g.edges, g.f_node.shape[0]
            live = (jnp.arange(e.shape[0]) < g.n_edges) & (e[:, 0] != e[:, 1])
            e = jnp.where(live[:, None], e, -1)
            u = jnp.unique(e, axis=0, size=e.shape[0], fill_value=-1)
            ok = u[:, 0] >= 0
            deg = (
                jax.ops.segment_sum(ok.astype(jnp.int32), jnp.where(ok, u[:, 0], n), n + 1)
                + jax.ops.segment_sum(ok.astype(jnp.int32), jnp.where(ok, u[:, 1], n), n + 1)
            )[:n]
            return ok.sum(), deg.max()
        # cond_print(
        #     ~fully_clustered(jax.tree.map(lambda x: x[-1], clustered_graphs)),
        #     "[{}], Warning! Couldn't fully cluster graph ({}, {}) in {} poolings\n"
        #     "{}",
        #     fully_clustered(jax.tree.map(lambda x: x[-1], clustered_graphs)),
        #     graph.n_nodes, graph.n_edges, max_depth,
        #     (clustered_graphs.n_nodes, jax.vmap(count_distinct)(clustered_graphs.edges), residual_stats(jax.tree.map(lambda x: x[-1], clustered_graphs)))
        # )
        # jax.debug.print('steps: {}',
        #     (clustered_graphs.n_nodes, jax.vmap(count_distinct)(clustered_graphs.edges))
        # )
        graphs = jax.tree.map(
            lambda x, xs: jnp.concatenate([x[None], xs], axis=0),
            graph,
            clustered_graphs
        )
        # n_unmerged = (
        #     graphs[depth].edges[:, 0] != graphs[depth].edges[:, 1]
        # ).sum()
        # graphs = eqx.tree_at(lambda g: g.f_node, graphs, jnp.where(
        #     n_unmerged == 0,
        #     graphs.f_node,
        #     jnp.full_like(graphs.f_node, jnp.nan)
        # ))

        top_depth = jnp.argmin(jnp.concatenate([
            clusterings[:,0] > -1,
            jnp.array([False])
        ]))
        # jax.debug.print('n_nodes: {}', graphs.n_nodes)
        # jax.debug.print('top_depth: {}', top_depth)
        key, subkey = jax.random.split(key)
        # Apply CC Step to the graph reached at the end of clustering
        graph_top = self.cc_step(
            subkey,
            graphs[top_depth],
            inference
        )
        # Update the 'graphs' stack at the final depth
        graphs = jax.tree.map(
            lambda x, y: x.at[top_depth].set(y),
            graphs,
            graph_top
        )

        def down_step_fn(carry, val):
            nonlocal graphs
            depth, clustering = val
            cluster_graph, state, key = carry
            node_graph = graphs[depth]

            def body_fn(carry):
                node_graph, cluster_graph, state, key = carry

                x_c = cluster_graph.f_node[clustering]
                x_n = node_graph.f_node

                weight = jax.vmap(self.reduce_weight)(
                    jnp.concatenate([x_n, x_c], axis=1)
                )
                weight = jax.nn.sigmoid(weight)
                upd = weight[:, None] * jax.vmap(self.reduce_cluster)(x_c)
                key, subkey = jax.random.split(key)
                upd = self.dropout(upd, inference=inference, key=subkey)
                # upd = jax.vmap(self.down_norm)(x_c)

                x = (1 - weight[:, None]) * x_n + upd
                # x = jax.vmap(self.down_norm)(x)

                return x, state, key

            should_run = depth < top_depth
            x, state, key = jax.lax.cond(
                should_run,
                body_fn,
                lambda c: (c[0].f_node, c[2], c[3]),
                (node_graph, cluster_graph, state, key)
            )
            node_graph = eqx.tree_at(lambda g: g.f_node, node_graph, x)

            return (
                (node_graph, state, key),
                None
            )

        # Scan for max_depth steps
        (graph, state, _), _ = jax.lax.scan(
            down_step_fn,
            (graphs[-1], state, key),
            (jnp.arange(max_depth, 0, -1) - 1, clusterings[::-1])
        )
        return graph, state

    def __call__(
        self,
        layer: "GNNLayer",
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        key, subkey = jax.random.split(key)
        graph, state = self.stack_cluster(
            subkey, state, graph, inference
        )
        return layer.forward(key, state, graph, inference)
