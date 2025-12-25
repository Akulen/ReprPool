from functools import cache
from typing import Literal
from ogb.graphproppred import PygGraphPropPredDataset
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import numpy as np
import jax.numpy as jnp

torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, GlobalStorage
])

from ggnn.data.types import BatchedSample, Graph
from ggnn.data.dataset import Dataset

@cache
def ogbg_molhiv():
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/ogbg')
    return dataset, {
        'task': 'classification',
        'num_classes': 2,
        'eval_metric': 'rocauc'
    }

def to_jax_padded(xs, pad_len, pad_value=0):
    return jnp.array([
        torch.nn.functional.pad(x, (0, pad_len - x.shape[-1]), value=pad_value)
        for x in xs
    ])

def pyg_to_jax(pyg_graphs):
    max_nodes = max(graph.num_nodes for graph in pyg_graphs)
    max_edges = max(graph.num_edges for graph in pyg_graphs)
    return Graph(
        n_nodes=jnp.array([graph.num_nodes for graph in pyg_graphs]),
        n_edges=jnp.array([graph.num_edges for graph in pyg_graphs]),
        f_node=to_jax_padded(
            [graph.x.T for graph in pyg_graphs],
            max_nodes+1,
        ).transpose(0, 2, 1),
        f_edge=to_jax_padded(
            [graph.edge_attr.T for graph in pyg_graphs],
            max_edges
        ).transpose(0, 2, 1),
        f_graph=jnp.array([graph.y.squeeze() for graph in pyg_graphs]),
        edges=to_jax_padded(
            [graph.edge_index for graph in pyg_graphs],
            max_edges,
            max_nodes
        ).transpose(0, 2, 1)
    )

# from collections import Counter
# def get_stats(graph: Graph, rng, batch_size=64):
#     indices = np.arange(graph.n_nodes.shape[0])
#     rng.shuffle(indices)
#     indices = indices[:(len(indices) // batch_size) * batch_size]
#     indices.reshape((-1, batch_size))
#     cnt = Counter()
#     for batch_idx in indices:
#         n_nodes = graph.n_nodes[batch_idx].sum()
#         n_edges = graph.n_edges[batch_idx].sum()
#         n_nodes = 2 ** np.ceil(np.log2(n_nodes)).astype(np.int32)
#         n_edges = 2 ** np.ceil(np.log2(n_edges)).astype(np.int32)
#         # n_nodes = ((n_nodes + 63) // 64) * 64
#         # n_edges = ((n_edges + 63) // 64) * 64
#         cnt[(n_nodes.item(), n_edges.item())] += 1
#     return cnt

class DSMolhiv(Dataset):
    def __init__(
        self,
        category: Literal['train', 'valid', 'test']='train',
        connex_target: bool=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.connex_target = connex_target

        dataset, config = ogbg_molhiv()
        dataset: PygGraphPropPredDataset = dataset[ #type: ignore
            dataset.get_idx_split()[category]
        ]
        self.global_data = ()
        graphs = pyg_to_jax(dataset)
        self.samples = BatchedSample(
            data=graphs,
            target=graphs.f_graph
        )
        self.n_samples = len(dataset)

    def to_serializable(self):
        serial = super().to_serializable()
        serial['scalar']['connex_target'] = self.connex_target
        return serial
