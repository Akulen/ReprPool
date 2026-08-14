from functools import cache
from typing import List, Literal
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*`torch_geometric\.distributed` has been deprecated.*",
)
from ogb.graphproppred import PygGraphPropPredDataset
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import numpy as np
import jax.numpy as jnp
from ogb.graphproppred import Evaluator

torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, GlobalStorage
])

from ggnn.data.types import BatchedSample
from ggnn.data.dataset import Dataset
from ggnn.data.pyg import pyg_to_jax

@cache
def ogbg_molhiv():
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/ogbg')
    return dataset, {
        'task': 'classification',
        'num_classes': 2,
        'eval_metric': 'rocauc'
    }

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
        dataset, config = ogbg_molhiv()
        super().__init__(n_class=config['num_classes'], **kwargs)
        self.connex_target = connex_target

        self.data_type = category
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

    def self_init(self):
        super().self_init()
        self.custom_loss_name = 'rocauc'
        self.evaluator = Evaluator(name = "ogbg-molhiv")

    def custom_loss(self, y_true, y_pred):
        input_dict = {
            "y_true": np.array(y_true[:, None]),
            "y_pred": np.array(y_pred[:, None]),
        }
        result_dict = self.evaluator.eval(input_dict)
        return result_dict['rocauc']

    def to_serializable(self):
        serial = super().to_serializable()
        serial['scalar']['connex_target'] = self.connex_target
        return serial
