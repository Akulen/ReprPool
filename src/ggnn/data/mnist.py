from functools import cache
from typing import Literal

from torch_geometric.datasets import GNNBenchmarkDataset

from ggnn.data.types import BatchedSample
from ggnn.data.dataset import Dataset
from ggnn.data.pyg import pyg_to_jax

# Deprecated
# def dgl_to_jax(dataset):
#     graphs = dataset.graph_lists
#     n_nodes = [graph.num_nodes() for graph in graphs]
#     n_edges = [graph.num_edges() for graph in graphs]
#     max_nodes = max(n_nodes)
#     max_edges = max(n_edges)
#     f_node = [graph.ndata['feat'].T for graph in graphs]
#     f_edge = [graph.edata['feat'].T for graph in graphs]
#     f_graph = dataset.graph_labels
#     edges = [torch.stack(graph.edges()) for graph in graphs]
#     return Graph(
#         n_nodes=jnp.array(n_nodes),
#         n_edges=jnp.array(n_edges),
#         f_node=to_jax_padded(
#             f_node,
#             max_nodes+1,
#         ).transpose(0, 2, 1),
#         f_edge=to_jax_padded(
#             f_edge,
#             max_edges
#         ).transpose(0, 2, 1),
#         f_graph=jnp.array(f_graph),
#         edges=to_jax_padded(
#             edges,
#             max_edges,
#             max_nodes,
#             dtype=jnp.int32
#         ).transpose(0, 2, 1)
#     )

@cache
def loadPixDataset(name='MNIST', category='train'):
    return GNNBenchmarkDataset(
        'data', name, category if category != 'valid' else 'val'
    )

class DSMnist(Dataset):
    def __init__(
        self,
        category: Literal['train', 'valid', 'test']='train',
        **kwargs
    ):
        dataset = loadPixDataset(category=category)

        self.data_type = category
        super().__init__(n_class=10, **kwargs)

        self.global_data = ()
        graph = pyg_to_jax(dataset)
        self.samples = BatchedSample(
            data=graph,
            target=graph.f_graph
        )
        self.n_samples = len(dataset)

    def model_params(self):
        return {
            'd_in': self.samples.data.f_node.shape[-1], #type: ignore
            'd_in_edge': self.samples.data.f_edge.shape[-1], #type: ignore
            'n_classes': self.n_class
        }
