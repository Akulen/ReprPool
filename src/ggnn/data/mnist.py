import os

from functools import cache
from typing import Literal
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
import jax.numpy as jnp

from benchmarking.data.superpixels import SuperPixDataset

from ggnn.data.types import BatchedSample
from ggnn.data.dataset import Dataset
from ggnn.data.pyg import to_jax_padded, pyg_to_jax
from ggnn.data.types import Graph

def dgl_to_jax(dataset):
    graphs = dataset.graph_lists
    n_nodes = [graph.num_nodes() for graph in graphs]
    n_edges = [graph.num_edges() for graph in graphs]
    max_nodes = max(n_nodes)
    max_edges = max(n_edges)
    f_node = [graph.ndata['feat'].T for graph in graphs]
    f_edge = [graph.edata['feat'].T for graph in graphs]
    f_graph = dataset.graph_labels
    edges = [torch.stack(graph.edges()) for graph in graphs]
    return Graph(
        n_nodes=jnp.array(n_nodes),
        n_edges=jnp.array(n_edges),
        f_node=to_jax_padded(
            f_node,
            max_nodes+1,
        ).transpose(0, 2, 1),
        f_edge=to_jax_padded(
            f_edge,
            max_edges
        ).transpose(0, 2, 1),
        f_graph=jnp.array(f_graph),
        edges=to_jax_padded(
            edges,
            max_edges,
            max_nodes,
            dtype=jnp.int32
        ).transpose(0, 2, 1)
    )

@cache
def loadPixDataset(name='MNIST'):
    return SuperPixDataset(name)

class DSMnist(Dataset):
    def __init__(
        self,
        category: Literal['train', 'valid', 'test']='train',
        **kwargs
    ):
        os.chdir('../benchmarking-gnns')
        dataset = loadPixDataset()
        os.chdir('../GGNN')

        self.data_type = category
        match category:
            case 'train':
                dataset = dataset.train
            case 'valid':
                dataset = dataset.val
            case 'test':
                dataset = dataset.test
        super().__init__(n_class=10, **kwargs)

        self.global_data = ()
        graph = dgl_to_jax(dataset)
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

class DSMnistBugged(Dataset):
    def __init__(
        self,
        category: Literal['train', 'valid', 'test']='train',
        **kwargs
    ):
        dataset = GNNBenchmarkDataset(
            root='data',
            name='MNIST',
            split=category if category != 'valid' else 'val'
        )
        super().__init__(n_class=10, **kwargs)

        self.data_type = category
        self.global_data = ()
        graph = pyg_to_jax(dataset, f_node_concat=['pos'])
        self.samples = BatchedSample(
            data=graph,
            target=graph.f_graph
        )
        self.n_samples = len(dataset)

    def model_params(self):
        return {
            'd_in': self.samples.data.f_node.shape[-1], #type: ignore
            'n_classes': self.n_class
        }
