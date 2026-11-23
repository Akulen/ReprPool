from typing import Literal

from ggnn.data.dataset import Dataset
from ggnn.data.types import BatchedSample
from ggnn.data.pyg import pyg_to_jax
from ggnn.data.mnist import loadPixDataset

class DSCifar10(Dataset):
    def __init__(
        self,
        category: Literal['train', 'valid', 'test']='train',
        **kwargs
    ):
        dataset = loadPixDataset('CIFAR10', category)

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
