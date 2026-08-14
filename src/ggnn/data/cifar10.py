import os
from functools import cache
from typing import Literal

from benchmarking.data.superpixels import SuperPixDataset
from ggnn.data.dataset import Dataset
from ggnn.data.types import BatchedSample
from ggnn.data.mnist import dgl_to_jax

@cache
def loadPixDataset(name='CIFAR10'):
    return SuperPixDataset(name)

class DSCifar10(Dataset):
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
