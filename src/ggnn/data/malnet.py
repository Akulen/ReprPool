from torch_geometric.datasets import MalNetTiny
from typing import Literal

from ggnn.data.types import BatchedSample
from ggnn.data import Dataset
from ggnn.data.pyg import pyg_to_jax

class DSMalNetTiny(Dataset):
    def __init__(
        self,
        category: Literal['train', 'valid', 'test']='train',
        **kwargs
    ):
        if category == 'valid': category = 'val' # type: ignore
        dataset = MalNetTiny(
            'data',
            category
        )
        self.data_type = category
        super().__init__(n_class=dataset.num_classes, **kwargs)

        graphs = pyg_to_jax(dataset, dummy_node_feature=True)

        self.global_data = ()
        self.samples = BatchedSample(
            data=graphs,
            target=graphs.f_graph
        )
        self.n_samples = len(dataset)

    def model_params(self):
        return {
            'd_in': self.samples.data.f_node.shape[-1], #type: ignore
            'd_in_edge': self.samples.data.f_edge.shape[-1], #type: ignore
            'n_classes': self.n_class
        }
