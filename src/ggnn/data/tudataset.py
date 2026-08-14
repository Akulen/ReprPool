import numpy as np
from torch_geometric.datasets import TUDataset
from typing import Literal

from ggnn.data.types import BatchedSample
from ggnn.data.dataset import Dataset
from ggnn.data.pyg import pyg_to_jax

class DSTU(Dataset):
    def __init__(
        self,
        name: Literal['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'NCI1', 'COLLAB'],
        category: Literal['train', 'valid', 'test']='train',
        dummy_node_feature: bool=False,
        **kwargs
    ):
        dataset = TUDataset(
            'data',
            name
        )
        rng = np.random.default_rng(42)
        n = len(dataset)
        fold = n // 10
        n_train = n - 2 * fold
        idx = np.arange(n)
        rng.shuffle(idx)

        super().__init__(n_class=dataset.y.max().item()+1, **kwargs)

        self.data_type = category
        if category == 'train':
            dataset = dataset[idx[:n_train]]
        elif category == 'valid':
            dataset = dataset[idx[n_train:n_train + fold]]
        elif category == 'test':
            dataset = dataset[idx[n_train + fold:]]

        def get_pyg_to_jax_fn(ds):
            return pyg_to_jax(
                ds, dummy_node_feature=dummy_node_feature
            )

        graphs = self._get_graphs(dataset, get_pyg_to_jax_fn, **kwargs)

        # for g in dataset: # type: ignore
        #     e1 = set()
        #     e2 = set()
        #     for u, v in zip(g.edge_index[0], g.edge_index[1]):
        #         e1.add((u, v))
        #         e2.add((v, u))
        #     if e1 != e2:
        #         raise ValueError(f'Graph {g} is not undirected:\n{e1}\n{e2}')

        self.global_data = ()
        self.samples = BatchedSample(
            data=graphs,
            target=graphs.f_graph
        )
        self.n_samples = len(dataset)

    def _get_graphs(self, dataset, default_fn, **kwargs):
        return default_fn(dataset)

    def model_params(self):
        return {
            'd_in': self.samples.data.f_node.shape[-1], #type: ignore
            'd_in_edge': self.samples.data.f_edge.shape[-1], #type: ignore
            'n_classes': self.n_class
        }

class DSRedditB(DSTU):
    def __init__(
        self, category: Literal['train', 'valid', 'test']='train', *args,
        **kwargs
    ):
        super().__init__(
            'REDDIT-BINARY', category, dummy_node_feature=True, **kwargs
        )

class DSRedditM5K(DSTU):
    def __init__(
        self, category: Literal['train', 'valid', 'test']='train', *args,
        **kwargs
    ):
        super().__init__(
            'REDDIT-MULTI-5K', category, dummy_node_feature=True, **kwargs
        )

class DSRedditM12K(DSTU):
    def __init__(
        self, category: Literal['train', 'valid', 'test']='train', *args,
        **kwargs
    ):
        super().__init__(
            'REDDIT-MULTI-12K', category, dummy_node_feature=True, **kwargs
        )

class DSOH(DSTU):
    max_deg = None

    def _get_graphs(self, dataset, default_fn, **kwargs):
        import torch
        import torch.nn.functional as F
        from torch_geometric.utils import degree

        degrees = []
        for data in dataset:
            degrees.append(degree(
                data.edge_index[0], data.num_nodes, dtype=torch.long
            ))
        if self.max_deg is None:
            self.max_deg = max([deg.max().item() for deg in degrees])
            print(self.max_deg)
        else:
            assert self.max_deg >= max([deg.max().item() for deg in degrees])
        new_graphs = []
        for deg, data in zip(degrees, dataset):
            data.x = F.one_hot(deg, num_classes=self.max_deg+1).to(torch.float32)
            new_graphs.append(data)

        return default_fn(new_graphs)

class DSNCI1(DSOH):
    def __init__(
        self, category: Literal['train', 'valid', 'test']='train', *args,
        **kwargs
    ):
        self.max_deg = 4

        super().__init__(
            'NCI1', category, dummy_node_feature=False, **kwargs
        )

class DSCOLLAB(DSOH):
    def __init__(
        self, category: Literal['train', 'valid', 'test']='train', *args,
        **kwargs
    ):
        self.max_deg = 491

        super().__init__(
            'COLLAB', category, dummy_node_feature=False, **kwargs
        )
