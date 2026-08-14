from functools import partial
import time
import jax
import torch
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from typing import cast, Any, Dict, Tuple
from jaxtyping import Array, Bool, Float, PRNGKeyArray
import torch_geometric.data as pyg_data
from torch_geometric.datasets import Planetoid

from ggnn.data.types import BatchedData, BatchedSample, Graph
from ggnn.data.dataset import Dataset
from ggnn.node_training import NodeStepFN

class NodeDataset(Dataset):
    masks: Dict[str, Bool[Array, 'n']]
    task='node'

    def __init__(
        self,
        n_class: int=2,
        shuffle: bool=True,
        seed: int=42,
        verbose: bool=False
    ):
        super().__init__(n_class, shuffle, seed, verbose)

    def self_init(self):
        self.step_fn = None
        self.compiled_step_fn = None
        self.trainer = None

    def set_node_step_function(
        self,
        static: eqx.Module,
        step_fn: NodeStepFN,
        name: str='train'
    ):
        if self.step_fn is None:
            self.step_fn = {}
            self.compiled_step_fn = {}
        assert isinstance(self.compiled_step_fn, dict)
        def foo(
            params: eqx.Module,
            static: eqx.Module,
            state: eqx.nn.State,
            opt_state: Any,
            lr_scale: Float[Array, ''],
            batch: BatchedSample,
            cat_masks: Bool[Array, 'n'],
            key: PRNGKeyArray
        ) -> Tuple[
            eqx.Module, eqx.nn.State, Any, PRNGKeyArray, Float[Array, "3 L"]
        ]:
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            model, state, opt_state, losses = step_fn(
                model, state, opt_state, lr_scale, batch, cat_masks, subkey
            )
            params, _ = eqx.partition(model, eqx.is_array)
            return params, state, opt_state, key, losses
        self.step_fn[name] = partial(foo, static=static)
        self.compiled_step_fn[name] = None

    def node_precompile(
        self,
        params: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        key: PRNGKeyArray,
        name: str='train',
        verbose: int=0
    ):
        st = time.time()
        if self.step_fn is None or self.step_fn[name] is None:
            raise RuntimeError('Step function not set')
        assert isinstance(self.compiled_step_fn, dict)
        self.compiled_step_fn[name] = jax.jit(self.step_fn[name]).trace(
            params=params,
            state=state,
            opt_state=opt_state,
            lr_scale=1.0,
            batch=self.samples,
            cat_masks=jnp.stack([
                self.masks[cat]
                for cat in ['train', 'val', 'test']
            ]),
            key=key
        ).lower().compile()
        if verbose > 0:
            print(f'Precompiled {name} in {time.time() - st:.6f}s')

    def run_batches(
        self,
        params: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        key: PRNGKeyArray,
        batch_size: int=32,
        custom_loss: bool=False,
        lr_scale: Float[Array, ""]=jnp.array(1.0)
    ) -> Tuple[
        Tuple[eqx.Module, eqx.nn.State, Any, PRNGKeyArray],
        Float[Array, "L n_batches"]
    ]:
        raise NotImplementedError('run_batches should not be called for a node-level task')

    def step(
        self,
        params: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        key: PRNGKeyArray,
        lr_scale: Float[Array, ""]=jnp.array(1.0),
        name: str='train'
    ) -> Tuple[
        eqx.Module, eqx.nn.State, Any, PRNGKeyArray, Float[Array, "3 L"],
    ]:
        if self.compiled_step_fn is None or self.compiled_step_fn[name] is None:
            raise RuntimeError('Step function not compiled')
        return self.compiled_step_fn[name](
            params=params,
            state=state,
            opt_state=opt_state,
            lr_scale=lr_scale,
            batch=self.samples,
            cat_masks=jnp.stack([
                self.masks[cat]
                for cat in ['train', 'val', 'test']
            ]),
            key=key
        )

    def to_serializable(self):
        def to_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, jnp.ndarray):
                return np.array(x)
            return x

        serial = {
            'scalar': {
                'n_class': self.n_class,
                # 'n_samples': self.n_samples,
                # 'shuffle': self.shuffle,
                'task': self.task
            },
            'eqx_data': {
                'samples': jax.tree.map(to_host, self.samples),
                'global_data': jax.tree.map(to_host, self.global_data),
                'masks': {
                    tp: jax.tree.map(to_host, mask)
                    for tp, mask in self.masks.items()
                }
            },
            'rng': self.rng.bit_generator.state
        }
        return serial

    @classmethod
    def from_serialized(cls, serial):
        obj = object.__new__(cls)
        obj.self_init()

        def from_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, np.ndarray):
                return jnp.array(x)
            return x

        for k, v in serial['scalar'].items():
            setattr(obj, k, v)
        for k, v in serial['eqx_data'].items():
            if isinstance(v, dict):
                d = {}
                for k2, v2 in v.items():
                    d[k2] = jax.tree.map(from_host, v2)
                setattr(obj, k, d)
                continue
            setattr(obj, k, jax.tree.map(from_host, v))
        bit_rng = np.random.PCG64()
        bit_rng.state = serial['rng']
        obj.rng = np.random.Generator(bit_rng)
        return obj

    def model_params(self):
        d_in_edge = 0
        assert isinstance(self.samples.data, Graph)
        if self.samples.data.f_edge is not None:
            d_in_edge = self.samples.data.f_edge.shape[-1]
        return {
            'd_in': self.samples.data.f_node.shape[-1],
            'd_in_edge': d_in_edge,
            'n_classes': self.n_class
        }

def node_pyg_to_jax(g_pyg: pyg_data.data.Data) -> Graph:
    edges = set()
    assert g_pyg.edge_index is not None
    for u, v in g_pyg.edge_index.T:
        assert u != v
        u = u.item()
        v = v.item()
        if u > v:
            u, v = v, u
        edges.add((u, v))
    return Graph(
        n_nodes=jnp.array(g_pyg.num_nodes),
        n_edges=jnp.array(len(edges)),
        f_node=jnp.array(g_pyg.x),
        f_edge=None,
        f_graph=jnp.zeros(0),
        edges=jnp.array([[u, v] for u, v in edges])
    )

# Undirected!
class DSCora(NodeDataset):
    def __init__(
        self,
        **kwargs
    ):
        dataset = Planetoid(root='data', name='Cora')

        super().__init__(n_class=dataset.num_classes, **kwargs)

        g_pyg = cast(pyg_data.data.Data, dataset[0])
        g = node_pyg_to_jax(g_pyg)

        self.global_data = ()
        self.masks = {
            'train': jnp.array(g_pyg.train_mask),
            'val': jnp.array(g_pyg.val_mask),
            'test': jnp.array(g_pyg.test_mask)
        }

        # We store the node index in the samples
        self.samples = BatchedSample(
            data=g,
            target=jnp.array(g_pyg.y, dtype=jnp.int32)
        )

class DSCiteseer(NodeDataset):
    def __init__(
        self,
        **kwargs
    ):
        dataset = Planetoid(root='data', name='Citeseer')

        super().__init__(n_class=dataset.num_classes, **kwargs)

        g_pyg = cast(pyg_data.data.Data, dataset[0])
        g = node_pyg_to_jax(g_pyg)

        self.global_data = ()
        self.masks = {
            'train': jnp.array(g_pyg.train_mask),
            'val': jnp.array(g_pyg.val_mask),
            'test': jnp.array(g_pyg.test_mask)
        }

        # We store the node index in the samples
        self.samples = BatchedSample(
            data=g,
            target=jnp.array(g_pyg.y, dtype=jnp.int32)
        )

class DSPubMed(NodeDataset):
    def __init__(
        self,
        **kwargs
    ):
        dataset = Planetoid(root='data', name='PubMed')

        super().__init__(n_class=dataset.num_classes, **kwargs)

        g_pyg = cast(pyg_data.data.Data, dataset[0])
        g = node_pyg_to_jax(g_pyg)

        self.global_data = ()
        self.masks = {
            'train': jnp.array(g_pyg.train_mask),
            'val': jnp.array(g_pyg.val_mask),
            'test': jnp.array(g_pyg.test_mask)
        }

        # We store the node index in the samples
        self.samples = BatchedSample(
            data=g,
            target=jnp.array(g_pyg.y, dtype=jnp.int32)
        )
