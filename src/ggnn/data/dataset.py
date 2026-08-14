from abc import ABC
from functools import partial
import random
import time
import torch
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import warnings
from typing import Any, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from ggnn.data.types import BatchedData, BatchedSample
from ggnn.training import StepFN, Trainer
from ggnn.node_training import NodeTrainer

class Dataset(ABC):
    n_samples: int
    samples: BatchedSample
    global_data: Tuple[BatchedData, ...]
    trainer: Optional[Trainer]=None
    step_fn=None
    scan_fn={}
    compiled=set()
    custom_loss_subloss=[]
    task='graph'
    data_type=''

    def __init__(
        self,
        n_class: int=2,
        shuffle: bool=True,
        seed: int=42,
        verbose: bool=False
    ):
        self.self_init()

        self.n_class = n_class
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def self_init(self):
        self.raw_scan_fn = None
        self.step_fn = None
        self.trainer = None

    def shard_data(self, trainer: Optional[Trainer|NodeTrainer]=None):
        if trainer is not None:
            self.trainer = trainer
        if self.trainer is not None:
            pass
            # self.cards, self.sets = self.trainer.global_shard(
            #     self.cards, self.sets
            # )
            # self.drafts = trainer.shard_data(self.drafts)

    def model_params(self):
        return {}

    def custom_loss(self, y_true, y_pred):
        raise NotImplementedError

    def set_step_function(
        self,
        static: eqx.Module,
        step_fn: StepFN
    ):
        def foo(
            carry: Tuple[
                eqx.Module, eqx.nn.State, Any, Float[Array, ""],
                Tuple[BatchedData, ...], PRNGKeyArray
            ],
            batch: BatchedSample,
            static: eqx.Module
        ) -> Tuple[
            Tuple[
                eqx.Module, eqx.nn.State, Any, Float[Array, ""],
                Tuple[BatchedData, ...], PRNGKeyArray
            ],
            Tuple[Float[Array, "L"], Float[Array, "bs 2"]]
        ]:
            params, state, opt_state, lr_scale, data, key = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            model, state, opt_state, output, ys = step_fn(
                model, state, opt_state, lr_scale, batch, data, subkey
            )
            params, _ = eqx.partition(model, eqx.is_array)
            return (params, state, opt_state, lr_scale, data, key), (output, ys)
        self.step_fn = partial(foo, static=static)
        self.raw_scan_fn = partial(jax.lax.scan, f=self.step_fn)
        self.scan_fn = {}

    def prepare_indices(self, batch_size: int, shuffle: bool=False):
        assert self.rng is not None
        if batch_size == -1:
            assert self.trainer is not None
            n_devices = self.trainer.n_devices
            batch_size = (
                self.n_samples + n_devices - 1
            ) // n_devices * n_devices
        if 2 * self.n_samples < batch_size:
            raise ValueError(
                f"Batch size is too large for the dataset (n={self.n_samples})"
            )
        indices = np.arange(self.n_samples)
        # if self.n_samples % batch_size != 0:
        #     n_extra = batch_size - (self.n_samples % batch_size)
        #     if shuffle:
        #         duplicate = self.rng.choice(
        #             np.arange(self.n_samples),
        #             size=n_extra,
        #             replace=False
        #         )
        #     else:
        #         duplicate = np.arange(n_extra)
        #     indices = np.concatenate((indices, duplicate), axis=0)
        if shuffle:
            self.rng.shuffle(indices)
        n_extra = self.n_samples % batch_size
        if n_extra == 0:
            last_idx = np.array([])
        else:
            indices, last_idx = indices[:-n_extra], indices[-n_extra:]
        indices = jnp.array(indices.reshape((-1, batch_size)))
        last_idx = jnp.array(last_idx)
        padded_idx = jnp.concatenate([
            last_idx
        ] + [
            last_idx[:1]
            for _ in range(batch_size - last_idx.shape[0])
        ])
        return indices, last_idx, padded_idx

    def make_batches(
        self, batch_size: int, shuffle: bool=False
    ) -> BatchedSample:
        indices, last_idx, padded = self.prepare_indices(batch_size, shuffle)
        batches = self.samples[indices]
        batch_size = batches.target.shape[1]
        if last_idx.shape[0] > 0:
            batches = eqx.tree_at(
                lambda bs: bs.mask,
                batches,
                jnp.full_like(batches.target, True, dtype=bool)
            )
            last_batch = eqx.tree_at(
                lambda bs: bs.mask,
                self.samples[padded],
                jnp.arange(batch_size) < last_idx.shape[0]
            )
            batches = jax.tree.map(
                lambda bs, l: jnp.concatenate((bs, l[None])),
                batches,
                last_batch
            )
        return batches

    def _precompile(
        self,
        params: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        key: PRNGKeyArray,
        batches: BatchedSample,
    ):
        assert self.raw_scan_fn is not None
        batch_size = batches.target.shape[1]
        if batch_size in self.scan_fn:
            return
        self.scan_fn[batch_size] = jax.jit(self.raw_scan_fn).trace(
            init=(params, state, opt_state, 1.0, self.global_data, key),
            xs=batches
        ).lower().compile()

    def precompile(
        self,
        params: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        key: PRNGKeyArray,
        batch_size: int=32,
        verbose: int=0
    ):
        st = time.time()
        batches = self.make_batches(batch_size)
        if self.trainer is not None:
            batches = self.trainer.batch_shard(batches)
        self._precompile(params, state, opt_state, key, batches)
        if verbose > 0:
            print(f'Precompiled {self.data_type} loop in {time.time() - st:.6f}s')

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
        if self.step_fn is None:
            raise ValueError("step_fn is not set")

        batches = self.make_batches(batch_size)
        batch_size = batches.target.shape[1]
        if self.trainer is not None:
            batches = self.trainer.batch_shard(batches)

        if batch_size not in self.scan_fn:
            warnings.warn(
                f"Running batch for uncompiled batch_size={batch_size}",
                RuntimeWarning
            )

        (
            (params, state, opt_state, _, _, key), (outputs, ys)
        ) = self.scan_fn[batch_size](
            init=(params, state, opt_state, lr_scale, self.global_data, key),
            xs=batches
        )
        ys = ys.reshape((-1, 2))
        outputs = outputs.T

        if custom_loss:
            y_true, y_pred = ys[:, 0], ys[:, 1]
            outputs = jnp.concatenate([
                outputs,
                jnp.tile(
                    jnp.array([self.custom_loss(y_true, y_pred)]),
                    (1, outputs.shape[1])
                )
            ], axis=0)

        return (params, state, opt_state, key), outputs

    def to_serializable(self):
        def to_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, jnp.ndarray):
                return np.array(x)
            return x

        serial = {
            'scalar': {
                'n_class': self.n_class,
                'n_samples': self.n_samples,
                'shuffle': self.shuffle,
                'task': self.task,
                'data_type': self.data_type
            },
            'eqx_data': {
                'samples': jax.tree.map(to_host, self.samples),
                'global_data': jax.tree.map(to_host, self.global_data),
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
            setattr(obj, k, jax.tree.map(from_host, v))
        bit_rng = np.random.PCG64()
        bit_rng.state = serial['rng']
        obj.rng = np.random.Generator(bit_rng)
        return obj
