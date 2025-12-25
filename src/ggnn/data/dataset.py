from abc import abstractmethod, ABC
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from ggnn.data.types import BatchedData, BatchedSample
from ggnn.training import Trainer

class Dataset(ABC):
    n_samples: int
    samples: BatchedSample
    global_data: Tuple[BatchedData, ...]
    trainer: Optional[Trainer]=None
    step_fn=None
    scan_fn=None
    compiled=False

    def __init__(
        self,
        batch_size: int=32,
        shuffle: bool=True,
        seed: int=42,
        verbose: bool=False
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def set_step_function(
        self,
        static: eqx.Module,
        step_fn: Callable[
            [
                eqx.Module, eqx.nn.State, Any,
                BatchedSample, Tuple[BatchedData, ...],
                PRNGKeyArray
            ],
            Tuple[eqx.Module, eqx.nn.State, Any, Float[Array, "L"]]
        ]
    ):
        def foo(
            carry: Tuple[
                eqx.Module, eqx.nn.State, Any,
                Tuple[BatchedData, ...], PRNGKeyArray
            ],
            batch: BatchedSample,
            static: eqx.Module
        ) -> Tuple[
            Tuple[
                eqx.Module, eqx.nn.State, Any,
                Tuple[BatchedData, ...], PRNGKeyArray
            ],
            Float[Array, "L"]
        ]:
            params, state, opt_state, data, key = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            model, state, opt_state, output = step_fn(
                model, state, opt_state, batch, data, subkey
            )
            params, _ = eqx.partition(model, eqx.is_array)
            return (params, state, opt_state, data, key), output
        self.step_fn = partial(foo, static=static)
        self.scan_fn = partial(jax.lax.scan, f=self.step_fn)
        self.compiled = False

    def precompile(
        self,
        params: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        key: PRNGKeyArray
    ):
        if self.compiled:
            return
        indices = jnp.arange(self.n_samples).reshape((-1, self.batch_size))
        batches = self.samples[indices]
        if self.trainer is not None:
            batches = self.trainer.batch_shard(batches)
        assert self.scan_fn is not None
        self.scan_fn = jax.jit(self.scan_fn).trace(
            init=(params, state, opt_state, self.global_data, key),
            xs=batches
        ).lower().compile()
        self.compiled = True

    def run_batches(
        self,
        params: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        key: PRNGKeyArray
    ) -> Tuple[
        Tuple[eqx.Module, eqx.nn.State, Any, PRNGKeyArray],
        Float[Array, "n_batches L"]
    ]:
        if self.step_fn is None:
            raise ValueError("step_fn is not set")
        assert self.scan_fn is not None

        indices = np.arange(self.n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = jnp.array(indices).reshape((-1, self.batch_size))

        batches = self.samples[indices]
        if self.trainer is not None:
            batches = self.trainer.batch_shard(batches)

        (params, state, opt_state, _, key), outputs = self.scan_fn(
            init=(params, state, opt_state, self.global_data, key),
            xs=batches
        )
        self.compiled = True
        return (params, state, opt_state, key), outputs

    def to_serializable(self):
        def to_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, jnp.ndarray):
                return np.array(x)
            return x

        serial = {
            'scalar': {
                'batch_size': self.batch_size,
                'n_samples': self.n_samples,
                'shuffle': self.shuffle,
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
        obj.step_fn = None
        obj.trainer = None
        obj.compiled = False
        return obj
