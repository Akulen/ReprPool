from functools import partial
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import cast, Any, Callable, List, Literal, Optional, Tuple
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ggnn.data.types import BatchedData, BatchedSample

Loss = Callable[
    [Float[Array, "bs"], Float[Array, "bs"], Optional[Bool[Array, "bs"]]],
    Float[Array, ""]
]

def MSE_loss(
    y_pred: Float[Array, "bs"],
    y_true: Float[Array, "bs"],
    mask: Optional[Bool[Array, "bs"]]=None
) -> Float[Array, ""]:
    if mask is None:
        mask = jnp.ones_like(y_pred)
    return (mask * (y_pred - y_true.reshape((-1, 1))) ** 2).sum() / mask.sum()

LOSSES = {
    'MSE': MSE_loss
}

class Trainer:
    def __init__(
        self,
        tx: optax.GradientTransformation,
        loss: Loss | Literal['MSE'] | List[Loss | Literal['MSE']]=MSE_loss,
        n_devices: Optional[int]=None
    ):
        if n_devices is None:
            n_devices = jax.device_count()
        self.mesh = jax.make_mesh((n_devices,), ("batch",))
        self.global_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec()
        )
        self.batch_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, "batch")
        )
        self.compute_loss_grad = eqx.filter_grad(
            self.compute_loss, has_aux=True
        )
        self.tx = tx
        if not isinstance(loss, list):
            loss = [loss]
        self.loss = []
        for l in loss:
            if isinstance(l, str):
                if l in LOSSES:
                    l= LOSSES[l]
                else:
                    raise NotImplementedError(f'Loss {l} not implemented')
            self.loss.append(l)

    def global_shard(self, *data):
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.global_sharding)

    def batch_shard(self, *data):
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.batch_shard)

    def compute_loss(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        batch: BatchedSample,
        data: Tuple[BatchedData, ...],
        key: PRNGKeyArray,
        inference: bool=False
    ) -> Tuple[Float[Array, ""], Tuple[Float[Array, "L"], eqx.nn.State]]:
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        keys = jax.random.split(key, batch_size)
        outputs, state = jax.vmap(
            partial(cast(Callable, model), state, inference, *data)
        )(batch, keys)
        loss = []
        for l in self.loss:
            loss.append(l(outputs, batch.target, batch.mask))
        return loss[0], (jnp.stack(loss), state)

    @eqx.filter_jit(donate="all")
    def train_step(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        batch: BatchedSample,
        data: Tuple[BatchedData, ...],
        key: PRNGKeyArray
    ) -> Tuple[eqx.Module, eqx.nn.State, Any, Float[Array, "L"]]:
        model, state, opt_state, data = self.global_shard(
            model, state, opt_state, data
        )
        batch = self.batch_shard(batch)
        grads, (loss, state) = self.compute_loss_grad(
            model, state, batch, data, key
        )

        updates, opt_state = self.tx.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )

        model = eqx.apply_updates(model, updates)
        model, state, opt_state = self.global_shard(
            model, state, opt_state
        )
        return model, state, opt_state, loss

    @eqx.filter_jit(donate="all")
    def eval_step(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        batch: BatchedSample,
        data: Tuple[BatchedData, ...],
        key: PRNGKeyArray
    ) -> Tuple[eqx.Module, eqx.nn.State, Any, Float[Array, "L"]]:
        model, state, opt_state, data = self.global_shard(
            model, state, opt_state, data
        )
        _, (loss, state) = self.compute_loss(
            model, state, batch, data, key, inference=True
        )
        return model, state, opt_state, loss
