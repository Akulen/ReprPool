import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import cast, Any, Callable, List, Optional, Protocol, Tuple
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ggnn.data.types import BatchedSample
from ggnn.training import LOSSES

def node_loss(
    loss_fn,
    y_pred: Float[Array, "n n_class"],
    y_true: Int[Array, "n"],
    cat_masks: Optional[Bool[Array, "C n"]]=None
) -> Float[Array, "C"]:
    return jax.vmap(
        loss_fn, in_axes=(None, None, 0)
    )(y_pred, y_true, cat_masks)

class NodeStepFN(Protocol):
    def __call__(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        lr_scale: Float[Array, ""],
        batch: BatchedSample,
        cat_masks: Bool[Array, "3 n"],
        key: PRNGKeyArray,
    ) -> Tuple[
        eqx.Module, eqx.nn.State, Any, Float[Array, "3 L"]
    ]: ...

class NodeTrainer:
    def __init__(
        self,
        tx: optax.GradientTransformation,
        loss: str | List[str]='BCE',
        n_devices: Optional[int]=None
    ):
        if n_devices is None:
            n_devices = jax.device_count()
        self.n_devices = n_devices
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
            if l in LOSSES:
                l = LOSSES[l]
            else:
                raise NotImplementedError(f'Loss {l} not implemented')
            self.loss.append(l)

    def global_shard(self, *data):
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.global_sharding)

    def batch_shard(self, *data):
        raise NotImplementedError
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.batch_sharding)

    def compute_loss(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        batch: BatchedSample,
        cat_masks: Bool[Array, "3 n"],
        key: PRNGKeyArray,
        inference: bool=False
    ) -> Tuple[
        Float[Array, ""],
        Tuple[Float[Array, "3 L"], eqx.nn.State]
    ]:
        logits, state = cast(Callable, model)(key, batch.data, state, inference)
        loss = []
        for l in self.loss:
            loss.append(node_loss(l, logits, batch.target, cat_masks))
        return loss[0][0], (
            jnp.stack(loss).T,
            state
        )

    @eqx.filter_jit(donate="all")
    def train_step(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        lr_scale: Float[Array, ""],
        batch: BatchedSample,
        cat_masks: Bool[Array, "3 n"],
        key: PRNGKeyArray,
    ) -> Tuple[
        eqx.Module,
        eqx.nn.State,
        Any,
        Float[Array, "3 L"],
    ]:
        # (
        #     model, state, opt_state, train_mask, val_mask, test_mask
        # ) = self.global_shard(
        #     model, state, opt_state, train_mask, val_mask, test_mask
        # )
        # batch = self.batch_shard(batch[None])[0] #type: ignore
        grads, (loss, state) = self.compute_loss_grad(
            model, state, batch, cat_masks, key
        )

        updates, opt_state = self.tx.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        updates = optax.tree.scale(lr_scale, updates)

        model = eqx.apply_updates(model, updates)
        model, state, opt_state = self.global_shard(
            model, state, opt_state
        )
        return model, state, opt_state, loss

    @eqx.filter_jit(donate='all')
    def eval_step(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        lr_scale: Float[Array, ""],
        batch: BatchedSample,
        cat_masks: Bool[Array, "3 n"],
        key: PRNGKeyArray,
    ) -> Tuple[
        eqx.Module,
        eqx.nn.State,
        Any,
        Float[Array, "3 L"],
    ]:
        _, (loss, _) = self.compute_loss(
            model, state, batch, cat_masks, key, True
        )
        return model, state, opt_state, loss
