from functools import partial
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import cast, Any, Callable, List, Literal, Optional, Protocol, Tuple
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Real

from ggnn.data.types import BatchedData, BatchedSample, Graph
from ggnn.models.gnn.metrics import modularity, modularity_loss

Loss = Callable[
    [Float[Array, "bs"], Float[Array, "bs"], Optional[Bool[Array, "bs"]]],
    Float[Array, ""]
]

# y_pred is assumed to be pre-sigmoid
def binary_accuracy_loss(
    y_pred: Float[Array, "bs"],
    y_true: Int[Array, "bs"],
    mask: Optional[Bool[Array, "bs"]]=None
) -> Float[Array, ""]:
    if mask is None:
        mask = jnp.ones_like(y_true)
    return (mask * (y_pred > 0) == y_true).sum() / mask.sum()

def accuracy_loss(
    y_pred: Float[Array, "bs n_class"],
    y_true: Int[Array, "bs"],
    mask: Optional[Bool[Array, "bs"]]=None
) -> Float[Array, ""]:
    if y_pred.shape == y_true.shape:
        return binary_accuracy_loss(y_pred, y_true, mask)
    assert(len(y_pred.shape) == len(y_true.shape) + 1)
    if mask is None:
        mask = jnp.ones_like(y_true)
    return (mask * (y_pred.argmax(-1) == y_true)).sum() / mask.sum()

def CrossEntropy_loss(
    y_pred: Float[Array, "bs n_class"],
    y_true: Int[Array, "bs"],
    mask: Optional[Bool[Array, "bs"]]=None
) -> Float[Array, ""]:
    assert y_pred.shape[:-1] == y_true.shape
    if mask is None:
        mask = jnp.ones_like(y_true)
    l = optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true)
    return (l * mask).sum() / mask.sum()

def MSE_loss(
    y_pred: Float[Array, "bs"],
    y_true: Float[Array, "bs"],
    mask: Optional[Bool[Array, "bs"]]=None
) -> Float[Array, ""]:
    if mask is None:
        mask = jnp.ones_like(y_pred)
    return (mask * (y_pred - y_true.reshape((-1, 1))) ** 2).sum() / mask.sum()

def BCE_loss(
    y_pred: Float[Array, "bs"],
    y_true: Int[Array, "bs"],
    mask: Optional[Bool[Array, "bs"]]=None
) -> Float[Array, ""]:
    assert y_pred.shape == y_true.shape
    if mask is None:
        mask = jnp.ones_like(y_pred)
    l = optax.losses.sigmoid_binary_cross_entropy(y_pred, y_true)
    return (l * mask).sum() / mask.sum()

# modularity and modularity_loss imported from ggnn.models.gnn.metrics

LOSSES = {
    'accuracy': accuracy_loss,
    'CrossEntropy': CrossEntropy_loss,
    'BCE': BCE_loss,
    'modularity_loss': modularity_loss,
    'modularity': modularity
}

class StepFN(Protocol):
    def __call__(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        lr_scale: Float[Array, ""],
        batch: BatchedSample,
        data: Tuple[BatchedData, ...],
        key: PRNGKeyArray,
    ) -> Tuple[
        eqx.Module, eqx.nn.State, Any, Float[Array, "L"],
        Float[Array, "bs 2"]
    ]: ...

def make_masks(
    batch: BatchedSample
) -> Tuple[Bool[Array, "n"], Bool[Array, "m"]]:
    assert isinstance(batch.data, Graph)
    node_mask = jnp.arange(batch.data.f_node.shape[0]) < batch.data.n_nodes
    edge_mask = jnp.arange(batch.data.f_edge.shape[0]) < batch.data.n_edges
    if batch.mask is not None:
        node_mask = node_mask & batch.mask
        edge_mask = edge_mask & batch.mask
    return node_mask, edge_mask

class Trainer:
    def __init__(
        self,
        tx: optax.GradientTransformation,
        loss: Loss | Literal['BCE'] | List[Loss | Literal['BCE']]=BCE_loss,
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
        clustering_metric = False
        for l in loss:
            if isinstance(l, str):
                if l in LOSSES:
                    if l == 'modularity' or l == 'modularity_loss':
                        clustering_metric = True
                    l = LOSSES[l]
                elif l == 'dataset':
                    continue
                else:
                    raise NotImplementedError(f'Loss {l} not implemented')
            self.loss.append(l)
        self.clustering_metric = clustering_metric
        # print(f"Trainer initialized with clustering_metric={self.clustering_metric}")

    def global_shard(self, *data):
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.global_sharding)

    def batch_shard(self, *data):
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.batch_sharding)

    def compute_loss(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        batch: BatchedSample,
        data: Tuple[BatchedData, ...],
        key: PRNGKeyArray,
        inference: bool=False
    ) -> Tuple[
        Float[Array, ""],
        Tuple[Float[Array, "L"], Real[Array, "bs 2"], eqx.nn.State]
    ]:
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        keys = jax.random.split(key, batch_size)

        out_axes = (0, None)
        if self.clustering_metric:
            out_axes = (0, 0, None)
        res = jax.vmap(
            cast(Callable, model),
            in_axes=(0,) +(None,) * len(data) + (0,) + (None, None),
            out_axes=out_axes,
            axis_name='batch'
        )(keys, *data, batch.data, state, inference)
        node_mask = None
        edge_mask = None
        if self.clustering_metric:
            cluster, score, state = res
            outputs = cluster, score
            node_mask, edge_mask = jax.vmap(make_masks)(batch)
        else:
            outputs, state = res
        # outputs, state = jax.vmap(
        #     partial(cast(Callable, model), state, inference, *data),
        #     axis_name='batch'
        # )(batch.data, keys)
        loss = []
        for l in self.loss:
            if self.clustering_metric:
                new_loss = jax.vmap(l)(
                    outputs[0],
                    batch.data.edges, # type: ignore
                    outputs[1],
                    node_mask,
                    edge_mask
                ).sum()
                n_samples = outputs[0].shape[0]
                if batch.mask is not None:
                    n_samples = batch.mask.sum()
                new_loss = new_loss / n_samples
            else:
                new_loss = l(outputs, batch.target, batch.mask)
            loss.append(new_loss)
        targets = batch.target
        if self.clustering_metric:
            targets = jnp.stack([targets, targets]).T
        else:
            assert not isinstance(outputs, Tuple)
            if len(targets.shape) == 1 and len(outputs.shape) == 2:
                targets = jnp.tile(targets, (outputs.shape[-1], 1)).T
            targets = jnp.stack([targets, outputs]).T
        return loss[0], (
            jnp.stack(loss),
            targets,
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
        data: Tuple[BatchedData, ...],
        key: PRNGKeyArray,
    ) -> Tuple[
        eqx.Module,
        eqx.nn.State,
        Any,
        Float[Array, "L"],
        Float[Array, "bs 2"]
    ]:
        model, state, opt_state, data = self.global_shard(
            model, state, opt_state, data
        )
        batch = self.batch_shard(batch[None])[0] #type: ignore
        # jax.debug.inspect_array_sharding(batch, callback=print)
        grads, (loss, ys, state) = self.compute_loss_grad(
            model, state, batch, data, key
        )
        # jax.debug.print(
        #     '{x} {y}',
        #     x=jax.tree.leaves(jax.tree.map(
        #         lambda n: jnp.isnan(n).any(),
        #         grads
        #     )),
        #     y=jnp.isnan(ys).any()
        # )

        updates, opt_state = self.tx.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        updates = optax.tree.scale(lr_scale, updates)

        model = eqx.apply_updates(model, updates)
        model, state, opt_state = self.global_shard(
            model, state, opt_state
        )
        return model, state, opt_state, loss, ys

    @eqx.filter_jit(donate="all")
    def eval_step(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any,
        lr_scale: Float[Array, ""],
        batch: BatchedSample,
        data: Tuple[BatchedData, ...],
        key: PRNGKeyArray,
    ) -> Tuple[
        eqx.Module,
        eqx.nn.State,
        Any,
        Float[Array, "L"],
        Float[Array, "bs 2"]
    ]:
        del lr_scale
        model, state, opt_state, data = self.global_shard(
            model, state, opt_state, data
        )
        _, (loss, ys, _) = self.compute_loss(
            model, state, batch, data, key, inference=True
        )
        return model, state, opt_state, loss, ys
