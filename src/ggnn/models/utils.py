from collections.abc import Hashable, Sequence
import warnings
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx
from equinox._misc import default_floating_dtype
from equinox.nn._misc import named_scope
from typing import List, Literal, Optional
from jaxtyping import Array, Bool, Float, Int, Shaped, PRNGKeyArray

class AtomEncoder(eqx.Module):
    embedders: List[eqx.nn.Embedding]

    def __init__(
        self, key: PRNGKeyArray,
        d_model: int
    ):
        full_atom_feature_dims = get_atom_feature_dims()
        keys = jax.random.split(key, len(full_atom_feature_dims))
        self.embedders = [
            #TODO: Use xavier_uniform to init?
            eqx.nn.Embedding(dim, d_model, key=emb_key)
            for dim, emb_key in zip(full_atom_feature_dims, keys)
        ]

    def __call__(self, x: Int[Array, "d_node_in"]) -> Float[Array, "d_model"]:
        assert(x.shape[0] == len(self.embedders))
        y = jnp.zeros(self.embedders[0].embedding_size)
        for i, layer in enumerate(self.embedders):
            y += layer(x[i])
        return y

class BondEncoder(eqx.Module):
    embedders: List[eqx.nn.Embedding]

    def __init__(
        self, key: PRNGKeyArray,
        d_model: int
    ):
        full_bond_feature_dims = get_bond_feature_dims()
        keys = jax.random.split(key, len(full_bond_feature_dims))
        self.embedders = [
            eqx.nn.Embedding(dim, d_model, key=emb_key)
            for dim, emb_key in zip(full_bond_feature_dims, keys)
        ]

    def __call__(self, x: Int[Array, "d_edge_in"]) -> Float[Array, "d_model"]:
        assert(x.shape[0] == len(self.embedders))
        y = jnp.zeros(self.embedders[0].embedding_size)
        for i, layer in enumerate(self.embedders):
            y += layer(x[i])
        return y

def masked_mean(
    x: Float[Array, "n"],
    mask: Optional[Bool[Array, "n"]]=None,
) -> Float[Array, ""]:
    if mask is None:
        return jnp.mean(x)
    return jnp.sum(x * mask) / jnp.sum(mask)

def masked_var(
    x: Float[Array, "n"],
    mask: Optional[Bool[Array, "n"]]=None,
    mean: Optional[Float[Array, ""]]=None,
    ddof: int=0
) -> Float[Array, ""]:
    if mask is None:
        return jnp.var(x, ddof=ddof)
    if mean is None:
        mean = masked_mean(x, mask)
    return jnp.sum((x - mean) * jnp.conj(x - mean) * mask) / (jnp.sum(mask) - ddof)

def layer_norm(
    x: Float[Array, "n"],
    mask: Optional[Bool[Array, "n"]]=None,
    eps: float=1e-5,
    ddof: int=0,
) -> Float[Array, "n"]:
    mean = masked_mean(x, mask)
    variance = masked_var(x, mask, mean, ddof=ddof)
    variance = jnp.maximum(0.0, variance)
    inv = jax.lax.rsqrt(variance + eps)
    if mask is None:
        out = (x - mean) * inv
    else:
        out = jnp.where(
            mask,
            (x - mean) * inv,
            x
        )
    return out

class BatchNorm(eqx.nn._sequential.StatefulLayer):
    weight: Float[Array, "input_size"] | None
    bias: Float[Array, "input_size"] | None
    ema_first_time_index: None | eqx.nn._stateful.StateIndex[Bool[Array, ""]]
    ema_state_index: (
        None | eqx.nn._stateful.StateIndex[tuple[Float[Array, "input_size"], Float[Array, "input_size"]]]
    )
    batch_counter: None | eqx.nn._stateful.StateIndex[Int[Array, ""]]
    batch_state_index: (
        None | eqx.nn._stateful.StateIndex[tuple[Float[Array, "input_size"], Float[Array, "input_size"]]]
    )
    axis_name: Hashable | Sequence[Hashable]
    inference: bool
    input_size: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    channelwise_affine: bool = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    mode: Literal["ema", "batch"] = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        axis_name: Hashable | Sequence[Hashable],
        eps: float = 1e-5,
        channelwise_affine: bool = True,
        momentum: float = 0.99,
        inference: bool = False,
        dtype=None,
        mode: Literal["ema", "batch", "legacy"] = "legacy",
    ):
        """**Arguments:**

        - `input_size`: The number of channels in the input array.
        - `axis_name`: The name of the batch axis to compute statistics over, as passed
            to `axis_name` in `jax.vmap` or `jax.pmap`. Can also be a sequence (e.g. a
            tuple or a list) of names, to compute statistics over multiple named axes.
        - `eps`: Value added to the denominator for numerical stability.
        - `channelwise_affine`: Whether the module has learnable channel-wise affine
            parameters.
        - `momentum`: The rate at which to update the running statistics. Should be a
            value between 0 and 1 exclusive.
        - `inference`: If `False` then the batch means and variances will be calculated
            and used to update the running statistics. If `True` then the running
            statistics are directly used for normalisation. This may be toggled with
            [`equinox.nn.inference_mode`][] or overridden during
            [`equinox.nn.BatchNorm.__call__`][].
        - `dtype`: The dtype to use for the running statistics and the weight and bias
            if `channelwise_affine` is `True`. Defaults to either
            `jax.numpy.float32` or `jax.numpy.float64` depending on whether JAX is in
            64-bit mode.
        - `mode`: The variant of batch norm to use, either 'ema' or 'batch'.
        """
        if mode == "legacy":
            mode = "ema"
            warnings.warn(
                "When `eqx.nn.BatchNorm(..., mode=...)` is unspecified it defaults to "
                "'ema', for backward compatibility. This typically has a performance "
                "impact, and for new code the user is encouraged to use 'batch' "
                "instead. See `https://github.com/patrick-kidger/equinox/issues/659`."
            )
        if mode not in {"ema", "batch"}:
            raise ValueError("Invalid mode, must be 'ema' or 'batch'.")
        self.mode = mode
        dtype = default_floating_dtype() if dtype is None else dtype
        if channelwise_affine:
            self.weight = jnp.ones((input_size,), dtype=dtype)
            self.bias = jnp.zeros((input_size,), dtype=dtype)
        else:
            self.weight = None
            self.bias = None
        if mode == "ema":
            self.ema_first_time_index = eqx.nn._stateful.StateIndex(jnp.array(True))
            init_buffers = (
                jnp.empty((input_size,), dtype=dtype),
                jnp.empty((input_size,), dtype=dtype),
            )
            self.ema_state_index = eqx.nn._stateful.StateIndex(init_buffers)
            self.batch_counter = None
            self.batch_state_index = None
        else:
            self.batch_counter = eqx.nn._stateful.StateIndex(jnp.array(0))
            init_hidden = (
                jnp.zeros((input_size,), dtype=dtype),
                jnp.ones((input_size,), dtype=dtype),
            )
            self.batch_state_index = eqx.nn._stateful.StateIndex(init_hidden)
            self.ema_first_time_index = None
            self.ema_state_index = None
        self.inference = inference
        self.axis_name = axis_name
        self.input_size = input_size
        self.eps = eps
        self.channelwise_affine = channelwise_affine
        self.momentum = momentum

    @named_scope("eqx.nn.BatchNorm")
    def __call__(
        self,
        x: Shaped[Array, "*d"],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray | None = None,
        mask: Bool[Array, "*d"] | None = None,
        inference: bool | None = None,
    ) -> tuple[Array, eqx.nn.State]:
        """**Arguments:**

        - `x`: A JAX array of shape `(input_size, dim_1, ..., dim_N)`.
        - `state`: An [`equinox.nn.State`][] object (which is used to store the
            running statistics).
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        - `inference`: As per [`equinox.nn.BatchNorm.__init__`][]. If
            `True` or `False` then it will take priority over `self.inference`. If
            `None` then the value from `self.inference` will be used.

        **Returns:**

        A 2-tuple of:

        - A JAX array of shape `(input_size, dim_1, ..., dim_N)`.
        - An updated state object (storing the updated running statistics).

        **Raises:**

        A `NameError` if no `vmap`s are placed around this operation, or if this vmap
        does not have a matching `axis_name`.
        """
        del key

        if inference is None:
            inference = self.inference

        if mask is None:
            mask = jnp.ones_like(x)

        def masked_mean(y, mask, axis_name):
            mean = (y * mask).sum()
            mean = lax.psum(mean, axis_name)
            total = mask.sum()
            total = lax.psum(total, axis_name)
            mean = mean / total
            return mean

        def masked_var(y, mask, axis_name, mean=None, ddof=0):
            if mean is None:
                mean = masked_mean(y, mask, axis_name)
            var = ((y - mean) * jnp.conj(y - mean) * mask).sum()
            var = lax.psum(var, axis_name)
            total = mask.sum()
            total = lax.psum(total, axis_name)
            var = var / (total - ddof)
            return var

        def _stats(y, ddof=0):
            mean = masked_mean(y, mask, self.axis_name)
            var = masked_var(y, mask, self.axis_name, mean, ddof)
            var = jnp.maximum(0.0, var)
            return mean, var

        def _norm(y, m, v, w, b):
            out = (y - m) / jnp.sqrt(v + self.eps)
            if self.channelwise_affine:
                out = out * w + b
            return out

        if self.mode == "ema":
            assert self.ema_first_time_index is not None
            assert self.ema_state_index is not None
            if inference:
                mean, var = state.get(self.ema_state_index)
            else:
                first_time = state.get(self.ema_first_time_index)
                state = state.set(self.ema_first_time_index, jnp.array(False))
                batch_mean, batch_var = jax.vmap(_stats)(x)
                running_mean, running_var = state.get(self.ema_state_index)
                momentum = self.momentum
                mean = (1 - momentum) * batch_mean + momentum * running_mean
                var = (1 - momentum) * batch_var + momentum * running_var
                # since jnp.array(0) == False
                mean = lax.select(first_time, batch_mean, mean)
                var = lax.select(first_time, batch_var, var)
                state = state.set(self.ema_state_index, (mean, var))
        else:
            assert self.batch_state_index is not None
            assert self.batch_counter is not None
            counter = state.get(self.batch_counter)
            hidden_mean, hidden_var = state.get(self.batch_state_index)
            if inference:
                # Zero-debias approach: mean = hidden_mean / (1 - momentum^counter)
                # For simplicity we do the minimal version here (no warmup).
                # scale = jnp.where(counter == 0, 1, 1 - self.momentum**counter)
                scale = 1
                mean = hidden_mean / scale
                var = hidden_var / scale
            else:
                ub_mean, ub_var = jax.vmap(eqx.Partial(_stats, ddof=1))(x)
                mean, var = jax.vmap(_stats)(x)
                new_counter = counter + 1
                new_hidden_mean = hidden_mean * self.momentum + ub_mean * (
                    1 - self.momentum
                )
                new_hidden_var = hidden_var * self.momentum + ub_var * (1 - self.momentum)
                state = state.set(self.batch_counter, new_counter)
                state = state.set(
                    self.batch_state_index, (new_hidden_mean, new_hidden_var)
                )

        out = jax.vmap(_norm)(x, mean, var, self.weight, self.bias)
        return out, state

class GlorotLinear(eqx.nn.Linear):
    def __init__(self, in_features, out_features, use_bias=True, dtype=None, *args, key: PRNGKeyArray):
        key, subkey = jax.random.split(key)
        # in_features: int | Literal["scalar"],
        # out_features: int | Literal["scalar"],
        # use_bias: bool = True,
        # dtype=None,
        # *,
        # key: PRNGKeyArray,
        super().__init__(
            in_features, out_features, use_bias=use_bias, dtype=dtype,
            key=subkey
        )
        key, subkey = jax.random.split(key)
        self.weight = jax.nn.initializers.glorot_uniform()(
            subkey, self.weight.shape
        )
        if self.bias is not None:
            self.bias = jnp.zeros(self.bias.shape, dtype=jnp.float32)

def get_activation(activation: str):
    match activation:
        case 'relu':
            return jax.nn.relu
        case 'elu':
            return jax.nn.elu
        case 'none':
            return lambda x: x
    raise ValueError(f'Unknown activation {activation}')

