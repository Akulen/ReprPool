import jax
import equinox as eqx
from typing import List, Optional, Tuple
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ggnn.models.utils import BatchNorm

class MLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    bnss: List[BatchNorm | eqx.nn.LayerNorm | None]
    dropout: eqx.nn.Dropout

    def __init__(
        self, key: PRNGKeyArray,
        dims: List[int],
        p: float=0.0,
        batch_norm: bool=True,
        plain_last: bool=True,
        v2: bool=False
    ):
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(in_features=d_in, out_features=d_out, key=key)
            for d_in, d_out, key in zip(dims[:-1], dims[1:], keys)
        ]
        if batch_norm:
            if v2:
                self.bnss = [
                    eqx.nn.LayerNorm(d)
                    for d in (dims[1:-1] if plain_last else dims[1:])
                ]
            else:
                self.bnss = [
                    BatchNorm(
                        input_size=d_out,
                        axis_name=('batch'),
                        momentum=0.9,
                        mode='batch'
                    )
                    for d_out in (dims[1:-1] if plain_last else dims[1:])
                ]
        else:
            self.bnss = [None] * (len(self.layers) - (1 if plain_last else 0))
        self.dropout = eqx.nn.Dropout(p)

    def __call__(
        self,
        x: Float[Array, "n d_in"],
        key: PRNGKeyArray,
        state: eqx.nn.State,
        mask: Optional[Int[Array, "n"]]=None,
        inference: bool=False
    ) -> Tuple[
        Float[Array, "n d_out"],
        eqx.nn.State
    ]:
        final_key, layer_keys = jax.random.split(key, 2)
        keys = jax.random.split(layer_keys, len(self.bnss))
        for layer, bns, layer_key in zip(self.layers, self.bnss, keys):
            x = jax.vmap(layer)(x)
            if isinstance(bns, BatchNorm):
                x, state = bns(x.T, state, mask=mask, inference=inference)
                x = jax.nn.relu(x.T)
            elif bns is None:
                x = jax.nn.relu(x)
            else:
                x = jax.vmap(bns)(x)
                x = jax.nn.relu(x)
            x = self.dropout(x, inference=inference, key=layer_key)

        if len(self.bnss) < len(self.layers):
            x = jax.vmap(self.layers[-1])(x)
            x = self.dropout(x, inference=inference, key=final_key)
        return x, state

