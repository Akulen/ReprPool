import equinox as eqx
from typing import TYPE_CHECKING, Tuple
from jaxtyping import PRNGKeyArray

if TYPE_CHECKING:
    from ggnn.models.gnn.types import GNNLayer
from ggnn.data.types import Graph

class DefaultHook(eqx.Module):
    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        **kwargs
    ):
        pass

    def __call__(
        self,
        layer: "GNNLayer",
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        return layer.forward(key, state, graph, inference)
