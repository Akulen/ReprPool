from abc import abstractmethod, ABC
import equinox as eqx
from typing import Optional, Tuple
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ggnn.data.types import Graph
from ggnn.models.gnn.hooks import DefaultHook
from ggnn.models.gnn.uf import UFHook, GNHook

class GNNLayer(eqx.Module, ABC):
    hook: DefaultHook

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        layer_hook: str="DefaultHook",
        hook_args: dict={}
    ):
        match layer_hook:
            case "DefaultHook":
                hook_fn = DefaultHook
            case "UF":
                hook_fn = UFHook
            case "GN":
                hook_fn = GNHook
            case _:
                raise NotImplementedError(f"Unknown layer hook: {layer_hook}")
        self.hook = hook_fn(key, d_model, **hook_args)

    @abstractmethod
    def forward(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        pass

    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        return self.hook(self, key, state, graph, inference)

class PoolLayer(eqx.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        key: PRNGKeyArray,
        graph: Graph,
        mask: Optional[Bool[Array, "n"]]=None,
        inference: bool=False
    ) -> Float[Array, "d_model"]:
        pass

class ClusteringLayer(eqx.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        dropout: float=0.0
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        graph: Graph,
        inference: bool=False
    ) -> Tuple[Graph, eqx.nn.State]:
        pass
