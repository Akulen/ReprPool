import jax
import equinox as eqx
from typing import Optional, Self
from jaxtyping import Array, Bool, Int, Real, Shaped

class BatchedData(eqx.Module):
    def __getitem__(self, idx: Int[Array, "*bs"]) -> Self:
        return jax.tree.map(lambda x: x[idx], self)

class BatchedSample(BatchedData):
    data: BatchedData
    target: Shaped[Array, "bs"]
    mask: Optional[Bool[Array, "bs"]]=None

class Graph(BatchedData):
    n_nodes: Int[Array, "bs"]
    n_edges: Int[Array, "bs"]
    f_node: Real[Array, "bs max_nodes d_node"]
    f_edge: Real[Array, "bs max_edges d_edge"]
    f_graph: Real[Array, "bs d_graph"]
    edges: Int[Array, "bs max_edges 2"]
