import jax
import jax.numpy as jnp
from typing import Optional
from jaxtyping import Array, Bool, Float, Int

def modularity(
    clusters: Int[Array, "n"],
    edges: Int[Array, "m 2"],
    scores: Float[Array, "n"],
    node_mask: Optional[Bool[Array, "n"]]=None,
    edge_mask: Optional[Bool[Array, "m"]]=None
) -> Float[Array, ""]:
    del scores
    if node_mask is None:
        node_mask = jnp.ones_like(clusters)
    if edge_mask is None:
        edge_mask = jnp.ones(edges.shape[0], dtype=jnp.bool_)
    
    m = edge_mask.sum().astype(jnp.float32)
    m = jnp.maximum(m, 1.0)
    n = clusters.shape[0]

    # Degree of each node
    degree = jax.ops.segment_sum(
        edge_mask.astype(jnp.float32),
        edges[:, 0],
        num_segments=n
    )

    # Sum of (degree_u * degree_v) for all pairs (u, v) in the same cluster
    # This is equal to sum_c (sum_{i in c} degree_i)^2
    cluster_degree_sum = jax.ops.segment_sum(degree, clusters, num_segments=n)
    sum_deg_square = jnp.sum(cluster_degree_sum**2)

    # Number of edges within the same cluster
    # edges[:, 0] and edges[:, 1] are sources and targets. 
    # Check if they belong to the same cluster.
    intra_cluster_mask = (clusters[edges[:, 0]] == clusters[edges[:, 1]]) & edge_mask
    lc = intra_cluster_mask.sum().astype(jnp.float32)

    return (lc / m) - (sum_deg_square / (m**2))

def modularity_loss(
    clusters: Int[Array, "n"],
    edges: Int[Array, "m 2"],
    scores: Float[Array, "n"],
    node_mask: Optional[Bool[Array, "n"]]=None,
    edge_mask: Optional[Bool[Array, "m"]]=None
) -> Float[Array, ""]:
    if node_mask is None:
        node_mask = jnp.ones_like(clusters)
    if edge_mask is None:
        edge_mask = jnp.ones(edges.shape[0], dtype=jnp.bool_)
    
    m = edge_mask.sum().astype(jnp.float32)
    m = jnp.maximum(m, 1.0)
    n = clusters.shape[0]

    # scores_p acts as a soft weight for each node in its cluster
    scores_p = scores + 1.0

    # Weighted degree: d_i = sum_j A_ij
    degree = jax.ops.segment_sum(
        edge_mask.astype(jnp.float32),
        edges[:, 0],
        num_segments=n
    )

    # Soft cluster degree sum
    # sum_i in cluster scores_p[i] * degree_i
    weighted_degree = degree * scores_p
    cluster_weighted_degree_sum = jax.ops.segment_sum(weighted_degree, clusters, num_segments=n)
    
    # The O(N^2) term for K was: sum_c sum_{i,j in c} scores_p[i] * (deg_i * deg_j / m) * scores_p[j]
    # = (1/m) * sum_c (sum_{i in c} scores_p[i] * deg_i)^2
    term_K = jnp.sum(cluster_weighted_degree_sum**2) / m

    # The O(N^2) term for A was: sum_{i,j in same cluster} scores_p[i] * A_ij * scores_p[j]
    # = sum_{edges (u,v) where clusters[u] == clusters[v]} scores_p[u] * scores_p[v]
    u, v = edges[:, 0], edges[:, 1]
    intra_cluster_edge = (clusters[u] == clusters[v]) & edge_mask
    term_A = jnp.sum(jnp.where(intra_cluster_edge, scores_p[u] * scores_p[v], 0.0))

    return -(term_A - term_K) / m
