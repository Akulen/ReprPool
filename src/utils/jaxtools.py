import jax
import jax.numpy as jnp

def cond_print(cond, text, *args):
    def foo(c, *vals):
        if c:
            print(text.format(*vals))
            import os
            os._exit(0)
    jax.debug.callback(foo, cond, *args)

def count_distinct(arr):
    if arr.size == 0: return 0

    sorted_indices = jnp.lexsort(arr.T[::-1])
    sorted_arr = arr[sorted_indices]
    return 1 + jnp.sum(jnp.any(sorted_arr[1:] != sorted_arr[:-1], axis=-1))
