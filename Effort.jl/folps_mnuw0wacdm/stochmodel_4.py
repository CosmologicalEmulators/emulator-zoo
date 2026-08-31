import jax.numpy as jnp

def StochModel(k):
    return jnp.stack((jnp.zeros_like(k), jnp.zeros_like(k)), axis=-1)
