import jax.numpy as jnp

def StochModel(k):
    return jnp.stack((jnp.ones_like(k), k**2 / 3.0), axis=-1)
