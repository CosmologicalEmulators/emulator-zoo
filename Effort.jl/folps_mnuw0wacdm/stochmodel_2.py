import jax.numpy as jnp

def StochModel(k):
    return jnp.stack((jnp.zeros_like(k), 2.0 * k**2 / 3.0), axis=-1)
