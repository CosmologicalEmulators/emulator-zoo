import jax.numpy as jnp

def postprocessing(input, Pl):
    return Pl * (jnp.exp(input[0])*1e-10)**2
