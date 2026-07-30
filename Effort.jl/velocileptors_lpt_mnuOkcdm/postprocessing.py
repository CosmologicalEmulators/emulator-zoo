import jax.numpy as jnp
def postprocessing(input, output, D):
    return output * (jnp.exp(input[1]) * 1e-10 * D**2)
