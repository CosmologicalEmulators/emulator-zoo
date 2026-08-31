import jax

from biascombination import BiasCombination


JacobianBiasCombination = jax.jacfwd(BiasCombination)
