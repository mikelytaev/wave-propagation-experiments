import jax
import jax.numpy as jnp


@jax.jit
def bartlett(measure: jax.Array, replica: jax.Array) -> jax.Array:
    if measure.shape != replica.shape:
        raise ValueError(f'measure and replica must have the same shape {measure.shape} != {replica.shape}')
    w = replica / jnp.linalg.norm(replica)
    return abs(jnp.dot(w.conj(), measure) * jnp.dot(measure.conj(), w))
