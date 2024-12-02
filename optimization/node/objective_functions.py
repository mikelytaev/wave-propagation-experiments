import jax
import jax.numpy as jnp


@jax.jit
def bartlett(measure: jax.Array, replica: jax.Array) -> jax.Array:
    if measure.shape != replica.shape:
        raise ValueError(f'measure and replica must have the same shape {measure.shape} != {replica.shape}')
    w = replica / jnp.linalg.norm(replica)
    return abs(jnp.dot(w.conj(), measure) * jnp.dot(measure.conj(), w))


@jax.jit
def abs_bartlett(measure: jax.Array, replica: jax.Array) -> jax.Array:
    if measure.shape != replica.shape:
        raise ValueError(f'measure and replica must have the same shape {measure.shape} != {replica.shape}')
    return 1/(jnp.linalg.norm(measure)**2 - ((jnp.dot(abs(measure.conj()), abs(replica))) / jnp.linalg.norm(replica))**2)
