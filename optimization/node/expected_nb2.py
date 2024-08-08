import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from experiments.optimization.node.expected import expected_value_jacfwd, expected_value_jet, expected_value_quad


# @jax.jit
# def func(z):
#     return (jnp.sin(z)*jnp.exp(jnp.cos(z) / z) + jnp.cos(z)) / jnp.exp(10*(z + 1))


# @jax.jit
# def func(z):
#     return jnp.sin(z) / (z + 10) + jnp.sqrt(10 + jnp.cos(z + 1 / (z + 100)))


@jax.jit
def func(z):
    return jnp.sin(z / 1.5)


v_d = expected_value_jacfwd(func, 5.0, 1.5, 8)
v_j = expected_value_jet(func, 5.0, 1.5, 40)
v_q = expected_value_quad(func, 5.0, 1.5, 5001)

print(f'v_d: {v_d}, v_j: {v_j}, v_q: {v_q}')
