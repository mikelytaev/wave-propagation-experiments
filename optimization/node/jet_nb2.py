import jax
import jax.numpy as jnp

from jax.experimental import jet


h0 = 0, 0


def func(z0, z1):
    return jnp.sin(z0 + z1)


f0, (f1, f2) = jet.jet(func, h0, ((1.0, 0.0), (1.0, 0.0)))
print(f0,  func(*h0))

print(f1)
print(f2)
