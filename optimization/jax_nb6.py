import jax
import jax.numpy as jnp
from jax import random


def f():
    key = random.PRNGKey(0)
    while True:
        key = random.split(key, 1)[0]
        yield random.normal(key, shape=(3,))


