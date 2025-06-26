import jax
from jax import random, numpy as jnp, lax

from experimental.rwp_jax import PiecewiseLinearNProfileModel, EvaporationDuctModel


def surface_duct_N_profile_generator(grid, key=random.PRNGKey(17031993)):
    keys = random.split(key, 2)
    h = jax.random.uniform(keys[0], minval=20, maxval=150)
    p = PiecewiseLinearNProfileModel(
        jnp.array([0, h, h + 20]),
        jnp.array([jax.random.uniform(keys[1], minval=0.0, maxval=30), 0, 0]),
    )
    return p(grid)


def trilinear_duct_N_profile_generator(grid, key=random.PRNGKey(17031993)):
    keys = random.split(key, 5)
    z1 = jax.random.uniform(keys[0], minval=30, maxval=70)
    z2 = jax.random.uniform(keys[1], minval=75, maxval=125)
    n1 = jax.random.uniform(keys[2], minval=15, maxval=25)
    n2mn1 = jax.random.uniform(keys[3], minval=1, maxval=30)
    n0 = jax.random.uniform(keys[4], minval=-5, maxval=20)
    p = PiecewiseLinearNProfileModel(
        jnp.array([0, z1, z2, z2 + 20]),
        jnp.array([n0, n1, n1 - n2mn1, n1 - n2mn1]) - (n1 - n2mn1),
    )
    return p(grid)


def evaporation_duct_N_profile_generator(grid, key=random.PRNGKey(17031993)):
    key = random.split(key, 1)
    edm = EvaporationDuctModel(height_m=jax.random.uniform(key[0], minval=5, maxval=45))
    return edm(grid)


def N_profile_generator(grid, key=random.PRNGKey(17031993)):
    keys = random.split(key, 2)
    selector = jax.random.randint(keys[0], 1, minval=0, maxval=3)[0]
    return lax.cond(
        selector == 0,
        lambda: evaporation_duct_N_profile_generator(grid, keys[1]),
        lambda: lax.cond(
            selector == 1,
            lambda: surface_duct_N_profile_generator(grid, keys[1]),
            lambda: trilinear_duct_N_profile_generator(grid, keys[1])
        )
    )
