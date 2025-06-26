from experimental.rwp_jax import *
import jax.numpy as jnp
from utils import *


p1 = PiecewiseLinearNProfileModel(jnp.array([0, 50, 100]), jnp.array([10, 0, 0]))
p2 = PiecewiseLinearNProfileModel(jnp.array([0, 50, 75, 100]), jnp.array([10, 30, 0, 0]))
p3 = PiecewiseLinearNProfileModel(jnp.array([0, 50]), jnp.array([10, 10]))

profiles = [p1 * (1-t) + p2 * t for t in jnp.linspace(0, 1, 20)]
profiles += [p2 * (1-t) + p3 * t for t in jnp.linspace(0, 1, 20)]

z_grid = jnp.linspace(0, 150, 251)

inv_model = RWPModel(params=RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)