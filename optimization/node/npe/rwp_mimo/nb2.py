import numpy as np
from jax import numpy as jnp

from experimental.rwp_jax import PiecewiseLinearNProfileModel
from experiments.optimization.node.npe.rwp_mimo.common import MultiAngleRWPModel

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

freq_hz = 3E9
max_range_m = 5000
measure_points_z = [10, 11, 12]
angles_deg = np.linspace(-10, 10, 30)
src_height_m = 125
measure_model = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz,
    angles_deg=angles_deg,
    max_range_m=max_range_m,
    src_height_m=src_height_m
)

measure1 = measure_model.compute()
elevated_duct_N = PiecewiseLinearNProfileModel(
    jnp.array([0, 100, 150, 200, 300]),
    jnp.array([0, 0, 20, 0, 0])
)
measure_model.set_N_profile(elevated_duct_N)
measure2 = measure_model.compute()
abs_measure1, abs_measure2 = [], []
for m in measure1:
    abs_measure1 += [jnp.real(m[1])]
for m in measure2:
    abs_measure2 += [jnp.real(m[1])]

plt.plot(angles_deg, abs_measure1)
plt.plot(angles_deg, abs_measure2)
plt.show()