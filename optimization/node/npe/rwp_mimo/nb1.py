from jax import numpy as jnp

from experimental.rwp_jax import PiecewiseLinearNProfileModel
from experiments.optimization.node.flax.utils import MLPNProfileModel
from experiments.optimization.node.npe.rwp_mimo.common import MultiAngleRWPModel, add_noise, adam

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

elevated_duct_N = PiecewiseLinearNProfileModel(jnp.array([0, 100, 150, 300]), jnp.array([20, 20, 0, 0]))
freq_hz = 1E9
max_range_m = 5000
measure_points_z = [10, 11, 12]
angles_deg = [-5, 0.0, 3]
src_height_m = 50
measure_model = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz,
    N_profile=elevated_duct_N,
    angles_deg=angles_deg,
    max_range_m=max_range_m,
    src_height_m=src_height_m
)

measure = add_noise(measure_model.compute(), 30)

profile_model=MLPNProfileModel(z_max_m=200.0, layers=[50]*4)
model = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz,
    #N_profile=profile_model,
    angles_deg=angles_deg,
    max_range_m=max_range_m,
    src_height_m=src_height_m
)
elevated_duct_opt_res = adam(model, measure, learning_rate=0.05, gamma=1E-3, profile_model=profile_model)


z_grid_o = jnp.linspace(0, 250, 250)

f, ax = plt.subplots(1, 4, figsize=(6, 3.6), constrained_layout=True)
model.set_N_profile(elevated_duct_N)
ax[0].plot(model.env.M_profile(z_grid_o), z_grid_o, color='blue', label="Original")
model.set_N_profile(elevated_duct_opt_res.res_profile)
ax[0].plot(model.env.M_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[0].set_title("(a)")
ax[0].set_xlabel("M-profile")
ax[0].set_ylabel("Height (m)")
ax[0].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[0].grid(True)
plt.savefig('1.eps')
#plt.show()