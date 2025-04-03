import jax

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from experimental.rwp_jax import RWPGaussSourceModel, RWPComputationalParams, PiecewiseLinearNProfileModel
import jax.numpy as jnp

from experiments.optimization.node.flax.utils import MLPNProfileModel
from experiments.optimization.node.npe.common import adam, RWPModel, add_noise, PLNPM, surface_based_duct_N

jax.config.update("jax_enable_x64", True)

model = RWPModel(params=RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*18,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
)
measure = model.apply_profile(surface_based_duct_N)
measure = add_noise(measure, 30)

f1 = model.calc_field(surface_based_duct_N)
f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-50, -10)
extent = (model.fwd_model.x_output_grid()[0], model.fwd_model.x_output_grid()[-1]*1e-3, model.fwd_model.z_output_grid()[0], model.fwd_model.z_output_grid()[-1])
ax.imshow(20*jnp.log10(abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
f.tight_layout()
plt.show()

linear_opt_res = adam(model, measure, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=PLNPM(z_grid_m=jnp.linspace(0, 100, 50), N_vals=jnp.linspace(0, 1, 50)))
mlp_opt_res = adam(model, measure, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[50]*4))

z_grid_o = jnp.linspace(0, 250, 250)
plt.plot(surface_based_duct_N.M_Profile(z_grid_o), z_grid_o)
plt.plot(linear_opt_res.res_profile.M_Profile(z_grid_o), z_grid_o)
plt.plot(mlp_opt_res.res_profile.M_Profile(z_grid_o), z_grid_o)
plt.ylim([z_grid_o[0], z_grid_o[-1]])
plt.xlabel("M-profile")
plt.ylabel("Height (m)")
plt.show()

f, ax = plt.subplots(1, 2, figsize=(6, 3.2), constrained_layout=True)
ylim = [0.01, 100]
ax[1].plot(linear_opt_res.ground_truth_errors, label="Piecewise linear", color="blue", linestyle="-")
ax[1].plot(mlp_opt_res.ground_truth_errors[0:250], label="MLP (l=4; w=50)", color="red", linestyle="-")
ax[1].set_xlim([1, 10000])
ax[1].set_ylim(ylim)
ax[1].set_xlabel("Iteration number")
ax[1].set_ylabel("Rel. error")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].grid(True)
ax[1].legend()


#f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
ylim = [0.01, 1000]
ax[0].plot(linear_opt_res.loss_vals, label="Piecewise linear", color="blue", linestyle="-")
ax[0].plot(mlp_opt_res.loss_vals[0:250], label="MLP (l=4; w=50)", color="red", linestyle="-")
ax[0].set_xlim([1, 10000])
ax[0].set_ylim(ylim)
ax[0].set_xlabel("Iteration number")
ax[0].set_ylabel("Loss")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].grid(True)
ax[0].legend()