from cProfile import label

import jax

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.pyplot import title

from experimental.rwp_jax import RWPGaussSourceModel, RWPComputationalParams, PiecewiseLinearNProfileModel
import jax.numpy as jnp

from experiments.optimization.node.flax.utils import MLPNProfileModel
from experiments.optimization.node.npe.common import adam, RWPModel, add_noise, surface_based_duct_N, surface_duct_N, \
    elevated_duct_N, surface_based_duct2_N
from matplotlib.lines import Line2D

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

measure_surface_duct = model.apply_profile(surface_duct_N)
measure_surface_duct = add_noise(measure_surface_duct, 30)
measure_elevated_duct = model.apply_profile(elevated_duct_N)
measure_elevated_duct = add_noise(measure_elevated_duct, 30)
measure_surface_based_duct = model.apply_profile(surface_based_duct_N)
measure_surface_based_duct = add_noise(measure_surface_based_duct, 30)
measure_surface_based_duct2 = model.apply_profile(surface_based_duct2_N)
measure_surface_based_duct2 = add_noise(measure_surface_based_duct2, 30)

layers = [50]*4
surface_duct_opt_res = adam(model, measure_surface_duct, ground_truth_profile=surface_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=layers))
elevated_duct_opt_res = adam(model, measure_elevated_duct, ground_truth_profile=elevated_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=200.0, layers=layers))
surface_based_duct_opt_res = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=layers))
surface_based_duct2_opt_res = adam(model, measure_surface_based_duct2, ground_truth_profile=surface_based_duct2_N, learning_rate=0.05, gamma=1E-2, profile_model=MLPNProfileModel(z_max_m=200.0, layers=[100]*4))

z_grid_o = jnp.linspace(0, 250, 250)

f, ax = plt.subplots(1, 4, figsize=(6, 3.4), constrained_layout=True)
model.apply_profile(surface_duct_N)
ax[0].plot(model.env.M_profile(z_grid_o), z_grid_o, color='blue', label="Original")
model.apply_profile(surface_duct_opt_res.res_profile)
ax[0].plot(model.env.M_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[0].set_xlabel("M-profile")
ax[0].set_ylabel("Height (m)")
ax[0].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[0].grid(True)

model.apply_profile(elevated_duct_N)
ax[1].plot(model.env.M_profile(z_grid_o), z_grid_o, color='blue', label="Original")
model.apply_profile(elevated_duct_opt_res.res_profile)
ax[1].plot(model.env.M_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[1].set_xlabel("M-profile")
ax[1].set_yticklabels([])
ax[1].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[1].grid(True)

model.apply_profile(surface_based_duct_N)
ax[2].plot(model.env.M_profile(z_grid_o), z_grid_o, color='blue', label="Original")
model.apply_profile(surface_based_duct_opt_res.res_profile)
ax[2].plot(model.env.M_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[2].set_xlabel("M-profile")
ax[2].set_yticklabels([])
ax[2].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[2].grid(True)

model.apply_profile(surface_based_duct2_N)
ax[3].plot(model.env.M_profile(z_grid_o), z_grid_o, color='blue', label="Original")
model.apply_profile(surface_based_duct2_opt_res.res_profile)
ax[3].plot(model.env.M_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[3].set_xlabel("M-profile")
ax[3].set_yticklabels([])
ax[3].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[3].grid(True)

legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Original'),
                   Line2D([0], [0], color='red', lw=1, label='Inverted')]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)
f.tight_layout()

plt.show()

vis_model = RWPModel(params=RWPComputationalParams(
        max_range_m=102000,
        max_height_m=250,
        dx_m=100,
        dz_m=1),
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
f1 = vis_model.calc_field(surface_based_duct2_N)
f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-70, -10)
extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1]*1e-3, vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
im = ax.imshow(20*jnp.log10(abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax.set_xlabel("Range (km)")
ax.set_ylabel("Height (m)")
ax.grid(True)
ax.set_ylim([0, 250])
f.colorbar(im, ax=ax, shrink=0.9, location='right')
#f.tight_layout()
plt.show()

f2 = vis_model.calc_field(surface_based_duct2_opt_res.res_profile)
f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-70, -10)
extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1]*1e-3, vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
im = ax.imshow(20*jnp.log10(abs(f2+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax.set_xlabel("Range (km)")
ax.set_ylabel("Height (m)")
ax.grid(True)
ax.set_ylim([0, 250])
f.colorbar(im, ax=ax, shrink=0.9, location='right')
#f.tight_layout()
plt.show()


f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-30, 30)
extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1]*1e-3, vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
im = ax.imshow((20*jnp.log10(abs(f1+1e-16)) - 20*jnp.log10(abs(f2+1e-16))).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('bwr'))
ax.set_xlabel("Range (km)")
ax.set_ylabel("Height (m)")
ax.grid(True)
ax.set_ylim([0, 250])
f.colorbar(im, ax=ax, shrink=0.9, location='right')
#f.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(surface_duct_opt_res.loss_vals, label="Surface duct")
plt.plot(elevated_duct_opt_res.loss_vals, label="Elevated duct")
plt.plot(surface_based_duct2_opt_res.loss_vals, label="Surface based duct")
plt.plot(surface_based_duct_opt_res.loss_vals, label="Mixed duct")
plt.xlim([0, 250])
plt.xlabel("Iteration number")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(surface_duct_opt_res.ground_truth_errors, label="Surface duct")
plt.plot(elevated_duct_opt_res.ground_truth_errors, label="Elevated duct")
plt.plot(surface_based_duct2_opt_res.ground_truth_errors, label="Surface based duct")
plt.plot(surface_based_duct_opt_res.ground_truth_errors, label="Mixed duct")
plt.xlim([0, 250])
plt.xlabel("Iteration number")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()