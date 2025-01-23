from cProfile import label

import jax

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.pyplot import title

from experimental.helmholtz_jax import PiecewiseLinearWaveSpeedModel
from experimental.rwp_jax import RWPGaussSourceModel, RWPComputationalParams, PiecewiseLinearNProfileModel
import jax.numpy as jnp

from experimental.uwa_jax import UWAComputationalParams
from experiments.optimization.node.flax.utils import MLPNProfileModel, MLPWaveSpeedModel
from experiments.optimization.node.npe.common import adam, RWPModel, add_noise, surface_based_duct_N, surface_duct_N, \
    elevated_duct_N, surface_based_duct2_N, UWAModel
from matplotlib.lines import Line2D

jax.config.update("jax_enable_x64", True)

model = UWAModel(params=UWAComputationalParams(
        max_range_m=10000,
        max_depth_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*18,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
)

p1 = PiecewiseLinearWaveSpeedModel(
            z_grid_m=jnp.array([0.0, 75, 200]),
            sound_speed=jnp.array([1500, (1510+1500.0)/2, 1510])
        )

measure_p1 = add_noise(model.apply_profile(p1), 30)

layers = [50]*8
p1_opt_res = adam(model, measure_p1, learning_rate=0.002, gamma=100.0, profile_model=MLPWaveSpeedModel(z_max_m=200.0, layers=layers, c0=1510.0))
z_grid_o = jnp.linspace(0, 200, 250)

f, ax = plt.subplots(1, 4, figsize=(6, 3.4), constrained_layout=True)
ax[0].plot(p1(z_grid_o), z_grid_o, color='blue', label="Original")
ax[0].plot(p1_opt_res.res_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[0].set_xlabel("M-profile")
ax[0].set_ylabel("Depth (m)")
ax[0].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[0].grid(True)


legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Original'),
                   Line2D([0], [0], color='red', lw=1, label='Inverted')]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)
f.tight_layout()

plt.show()

vis_model = UWAModel(params=UWAComputationalParams(
        max_range_m=30000,
        max_depth_m=250,
        dx_m=100,
        dz_m=1),
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