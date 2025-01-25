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
    elevated_duct_N, surface_based_duct2_N, UWAModel, munk_profile, ssp_1, thermocline_profile, slope_profile, PLWSM
from matplotlib.lines import Line2D

jax.config.update("jax_enable_x64", True)

model = UWAModel(params=UWAComputationalParams(
        max_range_m=5000,
        max_depth_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*17,
    measure_points_z=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
)

measure_p1 = add_noise(model.apply_profile(ssp_1), 30)

z_grid_o = jnp.linspace(0, 200, 250)
layers = [50]*4
ssp1_opt_res = adam(model, measure_p1, learning_rate=0.2, gamma=0.0,
                    profile_model=PLWSM(z_grid_m=z_grid_o, sound_speed=ssp_1(z_grid_o)[::-1]))

f, ax = plt.subplots(1, 4, figsize=(6, 3.4), constrained_layout=True)
ax[0].plot(ssp_1(z_grid_o)[::-1], z_grid_o[::-1], color='blue', label="Original")
ax[0].plot(ssp1_opt_res.res_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[0].set_xlabel("SSP (m/s)")
ax[0].set_ylabel("Depth (m)")
ax[0].set_ylim([z_grid_o[-1], z_grid_o[0]])
ax[0].grid(True)

ax[1].plot(munk_profile(z_grid_o)[::-1], z_grid_o[::-1], color='blue', label="Original")
#ax[1].plot(p1_opt_res.res_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[1].set_xlabel("SSP (m/s)")
ax[1].set_yticklabels([])
ax[1].set_ylim([z_grid_o[-1], z_grid_o[0]])
ax[1].grid(True)

ax[2].plot(thermocline_profile(z_grid_o)[::-1], z_grid_o[::-1], color='blue', label="Original")
#ax[2].plot(p1_opt_res.res_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[2].set_xlabel("SSP (m/s)")
ax[2].set_yticklabels([])
ax[2].set_ylim([z_grid_o[-1], z_grid_o[0]])
ax[2].grid(True)

ax[3].plot(slope_profile(z_grid_o)[::-1], z_grid_o[::-1], color='blue', label="Original")
#ax[2].plot(p1_opt_res.res_profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[3].set_xlabel("SSP (m/s)")
ax[3].set_yticklabels([])
ax[3].set_ylim([z_grid_o[-1], z_grid_o[0]])
ax[3].grid(True)


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
