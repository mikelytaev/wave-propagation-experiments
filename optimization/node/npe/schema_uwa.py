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

vis_model = UWAModel(params=UWAComputationalParams(
        max_range_m=10000,
        max_depth_m=250,
        dx_m=100,
        dz_m=1
    ),
)


f1 = vis_model.calc_field(munk_profile)
f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-80, -20)
extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1]*1e-3, vis_model.fwd_model.z_output_grid()[-1], vis_model.fwd_model.z_output_grid()[0])
im = ax.imshow(20*jnp.log10(abs(f1+1e-16)).T[:,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax.set_xticklabels([])
ax.set_yticklabels([])
f.tight_layout()
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
