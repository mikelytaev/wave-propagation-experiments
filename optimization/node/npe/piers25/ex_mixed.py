from cProfile import label

import jax

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.pyplot import title

from experimental.helmholtz_jax import PiecewiseLinearTerrainModel
from experimental.rwp_jax import RWPGaussSourceModel, RWPComputationalParams, EvaporationDuctModel, \
    PiecewiseLinearNProfileModel, TroposphereModel
import jax.numpy as jnp

from experiments.optimization.node.flax.utils import MLPNProfileModel
from experiments.optimization.node.npe.common import adam, RWPModel, add_noise, elevated_evaporation_duct_N
from matplotlib.lines import Line2D

from rwp.terrain import get_elevation_gmap
from scipy.interpolate import interp1d
import numpy as np

jax.config.update("jax_enable_x64", True)

env = TroposphereModel()
src = RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)

model = RWPModel(
    params=RWPComputationalParams(
        max_range_m=5000+100,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    src = src,
    env = env,
    measure_points_x=[-1] * 20,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
)

measure_evaporation_duct = model.apply_profile(elevated_evaporation_duct_N)
measure_evaporation_duct = add_noise(measure_evaporation_duct, 30)

layers = [50]*4
mixed_opt_res = adam(
    model,
    measure_evaporation_duct,
    ground_truth_profile=elevated_evaporation_duct_N,
    learning_rate=0.05,
    gamma=1e-3,
    profile_model=MLPNProfileModel(z_max_m=100.0, layers=layers)
)

model.apply_profile(mixed_opt_res.res_profile)

# vis_model = RWPModel(params=RWPComputationalParams(
#         max_range_m=50000,
#         max_height_m=250,
#         dx_m=100,
#         dz_m=1),
#     src=src,
#     env=env,
# )
# f1 = vis_model.calc_field(elevated_evaporation_duct_N)
# f, ax = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
# norm = Normalize(-70, -10)
# extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1]*1e-3, vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
# ax[0].imshow(20*jnp.log10(abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
# ax[0].set_xlabel("Range (km)")
# ax[0].set_ylabel("Height (m)")
# ax[0].grid(True)
# ax[0].set_ylim([0, 250])
#
# f1 = vis_model.calc_field(mixed_opt_res.res_profile)
# im = ax[1].imshow(20*jnp.log10(abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
# ax[1].set_xlabel("Range (km)")
# ax[1].set_yticklabels([])
# ax[1].grid(True)
# ax[1].set_ylim([0, 250])
# f.colorbar(im, ax=ax, shrink=0.9, location='right')
#
# for c in model.measure_points_z:
#     z = model.fwd_model.z_output_grid()[c]
#     x = model.fwd_model.x_output_grid()[-1]
#     ax[0].plot(x*1e-3, z, '*', color='white', mew=0.1)
#     ax[1].plot(x * 1e-3, z, '*', color='white', mew=0.1)
#
# plt.show()
