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
from experiments.optimization.node.npe.common import adam, RWPModel, add_noise, surface_based_duct_N
from matplotlib.lines import Line2D

from rwp.terrain import get_elevation_gmap
from scipy.interpolate import interp1d
import numpy as np

jax.config.update("jax_enable_x64", True)

start = (60.1, 30)
end = (60.1, 31)

elevations, distances = get_elevation_gmap(start, end, samples=5000)
terrain_func = interp1d(x=np.array(distances), y=[e/2 if e > 0 else 0.0 for e in elevations], fill_value="extrapolate")

env = TroposphereModel(
        terrain=terrain_func
    )
src = RWPGaussSourceModel(freq_hz=3E9, height_m=15.0, beam_width_deg=3.0)

model = RWPModel(
    params=RWPComputationalParams(
        max_range_m=10000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    src = src,
    env = env,
    measure_points_x=[-1]*25,
    measure_points_z=round(jnp.linspace(30, 150, 25)),
)

measure_terrain = model.apply_profile(surface_based_duct_N)
measure_terrain = add_noise(measure_terrain, 30)

layers = [50]*4
terrain_opt_res = adam(
    model,
    measure_terrain,
    ground_truth_profile=surface_based_duct_N,
    learning_rate=0.1,
    gamma=1e-3,
    profile_model=MLPNProfileModel(z_max_m=100.0, layers=layers)
)

model.apply_profile(terrain_opt_res.res_profile)

# vis_model = RWPModel(params=RWPComputationalParams(
#         max_range_m=50000,
#         max_height_m=250,
#         dx_m=100,
#         dz_m=1),
#     src=src,
#     env=env,
# )
# f1 = vis_model.calc_field(surface_based_duct_N)
# f, ax = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
# norm = Normalize(-70, -10)
# extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1]*1e-3, vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
# ax[0].imshow(20*jnp.log10(abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
# ax[0].set_xlabel("Range (km)")
# ax[0].set_ylabel("Height (m)")
# ax[0].grid(True)
# ax[0].set_ylim([0, 250])
# terrain_grid = np.array([terrain_func(v) for v in vis_model.fwd_model.x_output_grid()])
# ax[0].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
# ax[0].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')
#
# f1 = vis_model.calc_field(terrain_opt_res.res_profile)
# im = ax[1].imshow(20*jnp.log10(abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
# ax[1].set_xlabel("Range (km)")
# ax[1].set_yticklabels([])
# ax[1].grid(True)
# ax[1].set_ylim([0, 250])
# f.colorbar(im, ax=ax, shrink=0.9, location='right')
# ax[1].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
# ax[1].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')
#
# for c in model.measure_points_z:
#     z = model.fwd_model.z_output_grid()[c]
#     x = model.fwd_model.x_output_grid()[-1]
#     ax[0].plot(x*1e-3, z, '*', color='white', mew=0.1)
#     ax[1].plot(x * 1e-3, z, '*', color='white', mew=0.1)
#
# plt.show()
