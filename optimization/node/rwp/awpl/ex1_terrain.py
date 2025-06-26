from common import *
from matplotlib.colors import Normalize
from matplotlib.pyplot import figure

from experimental.rwp_jax import PiecewiseLinearNProfileModel, EmptyNProfileModel
import jax.numpy as jnp
from utils import *
import matplotlib.pyplot as plt
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
inv_model_terrain = RWPModel(params=RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    env=env,
)
inverted_profiles, inversion_time, nfev_list, _ = realtime(inv_model_terrain, profiles, gamma=5e-4)

inv_model_terrain.env.N_profile = profiles[35]
plt.plot(inv_model_terrain.env.M_profile(z_grid), z_grid)
inv_model_terrain.env.N_profile = inverted_profiles[35]
plt.plot(inv_model_terrain.env.M_profile(z_grid), z_grid)
plt.show()

f, ax = plt.subplots(2, 1, figsize=(6, 3.2), constrained_layout=True)
for i, true_profile in enumerate(profiles):
    inv_model_terrain.env.N_profile = true_profile
    ax[0].plot(inv_model_terrain.env.M_profile(z_grid) + 5 * i, z_grid)
ax[0].set_title('Original M-profiles')
ax[0].set_xticklabels([])
ax[0].set_ylabel("Height (m)")
ax[0].set_xlim([325, 554])
ax[0].set_ylim([z_grid[0], z_grid[-1]])
ax[0].grid(True)

for i, inverted_profile in enumerate(inverted_profiles):
    inv_model_terrain.env.N_profile = inverted_profile
    ax[1].plot(inv_model_terrain.env.M_profile(z_grid) + 5 * i, z_grid)
ax[1].set_title('Inverted M-profiles')
ax[1].set_ylabel("Height (m)")
ax[1].set_xlabel("M-profile (M units)")
ax[1].set_xlim([325, 554])
ax[1].set_ylim([z_grid[0], z_grid[-1]])
ax[1].grid(True)
plt.show()

f, ax = plt.subplots(1, 4, figsize=(6, 3.2), constrained_layout=True)
inds = [10, 20, 30, 40]
ax[0].set_ylabel("Height (m)")
for i in range(4):
    ax[i].set_title(f't = {inds[i]}')
    inv_model_terrain.env.N_profile = profiles[inds[i] - 1]
    ax[i].plot(inv_model_terrain.env.M_profile(z_grid), z_grid, color='blue')
    inv_model_terrain.env.N_profile = inverted_profiles[inds[i] - 1]
    ax[i].plot(inv_model_terrain.env.M_profile(z_grid), z_grid, color='red')
    ax[i].set_xlim([330, 358])
    ax[i].set_ylim([z_grid[0], z_grid[-1]])
    ax[i].set_xlabel("M-profile")
    ax[i].grid(True)
for i in range(1, 4):
    ax[i].set_yticklabels([])
legend_elements = [Line2D([0], [0], color='blue', lw=1, label='True profile'),
                   Line2D([0], [0], color='red', lw=1, label='Inverted profile')]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)
f.tight_layout()
plt.show()

plot_rel_error(profiles, inverted_profiles, z_grid)

vis_model = RWPModel(params=RWPComputationalParams(
        max_range_m=50000,
        max_height_m=250,
        dx_m=100,
        dz_m=0.5
    ),
    env=env,
)
f, ax = plt.subplots(1, 2, figsize=(6, 2.8), constrained_layout=True)
f1 = vis_model.calc_field(profiles[10])
extent = (vis_model.fwd_model.x_output_grid()[0]*1e-3, vis_model.fwd_model.x_output_grid()[-1]*1e-3,
          vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
terrain_grid = np.array([terrain_func(v) for v in vis_model.fwd_model.x_output_grid()])
norm = Normalize(-70, -10)
im = ax[0].imshow(20*jnp.log10(jnp.abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0].grid(True)
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Height (m)')
ax[0].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
ax[0].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

f1_i = vis_model.calc_field(inverted_profiles[10])
norm = Normalize(-15, 15)
im2 = ax[1].imshow((20*jnp.log10(jnp.abs(f1+1e-16)) - 20*jnp.log10(jnp.abs(f1_i+1e-16))).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('seismic'))
ax[1].grid(True)
ax[1].set_xlabel('Range (km)')
ax[1].set_yticklabels([])
ax[1].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
ax[1].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, ax=ax[0], shrink=0.7, location='bottom')
f.colorbar(im2, ax=ax[1], shrink=0.7, location='bottom')
plt.show()

r = 5000
r_i = abs(vis_model.fwd_model.x_output_grid() - r).argmin()

f, ax = plt.subplots(1, 2, figsize=(6, 3.2), constrained_layout=True)
ax[0].plot(20*jnp.log10(jnp.abs(f1[r_i,0:250]+1e-16)), vis_model.fwd_model.z_output_grid()[0:250])
ax[0].plot(20*jnp.log10(jnp.abs(f2[r_i,0:250]+1e-16)), vis_model.fwd_model.z_output_grid()[0:250])
ax[0].set_xlim([-70, -20])
ax[0].set_ylim([0, vis_model.fwd_model.z_output_grid()[250]])
ax[0].grid(True)

ax[1].plot((jnp.angle(f1[r_i,0:250]+1e-16)), vis_model.fwd_model.z_output_grid()[0:250])
ax[1].plot((jnp.angle(f2[r_i,0:250]+1e-16)), vis_model.fwd_model.z_output_grid()[0:250])
ax[1].set_ylim([0, vis_model.fwd_model.z_output_grid()[250]])
ax[1].grid(True)
plt.show()