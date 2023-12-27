import logging

from matplotlib.colors import Normalize
from scipy.interpolate import interp1d

from rwp.environment import Troposphere, Terrain, gauss_hill_func, WetGround, Impediment, CustomMaterial
from rwp.vis import FieldVisualiser
from utils import solution, get_elevation_func, AntennaParams
import numpy as np

logging.basicConfig(level=logging.DEBUG)

elevation_func = get_elevation_func(21.426706, -158.165515, 21.837204, -157.806865, 5000)
environment = Troposphere()
environment.terrain = Terrain(
    elevation=lambda x: elevation_func(x + 1500)*0.75,
    #ground_material=WetGround()
)
# environment.vegetation = [Impediment(
#     left_m=2.5E3,
#     right_m=32.5E3,
#     height_m=25,
#     material=CustomMaterial(eps=1.004, sigma=180e-6)
# )]
# surface_based_duct = interp1d(
#     x=[0, 900, 1200, 1500],
#     y=[5, 40, 0, 15],
#     fill_value="extrapolate")
# environment.M_profile = lambda x, z: surface_based_duct(z)

logging.basicConfig(level=logging.DEBUG)
src_vis, dst_vis, src_bw_vis, dst_bw_vis, merge_vis, opt_vis = solution(
    freq_hz=900e6,
    polarz="H",
    src_params=AntennaParams(
        power_dBm=20,
        gain_dBi=15,
        sensitivity_dBm=-105,
        height_m=30,
        beam_width_deg=30
    ),
    drone_params=AntennaParams(
        power_dBm=16,
        gain_dBi=15,
        sensitivity_dBm=-104
    ),
    dst_params=AntennaParams(
        power_dBm=15,
        gain_dBi=14,
        sensitivity_dBm=-103,
        height_m=25,
        beam_width_deg=30
    ),
    drone_max_height_m=1500,
    drone_max_range_m=100e3,
    dst_range_m=36e3,
    env=environment
)

plt = src_vis.plot2d(min=60, max=200, show_terrain=True, cmap="jet_r")
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

plt = dst_vis.plot2d(min=60, max=200, show_terrain=True, cmap="jet_r")
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

plt = src_bw_vis.plot2d(min=0.5, max=0.51, show_terrain=True, cmap='gray')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

plt = dst_bw_vis.plot2d(min=0.5, max=0.51, show_terrain=True, cmap='gray')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

plt = merge_vis.plot2d(min=0.5, max=0.51, show_terrain=True, cmap='gray')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

plt = opt_vis.plot2d(min=0, max=25, show_terrain=True, cmap='gnuplot')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

####################################

plt.rcParams['font.size'] = '10'
f, ax = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=True)
norm = Normalize(-50, 0)
extent = [src_vis.x_grid[0], src_vis.x_grid[-1], src_vis.z_grid[0], src_vis.z_grid[-1]]
im = ax[0].imshow(src_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
terrain_grid = np.array([src_vis.env.terrain.elevation(v) for v in src_vis.x_grid / src_vis.x_mult])
ax[0].plot(src_vis.x_grid, terrain_grid, 'k')
ax[0].fill_between(src_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
f.colorbar(im, ax=ax[:], fraction=0.046, location='bottom')
ax[0].set_xlabel("Range (km)", fontsize=10)
ax[0].set_ylabel("Height (m)", fontsize=10)
#ax[0].set_title("L, dB", fontsize=13)
ax[0].grid(True)

extent = [dst_vis.x_grid[0], dst_vis.x_grid[-1], dst_vis.z_grid[0], dst_vis.z_grid[-1]]
im = ax[1].imshow(dst_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
terrain_grid = np.array([src_vis.env.terrain.elevation(v) for v in src_vis.x_grid / src_vis.x_mult])
ax[1].plot(src_vis.x_grid, terrain_grid, 'k')
ax[1].fill_between(src_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
#f.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
ax[1].set_xlabel("Range (km)", fontsize=10)
#ax[1].set_title("E[L], dB", fontsize=10)
ax[1].set_yticklabels([])
ax[1].grid(True)
#f.tight_layout()
for a in ax[:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(10)
plt.savefig('ex1.1.eps')

#################################################

plt.rcParams['font.size'] = '10'
f, ax = plt.subplots(1, 2, figsize=(6.5, 2.8), constrained_layout=True)
norm = Normalize(0.5, 0.51)
extent = [src_bw_vis.x_grid[0], src_bw_vis.x_grid[-1], src_bw_vis.z_grid[0], src_bw_vis.z_grid[-1]]
im = ax[0].imshow(src_bw_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('gray'))
terrain_grid = np.array([src_bw_vis.env.terrain.elevation(v) for v in src_bw_vis.x_grid / src_bw_vis.x_mult])
ax[0].plot(src_bw_vis.x_grid, terrain_grid, 'k')
ax[0].fill_between(src_bw_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[0].set_xlabel("Range (km)", fontsize=10)
ax[0].set_ylabel("Height (m)", fontsize=10)
#ax[0].set_title("L, dB", fontsize=13)
ax[0].grid(True)

extent = [dst_bw_vis.x_grid[0], dst_bw_vis.x_grid[-1], dst_bw_vis.z_grid[0], dst_bw_vis.z_grid[-1]]
im = ax[1].imshow(dst_bw_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('gray'))
terrain_grid = np.array([dst_bw_vis.env.terrain.elevation(v) for v in src_vis.x_grid / src_vis.x_mult])
ax[1].plot(dst_bw_vis.x_grid, terrain_grid, 'k')
ax[1].fill_between(dst_bw_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[1].set_xlabel("Range (km)", fontsize=10)
#ax[1].set_title("E[L], dB", fontsize=10)
ax[1].set_yticklabels([])
ax[1].grid(True)
#f.tight_layout()
for a in ax[:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(10)
plt.savefig('ex1.2.eps')

#################################################

plt.rcParams['font.size'] = '10'
f, ax = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=True)
norm = Normalize(0.5, 0.51)
extent = [merge_vis.x_grid[0], merge_vis.x_grid[-1], merge_vis.z_grid[0], merge_vis.z_grid[-1]]
im = ax[0].imshow(merge_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('gray'))
terrain_grid = np.array([merge_vis.env.terrain.elevation(v) for v in merge_vis.x_grid / merge_vis.x_mult])
ax[0].plot(merge_vis.x_grid, terrain_grid, 'k')
ax[0].fill_between(merge_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[0].set_xlabel("Range (km)", fontsize=10)
ax[0].set_ylabel("Height (m)", fontsize=10)
#ax[0].set_title("L, dB", fontsize=13)
ax[0].grid(True)

norm = Normalize(0, 25)
extent = [opt_vis.x_grid[0], opt_vis.x_grid[-1], opt_vis.z_grid[0], opt_vis.z_grid[-1]]
im = ax[1].imshow(opt_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('gnuplot'))
terrain_grid = np.array([src_vis.env.terrain.elevation(v) for v in src_vis.x_grid / src_vis.x_mult])
ax[1].plot(src_vis.x_grid, terrain_grid, 'k')
ax[1].fill_between(src_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
f.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
ax[1].set_xlabel("Range (km)", fontsize=10)
#ax[1].set_title("E[L], dB", fontsize=10)
ax[1].set_yticklabels([])
ax[1].grid(True)
#f.tight_layout()
for a in ax[:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(10)
plt.savefig('ex1.3.eps')
