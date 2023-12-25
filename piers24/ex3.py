import logging

from matplotlib.colors import Normalize
from scipy.interpolate import interp1d

from rwp.environment import Troposphere, Terrain, gauss_hill_func, WetGround, Impediment, CustomMaterial
from rwp.vis import FieldVisualiser
from utils import solution, get_elevation_func

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
    src_height_m=30,
    src_1m_power_db=50,
    drone_1m_power_db=50,
    drone_max_height_m=1500,
    drone_max_range_m=100e3,
    dst_height_m=25,
    dst_range_m=36e3,
    dst_1m_power_db=50,
    src_min_power_db=10,
    drone_min_power_db=10,
    dst_min_power_db=10,
    env=environment
)

plt = src_vis.plot2d(min=-50, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

plt = dst_vis.plot2d(min=-50, max=0, show_terrain=True)
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



plt.rcParams['font.size'] = '10'
f, ax = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=True)
norm = Normalize(-50, 0)
extent = [src_vis.x_grid[0], src_vis.x_grid[-1], src_vis.z_grid[0], src_vis.z_grid[-1]]
im = ax[0].imshow(src_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
f.colorbar(im, ax=ax[:], fraction=0.046, location='bottom')
ax[0].set_xlabel("Range (km)", fontsize=10)
ax[0].set_ylabel("Height (m)", fontsize=10)
#ax[0].set_title("L, dB", fontsize=13)
ax[0].grid(True)

norm = Normalize(-80, 0)
extent = [dst_vis.x_grid[0], dst_vis.x_grid[-1], dst_vis.z_grid[0], dst_vis.z_grid[-1]]
im = ax[1].imshow(dst_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
#f.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
ax[1].set_xlabel("Range (km)", fontsize=10)
#ax[1].set_title("E[L], dB", fontsize=10)
ax[1].set_yticklabels([])
ax[1].grid(True)
#f.tight_layout()
for a in ax[:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(10)
plt.show()
