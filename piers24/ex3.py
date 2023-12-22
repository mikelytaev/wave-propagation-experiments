import logging

from rwp.environment import Troposphere, Terrain, gauss_hill_func, WetGround
from rwp.vis import FieldVisualiser
from utils import solution, get_elevation_func

logging.basicConfig(level=logging.DEBUG)


environment = Troposphere()
environment.terrain = Terrain(
    elevation=get_elevation_func(-33.97242552291476, 18.355074679608087, -33.97736899254513, 18.558882296380858, 1000),
    #ground_material=WetGround()
)

logging.basicConfig(level=logging.DEBUG)
src_vis, dst_vis, src_bw_vis, dst_bw_vis, merge_vis = solution(
    freq_hz=2400e6,
    polarz="H",
    src_height_m=20,
    src_1m_power_db=50,
    drone_1m_power_db=50,
    drone_max_height_m=2000,
    drone_max_range_m=100e3,
    dst_height_m=20,
    dst_range_m=20e3,
    dst_1m_power_db=40,
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
