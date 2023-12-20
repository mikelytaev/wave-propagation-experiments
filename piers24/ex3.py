import logging

from rwp.environment import Troposphere, Terrain, gauss_hill_func, WetGround
from rwp.vis import FieldVisualiser
from utils import solution


logging.basicConfig(level=logging.DEBUG)


environment = Troposphere()
environment.terrain = Terrain(
    elevation=gauss_hill_func(
        height_m=750,
        length_m=40E3,
        x0_m=50E3
    ),
    #ground_material=WetGround()
)


src_vis, dst_vis, src_bw_vis = solution(
    freq_hz=2400e6,
    polarz="H",
    src_height_m=20,
    src_1m_power_db=50,
    drone_1m_power_db=50,
    drone_max_height_m=2000,
    drone_max_range_m=100e3,
    dst_height_m=20,
    dst_range_m=100e3,
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

plt = src_bw_vis.plot2d(min=0.5, max=0.51, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()