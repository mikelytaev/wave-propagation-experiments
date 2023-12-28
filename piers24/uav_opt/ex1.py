from rwp.sspade import *
from rwp.vis import *
from rwp.environment import Troposphere, \
    Terrain, gauss_hill_func


logging.basicConfig(level=logging.DEBUG)


environment = Troposphere()
environment.terrain = Terrain(
    elevation=gauss_hill_func(
        height_m=750,
        length_m=40E3,
        x0_m=50E3
    ),
    ground_material=WetGround()
)

# environment.vegetation = [Impediment(
#     left_m=00E3,
#     right_m=100E3,
#     height_m=25,
#     material=CustomMaterial(eps=1.004, sigma=180e-6)
# )]


params = RWPSSpadeComputationalParams(
    max_range_m=100E3,
    max_height_m=1000,
    dx_m = 100,  # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1
)


antenna = GaussAntenna(freq_hz=900E6,
                       height=50,
                       beam_width=0.5,
                       elevation_angle=-0.7,
                       polarz='V')

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis3ghz = FieldVisualiser(field, env=environment, x_mult=1E-3)

plt = vis3ghz.plot2d(min=-90, max=10, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()