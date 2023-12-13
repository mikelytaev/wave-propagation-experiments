from rwp.sspade import *
from rwp.vis import *
from rwp.environment import Troposphere, \
    Terrain, gauss_hill_func


logging.basicConfig(level=logging.DEBUG)


environment = Troposphere(flat=False)
environment.terrain = Terrain(
    ground_material=SaltWater()
)
elevated_duct = interp1d(
    x=[0, 100, 150, 300],
    y=[0, 32, 10, 45],
    fill_value="extrapolate")
environment.M_profile = lambda x, z: \
    elevated_duct(z)

params = RWPSSpadeComputationalParams(
    max_range_m=200E3,
    max_height_m=2000,
    dx_m = 100,  # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1
)


antenna = GaussAntenna(freq_hz=2400E6,
                       height=1500,
                       beam_width=0.5,
                       elevation_angle=1.2,
                       polarz='H')

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis3ghz = FieldVisualiser(field, env=environment, x_mult=1E-3)

plt = vis3ghz.plot2d(min=-50, max=10, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()