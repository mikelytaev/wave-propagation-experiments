import logging

import numpy as np

from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *

environment = Troposphere()
environment.terrain = Terrain(ground_material=SaltWater())

logging.basicConfig(level=logging.DEBUG)


antenna = GaussAntenna(freq_hz=10E9, height=10, beam_width=2, elevation_angle=0, polarz='H')

params = RWPSSpadeComputationalParams(
    max_range_m=300E3,
    max_height_m=100,
    dx_m = 100, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1
)

expected_height = 20.0
n = 10
heights = np.random.normal(expected_height, 5, n)
environment.M_profile = lambda x, z: evaporation_duct(height=expected_height, z_grid_m=z)
expected_field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
error = Field(expected_field.x_grid, expected_field.z_grid, freq_hz=antenna.freq_hz)

for height in heights:
    print(height)
    environment.M_profile = lambda x, z: evaporation_duct(height=height, z_grid_m=z)
    field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
    error.field += (10*np.log10(np.abs(field.field+1e-16)) - 10*np.log10(np.abs(expected_field.field+1e-16)))**2

error.field = np.sqrt(error.field/n)

vis = FieldVisualiser(expected_field, env=environment, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + Transparent BC', x_mult=1E-3)

plt = vis.plot2d(min=-50, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

vis_error = FieldVisualiser(error, env=environment, trans_func=lambda v: v, label='СКО', x_mult=1E-3)

plt = vis_error.plot2d(min=0, max=10)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.show()

plt = vis_error.plot_hors(1, 3, 10)
plt.xlabel('Range (km)')
plt.ylabel('10log|u|')
plt.tight_layout()
plt.grid(True)
plt.ylim([0, 15])
plt.show()