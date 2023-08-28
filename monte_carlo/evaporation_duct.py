from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *

environment = Troposphere()
environment.terrain = Terrain(ground_material=SaltWater())
max_range = 150e3

environment.M_profile = lambda x, z: evaporation_duct(height=10, z_grid_m=z)

antenna = GaussAntenna(freq_hz=10E9, height=10, beam_width=2, elevation_angle=0, polarz='H')

params = RWPSSpadeComputationalParams(
    max_range_m=150E3,
    max_height_m=50,
    dx_m = 100, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1
)

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis = FieldVisualiser(field, env=environment, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + Transparent BC', x_mult=1E-3)

plt = vis.plot_hor(30)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt = vis.plot2d(min=-50, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()