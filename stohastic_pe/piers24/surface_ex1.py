from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *




environment = Troposphere()
environment.terrain = Terrain(ground_material=WetGround())

profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 45], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)

antenna = GaussAntenna(
    freq_hz=3E9,
    height=150,
    beam_width=2,
    elevation_angle=0,
    polarz='H'
)

params = RWPSSpadeComputationalParams(
    max_range_m=250e3,
    max_height_m=300,
    dx_m = 100, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1
)


field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis = FieldVisualiser(field, env=environment, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3)

plt = vis.plot2d(min=-100, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()