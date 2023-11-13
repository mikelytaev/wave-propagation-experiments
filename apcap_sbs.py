from rwp.sspade import *
from rwp.vis import *
from rwp.environment import Troposphere, \
    Terrain, gauss_hill_func


logging.basicConfig(level=logging.DEBUG)


environment = Troposphere()

antenna = GaussAntenna(freq_hz=6E9,
                       height=30,
                       beam_width=2,
                       elevation_angle=0,
                       polarz='H')

params = RWPSSpadeComputationalParams(
    max_range_m=100E3,
    max_height_m=300,
    dx_m = 100, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1
)

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis3ghz = FieldVisualiser(field, env=environment, x_mult=1E-3)

plt = vis3ghz.plot2d(min=-90, max=10, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("1_std.eps")


###########

elevated_duct = interp1d(
    x=[0, 100, 150, 300],
    y=[0, 32, 10, 45],
    fill_value="extrapolate")
environment.M_profile = lambda x, z: \
    elevated_duct(z)

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis3ghz = FieldVisualiser(field, env=environment, x_mult=1E-3)

plt = vis3ghz.plot2d(min=-90, max=10, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("2_duct.eps")


#####

environment.terrain = Terrain(
    elevation=gauss_hill_func(
        height_m=125,
        length_m=40E3,
        x0_m=50E3
    ),
    ground_material=WetGround()
)


field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis3ghz = FieldVisualiser(field, env=environment, x_mult=1E-3)

plt = vis3ghz.plot2d(min=-90, max=10, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("3_terrain.eps")

###########

environment.vegetation = [Impediment(
    left_m=30E3,
    right_m=70E3,
    height_m=25,
    material=CustomMaterial(eps=1.004, sigma=180e-6)
)]

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis3ghz = FieldVisualiser(field, env=environment, x_mult=1E-3)

plt = vis3ghz.plot2d(min=-90, max=10, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("4_vegetation.eps")