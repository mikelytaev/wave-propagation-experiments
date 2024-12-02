from rwp.sspade import *
from rwp.vis import *
from rwp.environment import Troposphere, \
    Terrain, gauss_hill_func


logging.basicConfig(level=logging.DEBUG)


environment = Troposphere()
environment.terrain = InterpTerrain(
    edge_range=[0, 2000, 3000, 6000, 8500, 10000, 20000, 22000, 25000, 30000],
    edge_height=[50, 42, 38, 25, 20, 0, 0, 20, 25, 27],
    ground_material=WetGround()
)
elevated_duct = interp1d(
    x=[0, 100, 150, 300],
    y=[0, 32, 10, 45],
    fill_value="extrapolate")
environment.M_profile = lambda x, z: elevated_duct(z)

# environment.vegetation = [Impediment(
#     left_m=00E3,
#     right_m=100E3,
#     height_m=25,
#     material=CustomMaterial(eps=1.004, sigma=180e-6)
# )]


params = RWPSSpadeComputationalParams(
    max_range_m=30E3,
    max_height_m=200,
    dx_m = 100,  # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1
)


antenna = GaussAntenna(freq_hz=3000E6,
                       height=15,
                       beam_width=2,
                       elevation_angle=-0.7,
                       polarz='V')

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)

vis3ghz = FieldVisualiser(field, env=environment, x_mult=1E-3)

plt = vis3ghz.plot2d(min=-40, max=-5, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.show()

f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-50, -10)
extent = (field.x_grid[0], field.x_grid[-1]*1e-3, field.z_grid[0], field.z_grid[-1])
ax.imshow(20*np.log10(abs(field.field+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
terrain_grid = np.array([environment.terrain.elevation(v) for v in field.x_grid])
ax.plot(field.x_grid*1e-3, terrain_grid, 'k')
ax.fill_between(field.x_grid*1e-3, terrain_grid*0, terrain_grid, color='brown')
ax.set_xticklabels([])
ax.set_yticklabels([])
f.tight_layout()
plt.show()
