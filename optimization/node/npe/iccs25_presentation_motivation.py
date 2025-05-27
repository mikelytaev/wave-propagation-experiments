from rwp.sspade import *
from rwp.terrain import get_elevation_gmap
from rwp.vis import *
from rwp.environment import Troposphere, \
    Terrain, gauss_hill_func


logging.basicConfig(level=logging.DEBUG)


environment = Troposphere()
environment.terrain = Terrain(
    ground_material=WetGround()
)

params = RWPSSpadeComputationalParams(
    max_range_m=100E3,
    max_height_m=300,
    dx_m = 100,  # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m = 1,
    #dz_computational_grid_wl=2.0,
    max_propagation_angle_deg=3.0
)


antenna = GaussAntenna(freq_hz=3000E6,
                       height=80,
                       beam_width=2,
                       elevation_angle=-0.7,
                       polarz='V')


def plot(antenna, environment, field, shift=0):
    f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
    norm = Normalize(-80, -10)
    extent = (field.x_grid[0], field.x_grid[-1]*1e-3, field.z_grid[0], field.z_grid[-1])
    im = ax.imshow(20*np.log10(abs(field.field+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
    terrain_grid = np.array([environment.terrain.elevation(v) for v in field.x_grid])
    ax.plot(field.x_grid*1e-3, terrain_grid, 'k')
    ax.fill_between(field.x_grid*1e-3, terrain_grid*0, terrain_grid, color='brown')
    ax.plot(environment.n2m1_profile(0, field.z_grid, antenna.freq_hz) * (1/2*1e6) / 2 + 10 - shift, field.z_grid)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Height (m)")
    f.colorbar(im, ax=ax, shrink=0.6, location='right')
    plt.show()

field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
plot(antenna, environment, field)

elevated_duct = interp1d(
    x=[0, 100, 150, 300],
    y=[0, 32, 10, 45],
    fill_value="extrapolate")
environment.M_profile = lambda x, z: elevated_duct(z)
field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
plot(antenna, environment, field)

environment.M_profile = lambda x, z: evaporation_duct(30, z)
field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
plot(antenna, environment, field, 140)