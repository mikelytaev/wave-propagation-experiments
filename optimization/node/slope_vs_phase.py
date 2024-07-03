import logging

import numpy as np

from uwa.source import GaussSource
from uwa.environment import UnderwaterEnvironment, Bathymetry
from uwa.sspade import UWASSpadeComputationalParams, uwa_ss_pade
from uwa.vis import AcousticPressureFieldVisualiser2d
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
import math as fm
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG)
max_range_m = 25E3
env = UnderwaterEnvironment(
        bottom_profile=Bathymetry(ranges_m=[0], depths_m=[500]),
        bottom_sound_speed_m_s=2000,
        bottom_density_g_cm=1.5,
        bottom_attenuation_dm_lambda=0.5
    )

src = GaussSource(
    freq_hz=100,
    depth_m=100,
    beam_width_deg=3,
    elevation_angle_deg=0,
    multiplier=5
)

params = UWASSpadeComputationalParams(
    max_range_m=max_range_m,
    max_depth_m=700,
    dx_m=100, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m=5,
)

measures = []
bottom_ssps = np.linspace(1500, 1550, 51)
for bssp in bottom_ssps:
    print(bssp)
    ssp = interp1d(x=[0, 500], y=[1500, bssp], fill_value="extrapolate")
    env.sound_speed_profile_m_s = lambda _, z: ssp(z)
    field = uwa_ss_pade(
        src=src,
        env=env,
        params=params
    )
    measures += [field.nearest_value(max_range_m, 100)]


m2 = [m[2] for m in measures]
plt.plot((np.abs(m2-m2[20])))
plt.show()

vis = AcousticPressureFieldVisualiser2d(field=field, env=env)
vis.sound_speed_profile().show()
vis.plot2d(min_val=-80, max_val=0, grid=True, show_terrain=True).show()

plt.figure(figsize=(6, 3.2))
plt.imshow(
    np.angle(field.field).T,
    norm=Normalize(vmin=-fm.pi/2, vmax=fm.pi/2),
    aspect='auto',
    extent=[0, field.x_grid[-1], field.z_grid[-1], field.z_grid[0]],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()
