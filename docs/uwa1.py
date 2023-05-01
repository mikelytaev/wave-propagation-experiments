import os
#os.chdir('../../../')
from uwa.sspade import *
from uwa.vis import AcousticPressureFieldVisualiser2d


max_range_m = 3000
env = UnderwaterEnvironment(
    sound_speed_profile_m_s=lambda x, z: 1500 + z*0,
    bottom_profile=Bathymetry(
        ranges_m=[0, max_range_m],
        depths_m=[300, 300]),
    bottom_sound_speed_m_s = 1700,
    bottom_density_g_cm = 1.5,
    bottom_attenuation_dm_lambda = 0.01
)

src = GaussSource(
    freq_hz=1000,
    depth_m=100,
    beam_width_deg=1,
    eval_angle_deg=-30,
    multiplier=5
)

params = UWASSpadeComputationalParams(
    max_range_m=max_range_m,
    max_depth_m=500,
    dx_m=1000, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m=500,
)

field = uwa_ss_pade(
    src=src,
    env=env,
    params=params
)

vis = AcousticPressureFieldVisualiser2d(field=field, env=env)
plt = vis.plot2d(min_val=-50, max_val=-5, grid=True, show_terrain=True).show()
plt.show()

src = GaussSource(
    freq_hz=100,
    depth_m=150,
    beam_width_deg=10,
    eval_angle_deg=10,
    multiplier=5
)

field = uwa_ss_pade(
    src=src,
    env=env,
    params=params
)

vis = AcousticPressureFieldVisualiser2d(field=field, env=env)
plt = vis.plot2d(-50, -5, grid=True, show_terrain=True)
plt.show()