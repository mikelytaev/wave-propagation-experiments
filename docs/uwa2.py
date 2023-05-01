from uwa.sspade import *
from uwa.vis import AcousticPressureFieldVisualiser2d
import math as fm


max_range_m = 2E3
env = UnderwaterEnvironment(
    sound_speed_profile_m_s=lambda x, z: munk_profile(z),
    bottom_profile=Bathymetry(ranges_m=[0], depths_m=[5000]),
    bottom_sound_speed_m_s = 1700,
    bottom_density_g_cm = 1.5,
    bottom_attenuation_dm_lambda = 0.5
)

src = GaussSource(
    freq_hz=50,
    depth_m=100,
    beam_width_deg=3,
    eval_angle_deg=0,
    multiplier=5
)

params = UWASSpadeComputationalParams(
    max_range_m=max_range_m,
    max_depth_m=5500,
    dx_m=100, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m=5,
)

field = uwa_ss_pade(
    src=src,
    env=env,
    params=params
)

vis = AcousticPressureFieldVisualiser2d(field=field, env=env)

vis.sound_speed_profile().show()

vis.plot2d(min_val=-80, max_val=0, grid=True, show_terrain=True).show()