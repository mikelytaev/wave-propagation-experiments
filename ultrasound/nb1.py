from uwa.source import GaussSource
from uwa.environment import UnderwaterEnvironment, Bathymetry, sound_speed_mps
from uwa.sspade import UWASSpadeComputationalParams, uwa_ss_pade
from uwa.vis import AcousticPressureFieldVisualiser2d
from scipy.interpolate import interp1d
import logging


logging.basicConfig(level=logging.DEBUG)

max_range_m = 100
temp_profile = interp1d(x=[0, 6, 8, 20], y=[19, 19, 12, 7], fill_value="extrapolate")
env = UnderwaterEnvironment(
    sound_speed_profile_m_s=lambda x, z: sound_speed_mps(z, temp_profile(z), 0),
    bottom_profile=Bathymetry(
        ranges_m=[0, max_range_m],
        depths_m=[4, 1]),
    bottom_sound_speed_m_s=1700,
    bottom_density_g_cm=1.5,
    bottom_attenuation_dm_lambda=0.1
)

src = GaussSource(
    freq_hz=31.4E3,
    depth_m=2,
    beam_width_deg=20,
    eval_angle_deg=-60,
    multiplier=0.1
)

params = UWASSpadeComputationalParams(
    max_range_m=max_range_m,
    max_depth_m=6,
    dx_m=0.1,  # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m=0.1,
    precision=0.1
)

field = uwa_ss_pade(
    src=src,
    env=env,
    params=params
)

vis = AcousticPressureFieldVisualiser2d(field=field, env=env, label='Horizontal')
vis.sound_speed_profile().show()
vis.plot2d(min_val=-50, max_val=0, grid=True, show_terrain=True).show()

vis.plot_hor(2, y_lims=[-70, 10]).show()
