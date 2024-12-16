import logging

import jax.numpy as jnp
from matplotlib.colors import Normalize

from experimental.helmholtz_jax import LinearSlopeWaveSpeedModel
from experimental.uwa_jax import UWAComputationalParams, UWAGaussSourceModel, UnderwaterEnvironmentModel, \
    uwa_forward_task


import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG)


src = UWAGaussSourceModel(freq_hz=500, depth_m=100, beam_width_deg=5)
env = UnderwaterEnvironmentModel(
    sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(c0=1500.0, slope_degrees=1.0),
    #rho_profile=StaircaseRhoModel(heights=[0, 500.0], vals=[1.0, 5.5]),
    bottom_profile=lambda x: x*0 + 1000
)
params = UWAComputationalParams(
    max_range_m=50000,
    max_depth_m=500,
    x_output_points=100,
    z_output_points=500,
)

field = uwa_forward_task(src=src, env=env, params=params)

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(field.field+1e-16)).T,
    norm=Normalize(vmin=-80, vmax=-30),
    aspect='auto',
    extent=[0, field.x_grid[-1], field.z_grid[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()
