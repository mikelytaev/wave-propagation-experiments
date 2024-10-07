import logging

import jax.numpy as jnp
from matplotlib.colors import Normalize

from experimental.helmholtz_jax import PiecewiseLinearWaveSpeedModel, \
    ConstWaveSpeedModel
from experimental.uwa_jax import ComputationalParams, GaussSourceModel, UnderwaterEnvironmentModel, \
    uwa_get_model, uwa_forward_task, UnderwaterLayerModel
import math as fm

import matplotlib.pyplot as plt

from uwa.field import AcousticPressureField

logging.basicConfig(level=logging.DEBUG)


src = GaussSourceModel(freq_hz=200.0, depth_m=100.0, beam_width_deg=10.0)
env = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=1000.0,
            sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                z_grid_m=jnp.array([0.0, 1000.0]),
                sound_speed=jnp.array([1500.0, 1520.0])
            ),
            density=1.0,
            attenuation_dm_lambda=0.0
        ),
        UnderwaterLayerModel(
            height_m=jnp.inf,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
            density=1.5,
            attenuation_dm_lambda=0.0
        )
    ]
)

params = ComputationalParams(
    max_range_m=50000,
    max_depth_m=500,
    x_output_points=500,
    z_output_points=500,
)

field = uwa_forward_task(src=src, env=env, params=params)
plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(field.field+1e-16)).T,
    norm=Normalize(vmin=-80, vmax=-35),
    aspect='auto',
    extent=[0, field.x_grid[-1], field.z_grid[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()


model = uwa_get_model(src=src, env=env, params=params)

env.layers[0].sound_speed_profile_m_s = ConstWaveSpeedModel(c0=1500.0)
c0 = env.layers[0].sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, model.z_computational_grid())

f = model.compute(init)
field2 = AcousticPressureField(freq_hz=src.freq_hz, x_grid=model.x_output_grid(), z_grid=model.z_output_grid(),
                                 field=f)

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(field2.field+1e-16)).T,
    norm=Normalize(vmin=-80, vmax=-35),
    aspect='auto',
    extent=[0, field2.x_grid[-1], field2.z_grid[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()
