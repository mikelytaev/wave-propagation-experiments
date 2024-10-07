import jax.numpy as jnp
import math as fm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from experiments.optimization.node.expected import expected_value_jacfwd, expected_value_quad
from experimental.helmholtz_jax import LinearSlopeWaveSpeedModel
from experimental.uwa_jax import GaussSourceModel, UnderwaterEnvironmentModel, UnderwaterLayerModel, \
    ComputationalParams, uwa_get_model

# import jax
# jax.config.update("jax_enable_x64", True)

# @jax.jit
# def func(z):
#     return (jnp.sin(z)*jnp.exp(jnp.cos(z) / z) + jnp.cos(z)) / jnp.exp(10*(z + 1))
#
# v = expected_value(func, 1.0, 1.0, 10)


src = GaussSourceModel(freq_hz=50.0, depth_m=100.0, beam_width_deg=10.0)
env = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=1000.0,
            sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(
                c0=1500.0,
                slope_degrees=1.0
            ),
            density=1.0,
            attenuation_dm_lambda=0.0
        ),
        # UnderwaterLayerModel(
        #     height_m=jnp.inf,
        #     sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
        #     density=1.5,
        #     attenuation_dm_lambda=0.0
        # )
    ]
)

params = ComputationalParams(
    max_range_m=100000,
    max_depth_m=500,
    x_output_points=300,
    z_output_points=100,
)

model = uwa_get_model(src=src, env=env, params=params)

c0 = env.layers[0].sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, model.z_computational_grid())


def field_by_slope(slope: float):
    #model.wave_speed.uwem.layers[0].sound_speed_profile_m_s.c0 = c0
    f = model.compute(init, slope)# + 1e-7
    return jnp.abs(f)


v0 = expected_value_jacfwd(field_by_slope, 0.5, 0.2, 1)
v_q = expected_value_quad(field_by_slope, 0.5, 0.2, 51)
v_d = expected_value_jacfwd(field_by_slope, 0.5, 0.2, 5)

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(v_d.T+1e-10),
    norm=Normalize(vmin=-80, vmax=-35),
    aspect='auto',
    extent=[0, model.x_output_grid()[-1], model.z_output_grid()[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()

# v = field_by_c0(1500.0)
# v0 = field_by_c0(1511.0)


plt.figure(figsize=(6, 3.2))
plt.imshow(
    (20*jnp.log10(v_q) - 20*jnp.log10(v0)).T,
    norm=Normalize(vmin=-10, vmax=10),
    aspect='auto',
    extent=[0, model.x_output_grid()[-1], model.z_output_grid()[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 3.2))
plt.imshow(
    (20*jnp.log10(v_d+1e-16) - 20*jnp.log10(v0+1e-16)).T,
    norm=Normalize(vmin=-10, vmax=10),
    aspect='auto',
    extent=[0, model.x_output_grid()[-1], model.z_output_grid()[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()