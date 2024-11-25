import logging

from experimental.rwp_jax import GaussSourceModel, TroposphereModel, ComputationalParams, create_rwp_model, \
    EvaporationDuctModel
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
logging.basicConfig(level=logging.DEBUG)

src = GaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
env = TroposphereModel(
    N_profile=EvaporationDuctModel(height_m=20, truncate_height_m=200)
)
params = ComputationalParams(
    max_range_m=50000,
    max_height_m=250,
    dx_m=100,
    dz_m=1
)
model = create_rwp_model(src, env, params)
field = model.compute(src.aperture(model.z_computational_grid()))

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(field[:,::-1]+1e-16)).T,
    norm=Normalize(vmin=-70, vmax=-20),
    aspect='auto',
    extent=[0, model.x_output_grid()[-1], 0, model.z_output_grid()[-1]],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()


plt.figure()
z_grid_m = jnp.linspace(0, 300, 301)
plt.plot(env.M_profile(z_grid_m), z_grid_m)
plt.show()