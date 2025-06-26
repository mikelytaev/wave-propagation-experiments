import optax
from flax import nnx
from jax import numpy as jnp
from jax import random
from matplotlib.colors import Normalize
import matplotlib
from scipy.interpolate import interp1d

from experiments.optimization.node.npe.rwp_mimo.common import MultiAngleRWPModel, Proxy

import jax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.optimization.node.npe.deeponet.rwp.deeponet_inverse import DeepONet, learn_inverse_G
from experiments.optimization.node.npe.deeponet.rwp.profile_generators import surface_duct_N_profile_generator, \
    trilinear_duct_N_profile_generator, evaporation_duct_N_profile_generator, N_profile_generator

jax.config.update("jax_enable_x64", True)

freq_hz = 3E9
max_range_m = 30000
measure_points_z = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
angles_deg = [-0.7]
src_height_m = 75
terrain_func = interp1d(
    x=[0, 2000, 3000, 6000, 8500, 10000, 20000, 22000, 25000, 30000],
    y=[50, 42, 38, 25, 20, 30, 10, 5, 0, 0],
    fill_value="extrapolate")

model = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz,
    beam_width_deg=2.0,
    angles_deg=angles_deg,
    max_range_m=max_range_m,
    src_height_m=src_height_m,
    terrain=terrain_func,
)

max_height = 150
grid = jnp.linspace(0, max_height, 151)

proxy = Proxy(model, grid)

key = random.PRNGKey(345345)
m = []
for i in range(100):
    key = random.split(key, 1)[0]
    N_vals = N_profile_generator(grid, key)
    m += [proxy(N_vals)]

mean = jnp.mean(jnp.array(m), axis=0)
var = jnp.var(jnp.array(m), axis=0)

vis_grid = jnp.linspace(1, max_height, 151)
plt.figure(figsize=(6, 3.2))
p0 = trilinear_duct_N_profile_generator(grid, random.PRNGKey(1766567))
p1 = trilinear_duct_N_profile_generator(grid, random.PRNGKey(176666567))
p2 = evaporation_duct_N_profile_generator(grid, random.PRNGKey(176666567))
p3 = surface_duct_N_profile_generator(grid, random.PRNGKey(1677777))


model_vis = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz+0.00002,
    angles_deg=angles_deg,
    max_range_m=100000,
    src_height_m=src_height_m,
    terrain=terrain_func,
)
proxy_vis = Proxy(model_vis, grid)

field0 = proxy_vis.calc_field(p0)

field1 = proxy_vis.calc_field(p1)

field2 = proxy_vis.calc_field(p2)

field3 = proxy_vis.calc_field(p3)

f, ax = plt.subplots(1, 4, figsize=(12, 2.5), constrained_layout=True)
extent = (model_vis.fwd_model[0].x_output_grid()[0]*1e-3, model_vis.fwd_model[0].x_output_grid()[-1]*1e-3,
          model_vis.fwd_model[0].z_output_grid()[0], model_vis.fwd_model[0].z_output_grid()[-1])
#terrain_grid = np.array([terrain_func(v) for v in vis_model.fwd_model.x_output_grid()])
norm = Normalize(-70, -10)

im = ax[0].imshow(20*jnp.log10(jnp.abs(field0+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0].grid(True)
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Height (m)')
#ax[0].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
#ax[0].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[1].imshow(20*jnp.log10(jnp.abs(field1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[1].grid(True)
ax[1].set_xlabel('Range (km)')
ax[1].set_yticklabels([])
#ax[1].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
#ax[1].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[2].imshow(20*jnp.log10(jnp.abs(field2+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[2].grid(True)
ax[2].set_xlabel('Range (km)')
ax[2].set_yticklabels([])
#ax[2].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
#ax[2].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[3].imshow(20*jnp.log10(jnp.abs(field3+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[3].grid(True)
ax[3].set_xlabel('Range (km)')
ax[3].set_yticklabels([])
#ax[3].plot(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid, 'k')
#ax[3].fill_between(vis_model.fwd_model.x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, ax=ax[:], shrink=0.9, location='right')
plt.show()
