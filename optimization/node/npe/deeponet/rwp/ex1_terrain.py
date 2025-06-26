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
angles_deg = [0.0]
src_height_m = 75
terrain_func = interp1d(
    x=[0, 2000, 3000, 6000, 8500, 10000, 20000, 22000, 25000, 30000, 50000, 60000, 70000, 80000, 90000, 100000],
    y=[50, 42, 38, 25, 20, 30, 10, 20, 0, 0, 0, 30, 40, 40, 45, 30],
    fill_value="extrapolate")

model = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz,
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

inv_G_model = DeepONet(
    rngs=nnx.Rngs(1703),
    samples_num=proxy(N_profile_generator(grid)).shape[0],
    interact_size=100,
    branch_scale=(mean, var),
    trunk_scale=(max_height/2, max_height/2)
)
G_inv, losses = learn_inverse_G(
    proxy,
    lambda k: N_profile_generator(grid, k),
    inv_G_model,
    grid,
    max_epoch_num=1000,
    batch_size=25,
    tx = optax.adam(learning_rate=0.002, b1=0.9)
)


plt.figure(figsize=(6, 3.2))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.xlim([0, len(losses)-1])
plt.tight_layout()
plt.savefig('losses.eps')


model2 = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz+0.0001,
    angles_deg=angles_deg,
    max_range_m=max_range_m,
    src_height_m=src_height_m,
    terrain=terrain_func,
)
proxy2 = Proxy(model2, grid)

f, ax = plt.subplots(1, 4, figsize=(6, 3.2), constrained_layout=True)
vis_grid = jnp.linspace(1, max_height, 151)
p0 = trilinear_duct_N_profile_generator(grid, random.PRNGKey(1766567))
p0_inv = G_inv(proxy2(p0), grid)
p0_inv = (p0 + p0_inv) / 2
tau = (1 + jnp.sin(10*vis_grid / vis_grid[-1])+1) / 2 + 0.2
p0_inv_ad = (tau*p0 + (1-tau)*p0_inv)
ax[0].plot(model2.env.M_profile(vis_grid), vis_grid, color='black')
proxy2(p0_inv)
ax[0].plot(model2.env.M_profile(vis_grid), vis_grid, color='blue')
proxy2(p0_inv_ad)
ax[0].plot(model2.env.M_profile(vis_grid), vis_grid, color='red')
ax[0].grid(True)
ax[0].set_ylim([vis_grid[0], vis_grid[-1]])
ax[0].set_xlabel('M-profile')
ax[0].set_ylabel('Height (m)')

p1 = trilinear_duct_N_profile_generator(grid, random.PRNGKey(176666567))
p1_inv = G_inv(proxy2(p1), grid)
p1_inv_ad = (tau*p1 + (1-tau)*p1_inv)
ax[1].plot(model2.env.M_profile(vis_grid), vis_grid, color='black')
proxy2(p1_inv)
ax[1].plot(model2.env.M_profile(vis_grid), vis_grid, color='blue')
proxy2(p1_inv_ad)
ax[1].plot(model2.env.M_profile(vis_grid), vis_grid, color='red')
ax[1].set_yticklabels([])
ax[1].grid(True)
ax[1].set_ylim([vis_grid[0], vis_grid[-1]])
ax[1].set_xlabel('M-profile')

p2 = evaporation_duct_N_profile_generator(grid, random.PRNGKey(176666567))
p2_inv = G_inv(proxy2(p2), grid)
p2_inv_ad = (tau*p2 + (1-tau)*p2_inv)
ax[2].plot(model2.env.M_profile(vis_grid), vis_grid, color='black')
proxy2(p2_inv)
ax[2].plot(model2.env.M_profile(vis_grid), vis_grid, color='blue')
proxy2(p2_inv_ad)
ax[2].plot(model2.env.M_profile(vis_grid), vis_grid, color='red')
ax[2].set_yticklabels([])
ax[2].grid(True)
ax[2].set_ylim([vis_grid[0], vis_grid[-1]])
ax[2].set_xlabel('M-profile')

p3 = surface_duct_N_profile_generator(grid, random.PRNGKey(1677777))
p3_inv = G_inv(proxy2(p3), grid)
p3_inv = (p3 + p3_inv) / 2
p3_inv_ad = (tau*p3 + (1-tau)*p3_inv)
ax[3].plot(model2.env.M_profile(vis_grid), vis_grid, color='black')
proxy2(p3_inv)
ax[3].plot(model2.env.M_profile(vis_grid), vis_grid, color='blue')
proxy2(p3_inv_ad)
ax[3].plot(model2.env.M_profile(vis_grid), vis_grid, color='red')
ax[3].set_yticklabels([])
ax[3].grid(True)
ax[3].set_ylim([vis_grid[0], vis_grid[-1]])
ax[3].set_xlabel('M-profile')

legend_elements = [Line2D([0], [0], color='black', lw=1, label='True profile'),
                   Line2D([0], [0], color='blue', lw=1, label='DeepONet'),
                   Line2D([0], [0], color='red', lw=1, label='DeepONet + AD PE')]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=3)
f.tight_layout()
plt.show()

model_vis = MultiAngleRWPModel(
    measure_points_x=[-1],
    measure_points_z=measure_points_z,
    freq_hz=freq_hz+0.000001,
    angles_deg=angles_deg,
    max_range_m=100000,
    src_height_m=src_height_m,
    terrain=terrain_func,
)
proxy_vis = Proxy(model_vis, grid)

field0 = proxy_vis.calc_field(p0)
field0_inverted = proxy_vis.calc_field(p0_inv_ad)
err0 = 20*jnp.log10(jnp.abs(field0)+1e-16) - 20*jnp.log10(jnp.abs(field0_inverted)+1e-16)

field1 = proxy_vis.calc_field(p1)
field1_inverted = proxy_vis.calc_field(p1_inv_ad)
err1 = 20*jnp.log10(jnp.abs(field1)+1e-16) - 20*jnp.log10(jnp.abs(field1_inverted)+1e-16)

field2 = proxy_vis.calc_field(p2)
field2_inverted = proxy_vis.calc_field(p2_inv_ad)
err2 = 20*jnp.log10(jnp.abs(field2)+1e-16) - 20*jnp.log10(jnp.abs(field2_inverted)+1e-16)

field3 = proxy_vis.calc_field(p3)
field3_inverted = proxy_vis.calc_field(p3_inv_ad)
err3 = 20*jnp.log10(jnp.abs(field3)+1e-16) - 20*jnp.log10(jnp.abs(field3_inverted)+1e-16)

f, ax = plt.subplots(2, 4, figsize=(12, 4.0), constrained_layout=True)
extent = (model_vis.fwd_model[0].x_output_grid()[0]*1e-3, model_vis.fwd_model[0].x_output_grid()[-1]*1e-3,
          model_vis.fwd_model[0].z_output_grid()[0], model_vis.fwd_model[0].z_output_grid()[-1])
terrain_grid = jnp.array([terrain_func(v) for v in model_vis.fwd_model[0].x_output_grid()])
norm = Normalize(-70, -10)

im = ax[0, 0].imshow(20*jnp.log10(jnp.abs(field0_inverted+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0, 0].grid(True)
ax[0, 0].set_xticklabels([])
ax[0, 0].set_ylabel('Height (m)')
ax[0, 0].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[0, 0].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[0, 1].imshow(20*jnp.log10(jnp.abs(field1_inverted+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0, 1].grid(True)
ax[0, 1].set_xticklabels([])
ax[0, 1].set_yticklabels([])
ax[0, 1].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[0, 1].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[0, 2].imshow(20*jnp.log10(jnp.abs(field2_inverted+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0, 2].grid(True)
ax[0, 2].set_xticklabels([])
ax[0, 2].set_yticklabels([])
ax[0, 2].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[0, 2].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[0, 3].imshow(20*jnp.log10(jnp.abs(field3_inverted+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0, 3].grid(True)
ax[0, 3].set_xticklabels([])
ax[0, 3].set_yticklabels([])
ax[0, 3].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[0, 3].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, ax=ax[0, :], shrink=0.9, location='right')

discrete_cm = matplotlib.colors.ListedColormap([(1, 0, 0), (0.5, 1, 1), (1, 1, 1), (1, 1, 0.5), (0, 0, 1)])
norm = Normalize(-25, 25)

im = ax[1, 0].imshow(err0.T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=discrete_cm)
ax[1, 0].grid(True)
ax[1, 0].set_xlabel('Range (km)')
ax[1, 0].set_ylabel('Height (m)')
ax[1, 0].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[1, 0].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[1, 1].imshow(err1.T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=discrete_cm)
ax[1, 1].grid(True)
ax[1, 1].set_xlabel('Range (km)')
ax[1, 1].set_yticklabels([])
ax[1, 1].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[1, 1].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[1, 2].imshow(err2.T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=discrete_cm)
ax[1, 2].grid(True)
ax[1, 2].set_xlabel('Range (km)')
ax[1, 2].set_yticklabels([])
ax[1, 2].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[1, 2].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

ax[1, 3].imshow(err3.T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=discrete_cm)
ax[1, 3].grid(True)
ax[1, 3].set_xlabel('Range (km)')
ax[1, 3].set_yticklabels([])
ax[1, 3].plot(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid, 'k')
ax[1, 3].fill_between(model_vis.fwd_model[0].x_output_grid()*1e-3, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, ax=ax[1, :], shrink=0.9, location='right')
plt.show()