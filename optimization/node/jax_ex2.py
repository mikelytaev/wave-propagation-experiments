import time

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.random as jr
import optax
from matplotlib.colors import Normalize
import scipy


import matplotlib.pyplot as plt

from experiments.optimization.node.utils import RationalHelmholtzPropagator, EmptyRefractiveIndexModel, MunkProfileModel
import math as fm

from uwa.environment import munk_profile
from uwa.source import GaussSource


src = GaussSource(
    freq_hz=50,
    depth_m=500,
    beam_width_deg=10,
    elevation_angle_deg=0
)
k0 = 2*jnp.pi*src.freq_hz/1500
z_grid = np.linspace(0, 4000, 1000)

het = munk_profile(z_grid)
het = (1500/het)**2-1

x_max_m = 20000

model = RationalHelmholtzPropagator(
    k0=k0,
    dx_m=100,
    dz_m=z_grid[1]-z_grid[0],
    z_n=1000,
    order=(7, 8),
    x_max_m=x_max_m,
    refractive_index=MunkProfileModel(ref_depth=1300)
)

init = jnp.array(src.aperture(k0=k0, z=z_grid), dtype=complex)

start = time.time()
r = model.compute(init)
end = time.time()
print(f'time = {end-start}')

start = time.time()
#model.refractive_index = MunkProfileModel()
r = model.compute(init)
end = time.time()
print(f'time = {end-start}')
#
# start = time.time()
# #model.ref_depth = 1
# r = model.compute(init)
# end = time.time()
# print(f'time = {end-start}')

plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(r+1e-16)).T,
    norm=Normalize(vmin=-120, vmax=-40),
    aspect='auto',
    extent=[0, model.x_max_m, z_grid[-1], z_grid[0]],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 3.2))
plt.imshow(
    jnp.angle(r).T,
    norm=Normalize(vmin=-fm.pi/2, vmax=fm.pi/2),
    aspect='auto',
    extent=[0, model.x_max_m, z_grid[-1], z_grid[0]],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 3.2))
plt.imshow(
    jnp.unwrap(jnp.angle(r), axis=1).T,
    norm=Normalize(vmin=-100*fm.pi/2, vmax=100*fm.pi/2),
    aspect='auto',
    extent=[0, model.x_max_m, z_grid[-1], z_grid[0]],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()




@jax.value_and_grad
def loss_vertical(model: RationalHelmholtzPropagator, z_batch_perm, yi, init):
    y_pred = model.compute(init)[-1, z_batch_perm]
    m = 20*jnp.log10(jnp.abs((yi+1e-15)/(y_pred+1e-15)))
    m2 = jnp.diff(jnp.unwrap(jnp.angle(y_pred))) - jnp.diff(jnp.unwrap(jnp.angle(yi)))
    return jnp.mean(m ** 2) + jnp.mean(m2 ** 2)


@jax.value_and_grad
def loss_vertical_abs(model: RationalHelmholtzPropagator, z_batch_perm, yi, init):
    y_pred = model.compute(init)[-1, z_batch_perm]
    m = 20*jnp.log10(jnp.abs((yi+1e-15)/(y_pred+1e-15)))
    return jnp.mean(m**2)


@jax.value_and_grad
def loss_vertical_phase(model: RationalHelmholtzPropagator, z_batch_perm, yi, init):
    y_pred = model.compute(init)[-1, z_batch_perm]
    m2 = jnp.diff(jnp.unwrap(jnp.angle(y_pred))) - jnp.diff(jnp.unwrap(jnp.angle(yi)))
    return jnp.mean(m2**2)


def dataloader(yi, batch_size: int, *, key):
    indices = jnp.arange(int(yi.shape[0]/12)) + 550
    print(f'{z_grid[indices[0]]} {z_grid[indices[-1]]} {len(indices)}')
    while True:
        #perm = jr.permutation(key, indices)
        index = jr.randint(key, (1,), 0, len(indices)-batch_size)[0]
        (key,) = jr.split(key, 1)
        #end = batch_size
        #batch_perm = perm[0:end]
        batch_perm = indices[index:index+batch_size]
        yield batch_perm, yi[batch_perm]


key = jr.PRNGKey(12345)
batch_size = 83

c_abs = 1.0
c_phase = 1.0 * jnp.pi / 180.0

optim = optax.adabelief(100)
training_model = RationalHelmholtzPropagator(
    k0=k0,
    dx_m=100,
    dz_m=z_grid[1]-z_grid[0],
    z_n=1000,
    refractive_index=MunkProfileModel(
        ref_sound_speed=1455.0,
        ref_depth=1280.0),
    order=(7, 8),
    x_max_m=x_max_m
)
#training_model.compute(init)
opt_state = optim.init(training_model)
print(opt_state)


@jax.jit
def make_step(model, z_batch_perm, f_batch, opt_state, init):
    loss_abs, grads = loss_vertical_abs(model, z_batch_perm, f_batch, init)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    loss_phase, grads = loss_vertical_phase(model, z_batch_perm, f_batch, init)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    loss_abs, _ = loss_vertical_abs(model, z_batch_perm, f_batch, init)
    loss_phase, _ = loss_vertical_phase(model, z_batch_perm, f_batch, init)

    return loss_abs, loss_phase, model, opt_state


def xi_estimate(mean_sqr, n, sigma):
    p = scipy.stats.chi2.ppf(0.95, n)
    return sigma ** 2 * p / n


loader = dataloader(r[-1, :], batch_size, key=key)
for step in range(10000):
    z_batch_perm, f_batch = next(loader)
    start = time.time()
    loss_abs, loss_phase, training_model, opt_state = make_step(training_model, z_batch_perm, f_batch, opt_state, init)
    end = time.time()
    print(f'Loss abs = {loss_abs}, Loss phase = {loss_phase}, ref_sound_speed = {training_model.refractive_index.ref_sound_speed}, ref_depth = {training_model.refractive_index.ref_depth}, time = {end-start}')
    abs_est = xi_estimate(loss_abs, batch_size, c_abs/3)
    phase_est = xi_estimate(loss_phase, batch_size, c_phase/3)
    print(f'Loss abs xi est = {abs_est}, Loss phase xi est = {phase_est}')

    if loss_abs < abs_est and loss_phase < phase_est:
        print(f'finished! - {step} steps')
        break
