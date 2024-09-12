import jax.numpy as jnp
import numpy as np
from matplotlib.colors import Normalize

from experiments.optimization.node.helmholtz_jax import LinearSlopeWaveSpeedModel, PiecewiseLinearWaveSpeedModel, \
    ConstWaveSpeedModel, RationalHelmholtzPropagator
from experiments.optimization.node.uwa_jax import ComputationalParams, GaussSourceModel, UnderwaterEnvironmentModel, \
    uwa_get_model, uwa_forward_task, UnderwaterLayerModel
from experiments.optimization.node.objective_functions import bartlett
import math as fm
import jax
import jax.random as jr
from scipy.optimize import minimize, dual_annealing

import matplotlib.pyplot as plt

from uwa.field import AcousticPressureField

import optax

import logging
#logging.basicConfig(level=logging.DEBUG)


src = GaussSourceModel(freq_hz=500.0, depth_m=50.0, beam_width_deg=10.0)
env_simulated = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                z_grid_m=jnp.array([0.0, 100, 200.0]),
                sound_speed=jnp.array([1510.0, 1500.0, 1505.0])
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
    max_range_m=2000,
    max_depth_m=250,
    x_output_points=3,
    z_output_points=100,
)

field = uwa_forward_task(src=src, env=env_simulated, params=params)
plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(field.field+1e-16)).T,
    norm=Normalize(vmin=-70, vmax=-20),
    aspect='auto',
    extent=[0, field.x_grid[-1], field.z_grid[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()


model_points_num = 3

env_replica = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                z_grid_m=jnp.linspace(0, 200, model_points_num),
                sound_speed=jnp.ones(model_points_num, dtype=float) * 1500.0
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


training_model = uwa_get_model(src=src, env=env_replica, params=params)

c0 = env_simulated.layers[0].sound_speed_profile_m_s(src.depth_m)
k0 = 2 * fm.pi * src.freq_hz / c0
init = src.aperture(k0, training_model.z_computational_grid())

measure = field.field[-1,:]


#@jax.value_and_grad

etalol_loss0_v = 1 / bartlett(measure[2:390:10], measure[2:390:10])


def loss0(sound_speed_vals):
    env_replica.layers[0].sound_speed_profile_m_s.sound_speed = sound_speed_vals
    f = training_model.compute(init)[-1, :]
    return 1 / bartlett(measure[2:390:10], f[2:390:10]) - etalol_loss0_v


def loss1(sound_speed_vals):
    return jnp.linalg.norm(jnp.diff(sound_speed_vals) / (env_replica.layers[0].height_m / (model_points_num - 1)))**2 * 3


@jax.jit
def loss_func(sound_speed_vals):
     return loss0(sound_speed_vals) + loss1(sound_speed_vals)


sound_speed_vals = jnp.ones(model_points_num, dtype=float) * 1500.0


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


@jax.jit
def hessp_loss(x, v):
    return hvp(loss_func, (x,), (v,))


@jax.jit
def jac_loss(x):
    return jax.grad(loss_func)(x)


ideal = jnp.array([1500.0, 1505.0, 1510.0, 1505.0, 1500.0])
#print(f'ideal_loss0: {loss0(ideal)}, ideal_loss_reg: {loss1(ideal)}')


def dual_annealing_callback(x, f, ctx):
    reg_loss = loss1(x)
    print(f'x = {x}; loss = {f}; reg_loss = {reg_loss}; ctx = {ctx}')
    return f - reg_loss < 0.5


ret = dual_annealing(
    func=loss_func,
    maxiter=5000,
    bounds=[(1485.0, 1515.0)]*model_points_num,
    callback=dual_annealing_callback,
    maxfun=100000,
    minimizer_kwargs={
        'method': 'L-BFGS-B',
        #'fun': loss_func,
        #'x0'=sound_speed_vals,
        'jac': jac_loss,
        'hessp': hessp_loss,
        'callback': lambda xk: print(xk),
        'options': {
                'maxiter': 50,
                #'maxfun': 10
            }
        },
    #no_local_search=True
)

sound_speed_vals = ret.x
print(ret)

fff


env_replica.layers[0].sound_speed_profile_m_s.sound_speed = sound_speed_vals
f_replica = training_model.compute(init)
plt.figure(figsize=(6, 3.2))
plt.imshow(
    20*jnp.log10(jnp.abs(f_replica+1e-16)).T,
    norm=Normalize(vmin=-70, vmax=-20),
    aspect='auto',
    extent=[0, field.x_grid[-1], field.z_grid[-1], 0],
    cmap=plt.get_cmap('jet')
)
plt.colorbar()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(field.x_grid, 20*jnp.log10(jnp.abs(field.field[-1,:]+1e-16)))
plt.plot(field.x_grid, 20*jnp.log10(jnp.abs(f_replica[-1,:]+1e-16)))
plt.grid(True)
plt.ylim([-70, -20])
plt.show()


# m = minimize(
#     method='BFGS',
#     fun=loss_func,
#     x0=sound_speed_vals,
#     jac=jac_loss,
#     hessp=hessp_loss,
#     callback=lambda xk: print(xk),
#     options={
#         'disp': True,
#     }
# )

fff


vg_loss_func = optax.value_and_grad_from_state(loss_func)


#grad_loss_func = jax.jacrev(loss_func)


start_learning_rate = 10
#optimizer = optax.adam(start_learning_rate)
optimizer = optax.lbfgs()
# optimizer = optax.chain(
#    optax.sgd(learning_rate=1.),
#    optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
# )
opt_state = optimizer.init(sound_speed_vals)
print(opt_state)


def make_step(sound_speed_vals, z_batch_perm, opt_state):
    loss, grads = vg_loss_func(sound_speed_vals, state=opt_state)
    # loss = loss_func(sound_speed_vals, z_batch_perm)
    # grads = grad_loss_func(sound_speed_vals, z_batch_perm)
    updates, opt_state = optimizer.update(grads, opt_state, sound_speed_vals, value=loss, grad=grads, value_fn=loss_func)
    sound_speed_vals = optax.apply_updates(sound_speed_vals, updates)
    #sound_speed_vals = optax.projections.projection_box(sound_speed_vals, 1490.0*jnp.ones(model_points_num), 1510.0*jnp.ones(model_points_num))
    return loss, sound_speed_vals, opt_state


def dataloader(size: int, batch_size: int, key):
    indices = jnp.arange(size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        end = batch_size
        batch_perm = perm[0:end]
        yield batch_perm


loader = dataloader(len(measure), batch_size=len(measure), key=jr.PRNGKey(12345))
for step in range(50):
    z_batch_perm = next(loader)
    loss, sound_speed_vals, opt_state = make_step(sound_speed_vals, z_batch_perm, opt_state)
    print(f'i = {step}; Loss = {loss}')
    print(f'Loss1 = {loss1(sound_speed_vals)}')
    print(sound_speed_vals)


