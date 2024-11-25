import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import math as fm

from experimental.helmholtz_jax import PiecewiseLinearWaveSpeedModel, \
    ConstWaveSpeedModel
from experiments.optimization.node.flax.utils import MLPWaveSpeedModel
from experiments.optimization.node.objective_functions import bartlett
from experimental.uwa_jax import GaussSourceModel, UnderwaterEnvironmentModel, UnderwaterLayerModel, \
    ComputationalParams, uwa_forward_task, uwa_get_model


jax.config.update("jax_enable_x64", True)
#logging.basicConfig(level=logging.DEBUG)


src = GaussSourceModel(freq_hz=200.0, depth_m=50.0, beam_width_deg=10.0)
env_simulated = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                z_grid_m=jnp.array([0.0, 100, 200.0]),
                sound_speed=jnp.array([1505.0, 1500.0, 1510.0])
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
    max_range_m=10000,
    max_depth_m=250,
    dx_m=500,
    dz_m=1
)

field = uwa_forward_task(src=src, env=env_simulated, params=params)
measure_points_depth = jnp.arange(1, 150, 10)#jnp.array([5, 10, 20, 30, 40, 50, 60, 70])
measure = field.field[-1,:]
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

env_replica = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=MLPWaveSpeedModel(z_max_m=200),
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




#@jax.value_and_grad

etalol_loss0_v = 1 / bartlett(measure[measure_points_depth], measure[measure_points_depth])


def loss0(params):
    env_replica.layers[0].sound_speed_profile_m_s.params = params
    f = training_model.compute(init)[-1, :]
    return 1 / bartlett(measure[measure_points_depth], f[measure_points_depth]) - etalol_loss0_v


def loss1(params):
    env_replica.layers[0].sound_speed_profile_m_s.params = params
    z_grid_m = jnp.linspace(0, env_replica.layers[0].height_m, 201)
    dz = z_grid_m[1] - z_grid_m[0]
    return jnp.linalg.norm(jnp.diff(env_replica.layers[0].sound_speed_profile_m_s(z_grid_m)) / dz) ** 2


@jax.jit
def loss_func(params):
    l0 = loss0(params)
    l1 = loss1(params)
    return l0 + 1000*l1


def jac_loss(x):
    return jax.grad(loss_func)(x)


import optax
learning_rate = 0.002
tx = optax.adam(learning_rate=learning_rate)

opt_params = env_replica.layers[0].sound_speed_profile_m_s.params

opt_state = tx.init(opt_params)
loss_grad_fn = jax.value_and_grad(loss_func)

for i in range(5000):
    l0 = loss0(opt_params)
    l1 = loss1(opt_params)
    print(f'l0 = {l0}; l1 = {l1}')
    loss_val, grads = loss_grad_fn(opt_params)
    updates, opt_state = tx.update(grads, opt_state)
    opt_params = optax.apply_updates(opt_params, updates)
    if True:
        env_replica.layers[0].sound_speed_profile_m_s.params = opt_params
        print(f'Loss step {i}: {loss_val}; ssp(1): {env_replica.layers[0].sound_speed_profile_m_s([1.0])}; ssp(100): {env_replica.layers[0].sound_speed_profile_m_s([100.0])}')

plt.figure()
z_grid_m = jnp.linspace(0, 200, 201)
plt.plot(env_replica.layers[0].sound_speed_profile_m_s(z_grid_m), z_grid_m)
plt.plot(env_simulated.layers[0].sound_speed_profile_m_s(z_grid_m), z_grid_m)
plt.show()
