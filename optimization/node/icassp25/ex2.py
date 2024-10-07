import jax.numpy as jnp
from matplotlib.colors import Normalize

from experimental.helmholtz_jax import PiecewiseLinearWaveSpeedModel, \
    ConstWaveSpeedModel, RationalHelmholtzPropagator
from experimental.uwa_jax import ComputationalParams, GaussSourceModel, UnderwaterEnvironmentModel, \
    uwa_get_model, uwa_forward_task, UnderwaterLayerModel
from experiments.optimization.node.objective_functions import bartlett
import math as fm
import jax
from scipy.optimize import minimize

import matplotlib.pyplot as plt

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
    max_range_m=5000,
    max_depth_m=250,
    x_output_points=5,
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


replica_z_grid_m = jnp.linspace(0, 200, 20)
env_replica = UnderwaterEnvironmentModel(
        layers=[
            UnderwaterLayerModel(
                height_m=200.0,
                sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                    z_grid_m=replica_z_grid_m,
                    sound_speed=env_simulated.layers[0].sound_speed_profile_m_s(replica_z_grid_m)+jax.random.uniform(jax.random.PRNGKey(0), (20,))*0.2,
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
measure = field.field[-1, :]


def train_linear(training_model: RationalHelmholtzPropagator, env_replica: UnderwaterEnvironmentModel):
    model_points_num = len(env_replica.layers[0].sound_speed_profile_m_s.z_grid_m)
    etalon_loss0_v = 1 / bartlett(measure[2:390:10], measure[2:390:10])


    def loss0(sound_speed_vals):
        env_replica.layers[0].sound_speed_profile_m_s.sound_speed = sound_speed_vals
        f = training_model.compute(init)[-1, :]
        return 1 / bartlett(measure[2:390:10], f[2:390:10]) - etalon_loss0_v


    def loss1(sound_speed_vals):
        return jnp.linalg.norm(jnp.diff(sound_speed_vals) / (env_replica.layers[0].height_m / (model_points_num - 1)))**2 * 3


    def loss_func(sound_speed_vals):
         return loss0(sound_speed_vals) + loss1(sound_speed_vals)


    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f), primals, tangents)[1]


    @jax.jit
    def hessp_loss(x, v):
        return hvp(loss_func, (x,), (v,))


    @jax.jit
    def jac_loss(x):
        return jax.grad(loss_func)(x)


    def dual_annealing_callback(x, f, ctx):
        reg_loss = loss1(x)
        print(f'x = {x}; loss = {f}; reg_loss = {reg_loss}; ctx = {ctx}')
        return f - reg_loss < 0.5


    # ret = dual_annealing(
    #     func=loss_func,
    #     maxiter=50,
    #     bounds=[(1485.0, 1515.0)]*model_points_num,
    #     callback=dual_annealing_callback,
    #     maxfun=500,
    #     minimizer_kwargs={
    #         'method': 'L-BFGS-B',
    #         #'fun': loss_func,
    #         #'x0'=sound_speed_vals,
    #         'jac': jac_loss,
    #         'hessp': hessp_loss,
    #         'callback': lambda xk: print(xk),
    #         'options': {
    #                 'maxiter': 15,
    #                 #'maxfun': 10
    #             }
    #         },
    #     #no_local_search=True
    # )

    m = minimize(
        method='L-BFGS-B',
        fun=loss_func,
        x0=env_replica.layers[0].sound_speed_profile_m_s.sound_speed,
        jac=jac_loss,
        hessp=hessp_loss,
        callback=lambda xk: print(f'{xk}, {loss_func(xk)}'),
        options={
            #'disp': True,
        }
    )

    sound_speed_vals = m.x
    print(m)
    env_replica.layers[0].sound_speed_profile_m_s.sound_speed = sound_speed_vals
    return sound_speed_vals


# n_iters = 5
# for i in range(0, n_iters):
#     train_linear(training_model, env_replica)
#     if i < n_iters - 1:
#         z_grid_m_t = jnp.linspace(0, 200, len(env_replica.layers[0].sound_speed_profile_m_s.z_grid_m)+1)
#         env_replica.layers[0].sound_speed_profile_m_s.sound_speed = env_replica.layers[0].sound_speed_profile_m_s(z_grid_m_t)
#         env_replica.layers[0].sound_speed_profile_m_s.z_grid_m = z_grid_m_t


train_linear(training_model, env_replica)


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

