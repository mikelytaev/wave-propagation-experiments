import math as fm
import time
from copy import deepcopy
from typing import List

import jax
import matplotlib.pyplot as plt
from attr import dataclass
from jax import numpy as jnp
from scipy.optimize import minimize, basinhopping, dual_annealing

from experimental.helmholtz_jax import RationalHelmholtzPropagator, AbstractWaveSpeedModel, \
    PiecewiseLinearWaveSpeedModel, ConstWaveSpeedModel
from experiments.optimization.node.objective_functions import bartlett
from experimental.uwa_jax import GaussSourceModel, UnderwaterEnvironmentModel, UnderwaterLayerModel, \
    ComputationalParams, uwa_get_model

jax.config.update("jax_enable_x64", True)

src = GaussSourceModel(
        freq_hz=500,
        depth_m=50.0,
        beam_width_deg=10.0
    )

env_simulated = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                z_grid_m=jnp.array([0.0, 200.0]),
                sound_speed=jnp.array([1500.0, 1510.0])
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

max_depth_m = 250
computational_params = ComputationalParams(
    max_range_m=5000,
    max_depth_m=max_depth_m,
    dx_m=5000/5,
    dz_m=max_depth_m/100,
)

simulated_model = uwa_get_model(
        src=src,
        env=env_simulated,
        params=computational_params
    )

arrays_num = 1
measure_points_depth = jnp.array([5, 10, 20, 30, 40, 50, 60, 70])
measure_points_range = -jnp.arange(1, arrays_num+1, 1)

def get_field(model: RationalHelmholtzPropagator, src: GaussSourceModel, env: UnderwaterEnvironmentModel):
    c0 = env.layers[0].sound_speed_profile_m_s(src.depth_m)
    k0 = 2 * fm.pi * src.freq_hz / c0
    init = src.aperture(k0, model.z_computational_grid())
    return model.compute(init)


env_replica = env_simulated

training_model = simulated_model

z_grid_m = jnp.linspace(0, 200, 3)

@jax.jit
def loss0(sound_speed_vals):
    etalon_loss0_v = 1 / bartlett(measure[measure_points_depth], measure[measure_points_depth])
    env_replica.layers[0].sound_speed_profile_m_s.z_grid_m = z_grid_m
    env_replica.layers[0].sound_speed_profile_m_s.sound_speed = sound_speed_vals
    f = get_field(training_model, src, env_replica)[measure_points_range, :]
    f = jnp.ravel(f)
    return 1 / bartlett(measure[measure_points_depth], f[measure_points_depth]) - etalon_loss0_v


@jax.jit
def loss1(sound_speed_vals):
    return jnp.linalg.norm(jnp.diff(sound_speed_vals) / (env_replica.layers[0].height_m / (2 - 1))) ** 2


gamma = 100.0


@jax.jit
def loss_func(sound_speed_vals):
    return loss0(sound_speed_vals) + gamma * loss1(sound_speed_vals)


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


@jax.jit
def hessp_loss(x, v):
    return hvp(loss_func, (x,), (v,))


@jax.jit
def jac_loss(x):
    return jax.grad(loss_func)(x)


measure = get_field(simulated_model, src, env_simulated)[measure_points_range, :]
measure = jnp.ravel(measure)

snr = 10
if snr:
    signal_level = jnp.mean(abs(measure) ** 2)
    noise_var = signal_level / (10 ** (snr / 10))
    noise_sigma_r = jnp.sqrt(noise_var / 2)
    r_t = jax.random.normal(jax.random.PRNGKey(1703), (len(measure), 2)) * noise_sigma_r
    noise = r_t[:, 0] + 1j * r_t[:, 1]
    measure += noise


# cs = jnp.linspace(1450, 1550, 501)
# losses = [loss_func(c, measure) for c in cs]
#
# plt.plot(cs, jnp.log10(jnp.array(losses)+1e-10))
# plt.grid(True)
# plt.show()

# ret = basinhopping(
#     func=loss_func,
#     x0=jnp.array([1550.0]),
#     minimizer_kwargs={
#             'method': 'L-BFGS-B',
#             #'fun': loss_func,
#             'args': (measure,),
#             'jac': jac_loss,
#     },
#     niter=200,
#     disp=True,
# )
# print(ret)


def dual_annealing_callback(x, f, context):
    print(x)
    if jnp.linalg.norm(jnp.diff(x) - 10.0/(len(x))-1) < 0.1:
        return True

#x0 = jnp.ones(len(z_grid_m)) * 1520.0
x0 = env_replica.layers[0].sound_speed_profile_m_s(z_grid_m) + jnp.linspace(0, 1, len(z_grid_m))
jac_loss(x0)
hessp_loss(x0, jnp.ones(len(z_grid_m)))

t = time.time()
# ret = dual_annealing(
#     func=loss_func,
#     args=(measure,),
#     x0=x0,
#     bounds=[(1450, 1550)]*len(x0),
#     minimizer_kwargs={
#             'method': 'L-BFGS-B',
#             #'fun': loss_func,
#             'jac': jac_loss,
#     },
#     #no_local_search=True,
#     maxiter=1000,
#     callback=dual_annealing_callback
# )
# ret = minimize(
#     method='L-BFGS-B',
#     fun=loss_func,
#     args=(measure,),
#     x0=x0,
#     jac=jac_loss,
# )
# ret = minimize(
#     method='BFGS',
#     fun=loss_func,
#     args=(measure,),
#     x0=x0,
#     jac=jac_loss,
# )
ret = minimize(
    method='Newton-CG',
    fun=loss_func,
    x0=x0,
    jac=jac_loss,
    hessp=hessp_loss,
)
# ret = basinhopping(
#     func=loss_func,
#     x0=x0,
#     #bounds=[(t-3, t+3) for t in x0],
#     minimizer_kwargs={
#             'method': 'Newton-CG',
#             #'fun': loss_func,
#             'jac': jac_loss,
#             'hessp': hessp_loss,
#     },
#     #no_local_search=True,
#     niter=1000,
#     callback=dual_annealing_callback
# )
opt_time = time.time() - t
print(ret)
print(opt_time)