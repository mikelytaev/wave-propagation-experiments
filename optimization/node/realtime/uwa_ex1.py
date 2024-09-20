from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from jax.experimental.array_api import linspace
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
import time

import logging
#logging.basicConfig(level=logging.DEBUG)

jax.config.update("jax_enable_x64", True)


src = GaussSourceModel(
    freq_hz=200.0,
    depth_m=50.0,
    beam_width_deg=10.0
)
env_simulated = UnderwaterEnvironmentModel(
    layers=[
        UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=PiecewiseLinearWaveSpeedModel(
                z_grid_m=jnp.array([0.0, 200.0]),
                sound_speed=jnp.array([1500.0, 1500.0])
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

computational_params = ComputationalParams(
    max_range_m=10000,
    max_depth_m=250,
    x_output_points=5,
    z_output_points=100,
)

simulated_model = uwa_get_model(
    src=src,
    env=env_simulated,
    params=computational_params
)


def get_field(model: RationalHelmholtzPropagator, src: GaussSourceModel, env: UnderwaterEnvironmentModel):
    c0 = env.layers[0].sound_speed_profile_m_s(src.depth_m)
    #c0 = 1500.0
    k0 = 2 * fm.pi * src.freq_hz / c0
    init = src.aperture(k0, model.z_computational_grid())
    return model.compute(init)


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


training_model = uwa_get_model(
    src=src,
    env=env_replica,
    params=computational_params
)



model_points_num = len(env_replica.layers[0].sound_speed_profile_m_s.z_grid_m)


def loss0(sound_speed_vals, measure):
    etalon_loss0_v = 1 / bartlett(measure[2:390:10], measure[2:390:10])
    env_replica.layers[0].sound_speed_profile_m_s.sound_speed = sound_speed_vals
    f = get_field(training_model, src, env_replica)[-1, :]
    return 1 / bartlett(measure[2:390:10], f[2:390:10]) - etalon_loss0_v


def loss1(sound_speed_vals):
    return jnp.linalg.norm(jnp.diff(sound_speed_vals) / (env_replica.layers[0].height_m / (model_points_num - 1)))**2


@jax.jit
def loss_func(sound_speed_vals, measure):
     return loss0(sound_speed_vals, measure) + loss1(sound_speed_vals)


@jax.jit
def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


@jax.jit
def hessp_loss(x, v):
    return hvp(loss_func, (x,), (v,))


@jax.jit
def jac_loss(x, measure):
    return jax.grad(loss_func)(x, measure)


def get_opt_solution(measure, x0):
    m = minimize(
        method='L-BFGS-B',
        fun=loss_func,
        args=(measure,),
        x0=x0,
        jac=jac_loss,
        hessp=hessp_loss,
        # callback=lambda xk: print(f'{xk}, {loss_func(xk)}'),
    )
    return m

env_replica.layers[0].sound_speed_profile_m_s.sound_speed = replica_z_grid_m*0.0 + 1500.0 + jax.random.uniform(jax.random.PRNGKey(0), (20,))*0.2
simulated_ssp_list = []

for lower in linspace(1500.0, 1510, 41):
    env_simulated.layers[0].sound_speed_profile_m_s.sound_speed = jnp.array([1500.0, lower])
    simulated_ssp_list += [deepcopy(env_simulated.layers[0].sound_speed_profile_m_s)]

env_simulated.layers[0].sound_speed_profile_m_s.z_grid_m = jnp.array([0.0, 75, 200])
for upper in linspace(1500.0, 1510, 41):
    env_simulated.layers[0].sound_speed_profile_m_s.sound_speed = jnp.array([upper, (lower+1500.0)/2, lower])
    simulated_ssp_list += [deepcopy(env_simulated.layers[0].sound_speed_profile_m_s)]

ms = env_simulated.layers[0].sound_speed_profile_m_s(75.0)
for middle in linspace(ms, 1510, 20):
    env_simulated.layers[0].sound_speed_profile_m_s.sound_speed = jnp.array([upper, middle, lower])
    simulated_ssp_list += [deepcopy(env_simulated.layers[0].sound_speed_profile_m_s)]

inverted_ssp_list = []
nfev_list = []
njev_list = []
opt_time_list = []
ssp_error_list = []
rel_error_list = []
dz = replica_z_grid_m[1] - replica_z_grid_m[0]
for sssp_i, simulated_ssp in enumerate(simulated_ssp_list):
    env_simulated.layers[0].sound_speed_profile_m_s = simulated_ssp
    measure = get_field(simulated_model, src, env_simulated)[-1, :]

    t = time.time()
    m = get_opt_solution(measure=measure, x0=env_replica.layers[0].sound_speed_profile_m_s.sound_speed)
    opt_time_list += [time.time() - t]
    print(m)
    nfev_list += [m.nfev]
    njev_list += [m.njev]
    env_replica.layers[0].sound_speed_profile_m_s.sound_speed = m.x
    inverted_ssp_list += [deepcopy(env_replica.layers[0].sound_speed_profile_m_s)]

    d = simulated_ssp(replica_z_grid_m) - inverted_ssp_list[-1](replica_z_grid_m)
    ssp_error_list += [jnp.linalg.norm(jnp.diff(d) / dz) * jnp.sqrt(dz)]
    sim_norm = jnp.linalg.norm(jnp.diff(simulated_ssp(replica_z_grid_m)) / dz) * jnp.sqrt(dz)
    rel_error_list += [ssp_error_list[-1] / sim_norm]

    print(f'Step: {sssp_i}/{len(simulated_ssp_list)}; SSP rel. error = {rel_error_list[-1]}')


f, ax = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
for i, simulated_ssp in enumerate(simulated_ssp_list):
    ax[0].plot(simulated_ssp.sound_speed[::-1] + i, simulated_ssp.z_grid_m[::-1])
ax[0].set_xticklabels([])
ax[0].set_ylabel("Depth (m)")
ax[0].set_ylim([simulated_ssp.z_grid_m[-1], simulated_ssp.z_grid_m[0]])
ax[0].grid(True)

for i, inverted_ssp in enumerate(inverted_ssp_list):
    ax[1].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[1].set_xlabel("Sound speed (m/s)")
ax[1].set_ylabel("Depth (m)")
ax[1].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[1].grid(True)
plt.show()
#plt.savefig('ex1_ssp_dynamics.eps')


f, ax = plt.subplots(1, 1, figsize=(10, 3.2), constrained_layout=True)
for i in range(0, len(simulated_ssp_list)):
    d = simulated_ssp_list[i](replica_z_grid_m) - inverted_ssp_list[i](replica_z_grid_m)
    ax.plot(d[::-1] + 5*i, replica_z_grid_m[::-1])
ax.set_xlabel('SSP difference (m/s)')
ax.set_ylabel('Depth (m)')
ax.set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax.set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax.grid(True)
plt.show()
#plt.savefig('ex1_ssp_error_pw.eps')

plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(ssp_error_list)), ssp_error_list)
plt.xlabel('Number of iteration')
plt.xticks(range(0, len(ssp_error_list))[::2])
plt.xlim([0, len(ssp_error_list)-1])
plt.ylabel('||error||')
plt.grid(True)
plt.show()
#plt.savefig('ex1_ssp_error_norm.eps')

plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(rel_error_list)), rel_error_list)
plt.xlabel('Number of iteration')
plt.xticks(range(0, len(rel_error_list))[::2])
plt.xlim([0, len(rel_error_list)-1])
plt.ylabel('||rel. error||')
plt.grid(True)
plt.show()
#plt.savefig('ex1_ssp_rel_error_norm.eps')

plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(nfev_list)), nfev_list, label='nfev')
plt.plot(range(0, len(njev_list)), njev_list, label='njev')
plt.xlabel('Number of iteration')
plt.xticks(range(0, len(nfev_list))[::2])
plt.xlim([0, len(nfev_list)-1])
plt.ylim([0, max(nfev_list + njev_list)])
plt.ylabel('Number of evaluations')
plt.grid(True)
plt.legend()
plt.show()
#plt.savefig('ex1_n_evals.eps')

plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(opt_time_list)), opt_time_list)
plt.xlabel('Number of iteration')
plt.xticks(range(0, len(opt_time_list))[::2])
plt.xlim([0, len(opt_time_list)-1])
plt.ylim([0, max(opt_time_list)])
plt.ylabel('Time (s)')
plt.grid(True)
plt.show()
#plt.savefig('ex1_opt_time.eps')

env_vis = deepcopy(env_simulated)
vis_model = uwa_get_model(
    src=src,
    env=env_vis,
    params=ComputationalParams(
        max_range_m=20000,
        max_depth_m=210,
        x_output_points=500,
        z_output_points=500,
    )
)


env_vis.layers[0].sound_speed_profile_m_s = simulated_ssp_list[0]
f_sim_0 = get_field(vis_model, src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = simulated_ssp_list[21]
f_sim_1 = get_field(vis_model, src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = simulated_ssp_list[42]
f_sim_2 = get_field(vis_model, src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = inverted_ssp_list[0]
f_inv_0 = get_field(vis_model, src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = inverted_ssp_list[21]
f_inv_1 = get_field(vis_model, src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = inverted_ssp_list[42]
f_inv_2 = get_field(vis_model, src, env_vis)


f, ax = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)
norm = Normalize(vmin=-60, vmax=-20)
cmap = plt.get_cmap('jet')

ax[0, 0].imshow(
    20*jnp.log10(jnp.abs(f_sim_0+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[0, 0].set_xticklabels([])
ax[0, 0].set_ylabel('Depth (m)')
ax[0, 0].grid(True)

ax[0, 1].imshow(
    20*jnp.log10(jnp.abs(f_sim_1+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[0, 1].set_xticklabels([])
ax[0, 1].set_yticklabels([])
ax[0, 1].grid(True)

ax[0, 2].imshow(
    20*jnp.log10(jnp.abs(f_sim_2+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[0, 2].set_xticklabels([])
ax[0, 2].set_yticklabels([])
ax[0, 2].grid(True)

ax[1, 0].imshow(
    20*jnp.log10(jnp.abs(f_inv_0+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[1, 0].set_xlabel("Range (km)")
ax[1, 0].set_ylabel('Depth (m)')
ax[1, 0].grid(True)

ax[1, 1].imshow(
    20*jnp.log10(jnp.abs(f_inv_1+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[1, 1].set_xlabel("Range (km)")
ax[1, 1].set_yticklabels([])
ax[1, 1].grid(True)

im = ax[1, 2].imshow(
    20*jnp.log10(jnp.abs(f_inv_2+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[1, 2].set_xlabel("Range (km)")
ax[1, 2].set_yticklabels([])
ax[1, 2].grid(True)

f.colorbar(im, ax=ax[1,:], shrink=0.3, location='bottom')

plt.show()
#plt.savefig('ex1_2d.eps')