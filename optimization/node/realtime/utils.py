import math as fm
import time
from copy import deepcopy
from typing import List

import jax
from attr import dataclass
from jax import numpy as jnp
from scipy.optimize import minimize

from experimental.helmholtz_jax import RationalHelmholtzPropagator, AbstractWaveSpeedModel, \
    PiecewiseLinearWaveSpeedModel, ConstWaveSpeedModel
from experiments.optimization.node.objective_functions import bartlett
from experimental.uwa_jax import UWAGaussSourceModel, UnderwaterEnvironmentModel, UnderwaterLayerModel, \
    UWAComputationalParams, uwa_get_model


jax.config.update("jax_enable_x64", True)


def get_field(model: RationalHelmholtzPropagator, src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel):
    c0 = env.layers[0].sound_speed_profile_m_s(src.depth_m)
    k0 = 2 * fm.pi * src.freq_hz / c0
    init = src.aperture(k0, model.z_computational_grid())
    return model.compute(init)


@dataclass
class RealtimeInversionModelResult:
    inverted_ssp_list: List[AbstractWaveSpeedModel]
    abs_error_list: List[float]
    rel_error_list: List[float]
    opt_time_list: List[float]
    nfev_list: List[float]
    njev_list: List[float]
    env: UnderwaterEnvironmentModel
    src: UWAGaussSourceModel


def realtime_inversion_model(freq_hz, range_to_vla_m, simulated_ssp_list, snr=None, gamma=1.0, arrays_num=1) -> RealtimeInversionModelResult:
    src = UWAGaussSourceModel(
        freq_hz=freq_hz,
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

    max_depth_m = 250
    computational_params = UWAComputationalParams(
        max_range_m=range_to_vla_m,
        max_depth_m=max_depth_m,
        dx_m=range_to_vla_m/5,
        dz_m=max_depth_m/100,
    )

    measure_points_depth = jnp.array([5, 10, 20, 30, 40, 50, 60, 70])
    measure_points_range = -jnp.arange(1, arrays_num+1, 1)

    simulated_model = uwa_get_model(
        src=src,
        env=env_simulated,
        params=computational_params
    )

    replica_z_grid_m = jnp.linspace(0, 200, 20)
    env_replica = deepcopy(env_simulated)
    env_replica.layers[0].sound_speed_profile_m_s.z_grid_m = replica_z_grid_m
    env_replica.layers[0].sound_speed_profile_m_s.sound_speed = env_simulated.layers[0].sound_speed_profile_m_s(replica_z_grid_m)+jax.random.uniform(jax.random.PRNGKey(0))

    training_model = uwa_get_model(
        src=src,
        env=env_replica,
        params=computational_params
    )

    model_points_num = len(env_replica.layers[0].sound_speed_profile_m_s.z_grid_m)

    def loss0(sound_speed_vals, measure):
        etalon_loss0_v = 1 / bartlett(measure[measure_points_depth], measure[measure_points_depth])
        env_replica.layers[0].sound_speed_profile_m_s.sound_speed = sound_speed_vals
        f = get_field(training_model, src, env_replica)[measure_points_range, :]
        f = jnp.ravel(f)
        return 1 / bartlett(measure[measure_points_depth], f[measure_points_depth]) - etalon_loss0_v


    def loss1(sound_speed_vals):
        return jnp.linalg.norm(jnp.diff(sound_speed_vals) / (env_replica.layers[0].height_m / (model_points_num - 1)))**2


    @jax.jit
    def loss_func(sound_speed_vals, measure):
         return loss0(sound_speed_vals, measure) + gamma*loss1(sound_speed_vals)


    @jax.jit
    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f), primals, tangents)[1]


    @jax.jit
    def jac_loss(x, measure):
        return jax.grad(loss_func)(x, measure)


    def get_opt_solution(measure, x0):
        return minimize(
            method='L-BFGS-B',
            fun=loss_func,
            args=(measure,),
            x0=x0,
            jac=jac_loss,
        )

    env_replica.layers[0].sound_speed_profile_m_s.sound_speed = simulated_ssp_list[0](replica_z_grid_m) + jax.random.uniform(jax.random.PRNGKey(0), (20,))*0.01

    inverted_ssp_list = []
    nfev_list = []
    njev_list = []
    opt_time_list = []
    abs_error_list = []
    rel_error_list = []
    dz = replica_z_grid_m[1] - replica_z_grid_m[0]
    for sssp_i, simulated_ssp in enumerate(simulated_ssp_list):
        env_simulated.layers[0].sound_speed_profile_m_s = simulated_ssp
        measure = get_field(simulated_model, src, env_simulated)[measure_points_range, :]
        measure = jnp.ravel(measure)

        if snr:
            signal_level = jnp.mean(abs(measure)**2)
            noise_var = signal_level / (10 ** (snr / 10))
            noise_sigma_r = jnp.sqrt(noise_var / 2)
            r_t = jax.random.normal(jax.random.PRNGKey(1703), (len(measure), 2)) * noise_sigma_r
            noise = r_t[:, 0] + 1j*r_t[:, 1]
            measure += noise

        t = time.time()
        x0 = env_replica.layers[0].sound_speed_profile_m_s.sound_speed
        x0 += simulated_ssp_list[0](0) - x0[0]
        m = get_opt_solution(measure=measure, x0=x0)
        opt_time_list += [time.time() - t]
        print(m)
        nfev_list += [m.nfev]
        njev_list += [m.njev]
        env_replica.layers[0].sound_speed_profile_m_s.sound_speed = m.x
        inverted_ssp_list += [deepcopy(env_replica.layers[0].sound_speed_profile_m_s)]

        d = simulated_ssp(replica_z_grid_m) - inverted_ssp_list[-1](replica_z_grid_m)
        ssp_error = jnp.linalg.norm(jnp.diff(d) / dz) * jnp.sqrt(dz)
        abs_error_list += [ssp_error]
        sim_norm = jnp.linalg.norm(jnp.diff(simulated_ssp(replica_z_grid_m)) / dz) * jnp.sqrt(dz)
        rel_error_list += [ssp_error / sim_norm]

        print(f'Step: {sssp_i}/{len(simulated_ssp_list)}; SSP abs. error = {abs_error_list[-1]:.2f}; SSP rel. error = {rel_error_list[-1]:.2f}')

    return RealtimeInversionModelResult(
        inverted_ssp_list=inverted_ssp_list,
        abs_error_list=abs_error_list,
        rel_error_list=rel_error_list,
        opt_time_list=opt_time_list,
        nfev_list=nfev_list,
        njev_list=njev_list,
        env=deepcopy(env_simulated),
        src=deepcopy(src)
    )
