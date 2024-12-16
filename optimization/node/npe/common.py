import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Sequence

import jax
import numpy as np
import optax
from jax import numpy as jnp, tree_util

from experimental.helmholtz_jax import RationalHelmholtzPropagator, AbstractWaveSpeedModel, \
    PiecewiseLinearWaveSpeedModel, ConstWaveSpeedModel
from experimental.rwp_jax import RWPGaussSourceModel, TroposphereModel, RWPComputationalParams, create_rwp_model, \
    AbstractNProfileModel, PiecewiseLinearNProfileModel
from experimental.uwa_jax import UWAGaussSourceModel, UWAComputationalParams, uwa_get_model, UnderwaterEnvironmentModel, \
    UnderwaterLayerModel

from experiments.optimization.node.objective_functions import bartlett


@dataclass
class OptResult:
    res_profile: AbstractNProfileModel
    start_profile: AbstractNProfileModel
    loss_vals: Sequence[float]
    time_s: float
    ground_truth_errors: Sequence[float] = None


def adam(model, measure, profile_model, gamma=1E-3, learning_rate=0.002, ground_truth_profile: AbstractNProfileModel=None):
    model.apply_profile(profile_model)
    tx = optax.adam(learning_rate=learning_rate)

    opt_params = model.env.N_profile.params
    opt_params_0 = deepcopy(opt_params)
    opt_state = tx.init(opt_params)
    loss_grad_fn = jax.value_and_grad(loss)

    best_loss = np.inf
    best_params = None
    best_counter = 0

    loss_vals = []
    if ground_truth_profile is not None:
        z_grid = jnp.linspace(0, profile_model.max_height_m(), 200)
        ground_truth_errors = []
    else:
        ground_truth_errors = None

    t_profile = deepcopy(profile_model)
    t = time.time()
    for i in range(10000):
        #l0 = loss0(opt_params, model, measure)
        #l1 = loss1(opt_params, model)
        #print(f'l0 = {l0}; l1 = {l1}')
        loss_val, grads = loss_grad_fn(opt_params, model, measure, gamma)
        updates, opt_state = tx.update(grads, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

        loss_vals += [loss_val]
        if ground_truth_profile is not None:
            t_profile.params = opt_params
            ground_truth_errors += [jnp.linalg.norm(ground_truth_profile(z_grid) - t_profile(z_grid)) / jnp.linalg.norm(ground_truth_profile(z_grid))]

        if i % 25 == 0:
            print(f'i = {i}; Loss = {loss_val}')

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = deepcopy(opt_params)
            best_counter = 0
        else:
            best_counter += 1

        if best_counter > 25:
            break

    time_s = time.time() - t

    res_profile = deepcopy(profile_model)
    res_profile.params = best_params
    start_profile = deepcopy(profile_model)
    start_profile.params = opt_params_0
    return OptResult(
        loss_vals=loss_vals,
        start_profile=start_profile,
        res_profile=res_profile,
        time_s=time_s,
        ground_truth_errors=ground_truth_errors
    )


@dataclass
class RWPModel:
    measure_points_x: List[int] = None
    measure_points_z: List[int] = None
    fwd_model: RationalHelmholtzPropagator = None
    src: RWPGaussSourceModel = RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
    env: TroposphereModel = TroposphereModel()
    params: RWPComputationalParams = RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    )

    def __post_init__(self):
        self.fwd_model = create_rwp_model(self.src, self.env, self.params)

    def calc_field(self, N_profile: AbstractNProfileModel):
        self.env.N_profile = N_profile
        return self.fwd_model.compute(self.src.aperture(self.fwd_model.z_computational_grid()))

    def get_replica(self):
        f = self.fwd_model.compute(self.src.aperture(self.fwd_model.z_computational_grid()))
        return f[self.measure_points_x, self.measure_points_z]


    def apply_profile(self, N_profile: AbstractNProfileModel):
        self.env.N_profile = N_profile
        return self.get_replica()


@dataclass
class UWAModel:
    measure_points_x: List[int] = None
    measure_points_z: List[int] = None
    fwd_model: RationalHelmholtzPropagator = None
    src: UWAGaussSourceModel = UWAGaussSourceModel(freq_hz=200, depth_m=10.0, beam_width_deg=3.0)
    env: UnderwaterEnvironmentModel = UnderwaterEnvironmentModel(
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
    params: UWAComputationalParams = UWAComputationalParams(
        max_range_m=5000,
        max_depth_m=250,
        dx_m=100,
        dz_m=1
    )

    def __post_init__(self):
        self.fwd_model = uwa_get_model(self.src, self.env, self.params)

    def calc_field(self, sound_speed: AbstractWaveSpeedModel):
        self.env.layers[0].sound_speed_profile_m_s = sound_speed
        c0 = self.env.layers[0].sound_speed_profile_m_s(jnp.array([self.src.depth_m]))[0]
        k0 = 2 * jnp.pi * self.src.freq_hz / c0
        return self.fwd_model.compute(self.src.aperture(k0, self.fwd_model.z_computational_grid()))

    def get_replica(self):
        c0 = self.env.layers[0].sound_speed_profile_m_s(jnp.array([self.src.depth_m]))[0]
        k0 = 2 * jnp.pi * self.src.freq_hz / c0
        f = self.fwd_model.compute(self.src.aperture(k0, self.fwd_model.z_computational_grid()))
        return f[self.measure_points_x, self.measure_points_z]


    def apply_profile(self, sound_speed: AbstractWaveSpeedModel):
        self.env.layers[0].sound_speed_profile_m_s = sound_speed
        return self.get_replica()


def Bartlett_loss(val, measure):
    etalon_loss0_v = 1 / bartlett(measure, measure)
    f = jnp.ravel(val)
    return 1 / bartlett(measure, f) - etalon_loss0_v


def add_noise(measure, snr, key=jax.random.PRNGKey(1703)):
    signal_level = jnp.mean(abs(measure) ** 2)
    noise_var = signal_level / (10 ** (snr / 10))
    noise_sigma_r = jnp.sqrt(noise_var / 2)
    r_t = jax.random.normal(key, (len(measure), 2)) * noise_sigma_r
    noise = r_t[:, 0] + 1j * r_t[:, 1]
    measure += noise
    return measure


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def operator(params, model: RWPModel):
    model.env.N_profile.params = params
    return model.get_replica()


def loss0(params, model: RWPModel, measure):
    val = operator(params, model)
    return Bartlett_loss(val, measure)


def loss1(params, model: RWPModel):
    model.env.N_profile.params = params
    z_grid_m = jnp.linspace(0, model.env.max_height_m()+20, 201)
    return jnp.linalg.norm(jnp.diff(model.env.N_profile(z_grid_m))) ** 2


def loss(params, model: RWPModel, measure, gamma):
    return loss0(params, model, measure) + gamma*loss1(params, model)


class PLNPM(PiecewiseLinearNProfileModel):

    @property
    def params(self):
        return self.N_vals

    @params.setter
    def params(self, value):
        self.N_vals = value

    def _tree_flatten(self):
        dynamic = (self.z_grid_m, self.N_vals)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(z_grid_m=dynamic[0], N_vals=dynamic[1])


tree_util.register_pytree_node(PLNPM,
                               PLNPM._tree_flatten,
                               PLNPM._tree_unflatten)

surface_duct_N = PiecewiseLinearNProfileModel(jnp.array([0, 50, 100]), jnp.array([20.0, 0, 0]))
elevated_duct_N = PiecewiseLinearNProfileModel(jnp.array([0, 100, 150, 300]), jnp.array([20, 20, 0, 0]))
surface_based_duct_N = PiecewiseLinearNProfileModel(jnp.array([0, 50, 75, 100]), jnp.array([10.0, 30, 0, 0]))
surface_based_duct2_N = PiecewiseLinearNProfileModel(jnp.array([0, 50, 120, 100]), jnp.array([30.0, 30, 0, 0]))
